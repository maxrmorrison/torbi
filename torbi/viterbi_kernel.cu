#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>

#include <vector>

#define NUM_THREADS 1024
#define WARP_SIZE 32
#define NUM_WARPS 32

#define FULL_MASK 0xffffffff

// Is this a perfect kernel? Maybe not. Does it work? Yes. Yes it does.
// Each thread block processes one input sequence at a time.
// The kernel loops over timesteps and then states, with one entire warp assigned to each
// state. Each warp computes part of the posterior distribution and then performs
// a parallel argmax using warp shift operations to find current next best state
// from the starting state (the state the warp is assigned to).
// Some fun pointer tricks save us from having to store the entire posterior
// distribution.
__global__ void viterbi_make_trellis_kernel(
    float* __restrict__ observation, // BATCH x FRAMES x STATES
    int* __restrict__ batch_frames, // BATCH
    float* __restrict__ transition, // STATES x STATES
    float* __restrict__ initial, // STATES
    float* __restrict__ posterior, // BATCH x STATES
    int* __restrict__ memory, // BATCH x FRAMES x STATES
    int max_frames,
    int states
) {

    // Handle batch
    int batch_id = blockIdx.x;
    int frames = batch_frames[batch_id]; // Get number of frames for this batch item
    observation += batch_id * max_frames * states;
    posterior += batch_id * states;
    memory += batch_id * max_frames * states;

    // The id of the warp to which this thread belongs
    int warp_id = threadIdx.x / WARP_SIZE;
    // The id of this thread within its warp
    int thread_warp_id = threadIdx.x % WARP_SIZE;

    extern __shared__ float posterior_cache[];

    float *posterior_current = posterior_cache;
    float *posterior_next = posterior_cache+states;

    // Set initial
    for (int i=threadIdx.x; i<states; i+=NUM_THREADS) {
        posterior_current[i] = observation[i] + initial[i];
    }
    __syncthreads();

    for (int t=1; t<frames; t++) {
        // Get optimal
        // Iterate rows by warp (each warp gets assigned a row)
        int max_index;
        float max_value;
        for (int j=warp_id; j<states; j+=NUM_WARPS) {
            // Indices start out as just 0-WARP_SIZE for the first WARP_SIZE elements in the array
            max_index = thread_warp_id;
            // Values start as the first WARP_SIZE elements in the row, with row selected by j
            max_value = posterior_current[thread_warp_id] + transition[j*states+thread_warp_id];

            // Slide the warp over the row in a linear argmax search (parallelized by threads within the warp)
            // Note that we start here offset by the WARP_SIZE since we already initialized using the first chunk
            for (int i=thread_warp_id+WARP_SIZE; i<states; i+=WARP_SIZE) {
                // Get the new value from the current row at the current offset
                float new_value = posterior_current[i] + transition[j*states + i];
                if (new_value > max_value) {
                    max_index = i;
                    max_value = new_value;
                }
            }
            __syncwarp();

            // Parallel reduction
            for (int offset=WARP_SIZE/2; offset>0; offset/=2) {
                float new_value = __shfl_down_sync(FULL_MASK, max_value, offset);
                int new_index = __shfl_down_sync(FULL_MASK, max_index, offset);
                if (new_value > max_value) {
                    max_value = new_value;
                    max_index = new_index;
                }
            }
            if (thread_warp_id == 0) {
                memory[(t)*states+j] = max_index;
                posterior_next[j] = observation[t*states+j] + max_value;
            }
        }
        float *posterior_last = posterior_current;
        posterior_current = posterior_next;
        posterior_next = posterior_last;
        __syncthreads();
    }

    // Write final posterior row
    for (int i=threadIdx.x; i<states; i+=NUM_THREADS) {
        posterior[i] = posterior_current[i];
    }
    __syncthreads();
}

// Trace back through the trellis to find sequence with maximal
// probability.
// indices, memory, and batch frames are all data pointers
// to the batch_frames tensor and the indices and memory tensors
// created by the viterbi_make_trellis_kernel (see above).
// The loop in this kernel only executes a few times so the
// uncoalesced memory accesses are negligible.
__global__ void viterbi_backtrace_trellis_kernel(
    int *indices,
    int *memory,
    int *batch_frames,
    int batch_size,
    int max_frames,
    int states
) {
    int global_thread_id = blockIdx.x * NUM_THREADS + threadIdx.x;
    int b = global_thread_id;
    if (b < batch_size) {
        int *indices_b = indices + max_frames * b;
        int *memory_b = memory + max_frames * states * b;
        int frames = batch_frames[b];
        int index = indices_b[frames-1];
        for (int t=frames-1; t>=1; t--) {
            index = memory_b[t*states + index];
            indices_b[t-1] = index;
        }
    }
}

// Create trellis by working forward to determine most likely next states.
// The resulting graph is stored in the memory_tensor, and the final posterior
// distribution is stored in posterior_tensor.
// observation: BATCH x FRAMES x STATES
// batch_frames: BATCH
// transition: STATES x STATES
// initial: STATES
// posterior: BATCH x STATES
// memory: BATCH x FRAMES x STATES
void viterbi_make_trellis_cuda(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor memory
) {
    const int threads = NUM_THREADS;

    int batch_size = observation.size(0);
    int max_frames = observation.size(1);
    int states = observation.size(2);

    const dim3 blocks(batch_size);

    int device_num = observation.device().index();
    cudaSetDevice(device_num);

    viterbi_make_trellis_kernel<<<blocks, threads, 2*states*sizeof(float)>>>(
        observation.data_ptr<float>(),
        batch_frames.data_ptr<int>(),
        transition.data_ptr<float>(),
        initial.data_ptr<float>(),
        posterior.data_ptr<float>(),
        memory.data_ptr<int>(),
        max_frames,
        states
    );
}

// Trace back through the trellis to find sequence with maximal
// probability.
// indices, memory, and batch frames are all data pointers
// to the batch_frames tensor and the indices and memory tensors
// created by the viterbi_make_trellis_kernel (see above).
void viterbi_backtrace_trellis_cuda(
    int *indices,
    int *memory,
    int *batch_frames,
    int batch_size,
    int max_frames,
    int states
) {
    const int threads = NUM_THREADS;

    int num_blocks = (batch_size + NUM_THREADS) / NUM_THREADS;

    const dim3 blocks(num_blocks);

    viterbi_backtrace_trellis_kernel<<<blocks, threads>>>(
        indices,
        memory,
        batch_frames,
        batch_size,
        max_frames,
        states
    );
}
