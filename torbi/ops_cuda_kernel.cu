#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>

#include <vector>

#define NUM_THREADS 1024
#define WARP_SIZE 32
#define NUM_WARPS 32

#define FULL_MASK 0xffffffff

__global__ void viterbi_forward_kernel(
    float* __restrict__ observation,
    float* __restrict__ transition,
    float* __restrict__ initial,
    float* __restrict__ posterior,
    int* __restrict__ memory,
    int frames,
    int states
) {
    // // save to avoid re-computation
    // // the id of the current thread
    // int thread_id = threadIdx.x;
    // the id of the warp to which this thread belongs
    int warp_id = threadIdx.x / WARP_SIZE;
    int thread_warp_id = threadIdx.x % WARP_SIZE;

    extern __shared__ float posterior_cache[];

    float *posterior_current = posterior_cache;
    float *posterior_next = posterior_cache+states;

    // Set initial
    for (int i=threadIdx.x; i<states; i+=NUM_THREADS) {
        // posterior[i] = observation[i] + initial[i];
        posterior_current[i] = observation[i] + initial[i];
    }
    __syncthreads();

    for (int t=1; t<frames; t++) {
        // Get optimal
        // Iterate rows by warp (each warp gets assigned a row)
        int max_index;
        float max_value;
        for (int j=warp_id; j<states; j+=NUM_WARPS) {
            // __syncthreads();
            
            // Indices start out as just 0-WARP_SIZE for the first WARP_SIZE elements in the array
            max_index = thread_warp_id;
            // Values start as the first WARP_SIZE elements in the row, with row selected by j
            // max_value = posterior[(t-1)*states+thread_warp_id] + transition[j*states+thread_warp_id];
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

            // This is a first attempt at a parallel reduction
            for (int offset=WARP_SIZE/2; offset>0; offset/=2) {
                float new_value = __shfl_down_sync(FULL_MASK, max_value, offset);
                int new_index = __shfl_down_sync(FULL_MASK, max_index, offset);
                if (new_value > max_value) {
                    max_value = new_value;
                    max_index = new_index;
                }
            }
            if (thread_warp_id == 0) {
                memory[t*states+j] = max_index;
                // posterior[t*states+j] = observation[t*states+j] + max_value;
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
        posterior[(frames-1)*states+i] = posterior_current[i];
    }
    __syncthreads();
}


void viterbi_cuda_forward(
    torch::Tensor observation,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor memory,
    int frames,
    int states
) {
    const int threads = NUM_THREADS;
    const dim3 blocks(1);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    viterbi_forward_kernel<<<blocks, threads, 2*states*sizeof(float)>>>(
        observation.data<float>(),
        transition.data<float>(),
        initial.data<float>(),
        posterior.data<float>(),
        memory.data<int>(),
        frames,
        states
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "CUDA kernel elapsed time: " << milliseconds << " ms" << std::endl;
}