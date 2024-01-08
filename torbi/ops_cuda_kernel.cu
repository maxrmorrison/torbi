#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>

#include <vector>

#define NUM_THREADS 1024
#define WARP_SIZE 32
#define NUM_WARPS 32

__global__ void viterbi_forward_kernel(
    float* __restrict__ observation,
    float* __restrict__ transition,
    float* __restrict__ initial,
    float* __restrict__ posterior,
    int* __restrict__ memory,
    float* __restrict__ probability,
    int frames,
    int states
);


void viterbi_cuda_forward(
    torch::Tensor observation,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor memory,
    torch::Tensor probability,
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
    viterbi_forward_kernel<<<blocks, threads>>>(
        observation.data<float>(),
        transition.data<float>(),
        initial.data<float>(),
        posterior.data<float>(),
        memory.data<int>(),
        probability.data<float>(),
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

__global__ void viterbi_forward_kernel(
    float* __restrict__ observation,
    float* __restrict__ transition,
    float* __restrict__ initial,
    float* __restrict__ posterior,
    int* __restrict__ memory,
    float* __restrict__ probability,
    int frames,
    int states
) {
    // save to avoid re-computation
    int states2 = states*states;
    // the id of the current thread
    int thread_id = threadIdx.x;
    // the id of the warp to which this thread belongs
    int warp_id = thread_id / WARP_SIZE;
    int thread_warp_id = threadIdx.x % WARP_SIZE;

    // used in the parallel argmax operations
    __shared__ float argmax_values[NUM_THREADS];
    __shared__ int argmax_indices[NUM_THREADS];

    // Set initial
    for (int i=thread_id; i<states; i+=NUM_THREADS) {
        posterior[i] = observation[i] + initial[i];
    }
    __syncthreads();

    for (int t=1; t<frames; t++) {
        // Forward Pass
        for (int i=thread_id; i<states2; i+=NUM_THREADS) {
            int s1 = i % states;
            probability[i] = posterior[(t-1)*states+s1] + transition[i];
        }
        __syncthreads();

        // Get optimal
        // for (int j=thread_id; j<states; j+=NUM_THREADS) {
            // max_posterior = probability[j*states];
            // float best_state = 0;

            // for (int s3=1; s3<states; s3++) {
            //     if (probability[j*states+s3] > max_posterior) {
            //         max_posterior = probability[j*states+s3];
            //         // memory[t*states+j] = s3;
            //         best_state = s3;
            //     }
            // }

            // memory[t*states+j] = best_state;

            // posterior[t*states+j] = observation[t*states+j] + max_posterior;
        // }

        // Iterate rows by warp (each warp gets assigned a row)
        int max_index;
        float max_value;
        for (int j=warp_id; j<states; j+=NUM_WARPS) {
            
            // Indices start out as just 0-WARP_SIZE for the first WARP_SIZE elements in the array
            max_index = thread_warp_id;
            // Values start as the first WARP_SIZE elements in the row, with row selected by j
            max_value = probability[j*states+thread_warp_id];

            // Slide the warp over the row in a linear argmax search (parallelized by threads within the warp)
            // Note that we start here offset by the WARP_SIZE since we already initialized using the first chunk
            for (int i=thread_warp_id+WARP_SIZE; i<states; i+=WARP_SIZE) {
                // Get the new value from the current row at the current offset
                float new_value = probability[j*states+i];
                if (new_value > max_value) {
                    max_index = i;
                    max_value = new_value;
                }
            }

            argmax_indices[thread_id] = max_index;
            argmax_values[thread_id] = max_value;
            __syncwarp();

            // Perform reduction
            // This is the worst possible way to do this, I just hope it works.
            // if (thread_warp_id == 0) {
            //     int max_index = argmax_indices[warp_id*WARP_SIZE];
            //     float max_value = argmax_values[warp_id*WARP_SIZE];
            //     for (int i=warp_id*WARP_SIZE+1; i<(warp_id+1)*WARP_SIZE; i++) {
            //         float new_value = argmax_values[i];
            //         if (new_value > max_value) {
            //             max_value = new_value;
            //             max_index = argmax_indices[i];
            //         }
            //     }

            //     memory[t*states+j] = max_index;

            //     posterior[t*states+j] = observation[t*states+j] + max_value;
            // }
            // __syncwarp();

            // This is a first attempt at a parallel reduction
            //I don't think this pragma makes any difference here
            #pragma unroll
            for (int offset=WARP_SIZE/2; offset>0; offset/=2) {
                if (thread_warp_id < offset) {
                    float value0 = argmax_values[thread_id];
                    float value1 = argmax_values[thread_id+offset];
                    int index1 = argmax_indices[thread_id+offset];
                    if (value0 >= value1) {
                        argmax_values[thread_id] = value0;
                    } else {
                        argmax_values[thread_id] = value1;
                        argmax_indices[thread_id] = index1;
                    }
                }
                __syncwarp();
            }
            if (thread_warp_id == 0) {
                memory[t*states+j] = argmax_indices[thread_id];
                posterior[t*states+j] = observation[t*states+j] + argmax_values[thread_id];
            }
        }
        __syncthreads(); 
    }
}