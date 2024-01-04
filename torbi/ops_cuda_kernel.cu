#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>

#include <vector>

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
    const int threads = 1024;
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
    float max_posterior;
    int states2 = states*states;

    // Set initial
    // auto start = std::chrono::high_resolution_clock::now();
    for (int i=threadIdx.x; i<states; i+=blockDim.x) {
        posterior[i] = observation[i] + initial[i];
    }
    __syncthreads();
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "Time taken by init: " << static_cast<double>(duration.count()) / 1e6 << " seconds" << std::endl;

    for (int t=1; t<frames; t++) {
        // Forward Pass
        for (int i=threadIdx.x; i<states2; i+=blockDim.x) {
            // note that i = s2*frames+s1
            int s1 = i % states;
            probability[i] = posterior[(t-1)*states+s1] + transition[i];
        }
        __syncthreads();

        // Get optimal
        for (int j=threadIdx.x; j<states; j+=blockDim.x) {
            max_posterior = probability[j*states];
            float best_state = 0;

            for (int s3=1; s3<states; s3++) {
                if (probability[j*states+s3] > max_posterior) {
                    max_posterior = probability[j*states+s3];
                    // memory[t*states+j] = s3;
                    best_state = s3;
                }
            }

            memory[t*states+j] = best_state;

            posterior[t*states+j] = observation[t*states+j] + max_posterior;
        }
        __syncthreads();
    }
}