
#include <torch/extension.h>
#include "viterbi.h"
#include "ATen/ATen.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

/******************************************************************************
C++ MPS implementation of Viterbi decoding
******************************************************************************/

/// C++ MPS implementation of first step of Viterbi decoding: making the
/// trellis matrix
///
/// Args:
///   observation: :math:`(N, T, S)` or :math:`(T, S)`
///     where `S = the number of states`,
///     `T = the length of the sequence`,
///     and `N = batch size`.
///     Time-varying categorical distribution
///   batch_frames :math:`(N)`
///     Sequence length of each batch item
///   transition :math:`(S, S)`
///     Categorical transition matrix
///   initial :math:`(S)`
///     Categorical initial distribution
///
/// Modifies:
///   posterior: :math:`(N, S)`
///     Minimum path costs
///   trellis: :math:`(N, T, S)`
///     Matrix of minimum path indices for backtracing
void viterbi_make_trellis_mps(
    const torch::Tensor& observation,
    const torch::Tensor& batch_frames,
    const torch::Tensor& transition,
    const torch::Tensor& initial,
    torch::Tensor& posterior,
    torch::Tensor& trellis_tensor,
    torch::Tensor& probability
) {

    int batch_size = observation.size(0);
    int max_frames = observation.size(1);
    int states = observation.size(2);

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Load the custom viterbi shader.
        id<MTLLibrary> kernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:viterbi_mps_lib.c_str()]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(kernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string kernel_name = std::string("viterbi_make_trellis_kernel");
        id<MTLFunction> viterbiDecodeFunction = [kernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(viterbiDecodeFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object for the viterbi decoding kernel.
        id<MTLComputePipelineState> viterbiPSO = [device newComputePipelineStateWithFunction:viterbiDecodeFunction error:&error];
        TORCH_CHECK(viterbiPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            // ensure the commandBuffer is free
            torch::mps::commit();

            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:viterbiPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(observation) offset:observation.storage_offset() * observation.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(batch_frames) offset:batch_frames.storage_offset() * batch_frames.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(transition) offset:transition.storage_offset() * transition.element_size() atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(initial) offset:initial.storage_offset() * initial.element_size() atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(posterior) offset:posterior.storage_offset() * posterior.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(trellis_tensor) offset:trellis_tensor.storage_offset() * trellis_tensor.element_size() atIndex:5];
            [computeEncoder setBytes:&max_frames length:sizeof(int) atIndex:6];
            [computeEncoder setBytes:&states length:sizeof(int) atIndex:7];

            // allocate shared memory posterior_cache object
            NSUInteger posteriorCacheBytes = 2 * states * sizeof(float);
            [computeEncoder setThreadgroupMemoryLength:posteriorCacheBytes atIndex:0];


            MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadGroupSize = viterbiPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > states) {
                threadGroupSize = states;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreadgroups:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }

}

/// C++ MPS implementation of the second step of Viterbi decoding: backtracing
/// the trellis to find the maximum likelihood path
///
/// Args:
///   trellis: :math:`(N, T, S)`
///     Matrix of minimum path indices for backtracing
///   batch_frames :math:`(N)`
///     Sequence length of each batch item
///   batch_size
///     Number of observation sequences in the batch
///   max_frames
///     Maximum number of frames of any observation sequence in the batch
///   states
///     Number of categories in the categorical distribution being decoded
///
/// Modifies:
///   indices: :math:`(N, T)`
///     The decoded bin indices
void viterbi_backtrace_trellis_mps(
    const torch::Tensor& indices,
    const torch::Tensor& trellis,
    const torch::Tensor& batch_frames,
    int batch_size,
    int max_frames,
    int states
) {
   @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Load the custom viterbi shader.
        id<MTLLibrary> kernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:viterbi_mps_lib.c_str()]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(kernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string kernel_name = std::string("viterbi_backtrace_trellis_kernel");
        id<MTLFunction> viterbiDecodeFunction = [kernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(viterbiDecodeFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object for the viterbi decoding kernel.
        id<MTLComputePipelineState> viterbiPSO = [device newComputePipelineStateWithFunction:viterbiDecodeFunction error:&error];
        TORCH_CHECK(viterbiPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            // ensure the commandBuffer is free
            torch::mps::commit();

            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:viterbiPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(indices) offset:indices.storage_offset() * indices.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(trellis) offset:trellis.storage_offset() * trellis.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(batch_frames) offset:batch_frames.storage_offset() * batch_frames.element_size() atIndex:2];
            [computeEncoder setBytes:&max_frames length:sizeof(int) atIndex:3];
            [computeEncoder setBytes:&states length:sizeof(int) atIndex:4];

            MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadGroupSize = viterbiPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > batch_size) {
                threadGroupSize = batch_size;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }
}

// TODO remove
void viterbi_backtrace_trellis_cpu(
    int *indices,
    int *trellis,
    int *batch_frames,
    int batch_size,
    int max_frames,
    int states) {
    // #pragma omp parallel for
    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end){
        for (int b = begin; b < end; b++) {
            int *indices_b = indices + max_frames * b;
            int *trellis_b = trellis + max_frames * states * b;
            int frames = batch_frames[b];
            int index = indices_b[frames - 1];
            for (int t = frames - 1; t >= 1; t--) {
                index = trellis_b[t * states + index];
                indices_b[t - 1] = index;
            }
        }
    });
}

// C++ op dispatching the Metal viterbi shader.
torch::Tensor viterbi_decode_mps(
    const torch::Tensor observation,
    const torch::Tensor batch_frames,
    const torch::Tensor transition,
    const torch::Tensor initial) {

    int batch_size = observation.size(0);
    int max_frames = observation.size(1);
    int states = observation.size(2);

    TORCH_CHECK(observation.device().is_mps(), "observation must be an MPS tensor");
    TORCH_CHECK(batch_frames.device().is_mps(), "batch_frames must be an MPS tensor");
    TORCH_CHECK(transition.device().is_mps(), "transition must be an MPS tensor");
    TORCH_CHECK(initial.device().is_mps(), "initial must be an MPS tensor");

    torch::Tensor observation_contig = observation.contiguous();
    torch::Tensor batch_frames_contig = batch_frames.contiguous();
    torch::Tensor transition_contig = transition.contiguous();
    torch::Tensor initial_contig = initial.contiguous();

    // Intermediate storage for path indices and costs
    torch::Tensor trellis = torch::zeros(
        {batch_size, max_frames, states},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS));
    torch::Tensor posterior = torch::zeros(
        {batch_size, states},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

    torch::Tensor probability = torch::zeros(
        {states*states},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

    // First step: make the minimum cost path trellis
    viterbi_make_trellis_mps(
        observation_contig,
        batch_frames_contig,
        transition_contig,
        initial_contig,
        posterior,
        trellis,
        probability);

    // return posterior;

    torch::Tensor indices = posterior.argmax(1);
    indices = indices.unsqueeze(1);
    indices = indices.repeat({1, max_frames});
    indices = indices.to(torch::kInt32);

    // torch::Tensor indices_cpu = indices.cpu();
    // torch::Tensor trellis_cpu = trellis.cpu();
    // torch::Tensor batch_frames_contig_cpu = batch_frames_contig.cpu();

    // viterbi_backtrace_trellis_cpu(
    //     indices_cpu.data_ptr<int>(),
    //     trellis_cpu.data_ptr<int>(),
    //     batch_frames_contig_cpu.data_ptr<int>(),
    //     batch_size,
    //     max_frames,
    //     states);

    // Second step: backtrace trellis to find maximum likelihood path
    // omp_set_num_threads(num_threads);
    viterbi_backtrace_trellis_mps(
        indices,
        trellis,
        batch_frames_contig,
        batch_size,
        max_frames,
        states);

    return indices;
}

TORCH_LIBRARY_IMPL(torbi, MPS, m) {
    m.impl("viterbi_decode", &viterbi_decode_mps);
}
