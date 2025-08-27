#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>

#include <vector>

#define FULL_MASK 0xffffffff

namespace torbi {

/******************************************************************************
Viterbi decoding CUDA kernels
******************************************************************************/

/// CUDA kernel for first step of Viterbi decoding: making the trellis matrix
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
///   max_frames
///     Maximum number of frames of any observation sequence in the batch
///   states
///     Number of categories in the categorical distribution being decoded
///
/// Modifies:
///   posterior: :math:`(N, S)`
///     Minimum path costs
///   trellis: :math:`(N, T, S)`
///     Matrix of minimum path indices for backtracing
///
/// Kernel description:
///   We parallelize over the batch dimension by concurrently utilizing
///   multiple GPU thread blocks. We loop over timesteps and then states; we
///   assign one warp to each state. Each warp computes part of the posterior
///   distribution and then performs a parallel argmax using warp shift
///   operations to find current next best state from the starting state (the
///   state the warp is assigned to).
__global__ void viterbi_make_trellis_kernel(
    float *__restrict__ observation,
    int *__restrict__ batch_frames,
    float *__restrict__ transition,
    float *__restrict__ initial,
    float *__restrict__ posterior,
    int *__restrict__ trellis,
    int max_frames,
    int states) {
    // Handle batch
    int batch_id = blockIdx.x;
    int frames = batch_frames[batch_id];  // Get number of frames for this batch item
    observation += batch_id * max_frames * states;
    posterior += batch_id * states;
    trellis += batch_id * max_frames * states;

    // The id of the warp to which this thread belongs
    int warp_id = threadIdx.x / warpSize;
    // The id of this thread within its warp
    int thread_warp_id = threadIdx.x % warpSize;

    extern __shared__ float posterior_cache[];

    float *posterior_current = posterior_cache;
    float *posterior_next = posterior_cache + states;

    // Set initial
    for (int i = threadIdx.x; i < states; i += blockDim.x) {
        posterior_current[i] = observation[i] + initial[i];
    }
    __syncthreads();

    const int num_warps = blockDim.x / warpSize;

    for (int t = 1; t < frames; t++) {
        // Get optimal
        // Iterate rows by warp (each warp gets assigned a row)
        int max_index;
        float max_value;
        for (int j = warp_id; j < states; j += num_warps) {
            // Indices start out as just 0-warpSize for the first warpSize elements in the array
            max_index = thread_warp_id;
            // Values start as the first warpSize elements in the row, with row selected by j
            max_value = posterior_current[thread_warp_id] + transition[j * states + thread_warp_id];

            // Slide the warp over the row in a linear argmax search (parallelized by threads within the warp)
            // Note that we start here offset by the warpSize since we already initialized using the first chunk
            for (int i = thread_warp_id + warpSize; i < states; i += warpSize) {
                // Get the new value from the current row at the current offset
                float new_value = posterior_current[i] + transition[j * states + i];
                if (new_value > max_value) {
                    max_index = i;
                    max_value = new_value;
                }
            }
            __syncwarp();

            // Parallel reduction
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                float new_value = __shfl_down_sync(FULL_MASK, max_value, offset);
                int new_index = __shfl_down_sync(FULL_MASK, max_index, offset);
                if (new_value > max_value) {
                    max_value = new_value;
                    max_index = new_index;
                }
            }
            if (thread_warp_id == 0) {
                trellis[(t)*states + j] = max_index;
                posterior_next[j] = observation[t * states + j] + max_value;
            }
        }
        float *posterior_last = posterior_current;
        posterior_current = posterior_next;
        posterior_next = posterior_last;
        __syncthreads();
    }

    // Write final posterior row
    for (int i = threadIdx.x; i < states; i += blockDim.x) {
        posterior[i] = posterior_current[i];
    }
    __syncthreads();
}

/// CUDA kernel for second step of Viterbi decoding: backtracing the trellis
/// to find the maximum likelihood path
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
__global__ void viterbi_backtrace_trellis_kernel(
    int *indices,
    int *trellis,
    int *batch_frames,
    int batch_size,
    int max_frames,
    int states) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int b = global_thread_id;
    if (b < batch_size) {
        // Get location to store maximum likelihood path
        int *indices_b = indices + max_frames * b;

        // Get trellis to backtrace
        int *trellis_b = trellis + max_frames * states * b;

        // Get number of frames
        int frames = batch_frames[b];

        // Backtrace
        int index = indices_b[frames - 1];
        for (int t = frames - 1; t >= 1; t--) {
            index = trellis_b[t * states + index];
            indices_b[t - 1] = index;
        }
    }
}

/******************************************************************************
C++ API for accessing Viterbi decoding CUDA kernels
******************************************************************************/

/// C++ API for first step of Viterbi decoding: making the trellis matrix
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
///   posterior
///     Overwritten with the minimum cost path matrix
///     shape=(batch, states)
///   trellis: :math:`(N, T, S)`
///     Matrix of minimum path indices for backtracing
void viterbi_make_trellis_cuda(
    const at::Tensor observation,
    const at::Tensor batch_frames,
    const at::Tensor transition,
    const at::Tensor initial,
    at::Tensor posterior,
    at::Tensor trellis)
{
    int batch_size = observation.size(0);
    int max_frames = observation.size(1);
    int states = observation.size(2);

    int device_num = observation.device().index();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_num);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int deviceWarpSize = prop.warpSize;
    int maxWarpsPerBlock = maxThreadsPerBlock / deviceWarpSize;

    int warps = maxWarpsPerBlock;
    if (warps > states) {
        warps = states;
    }

    int threads = warps * deviceWarpSize;

    const dim3 blocks(batch_size);
    cudaSetDevice(device_num);
    viterbi_make_trellis_kernel<<<blocks, threads, 2 * states * sizeof(float)>>>(
        observation.data_ptr<float>(),
        batch_frames.data_ptr<int>(),
        transition.data_ptr<float>(),
        initial.data_ptr<float>(),
        posterior.data_ptr<float>(),
        trellis.data_ptr<int>(),
        max_frames,
        states);
}

/// C++ API for second step of Viterbi decoding: backtracing the trellis
/// to find the maximum likelihood path
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
void viterbi_backtrace_trellis_cuda(
    const at::Tensor indices,
    const at::Tensor trellis,
    const at::Tensor batch_frames,
    int batch_size,
    int max_frames,
    int states)
{
    int device_num = indices.device().index();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_num);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    int threads = maxThreadsPerBlock;
    if (threads > batch_size) {
        threads = batch_size;
    }

    int num_blocks = (batch_size + threads - 1) / threads;
    const dim3 blocks(num_blocks);
    viterbi_backtrace_trellis_kernel<<<blocks, threads>>>(
        indices.data_ptr<int>(),
        trellis.data_ptr<int>(),
        batch_frames.data_ptr<int>(),
        batch_size,
        max_frames,
        states);
}

/// Decode time-varying categorical distributions using Viterbi decoding
///
/// Args:
///     observation: :math:`(N, T, S)` or :math:`(T, S)`
///         where `S = the number of states`, 
///         `T = the length of the sequence`,
///         and `N = batch size`.
///         Time-varying categorical distribution
///     batch_frames :math:`(N)`
///         Sequence length of each batch item
///     transition :math:`(S, S)`
///         Categorical transition matrix
///     initial :math:`(S)`
///         Categorical initial distribution
///
/// Return:
///     indices: :math:`(N, T)`
///         The decoded bin indices
at::Tensor viterbi_decode_cuda(
    at::Tensor observation,
    at::Tensor batch_frames,
    at::Tensor transition,
    at::Tensor initial
) {
    TORCH_INTERNAL_ASSERT(observation.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(batch_frames.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(transition.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(initial.device().type() == at::DeviceType::CUDA);
    int batch_size = observation.size(0);
    int max_frames = observation.size(1);
    int states = observation.size(2);

    auto device = observation.device();

    at::Tensor observation_contig = observation.contiguous();
    at::Tensor batch_frames_contig = batch_frames.contiguous();
    at::Tensor transition_contig = transition.contiguous();
    at::Tensor initial_contig = initial.contiguous();

    // Intermediate storage for path indices and costs
    at::Tensor trellis = at::zeros(
        {batch_size, max_frames, states},
        at::dtype(at::kInt).device(device));
    at::Tensor posterior = at::zeros(
        {batch_size, states},
        at::dtype(at::kFloat).device(device));

    // First step: make the minimum cost path trellis
    viterbi_make_trellis_cuda(
        observation_contig,
        batch_frames_contig,
        transition_contig,
        initial_contig,
        posterior,
        trellis);

    at::Tensor indices = posterior.argmax(1);
    indices = indices.unsqueeze(1);
    indices = indices.repeat({1, max_frames});
    indices = indices.to(at::kInt);

    // Second step: backtrace trellis to find maximum likelihood path
    viterbi_backtrace_trellis_cuda(
        indices,
        trellis,
        batch_frames_contig,
        batch_size,
        max_frames,
        states);

    return indices;
}

// Registers CUDA implementation(s)
TORCH_LIBRARY_IMPL(torbi, CUDA, m) {
    m.impl("viterbi_decode", &viterbi_decode_cuda);
}

}  // namespace torbi
