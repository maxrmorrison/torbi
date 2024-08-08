#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#define CPU_NO_WRAPPER

#include "viterbi_cpu.cpp"


/******************************************************************************
Forward definitions
******************************************************************************/


void viterbi_make_trellis_cuda(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor trellis
);

void viterbi_backtrace_trellis_cuda(
    int *indices,
    int *trellis,
    int *batch_frames,
    int batch_size,
    int max_frames,
    int states
);


/******************************************************************************
Macros
******************************************************************************/


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA torch::Tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/******************************************************************************
Device-agnostic C++ API
******************************************************************************/


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
///     num_threads (int, optional)
///         Number of threads to use if doing CPU decoding
///
/// Return:
///     indices: :math:`(N, T)`
///         The decoded bin indices
torch::Tensor viterbi_decode(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial,
    int num_threads=0
) {
    assert(batch_frames.dim() == 3);
    auto device = observation.device();
    assert(batch_frames.device() == device);
    assert(transition.device() == device);
    assert(initial.device() == device);
    int batch_size = observation.size(0);
    int max_frames = observation.size(1);
    int states = observation.size(2);

    // Intermediate storage for path indices and costs
    torch::Tensor trellis = torch::zeros(
        {batch_size, max_frames, states},
        torch::dtype(torch::kInt32).device(device));
    torch::Tensor posterior = torch::zeros(
        {batch_size, states},
        torch::dtype(torch::kFloat32).device(device));

    // First step: make the minimum cost path trellis
    if (device.is_cuda()) {
        CHECK_INPUT(observation);
        CHECK_INPUT(batch_frames);
        CHECK_INPUT(transition);
        CHECK_INPUT(initial);
        viterbi_make_trellis_cuda(
            observation,
            batch_frames,
            transition,
            initial,
            posterior,
            trellis);
    } else {
        omp_set_num_threads(num_threads);
        viterbi_make_trellis_cpu(
            observation,
            batch_frames,
            transition,
            initial,
            posterior,
            trellis);
    }

    torch::Tensor indices = posterior.argmax(1);
    indices = indices.unsqueeze(1);
    indices = indices.repeat({1, max_frames});
    indices = indices.to(torch::kInt32);

    // Second step: backtrace trellis to find maximum likelihood path
    if (device.is_cuda()) {
        CHECK_INPUT(indices);
        CHECK_INPUT(trellis);
        CHECK_INPUT(batch_frames);
        viterbi_backtrace_trellis_cuda(
            indices.data_ptr<int>(),
            trellis.data_ptr<int>(),
            batch_frames.data_ptr<int>(),
            batch_size,
            max_frames,
            states);
    } else {
        omp_set_num_threads(num_threads);
        viterbi_backtrace_trellis_cpu(
            indices.data_ptr<int>(),
            trellis.data_ptr<int>(),
            batch_frames.data_ptr<int>(),
            batch_size,
            max_frames,
            states);
    }

    return indices;
}


/******************************************************************************
Python binding
******************************************************************************/


PYBIND11_MODULE(viterbi, m) {
  m.def("decode", &viterbi_decode);
}
