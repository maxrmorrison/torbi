#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

void viterbi_cuda_forward(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor memory,
    int max_frames,
    int states
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void viterbi_forward(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor memory,
    int max_frames,
    int states
) {
    CHECK_INPUT(observation);
    CHECK_INPUT(batch_frames);
    CHECK_INPUT(transition);
    CHECK_INPUT(initial);
    CHECK_INPUT(posterior);
    CHECK_INPUT(memory);
    return viterbi_cuda_forward(
        observation,
        batch_frames,
        transition,
        initial,
        posterior,
        memory,
        max_frames,
        states
    );
}

PYBIND11_MODULE(cudaops, m) {
  m.def("forward", &viterbi_forward, "CUDA go BRRRRRRRRRRRRRRRR R R  R");
}