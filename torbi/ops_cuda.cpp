#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

void viterbi_cuda_forward(
    torch::Tensor observation,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor memory,
    torch::Tensor probability,
    int frames,
    int states
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void viterbi_forward(
    torch::Tensor observation,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor memory,
    torch::Tensor probability,
    int frames,
    int states
) {
    CHECK_INPUT(observation);
    CHECK_INPUT(transition);
    CHECK_INPUT(initial);
    CHECK_INPUT(posterior);
    CHECK_INPUT(memory);
    CHECK_INPUT(probability);
    return viterbi_cuda_forward(
        observation,
        transition,
        initial,
        posterior,
        memory,
        probability,
        frames,
        states
    );
}

PYBIND11_MODULE(cudaops, m) {
  m.def("forward", &viterbi_forward, "CUDA go BRRRRRRRRRRRRRRRR R R  R");
//   m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}