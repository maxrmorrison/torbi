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

void viterbi_backward(
    int *indices,
    int *memory,
    int *batch_frames,
    int batch_size,
    int max_frames,
    int states) {

    #pragma omp parallel for
    for (int b=0; b<batch_size; b++) {
        int *indices_b = indices + max_frames * b;
        int *memory_b = memory + max_frames * states * b;
        int frames = batch_frames[b];
        for (int t=frames-1; t>=0; t--) {
            indices_b[t-1] = memory_b[t*states+indices_b[t]];
        }
    }
}

torch::Tensor viterbi_forward(
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
    viterbi_cuda_forward(
        observation,
        batch_frames,
        transition,
        initial,
        posterior,
        memory,
        max_frames,
        states
    );


    int batch_size = observation.size(0);
    torch::Tensor indices = posterior.argmax(1);
    indices = indices.unsqueeze(1);
    indices = indices.repeat({1, max_frames});
    indices = indices.to(torch::kInt32).cpu();
    memory = memory.cpu();
    batch_frames = batch_frames.cpu();
    viterbi_backward(
        indices.data_ptr<int>(),
        memory.data_ptr<int>(),
        batch_frames.data_ptr<int>(),
        batch_size,
        max_frames,
        states
    );

    return indices;
}

PYBIND11_MODULE(cudaops, m) {
  m.def("forward", &viterbi_forward, "CUDA go BRRRRRRRRRRRRRRRR R R  R");
}