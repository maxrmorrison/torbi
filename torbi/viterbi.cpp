#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

void viterbi_forward_cuda(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor memory,
    int max_frames,
    int states
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA torch::Tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void viterbi_forward_cpu(
    torch::Tensor observation_tensor,
    torch::Tensor batch_frames_tensor,
    torch::Tensor transition_tensor,
    torch::Tensor initial_tensor,
    torch::Tensor posterior_tensor,
    torch::Tensor memory_tensor,
    int max_frames,
    int states
) {
    int batch_size = observation_tensor.size(0);
    float *observation_base = observation_tensor.data_ptr<float>();
    int *batch_frames = batch_frames_tensor.data_ptr<int>();
    float *transition = transition_tensor.data_ptr<float>();
    float *initial = initial_tensor.data_ptr<float>();
    float *posterior_base = posterior_tensor.data_ptr<float>();
    int *memory_base = memory_tensor.data_ptr<int>();

    int states2 = states*states;

    float *posterior_current = new float[states];
    float *posterior_next = new float[states];
    float *probability = new float[states2];

    int frames;
    float* observation;
    int* memory;
    float* posterior;

    float max_posterior;

    for (int b=0; b<batch_size; b++) {
        frames = batch_frames[b];
        observation = observation_base + b*max_frames*states;
        posterior = posterior_base + b*states;
        memory = memory_base + b*max_frames*states;

        #pragma omp parallel for schedule(static)
        for (int i=0; i<states; i++) {
            posterior_current[i] = observation[i] + initial[i];
        }

        for (int t=1; t<frames; t++) {

            #pragma omp parallel for simd schedule(static)
            for (int i=0; i<states2; i++) {
                int s1 = i % states;
                probability[i] = posterior_current[s1] + transition[i];
            }

            // Get optimal
            #pragma omp parallel for simd schedule(static) private(max_posterior)
            for (int j=0; j<states; j++) {
                max_posterior = probability[j*states];

                for (int s3=1; s3<states; s3++) {
                    if (probability[j*states+s3] > max_posterior) {
                        max_posterior = probability[j*states+s3];
                        memory[t*states+j] = s3;
                    }
                }
                posterior_next[j] = observation[t*states+j] + max_posterior;
            }
            float *posterior_last = posterior_current;
            posterior_current = posterior_next;
            posterior_next = posterior_last;
        }

        // #pragma omp parallel for simd schedule(static)
        for (int i=0; i<states; i++) {
            posterior[i] = posterior_current[i];
        }
    }
}

void viterbi_backtrace_cpu(
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
        for (int t=frames-1; t>=1; t--) {
            indices_b[t-1] = memory_b[t*states+indices_b[t]];
        }
    }
}

torch::Tensor viterbi_forward(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial
) {
    assert(batch_frames.dim() == 3);

    auto device = observation.device();
    assert(batch_frames.device() == device);
    assert(transition.device() == device);
    assert(initial.device() == device);

    int batch_size = observation.size(0);
    int max_frames = observation.size(1);
    int states = observation.size(2);

    // define output torch::Tensors memory and posterior
    torch::Tensor memory = torch::zeros(
        {batch_size, max_frames, states},
        torch::dtype(torch::kInt32).device(device)
    );
    torch::Tensor posterior = torch::zeros(
        {batch_size, states},
        torch::dtype(torch::kFloat32).device(device)
    );

    if (device.is_cuda()) {
        CHECK_INPUT(observation);
        CHECK_INPUT(batch_frames);
        CHECK_INPUT(transition);
        CHECK_INPUT(initial);
        viterbi_forward_cuda(
            observation,
            batch_frames,
            transition,
            initial,
            posterior,
            memory,
            max_frames,
            states
        );
    } else {
        viterbi_forward_cpu(
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

    torch::Tensor indices = posterior.argmax(1);
    indices = indices.unsqueeze(1);
    indices = indices.repeat({1, max_frames});
    indices = indices.to(torch::kInt32).cpu();
    memory = memory.cpu();
    batch_frames = batch_frames.cpu();
    viterbi_backtrace_cpu(
        indices.data_ptr<int>(),
        memory.data_ptr<int>(),
        batch_frames.data_ptr<int>(),
        batch_size,
        max_frames,
        states
    );

    return indices;
}

PYBIND11_MODULE(viterbi, m) {
  m.def("forward", &viterbi_forward, "Fast (batched) Viterbi decoding on both CPU and GPU");
}
