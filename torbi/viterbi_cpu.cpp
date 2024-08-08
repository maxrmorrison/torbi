#include <torch/extension.h>
#include <vector>

/******************************************************************************
C++ CPU implementation of Viterbi decoding
******************************************************************************/

/// C++ CPU implementation of first step of Viterbi decoding: making the
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
void viterbi_make_trellis_cpu(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor trellis_tensor
) {
    int batch_size = observation.size(0);
    int max_frames = observation.size(1);
    int states = observation.size(2);
    float *observation_base = observation.data_ptr<float>();
    int *batch_frames_ptr = batch_frames.data_ptr<int>();
    float *transition_ptr = transition.data_ptr<float>();
    float *initial_ptr = initial.data_ptr<float>();
    float *posterior_base = posterior.data_ptr<float>();
    int *trellis_base = trellis_tensor.data_ptr<int>();

    int states2 = states*states;

    float *posterior_current = new float[states];
    float *posterior_next = new float[states];
    float *probability = new float[states2];

    int frames;
    float* obs;
    int* trellis;
    float* posterior_ptr;

    float max_posterior;

    for (int b=0; b<batch_size; b++) {
        frames = batch_frames_ptr[b];
        obs = observation_base + b*max_frames*states;
        posterior_ptr = posterior_base + b*states;
        trellis = trellis_base + b*max_frames*states;

        #pragma omp parallel for schedule(static)
        for (int i=0; i<states; i++) {
            posterior_current[i] = obs[i] + initial_ptr[i];
        }

        for (int t=1; t<frames; t++) {

            #pragma omp parallel for simd schedule(static)
            for (int i=0; i<states2; i++) {
                int s1 = i % states;
                probability[i] = posterior_current[s1] + transition_ptr[i];
            }

            // Get optimal
            #pragma omp parallel for simd schedule(static) private(max_posterior)
            for (int j=0; j<states; j++) {
                max_posterior = probability[j*states];

                for (int s3=1; s3<states; s3++) {
                    if (probability[j*states+s3] > max_posterior) {
                        max_posterior = probability[j*states+s3];
                        trellis[t*states+j] = s3;
                    }
                }
                posterior_next[j] = obs[t*states+j] + max_posterior;
            }
            float *posterior_last = posterior_current;
            posterior_current = posterior_next;
            posterior_next = posterior_last;
        }

        // #pragma omp parallel for simd schedule(static)
        for (int i=0; i<states; i++) {
            posterior_ptr[i] = posterior_current[i];
        }
    }

    // clean up
    delete posterior_current;
    delete posterior_next;
    delete probability;
}


/// C++ CPU implementation of the second step of Viterbi decoding: backtracing
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
void viterbi_backtrace_trellis_cpu(
    int *indices,
    int *trellis,
    int *batch_frames,
    int batch_size,
    int max_frames,
    int states) {
    #pragma omp parallel for
    for (int b=0; b<batch_size; b++) {
        int *indices_b = indices + max_frames * b;
        int *trellis_b = trellis + max_frames * states * b;
        int frames = batch_frames[b];
        int index = indices_b[frames-1];
        for (int t=frames-1; t>=1; t--) {
            index = trellis_b[t*states + index];
            indices_b[t-1] = index;
        }
    }
}

// if standalone, create the wrappers
#ifndef CPU_NO_WRAPPER

//one helper macro for cpu only implementations
#define CHECK_NOT_CUDA(x) TORCH_CHECK(!(x.device().is_cuda()), #x " must NOT be a CUDA torch::Tensor")

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
///         Number of CPU threads to use
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
    CHECK_NOT_CUDA(observation);
    CHECK_NOT_CUDA(batch_frames);
    CHECK_NOT_CUDA(transition);
    CHECK_NOT_CUDA(initial);
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
    omp_set_num_threads(num_threads);
    viterbi_make_trellis_cpu(
        observation,
        batch_frames,
        transition,
        initial,
        posterior,
        trellis);

    torch::Tensor indices = posterior.argmax(1);
    indices = indices.unsqueeze(1);
    indices = indices.repeat({1, max_frames});
    indices = indices.to(torch::kInt32);

    // Second step: backtrace trellis to find maximum likelihood path
    omp_set_num_threads(num_threads);
    viterbi_backtrace_trellis_cpu(
        indices.data_ptr<int>(),
        trellis.data_ptr<int>(),
        batch_frames.data_ptr<int>(),
        batch_size,
        max_frames,
        states);

    return indices;
}


PYBIND11_MODULE(viterbi, m) {
  m.def("decode", &viterbi_decode);
}
#endif