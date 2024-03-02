#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

// forward definitions
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


// torch input validation macros
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA torch::Tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// Create trellis by working forward to determine most likely next states.
// The resulting graph is stored in the trellis_tensor, and the final posterior
// distribution is stored in posterior_tensor.
// observation_tensor: BATCH x FRAMES x STATES
// batch_frames_tensor: BATCH
// transition_tensor: STATES x STATES
// initial_tensor: STATES
// posterior_tensor: BATCH x STATES
// trellis_tensor: BATCH x FRAMES x STATES
void viterbi_make_trellis_cpu(
    torch::Tensor observation_tensor,
    torch::Tensor batch_frames_tensor,
    torch::Tensor transition_tensor,
    torch::Tensor initial_tensor,
    torch::Tensor posterior_tensor,
    torch::Tensor trellis_tensor
) {
    int batch_size = observation_tensor.size(0);
    int max_frames = observation_tensor.size(1);
    int states = observation_tensor.size(2);
    float *observation_base = observation_tensor.data_ptr<float>();
    int *batch_frames = batch_frames_tensor.data_ptr<int>();
    float *transition = transition_tensor.data_ptr<float>();
    float *initial = initial_tensor.data_ptr<float>();
    float *posterior_base = posterior_tensor.data_ptr<float>();
    int *trellis_base = trellis_tensor.data_ptr<int>();

    int states2 = states*states;

    float *posterior_current = new float[states];
    float *posterior_next = new float[states];
    float *probability = new float[states2];

    int frames;
    float* observation;
    int* trellis;
    float* posterior;

    float max_posterior;

    for (int b=0; b<batch_size; b++) {
        frames = batch_frames[b];
        observation = observation_base + b*max_frames*states;
        posterior = posterior_base + b*states;
        trellis = trellis_base + b*max_frames*states;

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
                        trellis[t*states+j] = s3;
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

    // clean up
    delete posterior_current;
    delete posterior_next;
    delete probability;
}

// Trace back through the trellis to find sequence with maximal
// probability.
// indices, trellis, and batch frames are all data pointers
// to the batch_frames tensor and the indices and trellis tensors
// created by the viterbi_make_trellis_kernel (see above).
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


// Decode a time-varying categorical distribution
//
//     Args:
//         observation: :math:`(N, T, S)` or :math:`(T, S)`
//             where `S = the number of states`, `T = the length of the sequence`,
//             and `N = batch size`.
//             Time-varying categorical distribution
//         batch_frames :math:`(N)`
//             Sequence length of each batch item
//         transition :math:`(S, S)`
//             Categorical transition matrix
//         initial :math:`(S)`
//             Categorical initial distribution
//         num_threads (int, optional)
//             Number of threads to use if doing CPU decoding
//
//     Return:
//         indices: :math:`(N, T)`
//             The decoded bin indices
torch::Tensor viterbi_decode(
    torch::Tensor observation, // BATCH x FRAMES x STATES
    torch::Tensor batch_frames, // BATCH
    torch::Tensor transition, // STATES x STATES
    torch::Tensor initial, // STATES
    int num_threads=0
) {
    omp_set_num_threads(num_threads);
    assert(batch_frames.dim() == 3);

    auto device = observation.device();
    assert(batch_frames.device() == device);
    assert(transition.device() == device);
    assert(initial.device() == device);

    int batch_size = observation.size(0);
    int max_frames = observation.size(1);
    int states = observation.size(2);

    // define output torch::Tensors trellis and posterior
    torch::Tensor trellis = torch::zeros(
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
        viterbi_make_trellis_cuda(
            observation,
            batch_frames,
            transition,
            initial,
            posterior,
            trellis
        );
    } else {
        omp_set_num_threads(num_threads);
        viterbi_make_trellis_cpu(
            observation,
            batch_frames,
            transition,
            initial,
            posterior,
            trellis
        );
    }

    torch::Tensor indices = posterior.argmax(1);
    indices = indices.unsqueeze(1);
    indices = indices.repeat({1, max_frames});
    indices = indices.to(torch::kInt32);

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
            states
        );
    } else {
        omp_set_num_threads(num_threads);
        viterbi_backtrace_trellis_cpu(
            indices.data_ptr<int>(),
            trellis.data_ptr<int>(),
            batch_frames.data_ptr<int>(),
            batch_size,
            max_frames,
            states
        );
    }

    return indices;
}

PYBIND11_MODULE(viterbi, m) {
  m.def("decode", &viterbi_decode);
}
