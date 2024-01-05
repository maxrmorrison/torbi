#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <chrono>
#include <omp.h>

namespace py = pybind11;

#define WORKERS 8


void cppforward(
    py::array_t<float> observation,
    py::array_t<float> transition,
    py::array_t<float> initial,
    py::array_t<float> posterior,
    py::array_t<int> memory,
    py::array_t<float> probability,
    int frames,
    int states
) {
    // frames x states
    py::buffer_info observation_buffer_info = observation.request();
    float *observation_ptr = static_cast<float *>(observation_buffer_info.ptr);

    // states x states
    py::buffer_info transition_buffer_info = transition.request();
    float *transition_ptr = static_cast<float *>(transition_buffer_info.ptr);

    // states
    py::buffer_info initial_buffer_info = initial.request();
    float *initial_ptr = static_cast<float *>(initial_buffer_info.ptr);

    // frames by states
    py::buffer_info posterior_buffer_info = posterior.request();
    float *posterior_ptr = static_cast<float *>(posterior_buffer_info.ptr);

    // frames by states
    py::buffer_info memory_buffer_info = memory.request();
    int *memory_ptr = static_cast<int *>(memory_buffer_info.ptr);

    // states by states
    py::buffer_info probability_buffer_info = probability.request();
    float *probability_ptr = static_cast<float *>(probability_buffer_info.ptr);

    float max_posterior;
    int a = 0;
    int b = 0;
    int states2 = states*states;
    float current_prob;
    int best_state = 0;

    // Set initial
    // auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for (int i=0; i<states; i++) {
        posterior_ptr[i] = observation_ptr[i] + initial_ptr[i];
    }
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "Time taken by init: " << static_cast<double>(duration.count()) / 1e6 << " seconds" << std::endl;

    for (int t=1; t<frames; t++) {
        // Forward Pass

        #pragma omp parallel for simd schedule(static)
        for (int i=0; i<states2; i++) {
            // note that i = s2*frames+s1
            int s1 = i % states;
            probability_ptr[i] = posterior_ptr[(t-1)*states+s1] + transition_ptr[i];
        }

        // Get optimal
        #pragma omp parallel for simd schedule(static) private(max_posterior)
        for (int j=0; j<states; j++) {
            max_posterior = probability_ptr[j*states];
            // memory_ptr[t*states+j] = 0; //initial best state is 0

            for (int s3=1; s3<states; s3++) {
                if (probability_ptr[j*states+s3] > max_posterior) {
                    max_posterior = probability_ptr[j*states+s3];
                    memory_ptr[t*states+j] = s3;
                }
            }

            posterior_ptr[t*states+j] = observation_ptr[t*states+j] + max_posterior;
        }
    }
}


PYBIND11_MODULE(fastops, m) {
    m.def("cppforward", &cppforward, "Viterbi go brrrrr");
}
