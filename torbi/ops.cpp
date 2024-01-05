#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <chrono>
#include <omp.h>

namespace py = pybind11;

#define WORKERS 8

float* argmax_2_pointers(float* pointer_0, float* pointer_1) {
    if (!pointer_0) {
        return pointer_1;
    }
    // std::cout << "\n\n" << "COMBINING " << pointer_0 << " AND " << pointer_1 << std::endl;
    if (*pointer_0 >= *pointer_1) {
        return pointer_0;
    } else {
        return pointer_1;
    }
}

inline float* argmax_pointers(float* start, float* stop) {
    float* max_pointer = start;
    float max = *start;
    float val;
    float* current = start + 1;
    for ( ; current < stop; current++) {
        val = *current;
        if (val > max) {
            max_pointer = current;
            max = val;
        }
    }
    return max_pointer;
}

int argmax_indices(int start_index, int stop_index, float* array) {
    int max_index = start_index;
    float max = array[start_index];
    float val;
    for (int current_index=start_index+1; current_index<stop_index; current_index++) {
        val = array[current_index];
        if (val > max) {
            max_index = current_index;
            max = val;
        }
    }
    return max_index;
}

int argmax_opt(py::array_t<float> pyarray) {
    auto start = std::chrono::high_resolution_clock::now();
    py::buffer_info buffer_info = pyarray.request();
    float *array = static_cast<float *>(buffer_info.ptr);
    size_t size = buffer_info.size;

    int output;
    for (int i=0; i<100; i++) {
        output = argmax_indices(0, size, array);
    }
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start);
    std::cout << "Time taken by argmax (indices): " << static_cast<double>(duration.count()) / 1e6 << " seconds" << std::endl;
    return output;
}

// #pragma omp declare reduction(argmax_indices: int : omp_out = argmax_2_indices(omp_out, omp_in, ))

// int parallel_argmax_opt(py::array_t<float> pyarray) {
//     auto start = std::chrono::high_resolution_clock::now();
//     py::buffer_info buffer_info = pyarray.request();
//     float *array = static_cast<float *>(buffer_info.ptr);
//     size_t size = buffer_info.size;
//     // std::cout << "array pointer: " << array << std::endl;

//     int max_index;
//     int work_per_thread = size/WORKERS;
//     for (int i=0; i<100; i++) {
//         #pragma omp parallel reduction(argmax_indices: max_pointer) num_threads(WORKERS)
//         {
//             int worker_id = omp_get_thread_num();
//             int offset = work_per_thread*worker_id;
//             if (worker_id == WORKERS-1) {
//                 max_pointer = argmax_pointers(array+offset, array_end);
//             } else{
//                 max_pointer = argmax_pointers(array+offset, array+offset+work_per_thread);
//             }
//             // #pragma omp critical
//             // {
//             // std::cout << "thread " << worker_id << " found pointer: " << max_pointer << "(index " << max_pointer-array << ")" << std::endl;
//             // }
//         }
//     }
//     // std::cout << "best overall pointer: " << max_pointer << std::endl;
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//     std::cout << "Time taken by parallel_argmax: " << static_cast<double>(duration.count()) / 1e6 << " seconds" << std::endl;
//     return static_cast<int>(max_pointer-array);
// }

int argmax(py::array_t<float> pyarray) {
    auto start = std::chrono::high_resolution_clock::now();
    py::buffer_info buffer_info = pyarray.request();
    float *array = static_cast<float *>(buffer_info.ptr);
    size_t size = buffer_info.size;
    float *stop = array + size;

    float *output;
    for (int i=0; i<100; i++) {
        output = argmax_pointers(array, stop);
    }
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start);
    std::cout << "Time taken by argmax: " << static_cast<double>(duration.count()) / 1e6 << " seconds" << std::endl;
    return static_cast<int>(output-array);
}

// #pragma omp declare reduction(argmax_pointers: float* : omp_out = argmax_2_pointers(omp_out, omp_in))
#pragma omp declare reduction(argmax_pointers: float* : omp_out = *omp_out >= *omp_in ? omp_out : omp_in)

int parallel_argmax(py::array_t<float> pyarray) {
    auto start = std::chrono::high_resolution_clock::now();
    py::buffer_info buffer_info = pyarray.request();
    float *array = static_cast<float *>(buffer_info.ptr);
    size_t size = buffer_info.size;
    // std::cout << "array pointer: " << array << std::endl;

    float* max_pointer;
    int work_per_thread = size/WORKERS;
    float* array_end = array + size;
    for (int i=0; i<10000; i++) {

        // This way is fastest so far
        // #pragma omp parallel reduction(argmax_pointers: max_pointer) num_threads(WORKERS)
        // {
        //     int worker_id = omp_get_thread_num();
        //     int offset = work_per_thread*worker_id;
        //     if (worker_id == WORKERS-1) {
        //         max_pointer = argmax_pointers(array+offset, array_end);
        //     } else{
        //         max_pointer = argmax_pointers(array+offset, array+offset+work_per_thread);
        //     }
        //     // #pragma omp critical
        //     // {
        //     // std::cout << "thread " << worker_id << " found pointer: " << max_pointer << "(index " << max_pointer-array << ")" << std::endl;
        //     // }
        // }

        // This way is slower but simpler. It can also be compiled with SIMD, but raises a segmentation fault if used?
        // #pragma omp parallel for reduction(argmax_pointers: max_pointer) num_threads(WORKERS)
        // for (int offset=0; offset<size; offset+=work_per_thread){
        //         max_pointer = argmax_pointers(array+offset, std::min(array+offset+work_per_thread, array_end));
        //     // #pragma omp critical
        //     // {
        //     // std::cout << "thread " << worker_id << " found pointer: " << max_pointer << "(index " << max_pointer-array << ")" << std::endl;
        //     // }
        // }

        // This way tries to use interleaving for better cache use
        #pragma omp parallel reduction(argmax_pointers: max_pointer) num_threads(WORKERS)
        {
            int worker_id = omp_get_thread_num();
            max_pointer = array+worker_id;
            float max_value = *max_pointer;
            float current_value;
            for (float *current_pointer=max_pointer+WORKERS; current_pointer<array_end; current_pointer+=WORKERS) {
                current_value = *current_pointer;
                if (current_value > max_value) {
                    max_pointer = current_pointer;
                    max_value = current_value;
                }
            }
        }
    }
    // std::cout << "best overall pointer: " << max_pointer << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by parallel_argmax: " << static_cast<double>(duration.count()) / 1e6 << " seconds" << std::endl;
    return static_cast<int>(max_pointer-array);
}

// void cppforward(
//     py::array_t<float> observation,
//     py::array_t<float> transition,
//     py::array_t<float> initial,
//     py::array_t<float> posterior,
//     py::array_t<int> memory,
//     py::array_t<float> probability,
//     int frames,
//     int states
// ) {
//     py::buffer_info observation_buffer_info = observation.request();
//     float *observation_ptr = static_cast<float *>(observation_buffer_info.ptr);

//     py::buffer_info transition_buffer_info = transition.request();
//     float *transition_ptr = static_cast<float *>(transition_buffer_info.ptr);

//     py::buffer_info initial_buffer_info = initial.request();
//     float *initial_ptr = static_cast<float *>(initial_buffer_info.ptr);

//     py::buffer_info posterior_buffer_info = posterior.request();
//     float *posterior_ptr = static_cast<float *>(posterior_buffer_info.ptr);

//     py::buffer_info memory_buffer_info = memory.request();
//     int *memory_ptr = static_cast<int *>(memory_buffer_info.ptr);

//     py::buffer_info probability_buffer_info = probability.request();
//     float *probability_ptr = static_cast<float *>(probability_buffer_info.ptr);

//     float max_posterior;
//     int a = 0;
//     int b = 0;
//     int states2 = states*states;
//     float current_prob;
//     int best_state = 0;

//     // Set initial
//     // auto start = std::chrono::high_resolution_clock::now();
//     // #pragma omp parallel for
//     for (int i=0; i<states; i++) {
//         posterior_ptr[i] = observation_ptr[i] + initial_ptr[i];
//     }
//     // auto stop = std::chrono::high_resolution_clock::now();
//     // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//     // std::cout << "Time taken by init: " << static_cast<double>(duration.count()) / 1e6 << " seconds" << std::endl;

//     for (int t=1; t<frames; t++) {
//         // Forward Pass
//         // start = std::chrono::high_resolution_clock::now();
//         // #pragma omp parallel for simd schedule(static)
//         for (int i=0; i<states2; i++) {
//             // note that i = s2*frames+s1
//             int s1 = i % frames;
//             probability_ptr[i] = posterior_ptr[(t-1)*frames+s1] + transition_ptr[i];
//         }
//         // stop = std::chrono::high_resolution_clock::now();
//         // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//         // a += duration.count();
//         // Get optimal
//         // start = std::chrono::high_resolution_clock::now();
//         // #pragma omp parallel for schedule(static) private(max_posterior)
//         for (int j=0; j<states; j++) {
//             max_posterior = probability_ptr[j*frames];
//             memory_ptr[t*frames+j] = 0; //initial best state is 0

//             // int max_idx = 0;
//             // memory_ptr[t*frames+j] = max_idx;
//             // max_posterior = probability_ptr[max_idx];

//             for (int s3=1; s3<states; s3++) {
//                 if (probability_ptr[j*frames+s3] > max_posterior) {
//                     max_posterior = probability_ptr[j*frames+s3];
//                     memory_ptr[t*frames+j] = s3;
//                 }
//             }

//             posterior_ptr[t*frames+j] = observation_ptr[t*frames+j] + max_posterior;
//         }
//         // stop = std::chrono::high_resolution_clock::now();
//         // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//         // b += duration.count();
//     }
//     // std::cout << "Time taken by 'a': " << static_cast<double>(a) / 1.0e6 << " seconds" << std::endl;
//     // std::cout << "Time taken by 'b': " << static_cast<double>(b) / 1.0e6 << " seconds" << std::endl;
// }

void print_numpy_array(const py::array_t<float> array) {
    auto buffer_info = array.request();
    float* ptr = static_cast<float*>(buffer_info.ptr);

    std::cout << "NumPy Array Shape: (";
    for (size_t dim : buffer_info.shape) {
        std::cout << dim << ", ";
    }
    std::cout << ")\n";

    std::cout << "Data:\n";
    for (size_t i = 0; i < buffer_info.size; ++i) {
        std::cout << ptr[i] << " ";
    }
    std::cout << "\n";
}

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
    m.def("parallel_argmax", &parallel_argmax, "Argmax go brrrrrrr");
    m.def("argmax", &argmax, "Argmax but slooooooow");
    m.def("argmax_opt", &argmax_opt, "Argmax (hopefully) optimal");
}