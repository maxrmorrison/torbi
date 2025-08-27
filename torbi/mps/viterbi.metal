#include <metal_stdlib>

using namespace metal;

kernel void viterbi_make_trellis_kernel(
    constant float* observation  [[buffer(0)]],
    device int* batch_frames [[buffer(1)]],
    device float* transition [[buffer(2)]],
    device float* initial [[buffer(3)]],
    device float* posterior [[buffer(4)]],
    device int* trellis [[buffer(5)]],
    constant int& max_frames [[buffer(6)]],
    constant int& states [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint batch_id [[threadgroup_position_in_grid]],
    uint warp_id [[simdgroup_index_in_threadgroup]],
    uint thread_warp_id [[thread_index_in_simdgroup]],
    uint warp_size [[threads_per_simdgroup]],
    uint num_threads [[threads_per_threadgroup]],
    threadgroup float* posterior_cache [[threadgroup(0)]] // shared memory
) {

    int num_warps = num_threads/warp_size;

    // Handle batch
    int frames = batch_frames[batch_id];  // Get number of frames for this batch item
    observation += batch_id * max_frames * states;
    posterior += batch_id * states;
    trellis += batch_id * max_frames * states;


    threadgroup float *posterior_current = posterior_cache;
    threadgroup float *posterior_next = posterior_cache + states;

    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // Set initial
    for (int i = tid; i < states; i += num_threads) {
        posterior_current[i] = observation[i] + initial[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (int t = 1; t < frames; t++) {
        // Get optimal
        // Iterate rows by warp (each warp gets assigned a row)
        int max_index;
        float max_value;
        for (int j = warp_id; j < states; j += num_warps) {
            // Indices start out as just 0-warp_size for the first warp_size elements in the array
            max_index = thread_warp_id;
            // Values start as the first warp_size elements in the row, with row selected by j
            max_value = posterior_current[thread_warp_id] + transition[j * states + thread_warp_id];

            // Slide the warp over the row in a linear argmax search (parallelized by threads within the warp)
            // Note that we start here offset by the warp_size since we already initialized using the first chunk
            for (int i = thread_warp_id + warp_size; i < states; i += warp_size) {
                // Get the new value from the current row at the current offset
                float new_value = posterior_current[i] + transition[j * states + i];
                if (new_value > max_value) {
                    max_index = i;
                    max_value = new_value;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Parallel reduction
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                float new_value = simd_shuffle_down(max_value, offset);
                int new_index = simd_shuffle_down(max_index, offset);
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
        threadgroup float *posterior_last = posterior_current;
        posterior_current = posterior_next;
        posterior_next = posterior_last;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // Write final posterior row
    for (int i = tid; i < states; i += num_threads) {
        posterior[i] = posterior_current[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}


kernel void viterbi_backtrace_trellis_kernel(
    device int* indices,
    device int* trellis,
    device int* batch_frames,
    constant int& max_frames,
    constant int& states,
    uint global_thread_id [[thread_position_in_grid]]
) {
    int b = global_thread_id;
    // Get location to store maximum likelihood path
    device int *indices_b = indices + max_frames * b;

    // Get trellis to backtrace
    device int *trellis_b = trellis + max_frames * states * b;

    // Get number of frames
    int frames = batch_frames[b];

    // Backtrace
    int index = indices_b[frames - 1];
    for (int t = frames - 1; t >= 1; t--) {
        index = trellis_b[t * states + index];
        indices_b[t - 1] = index;
    }
}