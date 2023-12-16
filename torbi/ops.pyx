cimport cython
from numpy.math cimport INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cforward(
    float[:, ::1] observation,
    float[:, ::1] transition,
    float[::1] initial,
    float[:, ::1] posterior,
    int[:, ::1] memory,
    int frames,
    int states
):
    cdef float max_posterior
    cdef int max_posterior_index

    # Add prior to first frame
    for i in range(states):
        posterior[0, i] = observation[0, i] + initial[i]

    # Forward pass
    for t in range(frames - 1):
        for j in range(states):

            # Get maximum of posterior distribution
            max_posterior = -INFINITY
            max_posterior_index = 0
            for k in range(states):
                value = posterior[t, k] + transition[j, k]
                if value > max_posterior:
                    max_posterior = value
                    max_posterior_index = k

            # Save index of maximum for backward pass
            memory[t + 1, j] = max_posterior_index

            # Update posterior distribution
            posterior[t + 1, j] = observation[t + 1, j] + max_posterior
