cimport cython
from cython.parallel import prange
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
) nogil:
    cdef float max_posterior
    cdef int max_posterior_index
    cdef int j = 0
    cdef int t = 1
    cdef int tm1 = 0

    # Initialize intermediate probabilities
    cdef array.array probability = array.array('f')
    memset(probability.data.as_voidptr, 0, 4 * states * states)

    # Add prior to first frame
    for i in prange(states, nogil=True):
        posterior[0, i] = observation[0, i] + initial[i]

    # Forward pass
    while t < frames:

        # Compute all possible updates
        for s1 in prange(states, nogil=True):
            for s2 in prange(states, nogil=True):
                probability[s1, s2] = posterior[t, s1] + transition[s1, s2]



            memory[t, s1] =

        # Get maximum of posterior distribution
        for k in prange(states, nogil=True):
            value = posterior[t, k] + transition[j, k]
            if value > max_posterior:
                max_posterior = value
                max_posterior_index = k

        j = 0
        while j < states:

            # Save index of maximum for backward pass
            memory[t1, j] = max_posterior_index

            # Update posterior distribution
            posterior[t1, j] = observation[t1, j] + max_posterior

            j += 1
        t += 1
        tm1 += 1

