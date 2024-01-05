cimport cython
from cython.parallel cimport prange
from numpy.math cimport INFINITY
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdio cimport printf

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cforward(
    float[:, :] observation,
    float[:, :] transition,
    float[:] initial,
    float[:, :] posterior,
    int[:, :] memory,
    float[:, :] probability,
    int frames,
    int states
) noexcept nogil:
    cdef float max_posterior
    cdef int i
    cdef int j
    cdef int s1
    cdef int s2
    cdef int s3
    cdef int t = 1
    cdef int tm1 = 0

    cdef timespec ts
    cdef double start
    cdef double a
    cdef double b = 0.
    cdef double c = 0.
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)

    # Add prior to first frame
    for i in prange(states, nogil=True):
        posterior[0, i] = observation[0, i] + initial[i]

    clock_gettime(CLOCK_REALTIME, &ts)
    a = (ts.tv_sec + (ts.tv_nsec / 1000000000.)) - start
    printf("%f\n", a)

    # Forward pass
    while t < frames:

        clock_gettime(CLOCK_REALTIME, &ts)
        start = ts.tv_sec + (ts.tv_nsec / 1000000000.)

        for s1 in prange(states, nogil=True):

            # Compute all possible updates
            for s2 in range(states):
                probability[s2, s1] = posterior[tm1, s1] + transition[s2, s1]

        clock_gettime(CLOCK_REALTIME, &ts)
        b = b + (ts.tv_sec + (ts.tv_nsec / 1000000000.)) - start

        clock_gettime(CLOCK_REALTIME, &ts)
        start = ts.tv_sec + (ts.tv_nsec / 1000000000.)

        j = 0
        while j < states:

            # Get optimal greedy update from current state
            s3 = 0
            max_posterior = -INFINITY
            while s3 < states:
                if probability[j, s3] > max_posterior:
                    max_posterior = probability[j, s3]
                    memory[t, j] = s3
                s3 = s3 + 1

            # Update posterior distribution
            posterior[t, j] = observation[t, j] + max_posterior

            j = j + 1
        t = t + 1
        tm1 = tm1 + 1

        clock_gettime(CLOCK_REALTIME, &ts)
        c = c + (ts.tv_sec + (ts.tv_nsec / 1000000000.)) - start

    printf("%f\n", b)
    printf("%f\n", c)