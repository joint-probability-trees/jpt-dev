# cython: auto_cpdef=True
# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonehcheck=False

cimport numpy as np


ctypedef double DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.int64_t SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from libc.math cimport nan as _libc_nan
cdef DTYPE_t nan = <DTYPE_t> _libc_nan


from libc.math cimport log as ln

# ----------------------------------------------------------------------------------------------------------------------


cdef inline DTYPE_t mean(DTYPE_t[::1] arr) nogil:
    cdef DTYPE_t result = 0
    cdef int i
    for i in range(arr.shape[0]):
        result += arr[i]
    return result / <DTYPE_t> arr.shape[0]


# ----------------------------------------------------------------------------------------------------------------------


cdef inline int alltrue(SIZE_t[::1] mask, SIZE_t[::1] pos) nogil:
    cdef SIZE_t i
    for i in range(pos.shape[0] if pos is not None else mask.shape[0]):
        if not mask[i if pos is None else mask[i]]:
            return False
    return True


# ----------------------------------------------------------------------------------------------------------------------
# Sorting alogorthm implementations from sklearn

cdef inline double ld(double x) nogil:
    return ln(x) / ln(2.0)


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
        return
    cdef int maxd = 2 * <int>ld(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
                      SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf,SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef inline void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[samples[maxind]] < Xf[samples[child]]:
            maxind = child
        if child + 1 < end and Xf[samples[maxind]] < Xf[samples[child] + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef inline void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1


cdef inline SIZE_t _bisect(DTYPE_t* Xf, DTYPE_t v, SIZE_t lower, SIZE_t upper) nogil:
    if Xf[lower] >= v:
        return lower
    elif Xf[upper] <= v:
        return upper + 1
    elif lower + 1 == upper and  Xf[lower] < v < Xf[upper]:
        return upper
    cdef SIZE_t pivot = (lower + upper) / 2
    if Xf[pivot] < v:
        return _bisect(Xf, v, pivot, upper)
    else:
        return _bisect(Xf, v, lower, pivot)


cdef inline SIZE_t bisect(DTYPE_t* Xf, DTYPE_t v, SIZE_t n) nogil:
    return _bisect(Xf, v, 0, n - 1)


# ----------------------------------------------------------------------------------------------------------------------

