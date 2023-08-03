# cython: auto_cpdef=True
# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonehcheck=False

import numpy as np
cimport numpy as np
from libc.math cimport isnan
from libc.math cimport log as ln

ctypedef double DTYPE_t                  # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.int64_t SIZE_t               # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef DTYPE_t nan, ninf, pinf

# ----------------------------------------------------------------------------------------------------------------------

cdef inline DTYPE_t mean(DTYPE_t[::1] arr) nogil:
    """
    Arithmetic mean in the vector ``arr``.
    
    :param arr: the array to compute the mean on.
    
    :return: The mean as double
    """
    cdef DTYPE_t result = 0
    cdef int i
    for i in range(arr.shape[0]):
        result += arr[i]
    return result / (<DTYPE_t> arr.shape[0])


# ----------------------------------------------------------------------------------------------------------------------

cdef inline int alltrue(SIZE_t[::1] mask, SIZE_t[::1] pos) nogil:
    """
    Check if all elements of this array are true-
    :param mask: 
    :param pos: 
    :return: 
    """
    cdef SIZE_t i
    for i in range(pos.shape[0] if pos is not None else mask.shape[0]):
        if not mask[i if pos is None else mask[i]]:
            return False
    return True


# ----------------------------------------------------------------------------------------------------------------------
# Sorting algorithm implementations taken from sklearn
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_splitter.pyx


cdef inline double ld(double x) nogil:
    """
    Calculate dual logarithm
    :param x: the function value
    :return: the dual logarithm of x
    """
    return ln(x) / ln(2.0)


cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    """
    Sort n-element arrays pointed to by Xf and samples, simultaneously,
    by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
    :param Xf: pointer to arrays
    :param samples: pointer to samples
    :param n: size of the array
    :return:
    """
    if n == 0:
        return
    cdef int maxd = 2 * <int>ld(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples, SIZE_t i, SIZE_t j) nogil:
    """
    Swap two elements at indices ``i`` and ``j`` in ``Xf`` 
    and their index positions in ``samples``.
    :param Xf: 
    :param samples: 
    :param i: 
    :param j: 
    :return: 
    """
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    """
    Median of three pivot selection, after Bentley and McIlroy (1993).
    Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    :param Xf: 
    :param n: 
    :return: 
    """
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


cdef inline void introsort(DTYPE_t* Xf, SIZE_t *samples,  SIZE_t n, int maxd) nogil:
    """
    Introsort with median of 3 pivot selection and 3-way partition function
    (robust to repeated elements, e.g. lots of zero features).
    :param Xf: 
    :param samples: 
    :param n: 
    :param maxd: 
    :return: 
    """
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
    """
    Restore heap order in Xf[start:end] by moving the max element to start.
    :param Xf: 
    :param samples: 
    :param start: 
    :param end: 
    :return: 
    """
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef inline void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    """
    Implementation of heapsort
    :param Xf: 
    :param samples: 
    :param n: 
    :return: 
    """
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


cpdef inline test_sort(DTYPE_t[::1] arr, SIZE_t[::1] indices, SIZE_t n=-1):
    """
    
    :param arr: 
    :param indices: 
    :param n: 
    :return: 
    """
    sort(&arr[0], &indices[0], arr.shape[0] if n == -1 else n)


cdef inline SIZE_t _bisect(DTYPE_t* Xf, DTYPE_t v, SIZE_t lower, SIZE_t upper) nogil:
    """
    
    :param Xf: 
    :param v: 
    :param lower: 
    :param upper: 
    :return: 
    """
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
    """
    
    :param Xf: 
    :param v: 
    :param n: 
    :return: 
    """
    return _bisect(Xf, v, 0, n - 1)


# ----------------------------------------------------------------------------------------------------------------------

cpdef inline DTYPE_t ifnan(DTYPE_t if_, DTYPE_t else_, transform=None):
    '''
    Returns the condition ``if_`` iff it is not ``Nan``, or if a transformation is
    specified, ``transform(if_)``. Returns ``else_`` if the condition is ``NaN``.
    ``transform`` can be any callable, which will be passed ``if_`` in case ``if_`` is not ``NaN``.
    '''
    if isnan(if_):
        return else_
    else:
        if transform is not None:
            return transform(if_)
        else:
            return if_


cdef inline np.int32_t equal(DTYPE_t x1, DTYPE_t x2, DTYPE_t tol=1e-7):
    """
    Check if two numbers are approximately equal.
    :param x1: the first number
    :param x2: the second number
    :param tol: the amount they may differ to be still considered equal
    :return: True if they are approximately equal, else if not.
    """
    return abs(x1 - x2) < tol


cpdef inline DTYPE_t[::1] linspace(DTYPE_t start, DTYPE_t stop, np.int64_t num):
    """
    Modification of the ``numpy.linspace`` function to return an array of ``num``
    equally spaced samples in the range of ``start`` and ``stop`` (both inclusive).

    In contrast to the original numpy function, this variant return the centroid of
    ``start`` and ``stop`` in the case where ``num`` is ``1``.
    
    :param start: 
    :param stop: 
    :param num: 
    :return: 
    """
    cdef DTYPE_t[::1] samples = np.ndarray(shape=num, dtype=np.float64)
    cdef DTYPE_t n
    cdef DTYPE_t space, val = start
    cdef np.int64_t i
    if num == 1:
        samples[0] = (stop - start) / 2
    else:
        n = <DTYPE_t> num - 1
        space = (stop - start) / n
        for i in range(num):
            samples[i] = val
            val += space
    return samples


# ----------------------------------------------------------------------------------------------------------------------

cdef class ConfInterval:
    """
    Represents a prediction interval with a predicted value, lower und upper bound
    """

    cdef readonly:
        DTYPE_t mean, lower, upper

    cpdef tuple totuple(self)

    cpdef DTYPE_t[::1] tomemview(self, DTYPE_t[::1] result=*)

