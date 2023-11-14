# cython: language_level=3

cdef class Impurity:
    """
    Class for fast, induction-like impurity measures.
    """

    cdef readonly float[:, ::1] data
    """
    The data used for calculations as a 2D C-contiguous float array.
    """

    cdef readonly int[::1] features
    """
    The indices of the features axis that are used to split on. 
    """