# cython: auto_cpdef=False
# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

__module__ = 'impurity.pyx'

import numpy as np
cimport numpy as np
import tabulate
from libc.math cimport isinf, isnan

from dnutils import mapstr

from ..base.cutils cimport DTYPE_t, SIZE_t, mean, nan, sort, ninf

# variables declaring that at num_samples[0] are the number of samples left of the split and vice versa
cdef int LEFT = 0
cdef int RIGHT = 1


# ----------------------------------------------------------------------------------------------------------------------


cdef inline DTYPE_t compute_var_improvements(
    DTYPE_t[::1] variances_total,
    DTYPE_t[::1] variances_left,
    DTYPE_t[::1] variances_right,
    SIZE_t samples_left,
    SIZE_t samples_right,
    SIZE_t skip_idx=-1) noexcept nogil:
    """
    Compute the variance improvement of a split. 
    
    :param variances_total: The variances before the split
    :param variances_left: The variances of the left side of the split
    :param variances_right: The variances of the right side of the split
    :param samples_left: The amount of samples on the left side of the split
    :param samples_right: The amount of samples on the right side of the split
    :param skip_idx: Skip the variable with this index for the computation
    
    :return: double describing the relative variance improvement
    """
    cdef SIZE_t i
    cdef DTYPE_t variance_impr = 0
    cdef DTYPE_t[::1] variances_old = variances_total
    cdef DTYPE_t n_samples = <DTYPE_t> samples_left + samples_right
    cdef DTYPE_t divisor = 0

    for i in range(variances_old.shape[0]):

        # skip the index specified from the signature or if variance old is 0, since then variance new has to be 0 too.
        # if this was not skipped, nans would pollute the sum.
        if skip_idx == i or variances_old[i] == 0:
            continue

        variance_impr = variance_impr * <DTYPE_t> divisor + (
                (
                    variances_old[i] -
                    (variances_left[i] * <DTYPE_t> samples_left + variances_right[i] * <DTYPE_t> samples_right) / n_samples
                ) / variances_old[i]
        ) / <DTYPE_t> (divisor + 1)
        divisor += 1

    return variance_impr


cpdef inline DTYPE_t _compute_var_improvements(
    DTYPE_t[::1] variances_total,
    DTYPE_t[::1] variances_left,
    DTYPE_t[::1] variances_right,
    SIZE_t samples_left,
    SIZE_t samples_right,
    SIZE_t skip_idx=-1
) noexcept nogil:
    """Python-callable version for testing only."""
    return compute_var_improvements(
        variances_total,
        variances_left,
        variances_right,
        samples_left,
        samples_right,
        skip_idx
    )


# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyPep8Naming
cdef inline void sum_at(
    DTYPE_t[:, ::1] M,
    SIZE_t[::1] rows,
    SIZE_t[::1] cols,
    DTYPE_t[::1] result
) noexcept nogil:
    """
    Sum rows at columns.
    
    :param M: Matrix with the raw data
    :type M; 2D contiguous view of array 
    :param rows: Indices of rows to sum
    :type rows: 1D contiguous view of rows
    :param cols: Indices of cols to sum
    :type cols: 1D contiguous view of columns
    :param result: Result to write into. Has to have the same length as ``cols``
    :type result: 1D contiguous view of numpy array 
    """
    result[...] = 0
    cdef SIZE_t i, j
    for j in range(cols.shape[0]):
        for i in range(rows.shape[0]):
            result[j] += M[rows[i], cols[j]]


cpdef inline void _sum_at(
    DTYPE_t[:, ::1] M,
    SIZE_t[::1] rows,
    SIZE_t[::1] cols,
    DTYPE_t[::1] result
) noexcept nogil:
    """Python-callable version for testing only."""
    sum_at(M, rows, cols, result)


# ----------------------------------------------------------------------------------------------------------------------

cdef inline void sq_sum_at(
    DTYPE_t[:, ::1] M,
    SIZE_t[::1] rows,
    SIZE_t[::1] cols,
    DTYPE_t[::1] result
) noexcept nogil:
    """
    Square the values in the rows and sum them.
    
    :param M: Matrix with the raw data
    :type M; 2D contiguous view of array 
    :param rows: Indices of rows to sum
    :type rows: 1D contiguous view of rows
    :param cols: Indices of cols to sum
    :type cols: 1D contiguous view of columns
    :param result: Result to write into. Has to have the same length as ``cols``
    :type result: 1D contiguous view of numpy array 
    """
    result[...] = 0
    cdef SIZE_t i, j
    cdef DTYPE_t v
    for j in range(cols.shape[0]):
        for i in range(rows.shape[0]):
            v = M[rows[i], cols[j]]
            result[j] += v * v


cpdef inline void _sq_sum_at(
    DTYPE_t[:, ::1] M,
    SIZE_t[::1] rows,
    SIZE_t[::1] cols,
    DTYPE_t[::1] result
) noexcept nogil:
    """Python-callable version for testing only."""
    sq_sum_at(M, rows, cols, result)


# ----------------------------------------------------------------------------------------------------------------------

cdef inline void variances(
    DTYPE_t[::1] sq_sums,
    DTYPE_t[::1] sums,
    SIZE_t n_samples,
    DTYPE_t[::1] result
) noexcept nogil:
    """
    Variance computation uses the proxy from sklearn: ::

    var = \sum_i^n (y_i - y_bar) ** 2
    = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2

    We do NOT use the unbiased variance calculation, because it does not coincide with the mean squared error which
    is the actual optimization target here. 

    See also: 'https://github.com/scikit-learn/scikit-learn/blob/de1262c35e2aa4ee062d050281ee576ce9e35c94/sklearn/
               tree/_criterion.pyx#L683'
    :param sq_sums: The square sums 
    :param sums: The ordinary sums
    :param n_samples: the number of samples
    :param result: The array to write into (result will be overwritten)
    """
    # prevent nan values due to division by zero in variance calculation of one-example splits
    if n_samples <= 1:
        result[:] = 0
        return

    result[:] = sq_sums
    cdef SIZE_t i
    for i in range(sums.shape[0]):
        result[i] -= sums[i] * sums[i] / <DTYPE_t> n_samples
        result[i] /= <DTYPE_t> n_samples


cpdef inline void _variances(
    DTYPE_t[::1] sq_sums,
    DTYPE_t[::1] sums,
    SIZE_t n_samples,
    DTYPE_t[::1] result
) noexcept nogil:
    """Python-callable version for testing only."""
    variances(sq_sums, sums, n_samples, result)


# ----------------------------------------------------------------------------------------------------------------------
# in-place vector addition

cdef inline void ivadd(DTYPE_t[::1] target, DTYPE_t[::1] arg, SIZE_t n, int sq=False) noexcept nogil:
    """
    Inplace vector addition
    :param target: the target vector
    :param arg: the vector to add to the target vector
    :param n: the size of the target vector
    :param sq: rather to square the numbers in ``arg`` before they are added or not
    """
    cdef SIZE_t i
    for i in range(n):
        target[i] += arg[i] if not sq else (arg[i] * arg[i])


# ----------------------------------------------------------------------------------------------------------------------

cdef inline void bincount(DTYPE_t[:, ::1] data,
                          SIZE_t[::1] rows,
                          SIZE_t[::1] cols,
                          SIZE_t[:, ::1] result) noexcept nogil:
    """
    Compute a histogram where the first dimension denotes the values of the column and the second denotes the column.
    The value stored at a specific position denotes the frequency.
    :param data: The raw data. In the raw data every accesses value by this method has to be index-able
    :param rows: The rows to select the datapoints from (array of indices)
    :type rows: Integer contiguous array
    :param cols: The columns to select the data from (array of indices)
    :type cols: Integer 1D contiguous array
    :param result: The result to write into
    :type result: 2D contiguous integer array 
    """
    result[...] = 0
    cdef SIZE_t i, j
    for i in range(rows.shape[0]):
        for j in range(cols.shape[0]):
            result[<SIZE_t> data[rows[i], cols[j]], j] += 1


# ----------------------------------------------------------------------------------------------------------------------

cdef class Impurity:
    """
    Class to implement fast impurity calculations on splits.

    Note:
        A general name convention is, that left and right refer to sides of a split made on a sorted array
        On this array with a split value of 104.25
        -----------------------------------------
        | 1  |  3 |  8.5 | 200  | 210  |  210.5 |
        ----------------------------------------
        left would be considered as 1,3,8.5 and right would be considered as 200,210,210.5


        Whenever an index is considered as invalid or not initialized '-1' is used.
    """

    # the raw 'read-only' data
    cdef DTYPE_t [:, ::1] data

    # the indices of the datapoints used for sorting, indexing and buffering current samples
    cdef readonly SIZE_t [::1] indices, index_buffer

    # the features to split on
    cdef readonly DTYPE_t[::1] feat

    # indices to mark start and end of a reading
    cdef SIZE_t start, end

    # array of indices that describe where to find what kind of target variable
    cdef SIZE_t[::1] numeric_vars, symbolic_vars, all_vars

    # array of indices describing what features are of symbolic and what are of numeric nature
    cdef SIZE_t[::1] numeric_features, symbolic_features

    # percentage of samples that have to be in a leaf to valid
    cdef public DTYPE_t min_samples_leaf

    # integer array describing the number of symbols per symbolic variable
    cdef SIZE_t[::1] symbols

    # integers holding the number of numeric targets, number of symbolic targets, maximum size of a symbolic domain,
    # number of targets, number of total numeric variables, number of total symbolic variable,
    # number of total variables
    cdef SIZE_t n_num_vars, n_sym_vars, max_sym_domain, n_vars, n_num_vars_total, n_sym_vars_total, n_vars_total

    # 2D integer array describing histograms total, left and right
    cdef SIZE_t[:, ::1] symbols_left, \
        symbols_right, \
        symbols_total

    # double array describing the gini improvements
    cdef DTYPE_t[::1] gini_improvements

    # double array describing the gini impurities
    cdef DTYPE_t[::1] gini_impurities

    # float array of gini impurities left of the split
    cdef DTYPE_t[::1] gini_left

    # float array of gini impurities right of the split
    cdef DTYPE_t[::1] gini_right

    # int array for storing booleans of which gini impurities should be inverted
    cdef SIZE_t[::1] invert_impurity

    # float arrays of all kinds of statistics left and right of a split
    cdef DTYPE_t[::1] variances_left, \
        variances_right, \
        variances_total, \
        variance_improvements, \
        sums_left, \
        sums_right, \
        sq_sums_left, \
        sq_sums_right, \
        sums_total, \
        sq_sums_total

    # double array containing the maximum variances of each numeric variable
    cdef DTYPE_t[::1] max_variances

    # integer array containing number of samples in left and right split
    cdef SIZE_t[::1] num_samples

    # integer array containing all indices of features
    cdef SIZE_t[::1] features

    cdef SIZE_t n_features

    # integer describing the best split position as index
    cdef readonly SIZE_t best_split_pos

    # integer describing the index of the best variable
    cdef readonly SIZE_t best_var

    # float describing the best impurity improvement
    cdef readonly  DTYPE_t max_impurity_improvement

    # percentage of numeric targets
    cdef DTYPE_t w_numeric

    # 2D integer array describing all dependencies that are considered under numeric variables
    cdef SIZE_t[:, ::1] numeric_dependency_matrix

    # 2D integer array describing all dependencies that are considered under symbolic variables
    cdef SIZE_t[:, ::1] symbolic_dependency_matrix

    @classmethod
    def from_tree(cls, tree):
        """
        Construct the impurity from a tree

        :param tree: the tree to take the parameters from
        :type tree: jpt.trees.JPT
        :return: the constructed impurity
        :rtype: Impurity
        """
        min_samples_leaf = tree.min_samples_leaf
        numeric_vars = Impurity.get_indices_of_numeric_targets_from_tree(tree)
        symbolic_vars = Impurity.get_indices_of_symbolic_targets_from_tree(tree)
        invert_impurity = Impurity.get_invert_impurity_from_tree(tree)
        n_sym_vars_total = Impurity.count_symbolic_variables_total_from_tree(tree)
        n_num_vars_total = Impurity.count_numeric_variables_total_from_tree(tree)
        numeric_features = Impurity.get_indices_of_numeric_features_from_tree(tree)
        symbolic_features = Impurity.get_indices_of_symbolic_features_from_tree(tree)
        symbols = Impurity.get_size_of_symbolic_variables_domain_from_tree(tree)
        max_variances = Impurity.get_max_variances_from_tree(tree)
        dependency_indices = Impurity.get_dependency_indices_from_tree(tree)
        return cls(min_samples_leaf, numeric_vars, symbolic_vars, invert_impurity,
                   n_sym_vars_total, n_num_vars_total, numeric_features, symbolic_features,
                   symbols, max_variances, dependency_indices)

    def __init__(self, min_samples_leaf, numeric_vars, symbolic_vars, invert_impurity,
                   n_sym_vars_total, n_num_vars_total, numeric_features, symbolic_features,
                   symbols, max_variances, dependency_indices):
        """
        Construct the impurity
        """

        # copy min_samples_leaf
        self.min_samples_leaf = min_samples_leaf

        # initialize data, features, index buffer and indices as None
        self.data = self.feat = self.index_buffer = self.indices = None

        # initialize start and end as -1
        self.start = self.end = -1

        # initialize best_variance index as -1
        self.best_var = -1

        # initialize best_split index as -1
        self.best_split_pos = -1

        # initialize max_impurity_improvement as 0
        self.max_impurity_improvement = 0

        # initialize array of indices of the numeric targets
        self.numeric_vars = numeric_vars

        # initialize array of indices of the symbolic targets
        self.symbolic_vars = symbolic_vars

        # store impurity inversion
        self.invert_impurity = invert_impurity

        # get the number of symbolic targets
        self.n_sym_vars = len(self.symbolic_vars)

        # get number of numeric targets
        self.n_num_vars = len(self.numeric_vars)

        # get the number of all symbolic variables
        self.n_sym_vars_total = n_sym_vars_total

        # get the number of all numeric and integer variables
        self.n_num_vars_total = n_num_vars_total

        # get indices of all targets
        self.all_vars = np.concatenate((self.numeric_vars, self.symbolic_vars))

        # number of all target variables
        self.n_vars = self.all_vars.shape[0]  # len(tree.variables)

        # number of all variables
        self.n_vars_total = self.n_sym_vars_total + self.n_num_vars_total

        # get the indices of numeric features
        self.numeric_features = numeric_features

        # get the indices of symbolic features
        self.symbolic_features = symbolic_features

        # construct all feature indices
        self.features = np.concatenate((self.numeric_features, self.symbolic_features))

        self.n_features = self.features.shape[0]

        # if symbolic targets exist
        if self.n_sym_vars:
            # Thread-invariant buffers

            # get the size of each symbolic variables domain
            self.symbols = symbols

            # get the maximum size of symbolic domains
            self.max_sym_domain = max(self.symbols)

            # initialize a 2D matrix of size (max_sym_domain, n_sym_vars) such that the histograms can be calculated
            self.symbols_total = np.ndarray(
                shape=(self.max_sym_domain, self.n_sym_vars),
                dtype=np.int64
            )
            
            # histograms for symbolic variables left and right of the splits
            self.symbols_left = np.ndarray(
                shape=(self.max_sym_domain, self.n_sym_vars),
                dtype=np.int64
            )
            self.symbols_right = np.ndarray(
                shape=(self.max_sym_domain, self.n_sym_vars),
                dtype=np.int64
            )

            # Symbolic targets require a buffer for improvement calculation
            # initialize the gini improvement per symbolic target
            self.gini_improvements = np.ndarray(
                shape=self.n_sym_vars,
                dtype=np.float64
            )

            # initialize the gini impurities per symbolic target
            self.gini_impurities = np.ndarray(
                shape=self.n_sym_vars,
                dtype=np.float64
            )

            # initialize gini impurities of symbolic targets left of the split
            self.gini_left = np.ndarray(
                shape=self.n_sym_vars,
                dtype=np.float64
            )

            # initialize gini impurities of symbolic targets right of the split
            self.gini_right = np.ndarray(
                shape=self.n_sym_vars,
                dtype=np.float64
            )

        # initialize number of samples in left and right split
        self.num_samples = np.ndarray(shape=2, dtype=np.int64)  # max(max(self.symbols) if self.n_sym_vars else 2, 2)

        # if numeric targets exist
        if self.n_num_vars:

            # Thread-invariant buffers
            self.sums_total = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.sq_sums_total = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.variances_total = np.ndarray(self.n_num_vars, dtype=np.float64)

            # calculate the prior variance of every variable
            self.max_variances = max_variances

            # buffers for left and right splits, similar to symbolic stuff
            self.sums_left = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.sums_right = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.sq_sums_left = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.sq_sums_right = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.variances_left = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.variances_right = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.variance_improvements = np.ndarray(self.n_num_vars, dtype=np.float64)

        # calculate percentage of numeric targets
        self.w_numeric = <DTYPE_t> self.n_num_vars / <DTYPE_t> self.n_vars

        # construct the dependency structure
        dependency_indices = dependency_indices
        cdef int idx_var
        cdef list idc_dep
        cdef SIZE_t[::1] indices


        # create numeric and symbolic dependency matrix.
        cdef SIZE_t n = self.n_vars_total
        cdef SIZE_t t = self.n_vars
        self.numeric_dependency_matrix = np.full(
            (n, t),
            -1,
            dtype=np.int64,
        )
        self.symbolic_dependency_matrix = self.numeric_dependency_matrix.copy()

        for idx_var in self.features:  # For all feature variables...
            # ...get the indices of the dependent numeric variables...
            indices = np.array([
                i_num for i_num, i_var in enumerate(self.numeric_vars) if i_var in dependency_indices[idx_var]
            ], dtype=np.int64)
            if indices.shape[0]:  # ...and store them in the numeric dependency matrix
                self.numeric_dependency_matrix[idx_var, :indices.shape[0]] = indices

            # Get the indices of the dependent numeric variables
            indices = np.array([
                i_sym for i_sym, i_var in enumerate(self.symbolic_vars) if i_var in dependency_indices[idx_var]
            ], dtype=np.int64)
            if indices.shape[0]:  # ... and store them in the numeric dependency matrix.
                self.symbolic_dependency_matrix[idx_var, :indices.shape[0]] = indices

    @classmethod
    def get_indices_of_numeric_targets_from_tree(cls, tree):
        """
        Get the indices of the numeric targets from a tree.
        """
        return np.array([
            <int> i for i, v in enumerate(tree.variables)
            if (v.numeric or v.integer) and v in tree.targets
        ], dtype=np.int64)

    @classmethod
    def get_indices_of_symbolic_targets_from_tree(cls, tree):
        """
        Get the indices of the symbolic targets from a tree.
        """
        return np.array([
            <int> i for i, v in enumerate(tree.variables)
            if v.symbolic and v in tree.targets
        ], dtype=np.int64)

    @classmethod
    def get_invert_impurity_from_tree(cls, tree):
        """
        Get the impurity inversion from a tree.
        """
        return np.array([
            <int> v.invert_impurity for v in tree.variables
            if v.symbolic and v in tree.targets
        ], dtype=np.int64)

    @classmethod
    def count_symbolic_variables_total_from_tree(cls, tree) -> int:
        return len([_ for _ in tree.variables if _.symbolic])

    @classmethod
    def count_numeric_variables_total_from_tree(cls, tree):
        return len([_ for _ in tree.variables if _.numeric or _.integer])

    @classmethod
    def get_indices_of_numeric_features_from_tree(cls, tree):
        return np.array([
            <int> i for i, v in enumerate(tree.variables)
            if (v.numeric or v.integer) and v in tree.features
        ], dtype=np.int64)

    @classmethod
    def get_indices_of_symbolic_features_from_tree(cls, tree):
        return np.array([
            <int> i for i, v in enumerate(tree.variables)
            if v.symbolic and v in tree.features
        ], dtype=np.int64)

    @classmethod
    def get_size_of_symbolic_variables_domain_from_tree(cls, tree):
        return np.array([
            v.domain.n_values
            for v in tree.variables if v.symbolic
        ], dtype=np.int64)

    @classmethod
    def get_max_variances_from_tree(cls, tree):
        return np.array(
                [v._max_std ** 2 for v in tree.variables if v.numeric],
                dtype=np.float64
            )

    @classmethod
    def get_dependency_indices_from_tree(cls, tree):
        # construct the dependency structure
        cdef dict dependency_indices = {}
        cdef int idx_var
        cdef list idc_dep
        cdef SIZE_t[::1] indices
        # convert variable dependency structure to index dependency structure for easy interpretation in cython
        for variable, dep_vars in tree.dependencies.items():
            # get the index version of the dependent variables and store them
            idx_var = tree.variables.index(variable)
            idc_dep = [tree.variables.index(var) for var in dep_vars]
            dependency_indices[idx_var] = idc_dep
        return dependency_indices

    cdef inline int check_max_variances(self, DTYPE_t[::1] variances) nogil:
        """
        Check if a variance in ``variances`` is higher than the initial variances
        
        :param variances: The calculated variances.
        :return: True if there is a higher variance, False if all are lower.
        """
        cdef int i
        for i in range(self.n_num_vars):
            if variances[i] > self.max_variances[i]:
                return True
        return False

    cpdef void setup(Impurity self, DTYPE_t[:, ::1] data, SIZE_t[::1] indices):
        """
        Set data and indices, update features and index_buffer
        
        :param data: the data to set
        :param indices: the indices to set
        """
        self.data = data
        self.feat = np.ndarray(
            shape=data.shape[0],
            dtype=np.float64
        )
        self.indices = indices
        self.index_buffer = np.ndarray(
            shape=indices.shape[0],
            dtype=np.int64
        )

    cpdef int has_numeric_vars_(Impurity self, SIZE_t except_var=-1):
        '''Python variant of ``has_numeric_vars()`` for testing purpose only.'''
        return self.has_numeric_vars(except_var)

    cdef inline int has_numeric_vars(Impurity self, SIZE_t except_var=-1) noexcept nogil:
        """
        :return: number of numeric targets, possibly reduced by 1 if ``except_var`` is
        passed and the variable with that index is also numeric.
        """
        cdef int offset = 0, i
        if except_var >= 0:
            for i in range(self.n_num_vars):
                if self.numeric_vars[i] == except_var:
                    offset = 1
                    break
        return self.n_num_vars - offset

    cdef inline int has_symbolic_vars(Impurity self) noexcept nogil:
        """
        :return: number of symbolic targets 
        """
        return self.n_sym_vars

    cdef inline int has_symbolic_features(Impurity self) noexcept nogil:
        """
        :return: number of symbolic features 
        """
        return self.n_sym_vars_total - self.n_sym_vars

    cdef inline int has_numeric_features(Impurity self) noexcept nogil:
        """
        :return: number of numeric features 
        """
        return self.n_num_vars_total - self.n_num_vars

    cdef inline void gini_impurity(Impurity self,
                                   SIZE_t[:, ::1] counts,
                                   SIZE_t n_samples,
                                   DTYPE_t[::1] result) noexcept nogil:
        """
        Calculate gini impurity of histogram
        
        Following the gini impurity measure (normalized by the number of possible symbolic values:
        ..
        In the uniform distribution: -Gini_u(C) = -|C|/|C| + |C|/|C|^2 = 1/|C| - 1
        
         Gini(C) = 1 / Gini_u(C) * \sum_c (P(c) * (1 - P(c)) = 1 / Gini_u(C) * \sum_c (P(c) - P(c)^2)
        -Gini(C) = 1 / Gini_u(C) * (\sum_c P(c)^2 - \sum_c P(c)) | \sum_c P(c) = 1 
        -Gini(C) = 1 / Gini_u(C) * (\sum_c P(c)^2 - 1)
         Gini(C) = 1 / -Gini_u(C) * (\sum_c P(c)^2 - 1)
         Gini(C) = (\sum_c P(c)^2 - 1) / (1 / |C| - 1)
         
        :param counts: TODO: histogram?
        :param n_samples: number of samples 
        :param result: resulting array to write into, will be overwritten completely
        """
        cdef SIZE_t i, j
        result[...] = 0
        for i in range(self.n_sym_vars):

            # skip variables that only have 1 possible value
            if self.symbols[i] == 1:
                continue

            for j in range(self.symbols[i]):
                result[i] += <DTYPE_t> counts[j, i] * counts[j, i]

            result[i] /= <DTYPE_t> (n_samples * n_samples)
            result[i] -= 1
            result[i] /= 1. / (<DTYPE_t> self.symbols[i]) - 1.

            if self.invert_impurity[i]:
                result[i] = 1 - result[i]

    cpdef SIZE_t _col_is_constant(Impurity self, SIZE_t start, SIZE_t end, SIZE_t col):
        '''For testing only.'''
        return self.col_is_constant(start, end, col)

    cdef inline SIZE_t col_is_constant(Impurity self, SIZE_t start, SIZE_t end, SIZE_t col) noexcept nogil:
        """
        Check if a column in self.data is a constant, i.e. only contains the same value in every row.
        The column is only evaluated between start and end
        :param start: start index of the rows
        :param end: end index of the rows
        :param col: the index of the column
        :return: 1 if it is constant, 0 else 
        """
        cdef DTYPE_t v_ = nan
        cdef DTYPE_t v
        cdef SIZE_t i
        if end - start <= 1:
            return True
        for i in range(start, end):
            v = self.data[self.indices[i], col]
            if isinf(v) or isnan(v):
                return -1
            if v_ == v:
                continue
            if v_ != v:
                if isinf(v_) or isnan(v_):
                    v_ = v
                else: return False
        return True

    cpdef DTYPE_t compute_best_split(self, SIZE_t start, SIZE_t end):
        """
        Calculate the best split on all variables.
        
        
        Note:
        Computation uses the impurity proxy from sklearn: ::

            var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2

        See also: 'https://github.com/scikit-learn/scikit-learn/blob/de1262c35e2aa4ee062d050281ee576ce9e35c94/
                   sklearn/tree/_criterion.pyx#L683'
                   
        :param start:  the start index used in ``self.indices``
        :param end: the end index in ``self.indices``
        :return: The best impurity improvement as a float
        """

        # initialize best variable index
        cdef int best_var = -1

        # take over parameters
        self.start = start
        self.end = end

        # calculate the number of samples
        cdef int n_samples = end - start

        # initialize impurity and gini index
        cdef np.float64_t impurity_total = 0
        cdef np.float64_t gini_total = 0

        # if numeric targets exist
        if self.has_numeric_vars():

            # reset the variances
            self.variances_total[:] = 0

            # calculate square sums of all current data
            sq_sum_at(self.data,
                      self.indices[self.start:self.end],
                      self.numeric_vars,
                      result=self.sq_sums_total)

            # calculate ordinary sums of all current data
            sum_at(self.data,
                   self.indices[self.start:self.end],
                   self.numeric_vars,
                   result=self.sums_total)

            # calculate variances from square and ordinary sums of all current data
            variances(self.sq_sums_total,
                      self.sums_total,
                      n_samples,
                      result=self.variances_total)

            # sanity check to see if the variances "make sense"
            # if not self.check_max_variances(self.variances_total):
            #     return 0

        # if symbolic targets exist
        if self.has_symbolic_vars():

            # compute histogram of all current data
            bincount(self.data,
                     self.indices[self.start:self.end],
                     self.symbolic_vars,
                     result=self.symbols_total)

            # calculate gini impurity of histogram
            self.gini_impurity(self.symbols_total, n_samples, self.gini_impurities)

            gini_total = mean(self.gini_impurities)

        else:
            gini_total = 0


        # int describing if the current variable is symbolic or not
        cdef int symbolic = 0

        # variable for tracking the index of symbolic variables
        cdef int symbolic_idx = -1

        cdef DTYPE_t impurity_improvement
        cdef int variable

        cdef SIZE_t split_pos

        self.index_buffer[:n_samples] = self.indices[self.start:self.end]

        # reset best impurity improvement
        self.max_impurity_improvement = ninf

        # for every feature
        for variable in self.features:

            # check if this variable is symbolic or not
            symbolic = variable in self.symbolic_features

            # increase symbolic index tracking by one if variable is symbolic
            symbolic_idx += symbolic

            # initialize split position
            split_pos = -1

            # evaluate the current variable
            impurity_improvement = self.evaluate_variable(
                variable,
                symbolic,
                symbolic_idx,
                self.variances_total if self.has_numeric_vars() else None,
                gini_total,
                self.index_buffer,
                &split_pos
            )

            # if the best impurity improvement of this variable is better than the current best
            if impurity_improvement > self.max_impurity_improvement:

                # update current best impurity
                self.max_impurity_improvement = impurity_improvement

                # update index of best variable
                self.best_var = variable

                # update the position of the split
                self.best_split_pos = split_pos

                # write back the sorted indices of the best split variable
                self.indices[self.start:self.end] = self.index_buffer[:n_samples]

        # if max impurity improvement has been updated at least once and the best variable is symbolic
        if not isinf(self.max_impurity_improvement) and self.best_var in self.symbolic_features:

            # Rearrange indices to contiguous subsets
            self.move_best_values_to_front(
                self.best_var,
                self.data[self.indices[start + self.best_split_pos],
                          self.best_var],
                &self.best_split_pos
            )

        # return the best improvement value
        return self.max_impurity_improvement

    cdef void move_best_values_to_front(
            self,
            SIZE_t var_idx,
            DTYPE_t value,
            SIZE_t* split_pos
    ) noexcept nogil:
        """
        Move all indices of data points with the specified value from ``split_pos`` on to the
        front of the index array.
        
        :param var_idx: 
        :param value: 
        :param split_pos: pointer to position of the split
        """
        cdef SIZE_t n_samples = self.end - self.start
        cdef int j
        cdef DTYPE_t v
        split_pos[0] = -1
        for j in range(n_samples):
            v = self.data[self.indices[self.start + j], var_idx]
            if v == value:
                v = -1
                split_pos[0] += 1
            self.feat[j] = v
        sort(&self.feat[0], &self.indices[self.start], n_samples)

    cdef DTYPE_t evaluate_variable(
            Impurity self,
            int var_idx,
            int symbolic,
            int symbolic_idx,
            DTYPE_t[::1] variances_total,
            DTYPE_t gini_total,
            SIZE_t[::1] index_buffer,
            SIZE_t* best_split_pos
    ) noexcept nogil:
        """
        Evaluate a variable w. r. t. its possible slit. Calculate the best split on this variable
        and the corresponding impurity.
        
        :param var_idx: the index of the variable in self.data
        :param symbolic: 1 if the variable is symbolic, 0 if numeric
        :param symbolic_idx: 
        :param variances_total: 
        :param gini_total: 
        :param index_buffer: 
        :param best_split_pos: pointer to the position of the best split
        :return: 
        """

        # copy data
        cdef DTYPE_t[:, ::1] data = self.data

        # copy features
        cdef DTYPE_t[::1] f = self.feat

        # initialize max impurity improvement of this variable
        cdef DTYPE_t max_impurity_improvement = ninf

        # copy start and end index
        cdef SIZE_t start = self.start, end = self.end

        # calculate the number of samples
        cdef SIZE_t n_samples = end - start

        # --------------------------------------------------------------------------------------------------------------
        # TODO: Check if sorting really needs a copy of the feature data
        cdef int i, j
        for j in range(n_samples):
            f[j] = data[index_buffer[j], var_idx]
        sort(&f[0], &index_buffer[0], n_samples)
        # --------------------------------------------------------------------------------------------------------------

        # description if this variable is numeric
        cdef int numeric = not symbolic

        # if this variable only contains the same values return ninf
        # if there was a NaN or infinity, return ninf
        cdef int is_constant = self.col_is_constant(start, end, var_idx)
        if is_constant == 1 or is_constant == -1:
            return ninf

        # Prepare the numeric stats
        if self.has_numeric_vars(var_idx):
            self.sums_left[...] = 0
            self.sums_right[...] = self.sums_total
            self.sq_sums_left[...] = 0
            self.sq_sums_right[...] = self.sq_sums_total

        # prepare the symbolic stats
        if self.has_symbolic_vars():
            self.symbols_left[...] = 0
            self.symbols_right[...] = self.symbols_total[...]

        # reset number of samples left and right of the split
        self.num_samples[:] = 0

        # counter for number of samples left and right of the split
        cdef SIZE_t samples_left, samples_right

        # initialize impurity improvement
        cdef DTYPE_t impurity_improvement = 0.

        cdef SIZE_t VAL_IDX
        cdef SIZE_t sample_idx
        cdef int last_iter
        cdef DTYPE_t min_samples
        cdef SIZE_t num_feat_idx = -1

        # check if currently evaluated variable is numeric target variable
        if self.has_numeric_vars(var_idx):
            for i in range(self.n_num_vars):
                if self.numeric_vars[i] == var_idx:
                    num_feat_idx = i
                    break

        cdef int subsequent_equal

        # for every split position as index
        for split_pos in range(n_samples):

            # get the index of the sample considered for the current split
            sample_idx = index_buffer[split_pos]

            # get if this is the last iteration
            last_iter = (symbolic and split_pos == n_samples - 1
                         or numeric and split_pos == n_samples - 2)

            # track number of samples left and right of the split
            self.num_samples[LEFT] += 1
            self.num_samples[RIGHT] = n_samples - self.num_samples[LEFT]
            samples_left = self.num_samples[LEFT]
            samples_right = n_samples - samples_left

            # if it is symbolic
            if symbolic:
                # get the symbolic value
                VAL_IDX = <SIZE_t> data[sample_idx, var_idx]

            # Compute the numeric impurity (variance)
            if self.has_numeric_vars(var_idx):
                self.update_numeric_stats_with_dependencies(
                    sample_idx,
                    self.numeric_dependency_matrix[var_idx, :]
                )

            # Compute the symbolic impurity (Gini index)
            if self.has_symbolic_vars():
                self.update_symbolic_stats_with_dependencies(
                    sample_idx,
                    self.symbolic_dependency_matrix[var_idx, :]
                )

            # Skip calculation for identical values (i.e. until next 'real' split point is reached:
            # for skipping, the sample must not be the last one (1) and consecutive values must be equal (2)
            if numeric or not last_iter and symbolic:
                subsequent_equal = data[index_buffer[split_pos], var_idx] == data[index_buffer[split_pos + 1], var_idx]
                if subsequent_equal:
                    if numeric and last_iter:
                        break
                    if not last_iter:
                        continue

            # reset impurity improvement
            impurity_improvement = 0.

            # if numeric targets exist
            if self.has_numeric_vars(var_idx):
                # calculate variance of left split
                variances(
                    self.sq_sums_left,
                    self.sums_left,
                    samples_left,
                    result=self.variances_left
                )
                # calculate variance of right split
                variances(
                    self.sq_sums_right,
                    self.sums_right,
                    samples_right,
                    result=self.variances_right
                )

                # compute the variance improvement for this split
                impurity_improvement += compute_var_improvements(
                    variances_total,
                    self.variances_left,
                    self.variances_right,
                    samples_left,
                    samples_right,
                    num_feat_idx
                ) * self.w_numeric

            # if symbolic targets exist
            if self.has_symbolic_vars():

                # if total gini impurity is not 0
                if gini_total:

                    # update gini impurity left and right of the split
                    self.gini_impurity(self.symbols_left, samples_left, self.gini_left)
                    self.gini_impurity(self.symbols_right, samples_right, self.gini_right)
                    impurity_improvement += (
                        (gini_total -
                        mean(self.gini_left) * (<DTYPE_t> samples_left / <DTYPE_t> n_samples) -
                        mean(self.gini_right) * (<DTYPE_t> samples_right / <DTYPE_t> n_samples)) / gini_total
                        * (1 - self.w_numeric)
                    )

            # if this variable is symbolic
            if symbolic:

                # if symbolic targets exist
                if self.has_symbolic_vars():
                    self.symbols_left[...] = 0
                    self.symbols_right[...] =  self.symbols_total[...]

                    self.gini_improvements[...] = 0
                    self.num_samples[...] = 0

                # if numeric targets exist
                if self.has_numeric_vars(var_idx):
                    self.sums_left[...] = 0
                    self.sq_sums_left[...] = 0

            # check if this split is legal, according to self.min_samples_leaf
            if samples_left < self.min_samples_leaf or samples_right < self.min_samples_leaf:
                impurity_improvement = ninf

            # check if this split is improving the impurity by the minimal required amount
            if impurity_improvement > max_impurity_improvement:
                max_impurity_improvement = impurity_improvement
                best_split_pos[0] = split_pos

            # break on the last iteration since the "real" last iteration is not needed
            if last_iter:
                break

        return max_impurity_improvement

    cdef inline void update_numeric_stats_with_dependencies(
            Impurity self,
            SIZE_t sample_idx,
            SIZE_t[::1] dependent_columns
    ) noexcept nogil:
        """
        Update the stats of all dependent numeric variables of a variable (numeric of symbolic)
        The sample is considered left of the split and the stats are updated that way.
        :param sample_idx: the index of the samples to retrieve the values from
        :param dependent_columns: the indices of the dependent symbolic variables
        """

        # initialize helper variables
        cdef SIZE_t var, i, dep_var
        cdef DTYPE_t y

        # for every dependent variable
        for i in range(dependent_columns.shape[0]):

            # get the index of the dependent variable
            dep_var = dependent_columns[i]

            # -1 marks the end of a dependency column, so end the method here
            if dep_var == -1:
                return

            # get the value of the variable at sample
            y = self.data[sample_idx, self.numeric_vars[dep_var]]

            # update stats left and right of the value
            self.sums_left[dep_var] += y
            self.sums_right[dep_var] = self.sums_total[dep_var] - self.sums_left[dep_var]
            self.sq_sums_left[dep_var] += y * y
            self.sq_sums_right[dep_var] = self.sq_sums_total[dep_var] - self.sq_sums_left[dep_var]

    cdef inline void update_symbolic_stats_with_dependencies(
            Impurity self,
            SIZE_t sample_idx,
            SIZE_t[::1] dependent_columns
    ) noexcept nogil:
        """
        Update the stats of all dependent symbolic variables of a variable (numeric of symbolic)
        The sample is considered left of the split and the stats are updated that way.
        :param sample_idx: the index of the samples to retrieve the values from
        :param dependent_columns: the indices of the dependent symbolic variables
        """

        # initialize helper variables
        cdef SIZE_t i, j, validx, dep_var
        # for dependent every variable
        for i in range(dependent_columns.shape[0]):

            # get the index of the dependent variable
            dep_var = dependent_columns[i]

            # -1 marks the end of a dependency column, so end the method here
            if dep_var == -1:
                return
            # get the value of the variable at sample
            validx = <SIZE_t> self.data[sample_idx, self.symbolic_vars[dep_var]]
            # update the histogram on the left split
            self.symbols_left[validx, dep_var] += 1

            # update the histogram on the right split
            self.symbols_right[validx, dep_var] = (
                self.symbols_total[validx, dep_var] -
                self.symbols_left[validx, dep_var]
            )

    def to_string(self):
        """
        :return: a table-like string describing the targets and features of this impurity
        """
        return tabulate.tabulate(zip([
            '# symbolic variables',
            'symbolic variables',
            '# numeric variables',
            'numeric variables',
            '# symbolic features',
            'symbolic features',
            '# numeric features',
            'numeric features'
        ], [
            len(self.symbolic_vars),
            ','.join(mapstr(self.symbolic_vars)),
            len(self.numeric_vars),
            ','.join(mapstr(self.numeric_vars)),
            len(self.symbolic_features),
            ','.join(mapstr(self.symbolic_features)),
            len(self.numeric_features),
            ','.join(mapstr(self.numeric_features))
        ]))

# cpdef class PCImpurity(Impurity)

