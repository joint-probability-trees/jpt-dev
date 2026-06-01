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

from ...base.cutils.cutils cimport DTYPE_t, SIZE_t, nan, sort, ninf

# variables declaring that at num_samples[0] are the number of samples left of the split and vice versa
cdef int LEFT = 0
cdef int RIGHT = 1

# Split validation modes: which targets contribute to impurity
cdef int SV_BOTH = 0        # all targets (training + evaluation)
cdef int SV_TRAINING = 1    # training targets only
cdef int SV_EVALUATION = 2  # evaluation targets only


# ----------------------------------------------------------------------------------------------------------------------


cdef inline DTYPE_t compute_var_improvements(
    DTYPE_t[::1] variances_total,
    DTYPE_t[::1] variances_left,
    DTYPE_t[::1] variances_right,
    SIZE_t samples_left,
    SIZE_t samples_right,
    SIZE_t[::1] dependent_columns) noexcept nogil:
    """
    Compute the mean relative variance improvement of a split across the
    dependent numeric targets.

    :param variances_total: The variances before the split
    :param variances_left: The variances of the left side of the split
    :param variances_right: The variances of the right side of the split
    :param samples_left: The amount of samples on the left side of the split
    :param samples_right: The amount of samples on the right side of the split
    :param dependent_columns: indices into the numeric-target buffers to
        iterate over; -1 acts as an end-of-row sentinel.

    :return: double describing the mean per-target relative variance
        improvement, in [0, 1] for non-empty inputs.
    """
    cdef SIZE_t j, i
    cdef DTYPE_t variance_impr = 0
    cdef DTYPE_t[::1] variances_old = variances_total
    cdef DTYPE_t n_samples = <DTYPE_t> samples_left + samples_right
    cdef DTYPE_t divisor = 0

    for j in range(dependent_columns.shape[0]):
        i = dependent_columns[j]

        # -1 marks the end of the dependency row.
        if i == -1:
            break

        # Skip targets with non-positive parent variance — nothing to
        # improve and dividing by zero (or a tiny negative produced by
        # float cancellation in `variances`) would otherwise blow up
        # the per-target ratio.
        if variances_old[i] <= 0:
            continue

        variance_impr = (
            variance_impr * divisor + (
                variances_old[i] -
                (
                    variances_left[i] * <DTYPE_t> samples_left
                    + variances_right[i] * <DTYPE_t> samples_right
                ) / n_samples
            ) / variances_old[i]
        ) / (divisor + 1)
        divisor += 1

    return variance_impr


cpdef inline DTYPE_t _compute_var_improvements(
    DTYPE_t[::1] variances_total,
    DTYPE_t[::1] variances_left,
    DTYPE_t[::1] variances_right,
    SIZE_t samples_left,
    SIZE_t samples_right,
    SIZE_t[::1] dependent_columns
) noexcept nogil:
    """Python-callable version for testing only."""
    return compute_var_improvements(
        variances_total,
        variances_left,
        variances_right,
        samples_left,
        samples_right,
        dependent_columns
    )


# ----------------------------------------------------------------------------------------------------------------------


cdef inline DTYPE_t compute_gini_improvement(
    DTYPE_t g_p,
    DTYPE_t g_l,
    DTYPE_t g_r,
    SIZE_t n_l,
    SIZE_t n_r,
) noexcept nogil:
    """
    Per-target normalised gini reduction in [0, 1] using the parent's raw
    ``ΣP² − 1`` as denominator. Caller must ensure ``g_p != 0``.
    """
    cdef DTYPE_t w_l = <DTYPE_t> n_l / <DTYPE_t> (n_l + n_r)
    cdef DTYPE_t w_r = 1.0 - w_l
    return (g_p - w_l * g_l - w_r * g_r) / g_p


cpdef inline DTYPE_t _compute_gini_improvement(
    DTYPE_t g_p,
    DTYPE_t g_l,
    DTYPE_t g_r,
    SIZE_t n_l,
    SIZE_t n_r,
) noexcept nogil:
    """Python-callable version for testing only."""
    return compute_gini_improvement(g_p, g_l, g_r, n_l, n_r)


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

    # minimum number of EVALUATION samples required in each
    # child partition when split validation is active in
    # SV_EVALUATION mode. Accepts the same int/float-in-(0,1)
    # convention as min_samples_leaf: C45Algorithm.learn()
    # resolves fractions to absolute counts; callers that
    # construct Impurity directly (e.g. via from_tree) may
    # assign the raw value, in which case the comparison
    # against the integer per-side sample count will still
    # behave correctly for absolute thresholds. 0 disables.
    cdef public DTYPE_t min_eval_samples

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

    # double array of per-feature minimum impurity improvement thresholds,
    # aligned with self.features (numeric features first, then symbolic)
    cdef DTYPE_t[::1] min_improvements

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

    # 2D integer array describing all dependencies that are considered under numeric variables
    cdef readonly SIZE_t[:, ::1] numeric_dependency_matrix

    # 2D integer array describing all dependencies that are considered under symbolic variables
    cdef readonly SIZE_t[:, ::1] symbolic_dependency_matrix

    # per-sample mask: 1 = training, 0 = evaluation; None = disabled (all training)
    cdef np.uint8_t[::1] validation_mask

    # split validation mode: SV_BOTH (0), SV_TRAINING (1), SV_EVALUATION (2)
    cdef int validation_mode

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
        min_eval_samples = getattr(tree, 'min_eval_samples', 0)
        numeric_vars = Impurity.get_indices_of_numeric_targets_from_tree(tree)
        symbolic_vars = Impurity.get_indices_of_symbolic_targets_from_tree(tree)
        invert_impurity = Impurity.get_invert_impurity_from_tree(tree)
        n_sym_vars_total = Impurity.count_symbolic_variables_total_from_tree(tree)
        n_num_vars_total = Impurity.count_numeric_variables_total_from_tree(tree)
        numeric_features = Impurity.get_indices_of_numeric_features_from_tree(tree)
        symbolic_features = Impurity.get_indices_of_symbolic_features_from_tree(tree)
        symbols = Impurity.get_size_of_symbolic_variables_domain_from_tree(tree)
        max_variances = Impurity.get_max_variances_from_tree(tree)
        min_improvements = Impurity.get_min_impurity_improvements_from_tree(tree)
        dependency_indices = Impurity.get_dependency_indices_from_tree(tree)
        return cls(min_samples_leaf, numeric_vars, symbolic_vars, invert_impurity,
                   n_sym_vars_total, n_num_vars_total, numeric_features, symbolic_features,
                   symbols, max_variances, min_improvements, dependency_indices,
                   min_eval_samples=min_eval_samples)

    def __init__(self, min_samples_leaf, numeric_vars, symbolic_vars, invert_impurity,
                   n_sym_vars_total, n_num_vars_total, numeric_features, symbolic_features,
                   symbols, max_variances, min_improvements, dependency_indices,
                   min_eval_samples=0):
        """
        Construct the impurity
        """

        # copy min_samples_leaf
        self.min_samples_leaf = min_samples_leaf

        # copy minimum evaluation samples per child
        self.min_eval_samples = min_eval_samples

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

        # per-feature minimum impurity improvement thresholds,
        # aligned with self.features (numeric features first, then symbolic)
        self.min_improvements = min_improvements

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

        # Exclude idx_var from its own dependency rows so a feature
        # that is also a target does not score impurity reduction
        # against itself (this replaces the old skip_idx parameter).
        for idx_var in self.features:  # For all feature variables...
            # ...get the indices of the dependent numeric variables...
            indices = np.array([
                i_num for i_num, i_var in enumerate(self.numeric_vars)
                if i_var in dependency_indices[idx_var] and i_var != idx_var
            ], dtype=np.int64)
            if indices.shape[0]:  # ...and store them in the numeric dependency matrix
                self.numeric_dependency_matrix[idx_var, :indices.shape[0]] = indices

            # Get the indices of the dependent symbolic variables
            indices = np.array([
                i_sym for i_sym, i_var in enumerate(self.symbolic_vars)
                if i_var in dependency_indices[idx_var] and i_var != idx_var
            ], dtype=np.int64)
            if indices.shape[0]:  # ... and store them in the symbolic dependency matrix.
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
            for v in tree.variables
            if v.symbolic and v in tree.targets
        ], dtype=np.int64)

    @classmethod
    def get_max_variances_from_tree(cls, tree):
        return np.array(
            [
                v._max_std ** 2 if v.numeric else 0.
                for v in tree.variables
                if (v.numeric or v.integer) and v in tree.targets
            ],
            dtype=np.float64
        )

    @classmethod
    def get_min_impurity_improvements_from_tree(cls, tree):
        """
        Return per-feature minimum impurity improvement thresholds as a
        float64 array aligned with ``self.features`` (numeric features
        first, then symbolic features).
        """
        return np.array(
            [
                v.min_impurity_improvement
                for v in tree.variables
                if (v.numeric or v.integer) and v in tree.features
            ] + [
                v.min_impurity_improvement
                for v in tree.variables
                if v.symbolic and v in tree.features
            ],
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
        cdef int any_limit = 0
        for i in range(self.n_num_vars):
            if self.max_variances[i] > 0:
                any_limit = 1
                if variances[i] > self.max_variances[i]:
                    return True
        # If no max_std constraints are set at all, never short-circuit splitting
        if not any_limit:
            return True
        return False

    cpdef void setup(Impurity self, DTYPE_t[:, ::1] data, SIZE_t[::1] indices,
                     np.uint8_t[::1] validation_mask=None, int validation_mode=0):
        """
        Set data and indices, update features and index_buffer

        :param data: the data to set
        :param indices: the indices to set
        :param validation_mask: per-sample mask (1=training, 0=evaluation), or None to disable
        :param validation_mode: SV_BOTH (0), SV_TRAINING (1), or SV_EVALUATION (2)
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
        self.validation_mask = validation_mask
        self.validation_mode = validation_mode

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
        Per-target raw ``ΣP(c)² − 1`` from the histogram counts.

        Always ≤ 0; equals 0 when the partition is pure (≤ 1 class
        actually present), the target has a single-value domain, or the
        partition is empty. Callers compose the per-target normalised
        reduction via ``compute_gini_improvement``.

        The whole ``result`` row is overwritten; slots that the caller
        did not feed valid counts for (e.g. non-dependent targets whose
        ``counts[:, i]`` was never updated) get a syntactically-valid
        but semantically-meaningless raw gini and must not be read.
        """
        cdef SIZE_t i, j, n_local
        cdef DTYPE_t sum_sq
        result[...] = 0
        for i in range(self.n_sym_vars):

            # skip targets that only have 1 possible value
            if self.symbols[i] == 1:
                continue

            # Count classes actually present in this partition;
            # if at most one is present, the partition is pure.
            n_local = 0
            sum_sq = 0
            for j in range(self.symbols[i]):
                if counts[j, i] > 0:
                    n_local += 1
                sum_sq += <DTYPE_t> counts[j, i] * counts[j, i]

            if n_local <= 1 or n_samples <= 0:
                result[i] = 0
            else:
                result[i] = sum_sq / <DTYPE_t> (n_samples * n_samples) - 1.0

    cpdef void _gini_impurity(
            Impurity self,
            SIZE_t[:, ::1] counts,
            SIZE_t n_samples,
            DTYPE_t[::1] result,
    ):
        """Python-callable version of ``gini_impurity`` for testing only."""
        self.gini_impurity(counts, n_samples, result)

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

        # ---- Split validation: build filtered row indices for impurity computation ----
        # When validation_mode != SV_BOTH, only a subset of samples contributes to
        # the impurity totals and running stats. Build that subset here.
        cdef SIZE_t[::1] impurity_rows
        cdef SIZE_t n_impurity_samples
        cdef SIZE_t ii

        if self.validation_mask is not None and self.validation_mode != SV_BOTH:
            # Count matching samples first
            n_impurity_samples = 0
            for ii in range(n_samples):
                if self.validation_mode == SV_TRAINING:
                    if self.validation_mask[self.indices[self.start + ii]] == 1:
                        n_impurity_samples += 1
                else:  # SV_EVALUATION
                    if self.validation_mask[self.indices[self.start + ii]] == 0:
                        n_impurity_samples += 1

            # Build the filtered index array
            impurity_rows = np.empty(n_impurity_samples, dtype=np.int64)
            n_impurity_samples = 0
            for ii in range(n_samples):
                if self.validation_mode == SV_TRAINING:
                    if self.validation_mask[self.indices[self.start + ii]] == 1:
                        impurity_rows[n_impurity_samples] = self.indices[self.start + ii]
                        n_impurity_samples += 1
                else:  # SV_EVALUATION
                    if self.validation_mask[self.indices[self.start + ii]] == 0:
                        impurity_rows[n_impurity_samples] = self.indices[self.start + ii]
                        n_impurity_samples += 1
        else:
            impurity_rows = self.indices[self.start:self.end]
            n_impurity_samples = n_samples

        # initialize impurity index
        cdef np.float64_t impurity_total = 0

        # if numeric targets exist
        if self.has_numeric_vars():

            # reset the variances
            self.variances_total[:] = 0

            # calculate square sums of all current data
            sq_sum_at(self.data,
                      impurity_rows,
                      self.numeric_vars,
                      result=self.sq_sums_total)

            # calculate ordinary sums of all current data
            sum_at(self.data,
                   impurity_rows,
                   self.numeric_vars,
                   result=self.sums_total)

            # calculate variances from square and ordinary sums of all current data
            variances(self.sq_sums_total,
                      self.sums_total,
                      n_impurity_samples,
                      result=self.variances_total)

            # if all numeric targets are within their max_std limits,
            # no split is needed for their benefit
            if not self.check_max_variances(self.variances_total):
                if not self.has_symbolic_vars():
                    return ninf

        # if symbolic targets exist
        if self.has_symbolic_vars():

            # compute histogram of all current data
            bincount(self.data,
                     impurity_rows,
                     self.symbolic_vars,
                     result=self.symbols_total)

            # calculate per-target raw ΣP² − 1 of the parent histogram;
            # this is the g_p input fed to compute_gini_improvement and
            # also the per-target pure-parent skip signal.
            self.gini_impurity(self.symbols_total, n_impurity_samples, self.gini_impurities)


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

        # index tracking position within self.features (for min_improvements lookup)
        cdef SIZE_t feat_idx = 0

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
                self.index_buffer,
                &split_pos,
                n_impurity_samples
            )

            # if the best impurity improvement of this variable is better than
            # the current best AND meets this variable's per-feature threshold
            if (
                impurity_improvement > self.max_impurity_improvement
                and impurity_improvement >= self.min_improvements[feat_idx]
            ):

                # update current best impurity
                self.max_impurity_improvement = impurity_improvement

                # update index of best variable
                self.best_var = variable

                # update the position of the split
                self.best_split_pos = split_pos

                # write back the sorted indices of the best split variable
                self.indices[self.start:self.end] = self.index_buffer[:n_samples]

            feat_idx += 1

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
            SIZE_t[::1] index_buffer,
            SIZE_t* best_split_pos,
            SIZE_t n_impurity_samples
    ) noexcept nogil:
        """
        Evaluate a variable w. r. t. its possible split. Calculate the best split on this variable
        and the corresponding impurity.

        :param var_idx: the index of the variable in self.data
        :param symbolic: 1 if the variable is symbolic, 0 if numeric
        :param symbolic_idx:
        :param variances_total:
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

        # reset number of samples left and right of the split (impurity-relevant samples)
        self.num_samples[:] = 0

        # counter for number of samples left and right of the split
        cdef SIZE_t samples_left, samples_right

        # total sample counters (all samples, for min_samples_leaf check)
        cdef SIZE_t total_left = 0, total_right

        # initialize impurity improvement
        cdef DTYPE_t impurity_improvement = 0.

        cdef SIZE_t VAL_IDX
        cdef SIZE_t sample_idx
        cdef int last_iter
        cdef DTYPE_t min_samples

        # ---- Active dependent target counts for the modality weighting ----
        # Per-variable counts (not per-tree): targets that actually carry
        # reducible impurity and that this variable is allowed to optimise
        # against. variances_total may be None when there are no numeric
        # targets at all.
        cdef SIZE_t[::1] num_dep_row = self.numeric_dependency_matrix[var_idx, :]
        cdef SIZE_t[::1] sym_dep_row = self.symbolic_dependency_matrix[var_idx, :]
        cdef SIZE_t n_num_active = 0
        cdef SIZE_t n_sym_active = 0
        cdef SIZE_t dep_idx

        if variances_total is not None:
            for j in range(num_dep_row.shape[0]):
                dep_idx = num_dep_row[j]
                if dep_idx == -1:
                    break
                if variances_total[dep_idx] > 0:
                    n_num_active += 1

        if self.has_symbolic_vars():
            for j in range(sym_dep_row.shape[0]):
                dep_idx = sym_dep_row[j]
                if dep_idx == -1:
                    break
                if self.gini_impurities[dep_idx] != 0:
                    n_sym_active += 1

        cdef DTYPE_t w_num_local = 0.
        if n_num_active + n_sym_active > 0:
            w_num_local = (
                <DTYPE_t> n_num_active
                / <DTYPE_t> (n_num_active + n_sym_active)
            )

        cdef DTYPE_t sym_score
        cdef DTYPE_t sym_sum
        cdef DTYPE_t imp_orig
        cdef DTYPE_t g_p

        cdef int subsequent_equal

        # ---- Split validation locals ----
        # has_mask: 1 if validation mask is active, 0 otherwise (safe for nogil)
        cdef int has_mask = self.validation_mask is not None
        cdef int sv_mode = self.validation_mode
        # is_training: whether current sample is a training sample
        cdef int is_training
        # should_update: whether current sample contributes to impurity stats
        cdef int should_update
        # is_candidate: whether current sample can be a split candidate
        cdef int is_candidate

        # for every split position as index
        for split_pos in range(n_samples):

            # get the index of the sample considered for the current split
            sample_idx = index_buffer[split_pos]

            # get if this is the last iteration
            last_iter = (symbolic and split_pos == n_samples - 1
                         or numeric and split_pos == n_samples - 2)

            # ---- Split validation: determine sample role ----
            if has_mask:
                is_training = self.validation_mask[sample_idx] == 1
                # Determine if this sample's targets should update running stats
                if sv_mode == SV_BOTH:
                    should_update = 1
                elif sv_mode == SV_TRAINING:
                    should_update = is_training
                else:  # SV_EVALUATION
                    should_update = not is_training
                # Only training feature values can be candidate split points
                is_candidate = is_training
            else:
                is_training = 1
                should_update = 1
                is_candidate = 1

            # track total number of samples left (for min_samples_leaf, always counted)
            total_left += 1
            total_right = n_samples - total_left

            # track number of impurity-relevant samples left and right of the split
            if should_update:
                self.num_samples[LEFT] += 1
            self.num_samples[RIGHT] = n_impurity_samples - self.num_samples[LEFT]
            samples_left = self.num_samples[LEFT]

            # if it is symbolic
            if symbolic:
                # get the symbolic value
                VAL_IDX = <SIZE_t> data[sample_idx, var_idx]

            # Update running stats only for impurity-relevant samples
            if should_update:

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

            # ---- Split validation: skip candidate evaluation for non-training samples ----
            if not is_candidate:
                # For symbolic variables, still need to reset stats after each unique value group
                if symbolic:
                    if self.has_symbolic_vars():
                        self.symbols_left[...] = 0
                        self.symbols_right[...] = self.symbols_total[...]
                        self.gini_improvements[...] = 0
                        self.num_samples[...] = 0
                    if self.has_numeric_vars(var_idx):
                        self.sums_left[...] = 0
                        self.sums_right[...] = self.sums_total
                        self.sq_sums_left[...] = 0
                        self.sq_sums_right[...] = self.sq_sums_total
                if last_iter:
                    break
                continue

            # reset impurity improvement
            impurity_improvement = 0.

            samples_right = self.num_samples[RIGHT]

            # if numeric targets exist and any of them is active
            if n_num_active > 0:
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

                # compute mean per-target variance improvement, restricted
                # to dependent numeric targets (the dep row already
                # excludes var_idx itself).
                impurity_improvement += compute_var_improvements(
                    variances_total,
                    self.variances_left,
                    self.variances_right,
                    samples_left,
                    samples_right,
                    num_dep_row
                ) * w_num_local

            # if symbolic targets exist and any of them is active
            if n_sym_active > 0:

                # raw ΣP² − 1 of each child histogram per target
                self.gini_impurity(self.symbols_left, samples_left, self.gini_left)
                self.gini_impurity(self.symbols_right, samples_right, self.gini_right)

                # mean of the per-target normalised reductions across the
                # active dependent symbolic targets; inverted targets
                # contribute 1 − imp_orig instead of imp_orig.
                sym_sum = 0.
                for j in range(sym_dep_row.shape[0]):
                    dep_idx = sym_dep_row[j]
                    if dep_idx == -1:
                        break
                    g_p = self.gini_impurities[dep_idx]
                    # pure parent → no reducible impurity; skip uniformly
                    # in both orientations (do NOT translate to 1 − 0 = 1)
                    if g_p == 0:
                        continue
                    imp_orig = compute_gini_improvement(
                        g_p,
                        self.gini_left[dep_idx],
                        self.gini_right[dep_idx],
                        samples_left,
                        samples_right,
                    )
                    if self.invert_impurity[dep_idx]:
                        sym_sum += 1.0 - imp_orig
                    else:
                        sym_sum += imp_orig

                sym_score = sym_sum / <DTYPE_t> n_sym_active
                impurity_improvement += sym_score * (1.0 - w_num_local)

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
                    self.sums_right[...] = self.sums_total
                    self.sq_sums_left[...] = 0
                    self.sq_sums_right[...] = self.sq_sums_total

            # check if this split is legal, according to self.min_samples_leaf
            # Use total sample counts (all samples), not just impurity-relevant ones
            if total_left < self.min_samples_leaf or total_right < self.min_samples_leaf:
                impurity_improvement = ninf

            # reject splits where either child has too few evaluation samples.
            # Only active in SV_EVALUATION mode: in that mode num_samples[LEFT/RIGHT]
            # already hold per-side eval-row counts (should_update == not is_training).
            if (has_mask
                    and sv_mode == SV_EVALUATION
                    and self.min_eval_samples > 0
                    and (samples_left < self.min_eval_samples
                         or samples_right < self.min_eval_samples)):
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

