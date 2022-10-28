# cython: auto_cpdef=True
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
from libc.stdio cimport printf

from dnutils import mapstr

from ..base.cutils cimport DTYPE_t, SIZE_t, mean, nan, sort

# variables declaring that at num_samples[0] are the number of samples left of the split and vice versa
cdef int LEFT = 0
cdef int RIGHT = 1


# ----------------------------------------------------------------------------------------------------------------------

cdef inline DTYPE_t compute_var_improvements(DTYPE_t[::1] variances_total,
                                   DTYPE_t[::1] variances_left,
                                   DTYPE_t[::1] variances_right,
                                   SIZE_t samples_left,
                                   SIZE_t samples_right) nogil:
    """
    Compute the variance improvement of a split. 
    
   :param variances_total: The variances before the split
   :param variances_left: The variances of the left side of the split
   :param variances_right: The variances of the right side of the split
   :param samples_left: The amount of samples on the left side of the split
   :param samples_right: The amount of samples on the right side of the split
   :return: double describing the relative variance improvement
   """
    # result[:] = variances_total
    cdef SIZE_t i
    cdef DTYPE_t result = mean(variances_total)
    cdef DTYPE_t variances_new = 0
    cdef DTYPE_t n_samples = <DTYPE_t> samples_left + samples_right

    for i in range(variances_total.shape[0]):
        variances_new += ((variances_left[i] * <DTYPE_t> samples_left
                           + variances_right[i] * <DTYPE_t> samples_right) / n_samples)
    variances_new /= <DTYPE_t> variances_total.shape[0]
    return (result - variances_new) / result
    # for i in range(variances_total.shape[0]):
    #     if variances_total[i]:
    #         result[i] /= variances_total[i]
    #     else:
    #         result[i] = 0


# ----------------------------------------------------------------------------------------------------------------------

cdef inline void sum_at(DTYPE_t[:, ::1] M,
                        SIZE_t[::1] rows,
                        SIZE_t[::1] cols,
                        DTYPE_t[::1] result) nogil:
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


# ----------------------------------------------------------------------------------------------------------------------

cdef inline void sq_sum_at(DTYPE_t[:, ::1] M,
                           SIZE_t[::1] rows,
                           SIZE_t[::1] cols,
                           DTYPE_t[::1] result) nogil:
    """
    Square the values in the rows and sum them..
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


# ----------------------------------------------------------------------------------------------------------------------

cdef inline void variances(DTYPE_t[::1] sq_sums,
                           DTYPE_t[::1] sums,
                           SIZE_t n_samples,
                           DTYPE_t[::1] result) nogil:
    """
    Variance computation uses the proxy from sklearn: ::

    var = \sum_i^n (y_i - y_bar) ** 2
    = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2

    See also: 'https://github.com/scikit-learn/scikit-learn/blob/de1262c35e2aa4ee062d050281ee576ce9e35c94/sklearn/
               tree/_criterion.pyx#L683'
    :param sq_sums: The square sums 
    :param sums: The ordinary sums
    :param n_samples: the number of samples
    :param result: The array to write into (result will be overwritten)
    """
    result[:] = sq_sums
    cdef SIZE_t i
    for i in range(sums.shape[0]):
        result[i] -= sums[i] * sums[i] / <DTYPE_t> n_samples
        result[i] /= n_samples - 1


# ----------------------------------------------------------------------------------------------------------------------
# in-place vector addition

cdef inline void ivadd(DTYPE_t[::1] target, DTYPE_t[::1] arg, SIZE_t n, int sq=False) nogil:
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
                          SIZE_t[:, ::1] result) nogil:
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

cdef inline void standardize(DTYPE_t[:, ::1] data,
                             DTYPE_t[::1] sums,
                             DTYPE_t[::1] variances,
                             DTYPE_t[:, ::1] result):
    """
    Standardize ``data`` with respect to mean and variance.
    :param data: the original data
    :param sums: the sums of each column of the data (not divided by the number of samples)
    :param variances: the variances of each column of the data
    :param result: the result to write into
    """

    # initialize indices
    cdef SIZE_t row_index
    cdef SIZE_t column_index

    # initialize statistics
    cdef DTYPE_t num_samples
    cdef DTYPE_t mean
    cdef DTYPE_t variance
    cdef DTYPE_t standard_deviation
    cdef DTYPE_t datapoint

    # get number of samples
    num_samples = <DTYPE_t> len(data)

    # for every column
    for column_index in range(0, result.shape[1]):

        # calculate mean
        mean = sums[column_index] / num_samples

        # get variance
        variance = variances[column_index]

        standard_deviation = (variance**0.5)

        # for every row
        for row_index in range(0, result.shape[0]):

            # get datapoint
            datapoint = data[row_index, column_index]

            # standardize data
            result[row_index, column_index] = (datapoint - mean) / standard_deviation


cdef inline void pca(DTYPE_t[:, ::1] data,  DTYPE_t[::1] eigenvalues_result, DTYPE_t[:, ::1] eigenvectors_result):
    """
    Compute PCA on a normalized dataset
    :param data: the data to decompose
    :param eigenvalues_result: the result to write the eigenvalues in
    :param eigenvectors_result: the result to write the eigenvectors in
    """

    # initialize covariance
    cdef DTYPE_t[:, ::1] covariance
    covariance = np.ndarray((data.shape[1], data.shape[1]), order="C")

    # initialize eigenvalues
    cdef DTYPE_t[:] eigenvalues
    eigenvalues = np.ndarray(data.shape[1])

    # initialize eigenvectors
    cdef DTYPE_t[:, :] eigenvectors
    eigenvectors = np.ndarray((data.shape[1], data.shape[1]))

    # compute covariance (seems to work)
    covariance = np.cov(data.T)

    # calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # copy eigenvalues to result as c contiguous array
    cdef SIZE_t index
    for index in range(len(eigenvalues)):
        eigenvalues_result[index] = eigenvalues[index]

    # copy eigenvectors to result as c contiguous array
    cdef SIZE_t row_index
    cdef SIZE_t column_index
    for row_index in range(eigenvectors.shape[0]):
        for column_index in range(eigenvectors.shape[1]):
            eigenvectors_result[row_index, column_index] = eigenvectors[row_index, column_index]


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

    # the indices of the sorted datapoints used for TODO
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

    # integer describing the best split position as index
    cdef readonly SIZE_t best_split_pos

    # integer describing the index of the best variable
    cdef readonly SIZE_t best_var

    # float describing the best impurity improvement
    cdef readonly  DTYPE_t max_impurity_improvement

    # percentage of numeric targets
    cdef DTYPE_t w_numeric

    # 2D integer array describing all dependencies that are considered under all variables
    cdef SIZE_t[:, ::1] dependency_matrix

    # 2D integer array describing all dependencies that are considered under numeric variables
    cdef SIZE_t[:, ::1] numeric_dependency_matrix

    # 2D integer array describing all dependencies that are considered under symbolic variables
    cdef SIZE_t[:, ::1] symbolic_dependency_matrix

    def __init__(self, tree):
        """
        Construct the impurity

        :param tree: the tree to take the parameters from
        :type tree: jpt.trees.JPT
        """

        # copy min_samples_leaf
        self.min_samples_leaf = tree.min_samples_leaf

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
        self.numeric_vars = np.array([<int> i for i, v in enumerate(tree.variables)
                                      if v.numeric and (tree.targets is None or v in tree.targets)],
                                     dtype=np.int64)

        # initialize array of indices of the symbolic targets
        self.symbolic_vars = np.array([<int> i for i, v in enumerate(tree.variables)
                                       if v.symbolic and (tree.targets is None or v in tree.targets)],
                                      dtype=np.int64)

        # get the number of symbolic targets
        self.n_sym_vars = len(self.symbolic_vars)

        # get number of numeric targets
        self.n_num_vars = len(self.numeric_vars)

        # get the number of all symbolic variables
        self.n_sym_vars_total = len([_ for _ in tree.variables if _.symbolic])

        # get the number of all numeric variables
        self.n_num_vars_total = len([_ for _ in tree.variables if _.numeric])

        # get indices of all targets
        self.all_vars = np.concatenate((self.numeric_vars, self.symbolic_vars))

        # number of all target variables
        self.n_vars = self.all_vars.shape[0]  # len(tree.variables)

        # number of all variables
        self.n_vars_total = self.n_sym_vars_total + self.n_num_vars_total

        # get the indices of numeric features
        self.numeric_features = np.array([i for i, v in enumerate(tree.variables)
                                         if v.numeric and (tree.targets is None or v not in tree.targets)],
                                         dtype=np.int64)
        # get the indices of symbolic features
        self.symbolic_features = np.array([i for i, v in enumerate(tree.variables)
                                          if v.symbolic and (tree.targets is None or v not in tree.targets)],
                                         dtype=np.int64)
        # construct all feature indices
        self.features = np.concatenate((self.numeric_features, self.symbolic_features))

        # if symbolic targets exist
        if self.n_sym_vars:
            # Thread-invariant buffers

            # get the size of each symbolic variables domain
            self.symbols = np.array([v.domain.n_values for v in tree.variables if v.symbolic], dtype=np.int64)

            # get the maximum size of symbolic domains
            self.max_sym_domain = max(self.symbols)

            # initialize a 2D matrix of size (max_sym_domain, n_sym_vars) such that the histograms can be calculated
            self.symbols_total = np.ndarray(shape=(self.max_sym_domain, self.n_sym_vars), dtype=np.int64)
            
            # histograms for symbolic variables left and right of the splits
            self.symbols_left = np.ndarray(shape=(self.max_sym_domain, self.n_sym_vars), dtype=np.int64)
            self.symbols_right = np.ndarray(shape=(self.max_sym_domain, self.n_sym_vars), dtype=np.int64)

            # Symbolic targets require a buffer for improvement calculation
            # initialize the gini improvement per symbolic target
            self.gini_improvements = np.ndarray(shape=self.n_sym_vars, dtype=np.float64)

            # initialize the gini impurities per symbolic target
            self.gini_impurities = np.ndarray(shape=self.n_sym_vars, dtype=np.float64)

            # initialize gini impurities of symbolic targets left of the split
            self.gini_left = np.ndarray(shape=self.n_sym_vars, dtype=np.float64)

            # initialize gini impurities of symbolic targets right of the split
            self.gini_right = np.ndarray(shape=self.n_sym_vars, dtype=np.float64)

        # initialize number of samples in left and right split
        self.num_samples = np.ndarray(shape=2, dtype=np.int64)  # max(max(self.symbols) if self.n_sym_vars else 2, 2)

        # if numeric targets exist
        if self.n_num_vars:

            # Thread-invariant buffers
            self.sums_total = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.sq_sums_total = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.variances_total = np.ndarray(self.n_num_vars, dtype=np.float64)

            # calculate the prior variance of every variable
            self.max_variances = np.array([v._max_std ** 2 for v in tree.variables if v.numeric], dtype=np.float64)

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

        # copy the dependency structure
        self.dependency_matrix = tree.dependency_matrix
        self.numeric_dependency_matrix = tree.numeric_dependency_matrix
        self.symbolic_dependency_matrix = tree.symbolic_dependency_matrix

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

    cpdef void setup(Impurity self, DTYPE_t[:, ::1] data, SIZE_t[::1] indices) except +:
        """
        Set data and indices, update features and index_buffer
        
        :param data: the data to set
        :param indices: the indices to set
        """
        self.data = data
        self.feat = np.ndarray(shape=data.shape[0], dtype=np.float64)
        self.indices = indices
        self.index_buffer = np.ndarray(shape=indices.shape[0], dtype=np.int64)

    cdef inline int has_numeric_vars(Impurity self) nogil:
        """
        :return: number of numeric targets
        """
        return self.n_num_vars

    cdef inline int has_symbolic_vars(Impurity self) nogil:
        """
        :return: number of symbolic targets 
        """
        return self.n_sym_vars

    cdef inline int has_symbolic_features(Impurity self) nogil:
        """
        :return: number of symbolic features 
        """
        return self.n_sym_vars_total - self.n_sym_vars

    cdef inline int has_numeric_features(Impurity self) nogil:
        """
        :return: number of numeric features 
        """
        return self.n_num_vars_total - self.n_num_vars

    cdef inline void gini_impurity(Impurity self, SIZE_t[:, ::1] counts, SIZE_t n_samples, DTYPE_t[::1] result) nogil:
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
            for j in range(self.symbols[i]):
                result[i] += <DTYPE_t> counts[j, i] * counts[j, i]
            result[i] /= <DTYPE_t> (n_samples * n_samples)
            result[i] -= 1
            result[i] /= 1. / (<DTYPE_t> self.symbols[i]) - 1.

    cdef inline SIZE_t col_is_constant(Impurity self, SIZE_t start, SIZE_t end, SIZE_t col) nogil except -1:
        """
        Check if a column in self.data is a constant, i.e. only contains the same value in every row.
        The column is only evaluated between start and end
        :param start: start index of the rows
        :param end: end index of the rows
        :param col: the index of the column
        :return: 1 if it is constant, 0 else 
        """
        cdef DTYPE_t v_ = nan, v
        cdef SIZE_t i
        if end - start <= 1:
            return True
        for i in range(start, end):
            v = self.data[self.indices[i], col]
            if v != v:
                return -1
            if v_ == v: continue
            if v_ != v:
                if v_ != v_: v_ = v  # NB: x != x is True iff x is NaN or inf
                else: return False
        return True

    cpdef DTYPE_t compute_best_split(self, SIZE_t start, SIZE_t end) except -1:
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

        # calculate number of samples
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
            if not self.check_max_variances(self.variances_total):
                return 0

        # if symbolic targets exist
        if self.has_symbolic_vars():

            # compute histogram of all current data
            bincount(self.data,
                     self.indices[self.start:self.end],
                     self.symbolic_vars,
                     result=self.symbols_total)

            # calculate gini impurity of histogram
            self.gini_impurity(self.symbols_total, n_samples, self.gini_impurities)

            # save total gini impurity as mean of all symbolic dimensions impurities
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
        self.max_impurity_improvement = 0

        # for every feature
        for variable in self.features:

            # check if this variable is symbolic or not
            symbolic = variable in self.symbolic_features

            # increase symbolic index tracking by one if variable is symbolic
            symbolic_idx += symbolic

            # initialize split position
            split_pos = -1

            # evaluate the current variable
            impurity_improvement = self.evaluate_variable(variable,
                                                          symbolic,
                                                          symbolic_idx,
                                                          self.variances_total if self.has_numeric_vars() else None,
                                                          gini_total,
                                                          self.index_buffer,
                                                          &split_pos)

            # if the best impurity improvement of this variable is better than the current best
            if impurity_improvement > self.max_impurity_improvement:

                # update current best impurity
                self.max_impurity_improvement = impurity_improvement

                # update index of best variable
                self.best_var = variable

                # update the position of the split
                self.best_split_pos = split_pos

                # TODO IDK
                self.indices[self.start:self.end] = self.index_buffer[:n_samples]

        # if max impurity improvement has been updated at least once and the best variable is symbolic
        if self.max_impurity_improvement and self.best_var in self.symbolic_features:

            # TODO IDK
            self.move_best_values_to_front(self.best_var,
                                           self.data[self.indices[start + self.best_split_pos],
                                                     self.best_var],
                                           &self.best_split_pos)

        # return the best improvement value
        return self.max_impurity_improvement

    cdef void move_best_values_to_front(self, SIZE_t var_idx, DTYPE_t value, SIZE_t* split_pos):  #nogil
        """
        TODO IDK
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

    cdef DTYPE_t evaluate_variable(Impurity self,
                                   int var_idx,
                                   int symbolic,
                                   int symbolic_idx,
                                   DTYPE_t[::1] variances_total,
                                   DTYPE_t gini_total,
                                   SIZE_t[::1] index_buffer,
                                   SIZE_t* best_split_pos) nogil except -1:
        """
        Evaluate a variable w. r. t. its possible slit. Calculate the best split on this variable
        and the corresponding impurity.
        
        TODO Document source code
        
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
        cdef DTYPE_t max_impurity_improvement = 0

        # copy start and end index
        cdef SIZE_t start = self.start, end = self.end

        # calculate number of samples
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

        # if this variable only contains the same values return 0
        cdef int is_constant = self.col_is_constant(start, end, var_idx)
        if is_constant == 1:
            return 0

        # if there was a NaN or infinity, return -1
        elif is_constant == -1:
            return -1

        # Prepare the numeric stats
        if self.has_numeric_vars():
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

        # initialize current impurity improvement
        cdef DTYPE_t tmp_impurity_impr

        cdef SIZE_t VAL_IDX
        cdef SIZE_t sample_idx
        cdef int last_iter
        cdef DTYPE_t min_samples

        # for every split position as index
        for split_pos in range(n_samples):

            # get the index of the sample considered for the current split
            sample_idx = index_buffer[split_pos]

            # get if this is the last iteration
            last_iter = (symbolic and split_pos == n_samples - 1
                         or numeric and split_pos == n_samples - 2)

            # if this variable is numeric
            if numeric:
                # track number of samples left and right of the split
                self.num_samples[LEFT] += 1
                self.num_samples[RIGHT] = <SIZE_t> n_samples - split_pos - 1
                samples_left = self.num_samples[LEFT]
                samples_right = self.num_samples[RIGHT]

            # if it is symbolic
            else:

                # get the symolbic value
                VAL_IDX = <SIZE_t> data[sample_idx, var_idx]

                # track number of samples left and right of the split
                self.num_samples[LEFT] += 1
                self.num_samples[RIGHT] = n_samples - self.num_samples[LEFT]
                samples_left = self.num_samples[LEFT]
                samples_right = n_samples - samples_left

            # Compute the numeric impurity (variance)
            if self.has_numeric_vars():
                self.update_numeric_stats_with_dependencies(sample_idx,
                                                            self.numeric_dependency_matrix[var_idx, :])

            # Compute the symbolic impurity (Gini index)
            if self.has_symbolic_vars():
                self.update_symbolic_stats_with_dependencies(sample_idx,
                                                             self.symbolic_dependency_matrix[var_idx, :])

            # Skip calculation for identical values (i.e. until next 'real' splitpoint is reached:
            # for skipping, the sample must not be the last one (1) and consequtive values must be equal (2)
            if not last_iter:
                if data[index_buffer[split_pos], var_idx] == data[index_buffer[split_pos + 1], var_idx]:
                    continue

            # reset impurity improvement
            impurity_improvement = 0.

            # if numeric targets exist
            if self.has_numeric_vars():

                # if there is more than one sample on the left side if the split
                if samples_left > 1:
                    # calculate variance of left split
                    variances(self.sq_sums_left, self.sums_left, samples_left, result=self.variances_left)
                else:
                    # reset variances left of the split
                    self.variances_left[:] = 0

                # if the variable considered for the split is numeric
                if numeric:

                    # if there is more than one sample on the right side if the split
                    if samples_right > 1:
                        # calculate variance of right split
                        variances(self.sq_sums_right, self.sums_right, samples_right, result=self.variances_right)
                    else:
                        # if there is only one sample reset the variances
                        self.variances_right[:] = 0

                    # compute the variance improvement for this split
                    impurity_improvement += compute_var_improvements(variances_total,
                                                                     self.variances_left,
                                                                     self.variances_right,
                                                                     samples_left,
                                                                     samples_right) * self.w_numeric

                # if the variable is symbolic
                else:
                    # TODO IDK
                    impurity_improvement += (mean(self.variances_total) - mean(self.variances_left)) / mean(self.variances_total)
                    impurity_improvement *= <DTYPE_t> samples_left / <DTYPE_t> n_samples * self.w_numeric

            # if symbolic targets exist
            if self.has_symbolic_vars():

                # if total gini impurity is not 0
                if gini_total:

                    # update gini impurity left and right of the split
                    self.gini_impurity(self.symbols_left, samples_left, self.gini_left)
                    self.gini_impurity(self.symbols_right, samples_right, self.gini_right)
                    impurity_improvement += ((gini_total -
                                              mean(self.gini_left) * (<DTYPE_t> samples_left / <DTYPE_t> n_samples) -
                                              mean(self.gini_right) * (<DTYPE_t> samples_right / <DTYPE_t> n_samples)) / gini_total
                                              * (1 - self.w_numeric))

            # if this variable is symbolic
            if symbolic:

                # if symbolic targets exist
                if self.has_symbolic_vars():
                    self.symbols_left[...] = 0
                    self.symbols_right[...] =  self.symbols_total[...]

                    self.gini_improvements[...] = 0
                    self.num_samples[...] = 0

                # if numeric targets exist
                if self.has_numeric_vars():
                    self.sums_left[...] = 0
                    self.sq_sums_left[...] = 0

            # check if this split is legal according to self.min_samples_leaf
            if samples_left < self.min_samples_leaf or samples_right < self.min_samples_leaf:
                impurity_improvement = 0.

            # check if this split is improves the impurity by the minimal required amount
            if impurity_improvement > max_impurity_improvement:
                max_impurity_improvement = impurity_improvement
                best_split_pos[0] = split_pos

            # break on the last iteration since the "real" last iteration is not needed
            if last_iter:
                break

        return max_impurity_improvement

    cdef inline void update_numeric_stats_with_dependencies(Impurity self,
                                                            SIZE_t sample_idx,
                                                            SIZE_t[::1] dependent_columns) nogil:
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

    cdef inline void update_symbolic_stats_with_dependencies(Impurity self,
                                                             SIZE_t sample_idx,
                                                             SIZE_t[::1] dependent_columns) nogil:
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
            self.symbols_right[validx, dep_var] = (self.symbols_total[validx, dep_var] -
                                                   self.symbols_left[validx, dep_var])

    def to_string(self):
        """
        :return: a table-like string describing the targets and features of this impurity
        """
        return tabulate.tabulate(zip(['# symbolic variables',
                                      'symbolic variables',
                                      '# numeric variables',
                                      'numeric variables',
                                      '# symbolic features',
                                      'symbolic features',
                                      '# numeric features',
                                      'numeric features'],
                                     [len(self.symbolic_vars),
                                      ','.join(mapstr(self.symbolic_vars)),
                                      len(self.numeric_vars),
                                      ','.join(mapstr(self.numeric_vars)),
                                      len(self.symbolic_features),
                                      ','.join(mapstr(self.symbolic_features)),
                                      len(self.numeric_features),
                                      ','.join(mapstr(self.numeric_features))]))


#-----------------------------------------------------------------------------------------------------------------------

cdef class PCAImpurity(Impurity):
    """
    Class to implement fast impurity and PCA calculations on splits.

    Note:
        A general name convention is, that left and right refer to sides of a split made on a sorted array
        On this array with a split value of 104.25
        -----------------------------------------
        | 1  |  3 |  8.5 | 200  | 210  |  210.5 |
        ----------------------------------------
        left would be considered as 1,3,8.5 and right would be considered as 200,210,210.5


        Whenever an index is considered as invalid or not initialized '-1' is used.
    """

    # the indices of all numeric variables (features and targets) in self.data
    cdef SIZE_t[::1] numeric_indices

    # the additional pca matrix calculating holding the current most informative linear relationship
    cdef DTYPE_t[:, ::1] pca_matrix

    # array holding the pca transformed and standardized copy of the original numeric data
    cdef readonly DTYPE_t[:, ::1] pca_data

    # array holding eigenvalues
    cdef readonly DTYPE_t[::1] eigenvalues

    # array holding eigenvectors
    cdef readonly DTYPE_t[:, ::1] eigenvectors

    # the variances before standardization so the inverse can be calculated later
    cdef readonly DTYPE_t[::1] pre_transformations_variances

    # the expectations before standardization so the inverse can be calculated later
    cdef readonly DTYPE_t[::1] pre_transformation_expectations

    def __init__(self, tree):
        super(PCAImpurity, self).__init__(tree)

        # initialize pca as identity matrix
        self.pca_matrix = np.identity(self.n_num_vars_total)

        # initialize pca data
        self.pca_data = None

        # get numeric indices
        self.numeric_indices = np.array([i for i,v in enumerate(tree.variables) if v.numeric])

        # initialize eigenvalues and eigenvectors
        self.eigenvalues = np.ndarray(shape=(self.n_num_vars_total,), order="C")
        self.eigenvectors = np.ndarray(shape=(self.n_num_vars_total, self.n_num_vars_total), order="C")

        # initialize pre transformation expectation and variance
        self.pre_transformation_expectations = np.ndarray((self.n_num_vars_total,), order="C")
        self.pre_transformations_variances = np.ndarray((self.n_num_vars_total,), order="C")


    cpdef void setup(PCAImpurity self, DTYPE_t[:, ::1] data, SIZE_t[::1] indices) except +:
        """
        Set data and indices, update features and index_buffer

        :param data: the data to set
        :param indices: the indices to set
        """
        self.data = data
        self.feat = np.ndarray(shape=data.shape[0], dtype=np.float64)
        self.indices = indices
        self.index_buffer = np.ndarray(shape=indices.shape[0], dtype=np.int64)
        self.pca_data = np.ndarray(shape=(data.shape[0], self.n_num_vars_total), dtype=np.float64,
                                   order="C")

    cdef inline void setup_pca_data(PCAImpurity self) nogil:
        """
        Copy the data that will be used for the PCA into self.pca_data
        """

        # initialize indices
        cdef SIZE_t column_index_data
        cdef SIZE_t row_index_data
        cdef SIZE_t column_index_pca
        cdef SIZE_t row_index_pca
        cdef SIZE_t[::1] row_indices

        row_indices = self.indices[self.start:self.end]

        for row_index_pca in range(0, len(row_indices)):
            for column_index_pca in range(0, len(self.numeric_indices)):
                row_index_data = row_indices[row_index_pca]
                column_index_data = self.numeric_indices[column_index_pca]
                self.pca_data[row_index_pca, column_index_pca] = self.data[row_index_data, column_index_data]


    cpdef DTYPE_t compute_best_split(self, SIZE_t start, SIZE_t end) except -1:
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

        # calculate number of samples
        cdef int n_samples = end - start

        # initialize impurity and gini index
        cdef np.float64_t impurity_total = 0
        cdef np.float64_t gini_total = 0

        # indices to later copy back from self.pca_data to self.data
        cdef SIZE_t row_index_pca, column_index_pca, row_index_data, column_index_data, index_expectation
        row_index_pca = column_index_pca = row_index_data = column_index_data = index_expectation = -1

        # if numeric targets exist
        if self.has_numeric_vars():
            print("setting up numeric structures")
            # reset the variances
            self.variances_total[:] = 0
            print("calculate square sums of all current data")
            # calculate square sums of all current data
            sq_sum_at(self.data,
                      self.indices[self.start:self.end],
                      self.numeric_vars,
                      result=self.sq_sums_total)
            print("calculate ordinary sums of all current data")
            # calculate ordinary sums of all current data
            sum_at(self.data,
                   self.indices[self.start:self.end],
                   self.numeric_vars,
                   result=self.sums_total)
            print("calculate variances from square and ordinary sums of all current data")
            # calculate variances from square and ordinary sums of all current data
            variances(self.sq_sums_total,
                      self.sums_total,
                      n_samples,
                      result=self.variances_total)

            #----------------------------------------------------------------------------------
            #--------------------------------PCA Calculations ---------------------------------
            #----------------------------------------------------------------------------------
            print("setting up pca structures")
            # setup data for pca processing
            self.setup_pca_data()

            # copy variances
            self.pre_transformations_variances = self.variances_total
            self.pre_transformation_expectations = self.sums_total

            # transform pre_transformation_expectations to real expectations
            for index_expectation in range(self.n_num_vars_total):
                self.pre_transformation_expectations[index_expectation] /= n_samples
            print("standardize data")
            # standardize data
            standardize(self.pca_data, self.sums_total, self.variances_total, self.pca_data)
            print("calculate pca")
            # calculate pca and save eigenvalues and vectors
            pca(self.pca_data, self.eigenvalues, self.eigenvectors)
            print("apply pca")
            # transform numeric data (mean is now 0)
            self.pca_data = np.dot(self.pca_data, self.eigenvectors)
            print("recalculate stats")
            # reset sums to 0
            self.sums_total[...] = 0.

            # recalculate square sums
            sq_sum_at(self.pca_data,
                      np.arange(self.pca_data.shape[0]),
                      np.arange(self.pca_data.shape[1]),
                      result=self.sq_sums_total)


            # recalculate variances
            variances(self.sq_sums_total,
                      self.sums_total,
                      n_samples,
                      result=self.variances_total)

            print("rewrite the results into self.data")
            # rewrite the results into self.data
            for row_index_pca in range(0, self.pca_data.shape[0]):
                row_index_data = row_index_pca + self.start
                for column_index_pca in range(0, self.pca_data.shape[1]):
                    column_index_data = self.numeric_indices[column_index_pca]
                    self.data[row_index_data, column_index_data] = self.pca_data[row_index_pca, column_index_pca]

            #----------------------------------------------------------------------------------
            #--------------------------- End of PCA Calculations ------------------------------
            #----------------------------------------------------------------------------------

        # if symbolic targets exist
        if self.has_symbolic_vars():
            print("compute histogram of all current data")
            # compute histogram of all current data
            bincount(self.data,
                     self.indices[self.start:self.end],
                     self.symbolic_vars,
                     result=self.symbols_total)

            print("calculate gini impurity of histogram")
            # calculate gini impurity of histogram
            self.gini_impurity(self.symbols_total, n_samples, self.gini_impurities)

            print("save total gini impurity as mean of all symbolic dimensions impurities")
            # save total gini impurity as mean of all symbolic dimensions impurities
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
        self.max_impurity_improvement = 0

        # for every feature
        for variable in self.features:
            print("evaluating variable", variable)
            # check if this variable is symbolic or not
            symbolic = variable in self.symbolic_features

            # increase symbolic index tracking by one if variable is symbolic
            symbolic_idx += symbolic

            # initialize split position
            split_pos = -1

            # evaluate the current variable
            impurity_improvement = self.evaluate_variable(variable,
                                                          symbolic,
                                                          symbolic_idx,
                                                          self.variances_total if self.has_numeric_vars() else None,
                                                          gini_total,
                                                          self.index_buffer,
                                                          &split_pos)

            # if the best impurity improvement of this variable is better than the current best
            if impurity_improvement > self.max_impurity_improvement:
                # update current best impurity
                self.max_impurity_improvement = impurity_improvement

                # update index of best variable
                self.best_var = variable

                # update the position of the split
                self.best_split_pos = split_pos

                # TODO IDK
                self.indices[self.start:self.end] = self.index_buffer[:n_samples]

        print("checking validity of split")
        # if max impurity improvement has been updated at least once and the best variable is symbolic
        if self.max_impurity_improvement and self.best_var in self.symbolic_features:
            # TODO IDK
            self.move_best_values_to_front(self.best_var,
                                           self.data[self.indices[start + self.best_split_pos],
                                                     self.best_var],
                                           &self.best_split_pos)

        print("all done")
        # return the best improvement value
        return self.max_impurity_improvement


