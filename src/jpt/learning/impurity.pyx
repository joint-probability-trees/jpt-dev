# cython: auto_cpdef=True
# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from libcpp.deque cimport deque

from dnutils import ifnone, out
from collections import deque as pydeque

from ..base.cutils cimport DTYPE_t, SIZE_t, mean, nan, sort

cdef int LEFT = 0
cdef int RIGHT = 1


# ----------------------------------------------------------------------------------------------------------------------


cdef inline void compute_var_improvements(DTYPE_t[::1] variances_total,
                                   DTYPE_t[::1] variances_left,
                                   DTYPE_t[::1] variances_right,
                                   SIZE_t samples_left,
                                   SIZE_t samples_right,
                                   DTYPE_t[::1] result) nogil:
    result[:] = variances_total
    cdef SIZE_t i
    cdef DTYPE_t n_samples = <DTYPE_t> samples_left + samples_right

    for i in range(variances_total.shape[0]):
        result[i] -= ((variances_left[i] * <DTYPE_t> samples_left
                       + variances_right[i] * <DTYPE_t> samples_right) / n_samples)

    for i in range(variances_total.shape[0]):
        if variances_total[i]:
            result[i] /= variances_total[i]
        else:
            result[i] = 0


# ----------------------------------------------------------------------------------------------------------------------

cdef inline void sum_at(DTYPE_t[:, ::1] M,
                        SIZE_t[::1] rows,
                        SIZE_t[::1] cols,
                        DTYPE_t[::1] result) nogil:
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
    '''
    Variance computation uses the proxy from sklearn: ::

        var = \sum_i^n (y_i - y_bar) ** 2
        = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2

    See also: https://github.com/scikit-learn/scikit-learn/blob/de1262c35e2aa4ee062d050281ee576ce9e35c94/sklearn/tree/_criterion.pyx#L683
    '''
    result[:] = sq_sums
    cdef SIZE_t i
    for i in range(sums.shape[0]):
        result[i] -= sums[i] * sums[i] / <DTYPE_t> n_samples
        result[i] /= n_samples - 1


# ----------------------------------------------------------------------------------------------------------------------
# in-place vector addition

cdef inline void ivadd(DTYPE_t[::1] target, DTYPE_t[::1] arg, SIZE_t n, int sq=False) nogil:
    cdef SIZE_t i
    for i in range(n):
        target[i] += arg[i] if not sq else (arg[i] * arg[i])


# ----------------------------------------------------------------------------------------------------------------------

cdef inline void bincount(DTYPE_t[:, ::1] data,
                          SIZE_t[::1] rows,
                          SIZE_t[::1] cols,
                          SIZE_t[:, ::1] result) nogil:
    result[...] = 0
    cdef SIZE_t i, j
    for i in range(rows.shape[0]):
        for j in range(cols.shape[0]):
            result[<SIZE_t> data[rows[i], cols[j]], j] += 1


# ----------------------------------------------------------------------------------------------------------------------


cdef class Impurity:

    cdef DTYPE_t [:, ::1] data
    cdef SIZE_t [::1] indices, index_buffer
    cdef DTYPE_t[::1] feat
    cdef SIZE_t start, end

    cdef SIZE_t[::1] numeric_vars, symbolic_vars, all_vars
    cdef public DTYPE_t min_samples_leaf
    cdef SIZE_t[::1] symbols
    cdef SIZE_t n_num_vars, n_sym_vars, max_sym_domain, n_vars

    cdef SIZE_t[:, ::1] symbols_left, \
        symbols_right, \
        symbols_total

    cdef DTYPE_t[::1] gini_improvements
    cdef DTYPE_t[::1] gini_impurities
    cdef DTYPE_t[::1] gini_left
    cdef DTYPE_t[::1] gini_right

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

    cdef DTYPE_t[::1] max_variances

    cdef SIZE_t[::1] num_samples
    cdef SIZE_t[::1] targets
    cdef deque[int] _best_split_pos
    cdef readonly int best_var
    cdef readonly  DTYPE_t max_impurity_improvement
    cdef DTYPE_t w_numeric

    cdef public list priors

    @property
    def best_split_pos(self):
        return pydeque([e for e in self._best_split_pos])

    def __init__(self, tree):
        self.min_samples_leaf = tree.min_samples_leaf

        self.data = self.feat = self.index_buffer = self.indices = None
        self.start = self.end = -1

        self.best_var = -1
        self.max_impurity_improvement = 0

        self.numeric_vars = np.array([<int> i for i, v in enumerate(tree.variables) if v.numeric], dtype=np.int64)
        self.symbolic_vars = np.array([<int> i for i, v in enumerate(tree.variables) if v.symbolic], dtype=np.int64)
        self.all_vars = np.concatenate((self.numeric_vars, self.symbolic_vars))
        self.n_vars = len(tree.variables)
        self.priors = []

        self.targets = np.array([i for i, v in enumerate(tree.variables) if tree.targets is None or v in tree.targets], dtype=np.int64)

        self.n_sym_vars = len(self.symbolic_vars)

        if self.n_sym_vars:
            # Thread-invariant buffers
            self.symbols = np.array([v.domain.n_values for v in tree.variables if v.symbolic], dtype=np.int64)
            self.max_sym_domain = max(self.symbols)
            self.symbols_total = np.ndarray(shape=(self.max_sym_domain, self.n_sym_vars), dtype=np.int64)
            
            # Thread-private buffers
            self.symbols_left = np.ndarray(shape=(self.max_sym_domain, self.n_sym_vars), dtype=np.int64)
            self.symbols_right = np.ndarray(shape=(self.max_sym_domain, self.n_sym_vars), dtype=np.int64)
            self.gini_improvements = np.ndarray(shape=self.n_sym_vars, dtype=np.float64)
            self.gini_impurities = np.ndarray(shape=self.n_sym_vars, dtype=np.float64)
            self.gini_left = np.ndarray(shape=self.n_sym_vars, dtype=np.float64)
            self.gini_right = np.ndarray(shape=self.n_sym_vars, dtype=np.float64)

        self.num_samples = np.ndarray(shape=max(max(self.symbols) if self.n_sym_vars else 2, 2), dtype=np.int64)
        self.n_num_vars = len(self.numeric_vars)

        if self.n_num_vars:
            # Thread-invariant buffers
            self.sums_total = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.sq_sums_total = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.variances_total = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.max_variances = np.array([v._max_std ** 2 for v in tree.variables if v.numeric], dtype=np.float64)

            # Thread-private buffers
            self.sums_left = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.sums_right = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.sq_sums_left = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.sq_sums_right = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.variances_left = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.variances_right = np.ndarray(self.n_num_vars, dtype=np.float64)
            self.variance_improvements = np.ndarray(self.n_num_vars, dtype=np.float64)

        self.w_numeric = <DTYPE_t> self.n_num_vars / <DTYPE_t> self.n_vars

    cdef inline int check_max_variances(self, DTYPE_t[::1] variances):  # nogil:
        cdef int i
        for i in range(self.n_num_vars):
            if variances[i] > self.max_variances[i]:
                return True
        return False

    cpdef void setup(Impurity self, DTYPE_t[:, ::1] data, SIZE_t[::1] indices) except +:
        self.data = data
        self.feat = np.ndarray(shape=data.shape[0], dtype=np.float64)
        self.indices = indices
        self.index_buffer = np.ndarray(shape=indices.shape[0], dtype=np.int64)

    cdef inline int has_numeric_vars(Impurity self) nogil:
        return self.numeric_vars.shape[0]

    cdef inline int has_symbolic_vars(Impurity self) nogil:
        return self.symbolic_vars.shape[0]

    cdef inline void gini_impurity(Impurity self, SIZE_t[:, ::1] counts, SIZE_t n_samples, DTYPE_t[::1] result) nogil:
        cdef SIZE_t i, j
        # cdef DTYPE_t[::1] buf2 = self.gini_buffer2
        result[...] = 0
        for i in range(self.n_sym_vars):
            for j in range(self.symbols[i]):
                result[i] += <DTYPE_t> counts[j, i] * counts[j, i]
            result[i] /= <DTYPE_t> (n_samples * n_samples)
            result[i] -= 1
            result[i] /= 1. / (<DTYPE_t> self.symbols[i]) - 1.
        # cdef DTYPE_t result = 0
        # for i in range(self.n_sym_vars):
        #     result += buf2[i]
        # return result / <DTYPE_t> self.n_sym_vars

    cdef inline int col_is_constant(Impurity self, long start, long end, long col) nogil:
        cdef DTYPE_t v_ = nan, v
        cdef long i
        for i in range(start, end):
            v = self.data[self.indices[i], col]
            if v_ == v: continue
            if v_ != v:
                if v_ != v_: v_ = v  # NB: x != x is True iff x is NaN
                else: return False
        return True

    cpdef compute_best_split(self, SIZE_t start, SIZE_t end):
        '''
        Computation uses the impurity proxy from sklearn: ::

            var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2

        See also: https://github.com/scikit-learn/scikit-learn/blob/de1262c35e2aa4ee062d050281ee576ce9e35c94/sklearn/tree/_criterion.pyx#L683
        '''
        cdef int best_var = -1
        self.start = start
        self.end = end
        cdef int n_samples = end - start
        cdef np.float64_t denom = 0

        cdef np.float64_t impurity_total = 0
        cdef np.float64_t gini_total = 0

        self.max_impurity_improvement = 0

        if self.has_numeric_vars():
            self.variances_total[:] = 0
            sq_sum_at(self.data,
                      self.indices[self.start:self.end],
                      self.numeric_vars,
                      result=self.sq_sums_total)

            sum_at(self.data,
                   self.indices[self.start:self.end],
                   self.numeric_vars,
                   result=self.sums_total)

            variances(self.sq_sums_total,
                      self.sums_total,
                      n_samples,
                      result=self.variances_total)

            if not self.check_max_variances(self.variances_total):
                return 0

            denom += 1
            # impurity_total += len(self.numeric_vars) * np.mean(self.variances_total)
            # impurity_total += self.w_numeric * np.mean(self.variances_total)

        if self.symbolic_vars.shape[0]:
            bincount(self.data,
                     self.indices[self.start:self.end],
                     self.symbolic_vars,
                     result=self.symbols_total)

            self.gini_impurity(self.symbols_total, n_samples, self.gini_impurities)
            gini_total = mean(self.gini_impurities)
            # impurity_total += len(self.symbolic_vars) * gini_total
            # impurity_total += (1 - self.w_numeric) * gini_total
            denom += 1
        else:
            gini_total = 0

        # impurity_total /= <DTYPE_t> denom * self.n_vars

        cdef int symbolic = 0
        cdef int symbolic_idx = -1

        cdef DTYPE_t impurity_improvement
        cdef int variable

        cdef deque[int] split_pos
        self.index_buffer[self.start:self.end] = self.indices[self.start:self.end]

        for variable in self.all_vars:
            symbolic = variable in self.symbolic_vars
            symbolic_idx += symbolic
            split_pos.clear()
            impurity_improvement = self.evaluate_variable(variable,
                                                          symbolic,
                                                          symbolic_idx,
                                                          denom,
                                                          self.variances_total if self.has_numeric_vars() else None,
                                                          gini_total,
                                                          impurity_total,
                                                          self.index_buffer[self.start:self.end],
                                                          &split_pos)

            if impurity_improvement > self.max_impurity_improvement:
                self.max_impurity_improvement = impurity_improvement
                self._best_split_pos.clear()
                for e in split_pos:
                    self._best_split_pos.push_back(e)

                self.best_var = variable
                self.indices[self.start:self.end] = self.index_buffer[self.start:self.end]
                # out('var impr.', np.asarray(self.variance_improvements), 'total:', self.max_impurity_improvement)
        return self.max_impurity_improvement

    cdef DTYPE_t evaluate_variable(Impurity self,
                               int var_idx,
                               int symbolic,
                               int symbolic_idx,
                               DTYPE_t denom,
                               DTYPE_t[::1] variances_total,
                               DTYPE_t gini_total,
                               DTYPE_t impurity_total,
                               SIZE_t[::1] index_buffer,
                               # DTYPE_t* best_split_val,
                               deque[int]* best_split_pos) nogil:
        cdef DTYPE_t[:, ::1] data = self.data
        cdef DTYPE_t[::1] f = self.feat
        # cdef SIZE_t[::1] indices = self.indices
        cdef DTYPE_t max_impurity_improvement = 0
        cdef SIZE_t start = self.start, end = self.end
        cdef SIZE_t n_samples = end - start
        # --------------------------------------------------------------------------------------------------------------
        # TODO: Check if sorting really needs a copy of the feature data
        cdef int i, j
        for j in range(n_samples):
            f[j] = data[index_buffer[j], var_idx]
        sort(&f[0], &index_buffer[0], n_samples)
        # --------------------------------------------------------------------------------------------------------------
        cdef int numeric = not symbolic
        
        if self.col_is_constant(start, end, var_idx):
            return 0
        # Prepare the stats
        if self.has_numeric_vars():
            self.sums_left[...] = 0
            self.sums_right[...] = self.sums_total
            self.sq_sums_left[...] = 0
            self.sq_sums_right[...] = self.sq_sums_total

        if self.has_symbolic_vars():
            self.symbols_left[...] = 0
            self.symbols_right[...] = self.symbols_total[...]

        self.num_samples[:] = 0
        cdef SIZE_t samples_left, samples_right
        cdef DTYPE_t impurity_improvement = 0
        cdef DTYPE_t tmp_impurity_impr

        cdef SIZE_t VAL_IDX
        cdef SIZE_t sample_idx
        cdef int last_iter
        cdef DTYPE_t min_samples

        for split_pos in range(n_samples):
            sample_idx = index_buffer[split_pos]
            last_iter = (symbolic and split_pos == n_samples - 1
                         or numeric and split_pos == n_samples - 2)

            if numeric:
                self.num_samples[LEFT] += 1
                self.num_samples[RIGHT] = <SIZE_t> n_samples - split_pos - 1
                samples_left = self.num_samples[LEFT]
                samples_right = self.num_samples[RIGHT]
            else:
                VAL_IDX = <SIZE_t> data[sample_idx, var_idx]
                self.num_samples[VAL_IDX] += 1
                samples_left = self.num_samples[VAL_IDX]
                samples_right = n_samples - samples_left

            # if numeric and last_iter:
            #     break

            # Compute the numeric impurity by variance approximation
            if self.has_numeric_vars():
                self.update_numeric_stats(sample_idx)

            # Compute the symbolic impurity by the Gini index
            if self.has_symbolic_vars():
                self.update_symbolic_stats(sample_idx)

            # Skip calculation for identical values (i.e. until next 'real' splitpoint is reached:
            # for skipping, the sample must not be the last one (1) and consequtive values must be equal (2)
            if not last_iter:
                if data[index_buffer[split_pos], var_idx] == data[index_buffer[split_pos + 1], var_idx]:
                    continue
                elif symbolic:
                    best_split_pos[0].push_back(split_pos)

            elif symbolic:
                best_split_pos[0].push_back(split_pos)

            if numeric:
                impurity_improvement = 0

            if self.has_numeric_vars():
                if samples_left > 1:
                    variances(self.sq_sums_left, self.sums_left, samples_left, result=self.variances_left)
                else:
                    self.variances_left[:] = 0

                if numeric:
                    if samples_right > 1:
                        variances(self.sq_sums_right, self.sums_right, samples_right, result=self.variances_right)
                    else:
                        self.variances_right[:] = 0
                    compute_var_improvements(variances_total,
                                             self.variances_left,
                                             self.variances_right,
                                             samples_left,
                                             samples_right,
                                             result=self.variance_improvements)
                    impurity_improvement += mean(self.variance_improvements) * self.w_numeric
                else:
                    # for j in range(self.n_num_vars):
                    #     if self.variances_total[j]:
                    #         impurity_improvement += (self.variances_total[j] - self.variances_left[j]) / self.variances_total[j]
                    impurity_improvement += (mean(self.variances_total) - mean(self.variances_left)) / mean(self.variances_total)
                    # impurity_improvement /= self.n_num_vars
                    impurity_improvement *= <DTYPE_t> self.num_samples[VAL_IDX] / <DTYPE_t> n_samples * self.w_numeric

            if self.has_symbolic_vars():
                if gini_total:
                    # print('gini:', np.asarray(self.gini_impurities))
                    self.gini_impurity(self.symbols_left, samples_left, self.gini_left)
                    # print('impr:', np.asarray(self.gini_improvements))
                    if numeric:
                        self.gini_impurity(self.symbols_right, samples_right, self.gini_right)
                        # gini_improvement = (gini_total - (<DTYPE_t> samples_left / <DTYPE_t> n_samples +
                        #                                   <DTYPE_t> samples_right / <DTYPE_t> n_samples * gini_right)) / gini_total
                        impurity_improvement += ((gini_total -
                                                  mean(self.gini_left) * (<DTYPE_t> samples_left / <DTYPE_t> n_samples) +
                                                  mean(self.gini_right) * (<DTYPE_t> samples_right / <DTYPE_t> n_samples)) / gini_total
                                                  * (1 - self.w_numeric))
                    else:
                        tmp_impurity_impr = 0
                        # for j in range(self.n_sym_vars):
                            # if self.gini_impurities[j]:
                            #     tmp_impurity_impr += (self.gini_impurities[j] - self.gini_improvements[j]) / self.gini_impurities[j]
                        # tmp_impurity_impr /= <DTYPE_t> self.n_sym_vars
                        tmp_impurity_impr += (gini_total - mean(self.gini_left)) / gini_total
                        tmp_impurity_impr *= <DTYPE_t> self.num_samples[VAL_IDX] / <DTYPE_t> n_samples * (1. - self.w_numeric)
                        impurity_improvement += tmp_impurity_impr
                        # self.gini_improvements[symbolic_idx] = self.gini_impurities - gini_left
                        # impurity_improvement += ((gini_left) / gini_total * 1. / <DTYPE_t> self.n_sym_vars

            if symbolic:
                self.symbols_left[...] = 0
                self.gini_improvements[...] = 0

                if self.has_numeric_vars():
                    self.sums_left[...] = 0
                    self.sq_sums_left[...] = 0

                if last_iter:
                    # impurity_improvement /= denom
                    # impurity_improvement = (impurity_total - impurity_improvement) / impurity_total
                    # print('total', impurity_improvement)
                    cnt = 0
                    min_samples = self.min_samples_leaf
                    for i in range(self.symbols[symbolic_idx]):
                        if self.num_samples[i] == 0:
                            cnt += 1
                        elif self.num_samples[i] < min_samples:
                            min_samples = self.num_samples[i]
                    if min_samples < self.min_samples_leaf / (self.symbols[sample_idx] - cnt):
                        impurity_improvement = 0
                        break
                    # if cnt >= self.symbols[symbolic_idx] - 1:  # Require splits with entropy > 0
                    #     impurity_improvement = 0
                    #     break

            else:  # numeric
                # impurity_improvement /= denom
                if (samples_left < self.min_samples_leaf
                    or samples_right < self.min_samples_leaf):
                    #or not self.check_max_variances(self.variances_total)):
                    impurity_improvement = 0

            if (numeric or last_iter) and (impurity_improvement > max_impurity_improvement):
                max_impurity_improvement = impurity_improvement
                if numeric:  # Clear the split pos as in numeric vars we only have exactly one
                    best_split_pos[0].clear()
                    best_split_pos[0].push_back(split_pos)

            if last_iter:
                break

        return max_impurity_improvement

    cdef inline void update_numeric_stats(Impurity self, SIZE_t sample_idx) nogil:
        cdef SIZE_t var, i
        cdef DTYPE_t y
        for i in range(self.n_num_vars):
            y = self.data[sample_idx, self.numeric_vars[i]]
            self.sums_left[i] += y
            self.sums_right[i] = self.sums_total[i] - self.sums_left[i]
            self.sq_sums_left[i] += y * y
            self.sq_sums_right[i] = self.sq_sums_total[i] - self.sq_sums_left[i]

    cdef inline void update_symbolic_stats(Impurity self, SIZE_t sample_idx) nogil:
        cdef SIZE_t i, j, validx
        for i in range(self.n_sym_vars):
            validx = <SIZE_t> self.data[sample_idx, self.symbolic_vars[i]]
            self.symbols_left[validx, i] += 1
            self.symbols_right[validx, i] = self.symbols_total[validx, i] - self.symbols_left[validx, i]

