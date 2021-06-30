# cython: auto_cpdef=True
# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libcpp.deque cimport deque

from dnutils import ifnone, out
from collections import deque as pydeque

from libc.math cimport log as ln

cdef double nan = np.nan
ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.int64_t SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef int LEFT = 0
cdef int RIGHT = 1

# ----------------------------------------------------------------------------------------------------------------------
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline void compute_var_improvements(DTYPE_t[::1] variances_total,
                                   DTYPE_t[::1] variances_left,
                                   DTYPE_t[::1] variances_right,
                                   DTYPE_t samples_left,
                                   DTYPE_t samples_right,
                                   DTYPE_t[::1] result) nogil:
    result[:] = variances_total
    cdef SIZE_t i
    cdef DTYPE_t n_samples = samples_left + samples_right
    for i in range(variances_total.shape[0]):
        result[i] -= ((variances_left[i] * samples_left + variances_right[i] * samples_right) / n_samples)
    for i in range(variances_total.shape[0]):
        if variances_total[i]:
            result[i] /= variances_total[i]
        else:
            result[i] = 0


# ----------------------------------------------------------------------------------------------------------------------

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline void sq_sum_at(DTYPE_t[:, ::1] M,
                           SIZE_t[::1] rows,
                           SIZE_t[::1] cols,
                           DTYPE_t[::1] result) nogil:
    result[...] = 0
    cdef SIZE_t i, j
    for j in range(cols.shape[0]):
        for i in range(rows.shape[0]):
            result[j] += M[rows[i], cols[j]] * M[rows[i], cols[j]]


# ----------------------------------------------------------------------------------------------------------------------

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline void variances(DTYPE_t[::1] sq_sums,
                           DTYPE_t[::1] sums,
                           DTYPE_t n_samples,
                           DTYPE_t[::1] result):
    '''
    Variance computation uses the proxy from sklearn: ::

        var = \sum_i^n (y_i - y_bar) ** 2
        = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2

    See also: https://github.com/scikit-learn/scikit-learn/blob/de1262c35e2aa4ee062d050281ee576ce9e35c94/sklearn/tree/_criterion.pyx#L683
    '''
    result[:] = sq_sums
    cdef i
    for i in range(sums.shape[0]):
        result[i] -= sums[i] * sums[i] / n_samples


# ----------------------------------------------------------------------------------------------------------------------
# in-place vector addition

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline void ivadd(DTYPE_t[::1] target, DTYPE_t[::1] arg, SIZE_t n, int sq=False) nogil:
    cdef SIZE_t i
    for i in range(n):
        target[i] += arg[i] if not sq else (arg[i] * arg[i])


# ----------------------------------------------------------------------------------------------------------------------
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline void bincount(DTYPE_t[:, ::1] data,
                          SIZE_t[::1] rows,
                          SIZE_t[::1] cols,
                          DTYPE_t[:, ::1] result) nogil:
    result[...] = 0
    cdef SIZE_t i, j
    for i in range(rows.shape[0]):
        for j in range(cols.shape[0]):
            result[<SIZE_t> data[rows[i], cols[j]], j] += 1.


cdef class Impurity:

    cdef DTYPE_t [:, ::1] data
    cdef SIZE_t [::1] indices
    cdef DTYPE_t[::1] feat
    cdef SIZE_t start, end
    cdef object tree
    cdef tuple variables
    cdef SIZE_t[::1] numeric_vars, symbolic_vars
    cdef DTYPE_t min_samples_leaf
    cdef SIZE_t[::1] symbols
    cdef object symbols_left, symbols_right, symbols_total, gini_buffer, gini_buffer2, sums_left, sums_right, sq_sums_left, sq_sums_right, sums_total, sq_sums_total
    cdef DTYPE_t[::1] variances_left
    cdef DTYPE_t[::1] variances_right
    cdef DTYPE_t[::1] variance_improvements
    cdef deque[int] _best_split_pos
    cdef readonly int best_var
    cdef readonly  DTYPE_t max_impurity_improvement

    @property
    def best_split_pos(self):
        return pydeque([e for e in self._best_split_pos])

    def __init__(self, tree, data, start, end):
        self.tree = tree
        self.data = data
        self.start = start
        self.end = end
        self.feat = np.ndarray(shape=end - start, dtype=np.float64)
        self.indices = tree.indices  # np.ndarray(shape=data.shape[0], dtype=np.int64)
        self.best_var = -1
        self.max_impurity_improvement = 0
        self.variables = tree.variables

        self.numeric_vars = np.array([<int> i for i, v in enumerate(self.variables) if v.numeric], dtype=np.int64)
        self.symbolic_vars = np.array([<int> i for i, v in enumerate(self.variables) if v.symbolic], dtype=np.int64)

        if self.symbolic_vars.shape[0]:
            self.symbols = np.array([v.domain.n_values for v in self.variables if v.symbolic], dtype=np.int64)
            self.symbols_left = np.zeros(shape=(max(self.symbols),
                                                len(self.symbolic_vars)), dtype=np.float64)
            self.symbols_right = np.zeros(shape=(max(self.symbols),
                                                 len(self.symbolic_vars)), dtype=np.float64)
            self.symbols_total = np.zeros(shape=(max(self.symbols),
                                                 len(self.symbolic_vars)), dtype=np.float64)
            self.gini_buffer = np.ndarray(shape=self.symbols_total.shape, dtype=np.float64)
            self.gini_buffer2 = np.ndarray(shape=self.symbols_total.shape[1], dtype=np.float64)
            self.gini_buffer = np.ndarray(shape=self.symbols_total.shape, dtype=np.float64)
            self.gini_buffer2 = np.ndarray(shape=self.symbols_total.shape[1], dtype=np.float64)

        if self.numeric_vars.shape[0]:
            self.sums_left = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sums_right = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sums_total = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sq_sums_left = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sq_sums_right = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sq_sums_total = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.variances_left = np.ndarray(len(self.numeric_vars), dtype=np.float64)
            self.variances_right = np.ndarray(len(self.numeric_vars), dtype=np.float64)
            self.variance_improvements = np.ndarray(len(self.numeric_vars), dtype=np.float64)

        self.min_samples_leaf = tree.min_samples_leaf

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef inline int has_numeric_vars(Impurity self):
        return self.numeric_vars.shape[0]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef inline int has_symbolic_vars(Impurity self):
        return self.symbolic_vars.shape[0]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef inline DTYPE_t gini_impurity(Impurity self, DTYPE_t[:, ::1] counts, DTYPE_t n_samples) nogil:
        cdef DTYPE_t[:, ::1] buffer = self.gini_buffer
        buffer[...] = counts
        cdef DTYPE_t[::1] buf2 = self.gini_buffer2
        cdef SIZE_t i, j
        for i in range(buffer.shape[0]):
            for j in range(self.symbols[i]):
                buffer[j, i] = buffer[j, i] * buffer[j, i]
        buf2[:] = 0
        for i in range(buffer.shape[0]):
            for j in range(self.symbols[i]):
                buf2[i] += buffer[j, i]
            buf2[i] /= (n_samples * n_samples)
            buf2[i] -= 1
            buf2[i] /= -self.symbols[i] / 4.
        # np.sum(buffer, axis=0, out=buf2)
        # buf2 /=
        # buf2 -= 1
        # buf2 /= -self.symbols * 1/4
        cdef DTYPE_t result = 0
        for i in range(buf2.shape[0]):
            result += buf2[i]
        result /= self.symbolic_vars.shape[0]
        return result

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef inline int col_is_constant(Impurity self, long start, long end, long col) nogil:
        cdef np.float64_t v_ = nan, v
        cdef long i
        for i in range(start, end):
            v = self.data[self.indices[i], col]
            if v_ == v: continue
            if v_ != v:
                if v_ != v_: v_ = v  # NB: x != x is True iff x is NaN
                else: return False
        return True

    cpdef compute_best_split(self):
        '''
        Computation uses the impurity proxy from sklearn: ::

            var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2

        See also: https://github.com/scikit-learn/scikit-learn/blob/de1262c35e2aa4ee062d050281ee576ce9e35c94/sklearn/tree/_criterion.pyx#L683
        '''
        cdef int best_var = -1
        # cdef np.float64_t best_split_val = -1
        # cdef np.float64_t max_impurity_improvement = -1
        cdef int n_samples = len(self.indices)
        cdef np.float64_t denom = 0

        cdef DTYPE_t[:, ::1] data = self.data

        # variances_total = [0]
        cdef np.float64_t impurity_total = 0
        variances_total = np.zeros(len(self.numeric_vars))
        cdef np.float64_t gini_total = 0

        if self.has_numeric_vars():
            sq_sum_at(data,
                      self.indices[self.start:self.end],
                      self.numeric_vars,
                      result=self.sq_sums_total)

            sum_at(data,
                   self.indices[self.start:self.end],
                   self.numeric_vars,
                   result=self.sums_total)

            variances(self.sq_sums_total,
                      self.sums_total,
                      n_samples,
                      result=variances_total)

            # variances_total = (self.sq_sums_total - (self.sums_total ** 2 / n_samples)) / n_samples
            denom += 1
            impurity_total += len(self.numeric_vars) * np.mean(variances_total)

        if self.symbolic_vars.shape[0]:
            bincount(data,
                     self.indices[self.start:self.end],
                     self.symbolic_vars,
                     result=self.symbols_total)

            gini_total = self.gini_impurity(self.symbols_total, n_samples)
            impurity_total += len(self.symbolic_vars) * gini_total
            denom += 1
        else:
            gini_total = 0

        impurity_total /= denom * len(self.variables)

        cdef int symbolic = 0
        cdef int symbolic_idx = -1

        cdef DTYPE_t impurity_improvement
        cdef int variable

        cdef deque[int] split_pos
        cdef SIZE_t[::1] index_buffer = np.ndarray(shape=self.end - self.start, dtype=np.int64)
        index_buffer[...] = self.indices[self.start:self.end]

        # print(np.asarray(self.numeric_vars))
        # print(np.asarray(self.symbolic_vars))

        for variable in np.concatenate((self.numeric_vars, self.symbolic_vars)):
            symbolic = variable in self.symbolic_vars
            symbolic_idx += symbolic
            split_pos.clear()
            impurity_improvement = self.evaluate_variable(variable,
                                                          symbolic_idx,
                                                          denom,
                                                          variances_total,
                                                          gini_total,
                                                          impurity_total,
                                                          index_buffer,
                                                          &split_pos)

            if impurity_improvement > self.max_impurity_improvement:
                self.max_impurity_improvement = impurity_improvement
                self._best_split_pos.clear()
                for e in split_pos:
                    self._best_split_pos.push_back(e)
                # print('new best:', variable, self.best_split_pos, impurity_improvement)

                self.best_var = variable
                self.indices[self.start:self.end] = index_buffer[...]

        return self.max_impurity_improvement

    cdef DTYPE_t evaluate_variable(Impurity self,
                               int var_idx,
                               int symbolic_idx,
                               DTYPE_t denom,
                               DTYPE_t[::1] variances_total,
                               DTYPE_t gini_total,
                               DTYPE_t impurity_total,
                               SIZE_t[::1] index_buffer,
                               # DTYPE_t* best_split_val,
                               deque[int]* best_split_pos) except -1:
        cdef DTYPE_t[:, ::1] data = self.data
        cdef DTYPE_t[::1] f = self.feat
        cdef SIZE_t[::1] indices = self.indices
        cdef DTYPE_t max_impurity_improvement = 0
        cdef SIZE_t start = self.start, end = self.end
        cdef DTYPE_t n_samples = <DTYPE_t> end - start
        # --------------------------------------------------------------------------------------------------------------
        # TODO: Check if sorting really needs a copy of the feature data
        cdef int i, j
        for j, i in enumerate(index_buffer):
            f[j] = data[i, var_idx]
        sort(&f[0], &index_buffer[0], <SIZE_t> n_samples)
        # --------------------------------------------------------------------------------------------------------------
        cdef int symbolic = var_idx in self.symbolic_vars
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

        cdef SIZE_t[::1] num_samples = np.zeros(shape=self.symbols_left.shape[0] if symbolic else 2, dtype=np.int64)
        cdef DTYPE_t samples_left, samples_right
        cdef DTYPE_t impurity_improvement = 0

        cdef SIZE_t VAL_IDX
        cdef SIZE_t sample_idx

        for pos in range(start, end):
            split_pos = pos - start
            sample_idx = index_buffer[split_pos]
            last_iter = pos + 1 == end and symbolic or pos == end and numeric

            if numeric:
                num_samples[LEFT] += 1
                num_samples[RIGHT] = <SIZE_t> n_samples - split_pos - 1
                samples_left = num_samples[LEFT]
                samples_right = num_samples[RIGHT]
            else:
                VAL_IDX = <SIZE_t> data[sample_idx, var_idx]
                num_samples[VAL_IDX] += 1
                samples_left = num_samples[VAL_IDX]
                samples_right = n_samples - samples_left

            if numeric and split_pos == n_samples - 1:
                break

            # Compute the numeric impurity by variance approximation
            if self.has_numeric_vars():
                self.update_numeric_stats(sample_idx)

            if self.has_symbolic_vars():
                # Compute the symbolic impurity by the Gini index
                for i, v_idx in enumerate(self.symbolic_vars):
                    self.symbols_left[<SIZE_t> data[sample_idx, v_idx], i] += 1
                self.symbols_right = self.symbols_total - self.symbols_left

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
                if samples_left:
                    variances(self.sq_sums_left, self.sums_left, samples_left, result=self.variances_left)
                else:
                    self.variances_left[:] = 0

                if samples_right:
                    variances(self.sq_sums_right, self.sums_right, samples_right, result=self.variances_right)
                else:
                    self.variances_right[:] = 0

                compute_var_improvements(variances_total,
                                         self.variances_left,
                                         self.variances_right,
                                         samples_left,
                                         samples_right,
                                         result=self.variance_improvements)

                # np.asarray(self.variance_improvements)[np.asarray(variances_total) == 0] = 0
                avg_variance_improvement = np.mean(self.variance_improvements)

                if numeric:
                    impurity_improvement += avg_variance_improvement
                else:
                    impurity_improvement += np.mean(self.variances_left) * <DTYPE_t> num_samples[VAL_IDX]\
                                               * len(self.numeric_vars) / (n_samples * len(self.variables))

            if self.has_symbolic_vars():
                if gini_total:
                    gini_left = self.gini_impurity(self.symbols_left, samples_left)
                    gini_right = self.gini_impurity(self.symbols_right, samples_right)
                    gini_improvement = (gini_total - (samples_left / n_samples * gini_left +
                                                      samples_right / n_samples * gini_right)) / gini_total
                    if numeric:
                        impurity_improvement += gini_improvement
                    else:
                        impurity_improvement += gini_left * <DTYPE_t> num_samples[VAL_IDX] * len(self.symbolic_vars) / (n_samples * len(self.variables))

            if symbolic:
                self.symbols_left[...] = 0

                if self.has_numeric_vars():
                    self.sums_left[...] = 0
                    self.sq_sums_left[...] = 0

                if last_iter:
                    impurity_improvement /= denom
                    impurity_improvement = (impurity_total - impurity_improvement) / impurity_total

                    if not all(not num_samples[i] or num_samples[i] >= self.min_samples_leaf for i in range(self.symbols[symbolic_idx])):
                        impurity_improvement = 0

            else:
                impurity_improvement /= denom
                if samples_left < self.min_samples_leaf or samples_right < self.min_samples_leaf:
                    impurity_improvement = 0

            if (numeric or split_pos == n_samples - 1) and (impurity_improvement > max_impurity_improvement):
                max_impurity_improvement = impurity_improvement
                if numeric:  # Clear the split pos as in numeric vars we only have exactly one
                    best_split_pos[0].clear()
                    best_split_pos[0].push_back(split_pos)

        return max_impurity_improvement

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef inline void update_numeric_stats(Impurity self, SIZE_t sample_idx):
        cdef SIZE_t var, i
        cdef DTYPE_t y
        for i, var in enumerate(self.numeric_vars):
            y = self.data[sample_idx, var]
            self.sums_left[i] += y
            self.sq_sums_left[i] += y * y
        self.sums_right[:] = self.sums_total - self.sums_left
        self.sq_sums_right[:] = self.sq_sums_total - self.sq_sums_left


# ----------------------------------------------------------------------------------------------------------------------

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline double ld(double x) nogil:
    return ln(x) / ln(2.0)


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
        return
    cdef int maxd = 2 * <int>ld(n)
    introsort(Xf, samples, n, maxd)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
                      SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
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


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
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


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
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

