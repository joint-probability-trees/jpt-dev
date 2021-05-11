
import numpy as np
from dnutils import ifnone, out
from dnutils.stats import stopwatch


class Impurity:

    def __init__(self, tree, indices):
        self.tree = tree
        self.data = tree.data
        self.indices = tuple(indices)
        # self.targets = ifnone(targets, tuple(range(len(indices))))
        self.variables = tree.variables
        self._numeric_vars = tuple(i for i, v in enumerate(self.variables) if v.numeric)

        self._symbolic_vars = tuple(i for i, v in enumerate(self.variables) if v.symbolic)
        self.symbols = np.array([v.domain.n_values for v in self.variables if v.symbolic])

        self.symbols_left = np.zeros(shape=(max(self.symbols),
                                            len(self.symbolic_vars)), dtype=np.float64)
        self.symbols_right = np.zeros(shape=(max(self.symbols),
                                             len(self.symbolic_vars)), dtype=np.float64)
        self.symbols_total = np.zeros(shape=(max(self.symbols),
                                             len(self.symbolic_vars)), dtype=np.float64)

        self.sums_left = np.zeros(len(self.numeric_vars), dtype=np.float64)
        self.sums_right = np.zeros(len(self.numeric_vars), dtype=np.float64)
        self.sums_total = np.zeros(len(self.numeric_vars), dtype=np.float64)
        self.sq_sums_left = np.zeros(len(self.numeric_vars), dtype=np.float64)
        self.sq_sums_right = np.zeros(len(self.numeric_vars), dtype=np.float64)
        self.sq_sums_total = np.zeros(len(self.numeric_vars), dtype=np.float64)

        self.gini_buffer = np.ndarray(shape=self.symbols_total.shape, dtype=np.float64)
        self.gini_buffer2 = np.ndarray(shape=self.symbols_total.shape[1], dtype=np.float64)
        self.min_samples_leaf = tree.min_samples_leaf

    @property
    def symbolic_vars(self):
        return self._symbolic_vars

    @property
    def numeric_vars(self):
        return self._numeric_vars

    def gini_impurity(self, counts, n_samples):
        buffer = self.gini_buffer
        buffer[...] = counts
        buf2 = self.gini_buffer2
        np.power(buffer, 2, out=buffer)
        np.sum(buffer, axis=0, out=buf2)
        buf2 /= n_samples * n_samples
        buf2 -= 1
        buf2 /= -self.symbols
        return sum(buf2) / len(self.symbolic_vars)

    def col_is_constant(self, indices, col):
        v_ = None
        for i in indices:
            v = self.data[i, col]
            if v_ == v: continue
            if v_ != v:
                if v_ is None: v_ = v
                else: return False
        return True

    def compute_best_split(self):
        '''
        Computation uses the impurity proxy from sklearn: ::

            var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2

        See also: https://github.com/scikit-learn/scikit-learn/blob/de1262c35e2aa4ee062d050281ee576ce9e35c94/sklearn/tree/_criterion.pyx#L683
        '''
        best_var = None
        best_split_pos = None
        best_split_val = None
        # samples = len(self.indices)
        max_impurity_improvement = -1
        n_samples = len(self.indices)

        data = self.data

        if self.numeric_vars:
            np.sum(data[self.indices, :][:, self.numeric_vars] ** 2, axis=0, out=self.sq_sums_total)
            np.sum(data[self.indices, :][:, self.numeric_vars], axis=0, out=self.sums_total)
            variances_total = (self.sq_sums_total - (self.sums_total ** 2 / n_samples)) / n_samples

        self.symbols_total[...] = 0
        for sym, var in enumerate(self.symbolic_vars):
            self.symbols_total[:self.symbols[sym], sym] = np.bincount(data[self.indices, var].astype(np.int32),
                                                                      minlength=self.symbols[sym]).T

        # with stopwatch('new'):
        gini_total = self.gini_impurity(self.symbols_total, n_samples)
        impurity_total = (np.mean(variances_total) + gini_total) / 2
        out(impurity_total)
        # out(self.indices)
        # with stopwatch('old'):
        #     gini_total_ = np.sum(4 / self.symbols * (1 - np.sum(self.symbols_total ** 2, axis=0) / n_samples ** 2)) / len(self.symbolic_vars)
        symbolic = 0
        for variable in self.numeric_vars + self.symbolic_vars:
            indices = tuple(sorted(self.indices, key=lambda i: data[i, variable]))
            symbolic += variable in self.symbolic_vars
            numeric = not symbolic

            if self.col_is_constant(indices, variable):  # np.unique(data[indices, variable]).shape[0] == 1:
                continue

            # Prepare the stats
            if self.numeric_vars:
                self.sums_left[...] = 0
                self.sums_right[...] = self.sums_total
                self.sq_sums_left[...] = 0
                self.sq_sums_right[...] = self.sq_sums_total

            if self.symbolic_vars:
                self.symbols_left[...] = 0
                self.symbols_right[...] = self.symbols_total

            samples = np.zeros(shape=self.symbols_left.shape[0])
            impurity_improvement = 0

            for split_pos, sample in enumerate(indices):
                pivot = data[sample, variable]
                if numeric:
                    samples[0] += 1
                    samples[1] = n_samples - split_pos - 1
                    samples_left = samples[0]
                    samples_right = samples[1]
                else:
                    samples[int(pivot)] += 1
                    samples_left = samples[int(pivot)]
                    samples_right = n_samples - samples_left


                if numeric and split_pos == n_samples:
                    break

                # Compute the numeric impurity by variance approximation
                if self.numeric_vars:
                    y = data[sample, self.numeric_vars]  # (NB: there are all vectors here!)
                    self.sums_left += y
                    self.sums_right = self.sums_total - self.sums_left
                    self.sq_sums_left += y ** 2
                    self.sq_sums_right = self.sq_sums_total - self.sq_sums_left

                if self.symbolic_vars:
                    # Compute the symbolic impurity by the Gini index
                    for v_idx in self.symbolic_vars:
                        self.symbols_left[int(data[sample, v_idx]), :] += 1
                    self.symbols_right = self.symbols_total - self.symbols_left

                if split_pos < n_samples - 1 and data[split_pos, variable] == data[split_pos + 1, variable]:
                    continue

                if numeric:
                    impurity_improvement = 0
                denom = 0

                if self.numeric_vars:
                    variances_left = (self.sq_sums_left - self.sums_left ** 2 / samples_left) / samples_left if samples_left else 0
                    variances_right = (self.sq_sums_right - self.sums_right ** 2 / samples_right) / samples_right if samples_right else 0
                    variance_improvements = (variances_total - (samples_left * variances_left + samples_right * variances_right) / n_samples) / variances_total
                    avg_variance_improvement = np.mean(variance_improvements)
                    # out(variances_total, variances_left, variances_right, variance_improvements, avg_variance_improvement)
                    impurity_improvement += avg_variance_improvement if numeric else (np.mean(variances_left) * samples[int(data[sample, variable])] / n_samples)
                    out(variable, 'numeric improvement', impurity_improvement)
                    denom += 1

                if self.symbolic_vars:
                    gini_left = self.gini_impurity(self.symbols_left, samples_left)
                    gini_right = self.gini_impurity(self.symbols_right, samples_right)
                    gini_improvement = (gini_total - (samples_left / n_samples * gini_left +
                                        samples_right / n_samples * gini_right)) / gini_total
                    # out(gini_total, gini_left, gini_right, gini_improvement)
                    impurity_improvement += gini_improvement if numeric else (gini_left * samples[int(data[sample, variable])] / n_samples)
                    out(variable, 'symbolic improvement', impurity_improvement)
                    denom += 1

                impurity_improvement /= denom
                if symbolic:
                    self.symbols_left[...] = 0
                    self.sums_left[...] = 0
                    self.sq_sums_left[...] = 0
                    if split_pos == n_samples - 1:
                        impurity_improvement = (impurity_total - impurity_improvement) / impurity_total
                        out(samples)
                        if not all(not samples[i] or samples[i] >= self.min_samples_leaf for i in range(self.symbols[symbolic-1])):
                            impurity_improvement = 0
                        out('symbolic', impurity_improvement)
                else:
                    if samples_left < self.min_samples_leaf or samples_right < self.min_samples_leaf:
                        impurity_improvement = 0

                if (numeric or split_pos == n_samples - 1) and (impurity_improvement > max_impurity_improvement):
                    max_impurity_improvement = impurity_improvement
                    best_var = variable
                    best_split_pos = split_pos
                    best_split_val = (data[sample, best_var] + data[indices[split_pos + 1], best_var]) / 2. if numeric else None
                    out(best_var, best_split_val, max_impurity_improvement, samples)
        return best_var, best_split_val, max_impurity_improvement
