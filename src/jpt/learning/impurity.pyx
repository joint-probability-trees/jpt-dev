
import numpy as np
from dnutils import ifnone, out
from dnutils.stats import stopwatch


class Impurity:

    def __init__(self, tree, data, indices):
        self.tree = tree
        self.data = data
        self.indices = tuple(indices)
        # self.targets = ifnone(targets, tuple(range(len(indices))))
        self.variables = tree.variables
        self._numeric_vars = tuple(i for i, v in enumerate(self.variables) if v.numeric)

        self._symbolic_vars = tuple(i for i, v in enumerate(self.variables) if v.symbolic)
        if self.symbolic_vars:
            self.symbols = np.array([v.domain.n_values for v in self.variables if v.symbolic])
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

        if self.numeric_vars:
            self.sums_left = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sums_right = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sums_total = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sq_sums_left = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sq_sums_right = np.zeros(len(self.numeric_vars), dtype=np.float64)
            self.sq_sums_total = np.zeros(len(self.numeric_vars), dtype=np.float64)

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
        buf2 /= -self.symbols * 1/4
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
        max_impurity_improvement = -1
        n_samples = len(self.indices)
        denom = 0

        data = self.data

        # variances_total = [0]
        impurity_total = 0

        if self.numeric_vars:
            np.sum(data[self.indices, :][:, self.numeric_vars] ** 2, axis=0, out=self.sq_sums_total)
            np.sum(data[self.indices, :][:, self.numeric_vars], axis=0, out=self.sums_total)
            variances_total = (self.sq_sums_total - (self.sums_total ** 2 / n_samples)) / n_samples
            denom += 1
            impurity_total += len(self.numeric_vars) * np.mean(variances_total)

        if self.symbolic_vars:
            self.symbols_total[...] = 0
            for sym, var in enumerate(self.symbolic_vars):
                self.symbols_total[:self.symbols[sym], sym] = np.bincount(data[self.indices, var].astype(np.int32),
                                                                          minlength=self.symbols[sym]).T
            gini_total = self.gini_impurity(self.symbols_total, n_samples)
            impurity_total += len(self.symbolic_vars) * gini_total
            denom += 1
        else:
            gini_total = 0

        impurity_total /= denom * len(self.variables)
        symbolic = 0
        symbolic_idx = 0

        for variable in self.numeric_vars + self.symbolic_vars:
            indices = tuple(sorted(self.indices, key=lambda i: data[i, variable]))
            symbolic = variable in self.symbolic_vars
            symbolic_idx += symbolic
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

            samples = np.zeros(shape=self.symbols_left.shape[0] if symbolic else 2)
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

                if numeric and split_pos == n_samples - 1:
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
                    for i, v_idx in enumerate(self.symbolic_vars):
                        self.symbols_left[int(data[sample, v_idx]), i] += 1
                    self.symbols_right = self.symbols_total - self.symbols_left

                # skip calculation for identical values (i.e. until next 'real' splitpoint is reached
                if (split_pos < n_samples - 1 and symbolic or split_pos < n_samples and numeric) \
                        and data[indices[split_pos], variable] == data[indices[split_pos + 1], variable]:
                    continue

                if numeric:
                    impurity_improvement = 0

                if self.numeric_vars:
                    variances_left = (self.sq_sums_left - self.sums_left ** 2
                                      / samples_left) / samples_left if samples_left else 0

                    variances_right = (self.sq_sums_right - self.sums_right ** 2
                                       / samples_right) / samples_right if samples_right else 0

                    variance_improvements = (variances_total - (samples_left * variances_left
                                                                + samples_right * variances_right)
                                             / n_samples) / variances_total
                    variance_improvements[variances_total == 0] = 0

                    avg_variance_improvement = np.mean(variance_improvements)
                    impurity_improvement += avg_variance_improvement if numeric else (np.mean(variances_left) * samples[int(pivot)] * len(self.numeric_vars) / (n_samples * len(self.variables)))

                if self.symbolic_vars:
                    if gini_total:
                        gini_left = self.gini_impurity(self.symbols_left, samples_left)
                        gini_right = self.gini_impurity(self.symbols_right, samples_right)
                        gini_improvement = (gini_total - (samples_left / n_samples * gini_left +
                                            samples_right / n_samples * gini_right)) / gini_total
                        impurity_improvement += gini_improvement if numeric else (gini_left * samples[int(pivot)] * len(self.symbolic_vars) / (n_samples * len(self.variables)))

                if symbolic:
                    self.symbols_left[...] = 0

                    if self.numeric_vars:
                        self.sums_left[...] = 0
                        self.sq_sums_left[...] = 0

                    if split_pos == n_samples - 1:
                        impurity_improvement /= denom
                        impurity_improvement = (impurity_total - impurity_improvement) / impurity_total
                        if not all(not samples[i] or samples[i] >= self.min_samples_leaf for i in range(self.symbols[symbolic_idx-1])):
                            impurity_improvement = 0
                else:
                    impurity_improvement /= denom
                    if samples_left < self.min_samples_leaf or samples_right < self.min_samples_leaf:
                        impurity_improvement = 0
                if (numeric or split_pos == n_samples - 1) and (impurity_improvement > max_impurity_improvement):
                    max_impurity_improvement = impurity_improvement
                    best_var = variable
                    best_split_pos = split_pos
                    best_split_val = (data[sample, best_var]
                                      + data[indices[split_pos + 1], best_var]) / 2. if numeric else None
        return best_var, best_split_val, max_impurity_improvement
