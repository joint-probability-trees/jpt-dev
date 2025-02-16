# cython: auto_cpdef=False,
# cython: infer_types=False,
# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
__module__ = 'quantiles.pyx'

from cmath import isnan
from operator import itemgetter, attrgetter
from typing import Iterable

from dnutils import ifnot, first, ifnone

from ...base.constants import eps
from ...base.intervals import R, EMPTY, EXC, INC

import numpy as np

from ...base.functions cimport PiecewiseFunction, LinearFunction, ConstantFunction, Undefined, Jump, Function
from ...base.intervals cimport ContinuousSet, RealSet
cimport numpy as np
from ...base.cutils cimport SIZE_t, DTYPE_t, sort

import warnings

from jpt.base.utils import pairwise, normalized
from jpt.base.errors import Unsatisfiability

from .cdfreg import CDFRegressor

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------------------------------

cdef class QuantileDistribution:
    '''
    Abstract base class for any quantile-parameterized cumulative data distribution.
    '''

    cdef public DTYPE_t epsilon
    cdef public np.int32_t verbose
    cdef public np.int32_t min_samples_mars
    cdef PiecewiseFunction _cdf, _pdf, _ppf

    def __init__(self, epsilon=.01, min_samples_mars=5, verbose=False):
        self.epsilon = epsilon
        self.verbose = verbose
        self.min_samples_mars = min_samples_mars
        self._cdf = None
        self._pdf = None
        self._ppf = None

    def __hash__(self):
        return hash((
            QuantileDistribution,
            self.epsilon,
            self.verbose,
            self.min_samples_mars,
            self._cdf
        ))

    def __eq__(self, o):
        if not isinstance(o, QuantileDistribution):
            raise TypeError('Illegal type: %s' % type(o).__name__)
        return (self.epsilon == o.epsilon and
                self.min_samples_mars == o.min_samples_mars and
                self.cdf == o.cdf)

    cpdef QuantileDistribution copy(QuantileDistribution self):
        cdef QuantileDistribution result = QuantileDistribution(self.epsilon,
                                                                min_samples_mars=self.min_samples_mars)
        result.cdf = self.cdf.copy()
        return result

    @staticmethod
    def from_cdf(cdf: PiecewiseFunction) -> QuantileDistribution:
        d = QuantileDistribution()
        d.cdf = cdf
        return d

    @staticmethod
    def from_pdf(pdf: PiecewiseFunction) -> QuantileDistribution:
        d = QuantileDistribution()
        d.pdf = pdf
        return d

    cpdef _assert_consistency(self):
        if self._cdf is not None:
            assert len(self.cdf.functions) > 1, self.cdf.pfmt()
            assert len(self.cdf.intervals) == len(self.cdf.functions), \
                '# intervals: %s != # functions: %s' % (len(self.cdf.intervals),
                                                        len(self.cdf.functions))

    @staticmethod
    def pdf_to_cdf(
            pdf: PiecewiseFunction,
            dirac_weights: Iterable[float] = None
    ) -> PiecewiseFunction:
        '''Convert a PDF into a CDF by piecewise integration'''
        cdf = PiecewiseFunction()
        pdf = pdf.rectify()
        head = 0
        dirac_weights = normalized(dirac_weights) if dirac_weights is not None else []
        dirac_offsets = []

        last_int: ContinuousSet = None
        for i, f in pdf.iter():
            if not f.value:
                f_ = ConstantFunction(head)
            elif np.isinf(f.value):
                dirac_offsets.append((
                    len(cdf),
                    dirac_offsets.pop(-1) if dirac_offsets else 1
                ))
                last_int = i
                continue
            else:
                f_ = LinearFunction(
                    f.value,
                    head - f.value * i.lower
                )
            head = f_.eval(i.upper)
            if last_int is not None and last_int.contiguous(i):
                i = last_int.union(i)
                last_int = None
            cdf.append(i.copy(), f_)

        # Weigh the dirac impulses proportionally
        residual = 1 - cdf.eval(cdf.intervals[-1].any_point())

        if residual < -1e-8:
            raise ValueError(
                'Illegal CDF value > 1: %f' % (1 - residual)
            )
        elif residual > 0:
            dirac_offsets = [(i, w * residual) for i, w in dirac_offsets]

        # Apply offsets by dirac impulses
        offset = 0
        for i, (_, f) in enumerate(cdf.iter()):
            if dirac_offsets and dirac_offsets[0][0] == i:
                offset += dirac_offsets.pop(0)[1]
            f += offset

        return cdf.stretch(
            1 / cdf.eval(
                cdf.intervals[-1].any_point()
            )
        )

    cpdef QuantileDistribution fit(
            self,
            DTYPE_t[:, ::1] data,
            SIZE_t[::1] rows,
            SIZE_t col,
            DTYPE_t leftmost = np.nan,
            DTYPE_t rightmost = np.nan
    ):
        """
        Fit the quantile distribution to the rows ``rows`` and column ``col`` of the
        data array ``data``.
        
        ``leftmost`` and ``rightmost`` can be optional "imaginary" points on the leftmost or
        rightmost side, respectively.
        
        :param data: 
        :param rows: 
        :param col: 
        :param leftmost: 
        :param rightmost: 
        :return: 
        """
        if rows is None:
            rows = np.arange(data.shape[0], dtype=np.int64)

        # We have to copy the data into C-contiguous array first
        cdef SIZE_t i, n_samples = rows.shape[0] + (0 if isnan(leftmost) else 1) + (0 if isnan(rightmost) else 1)
        cdef DTYPE_t[:, ::1] data_buffer = np.ndarray(
            shape=(2, n_samples),
            dtype=np.float64,
            order='C'
        )
        # Write the data points themselves into the first (upper) row of the array...
        if not isnan(leftmost):
            data_buffer[0, 0] = leftmost
        for i in range(n_samples - (not isnan(rightmost)) - (not isnan(leftmost))):
            if not isnan(leftmost):
                assert data[rows[i], col] > leftmost, 'leftmost: %s <= %s' % (data[rows[i], col], leftmost)
            if not isnan(rightmost):
                assert data[rows[i], col] < rightmost, 'rightmost: %s >= %s' % (data[rows[i], col], rightmost)
            data_buffer[0, i + (not isnan(leftmost))] = data[rows[i], col]
        if not isnan(rightmost):
            data_buffer[0, n_samples - 1] = rightmost
        np.asarray(data_buffer[0, :]).sort()

        # ... and the respective quantiles into the row below
        cdef SIZE_t count = 0,
        i = 0
        for i in range(n_samples):
            if i > 0 and data_buffer[0, i] == data_buffer[0, i - 1]:
                data_buffer[1, count - 1] += 1
            else:
                data_buffer[0, count] = data_buffer[0, i]
                data_buffer[1, count] = <DTYPE_t> i + 1
                count += 1

        for i in range(count):
            data_buffer[1, i] -= 1
            data_buffer[1, i] /= <DTYPE_t> (n_samples - 1)

        data_buffer = np.ascontiguousarray(data_buffer[:, :count])
        cdef DTYPE_t[::1] x, y
        n_samples = count

        self._ppf = self._pdf = None
        if n_samples > 1:
            regressor = CDFRegressor(eps=self.epsilon)
            regressor.fit(data_buffer)
            self._cdf = PiecewiseFunction()
            self._cdf.functions.append(ConstantFunction(0))
            self._cdf.intervals.append(ContinuousSet(-np.inf, np.inf, EXC, EXC))
            for left, right in pairwise(regressor.support_points):
                self._cdf.functions.append(LinearFunction.from_points(tuple(left), tuple(right)))
                self._cdf.intervals[-1].upper = left[0]
                self._cdf.intervals.append(ContinuousSet(left[0], right[0], 1, 2))

            # overwrite right most interval by an interval with an including right border
            self._cdf.intervals[-1] = ContinuousSet(self._cdf.intervals[-1].lower,
                                                    np.nextafter(self._cdf.intervals[-1].upper,
                                                                 self._cdf.intervals[-1].upper + 1), INC, EXC)

            self._cdf.functions.append(ConstantFunction(1))
            self._cdf.intervals.append(ContinuousSet(self._cdf.intervals[-1].upper, np.inf, INC, EXC))
        else:
            x = data_buffer[0, :]
            y = data_buffer[1, :]
            self._cdf = PiecewiseFunction()
            self._cdf.intervals.append(ContinuousSet(-np.inf, x[0], EXC, EXC))
            self._cdf.functions.append(ConstantFunction(0))
            self._cdf.intervals.append(ContinuousSet(x[0], np.inf, INC, EXC))
            self._cdf.functions.append(ConstantFunction(1))

        self._assert_consistency()

        return self

    cpdef crop(self, ContinuousSet interval):
        """
        Return a copy this quantile distribution that is cropped to the ``interval``.
        :param interval: The interval to crop to
        :return: A modified, normalized copy of this QuantileDistribution
        """
        if self._cdf is None:
            raise RuntimeError('No quantile distribution fitted. Call fit() first.')

        # if given interval is outside quantile boundaries,
        # cropping would result in a non-valid quantile
        if (interval.uppermost() < self._cdf.intervals[0].upper or
                interval.lowermost() > self._cdf.intervals[-1].lower):
            raise Unsatisfiability(
                'CDF has zero Probability in %s (should be in %s)' % (
                    interval,
                    ContinuousSet(
                        self._cdf.intervals[0].upper,
                        self._cdf.intervals[-1].lower, EXC, EXC
                    )
                )
            )

        # I: crop
        cdf_ = self.cdf.crop(interval)

        # II: add constant to move to
        cdf_ += -cdf_.eval(cdf_.intervals[0].lowermost())

        # everything left of the leftmost point of the cropped function evaluates to 0
        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet(-np.inf, cdf_.intervals[0].lower, EXC, EXC))
        cdf.functions.append(ConstantFunction(0.))

        # III: stretch function to represent a proper quantile fn
        alpha = cdf_.functions[-1].eval(interval.uppermost())

        f_ = cdf_.functions[0]
        for i, f in zip(cdf_.intervals, cdf_.functions):
            if f == ConstantFunction(0.):
                cdf.intervals[-1].upper = i.upper
                continue
            y = cdf.functions[-1].eval(i.lower)
            c = (f.eval(i.lower) - f_.eval(i.lower)) * alpha  # If the function is continuous (no jump), c = 0
            f_ = f
            cdf.intervals.append(ContinuousSet(i.lower, i.upper, INC, EXC))
            upper_ = np.nextafter(i.upper, i.upper - 1)

            if isinstance(f, ConstantFunction) or i.size() == 1:
                cdf.functions.append(ConstantFunction(y + c))
            else:
                cdf.intervals[-1].lower = cdf.intervals[-2].upper
                cdf.functions.append(LinearFunction.from_points((i.lower, y + c),
                                                                (upper_, (f.m / alpha) * (upper_ - i.lower) + y + c)))
        if cdf.functions[-1] == ConstantFunction(1.):
            cdf.intervals[-1].upper = np.inf
            cdf.intervals[-1].right = EXC
            if len(cdf.intervals) > 1:
                cdf.intervals[-1].lower = cdf.intervals[-2].upper
        else:
            if interval.uppermost() in cdf.intervals[-1]:
                cdf.intervals[-1].upper = np.nextafter(cdf.intervals[-1].upper, cdf.intervals[-1].upper - 1)

            # everything right of the rightmost point of the cropped function evaluates to 1
            cdf.functions.append(ConstantFunction(1.))
            cdf.intervals.append(ContinuousSet(cdf.intervals[-1].upper, np.inf, INC, EXC))

        # Clean the function segments that might have become empty
        intervals_ = []
        functions_ = []
        for i, f in zip(cdf.intervals, cdf.functions):
            if not i.isempty():
                intervals_.append(i.copy())
                functions_.append(f.copy())
        cdf.functions = functions_
        cdf.intervals = intervals_

        result = QuantileDistribution(self.epsilon, min_samples_mars=self.min_samples_mars)
        result.cdf = cdf

        result._assert_consistency()

        return result

    @property
    def cdf(self):
        return self._cdf

    @cdf.setter
    def cdf(self, cdf):
        self._cdf = cdf
        self._pdf = None
        self._ppf = None

    @property
    def pdf(self):
        if self._cdf is None:
            raise RuntimeError('No quantile distribution fitted. Call fit() first.')

        elif self._pdf is None:
            pdf = self._cdf.differentiate()
            if len(self._cdf.intervals) == 2:
                pdf.intervals.insert(
                    1,
                    ContinuousSet(
                        pdf.intervals[0].upper,
                        pdf.intervals[0].upper + eps,
                        INC,
                        EXC
                    )
                )
                pdf.intervals[-1].lower += eps
                pdf.functions.insert(1, ConstantFunction(np.inf))
            self._pdf = pdf
        return self._pdf

    @pdf.setter
    def pdf(self, pdf):
        self._pdf = pdf
        self._cdf = QuantileDistribution.pdf_to_cdf(pdf)
        self._ppf = None

    @property
    def ppf(self):
        cdef PiecewiseFunction ppf = PiecewiseFunction()
        cdef ContinuousSet interval
        cdef Function f
        cdef DTYPE_t one_plus_eps = np.nextafter(1, 2)

        if self._cdf is None:
            raise RuntimeError('No quantile distribution fitted. Call fit() first.')

        elif self._ppf is None:

            ppf.intervals.append(ContinuousSet(-np.inf, np.inf, EXC, EXC))

            self._assert_consistency()

            if len(self._cdf.functions) == 2:
                ppf.intervals[-1].upper = 1
                ppf.functions.append(Undefined())
                ppf.intervals.append(ContinuousSet(1, one_plus_eps, INC, EXC))
                ppf.functions.append(ConstantFunction(self._cdf.intervals[-1].lower))
                ppf.intervals.append(ContinuousSet(one_plus_eps, np.inf, INC, EXC))
                ppf.functions.append(Undefined())
                self._ppf = ppf
                return ppf

            y_cdf = 0
            for interval, f in zip(self._cdf.intervals, self._cdf.functions):

                if f.is_invertible():
                    ppf.functions.append(f.invert())
                    const = 0
                elif not f.c:
                    ppf.functions.append(Undefined())
                elif f(interval.lower) > y_cdf:
                    ppf.functions.append(ConstantFunction(interval.lower))
                else:
                    continue

                if not np.isinf(interval.upper):
                    ppf.intervals[-1].upper = min(one_plus_eps, f(interval.upper))
                else:
                    ppf.intervals[-1].upper = one_plus_eps

                ppf.intervals.append(ContinuousSet(ppf.intervals[-1].upper, np.inf, INC, EXC))
                y_cdf = min(one_plus_eps, f(interval.upper))

            ppf.intervals[-2].upper = ppf.intervals[-1].lower = one_plus_eps
            ppf.functions.append(Undefined())

            if not np.isnan(ppf.eval(one_plus_eps)):
                print(ppf.functions[ppf.idx_at(one_plus_eps)],
                      (<LinearFunction> ppf.functions[ppf.idx_at(one_plus_eps)]).eval(one_plus_eps),
                      ppf.eval(one_plus_eps))
                raise ValueError('ppf(%.16f) = %.20f [fct: %d]:\n %s\n===cdf\n%s' % (one_plus_eps,
                                                                                     ppf.eval(one_plus_eps),
                                                                                     ppf.idx_at(one_plus_eps),
                                                                                     ppf.pfmt(),
                                                                                     self._cdf.pfmt()))

            if np.isnan(ppf.eval(1.)):
                raise ValueError(
                    str(one_plus_eps) +
                    'ppf:\n %s\n===cdf\n%s\nval' % (
                        ppf.pfmt(),
                        (self._cdf.pfmt() + str(ppf.intervals[-1].lower)
                         + ' ' + str(ppf.intervals[-2].upper)
                         + ' ' + str(ppf.eval(1.))
                         + ' ' + str(ppf.eval(one_plus_eps)))
                    )
                )
            self._ppf = ppf
        return self._ppf

    @staticmethod
    def merge(distributions, weights=None):
        '''
        Construct a merged quantile-distribution from the passed distributions using the ``weights``.
        '''
        intervals = [ContinuousSet(-np.inf, np.inf, EXC, EXC)]
        functions = [ConstantFunction(0)]
        if weights is None:
            weights = [1. / len(distributions)] * len(distributions)

        if abs(sum(weights) - 1) > 1e-8:
            raise ValueError(
                'Weights must sum to 1, got sum %s; %s' % (
                    sum(weights),
                    str(weights)
                )
            )
        if any(np.isnan(w) for w in weights):
            raise ValueError(
                'Detected NaN in weight vector!'
            )

        # --------------------------------------------------------------------------------------------------------------
        # We preprocess the CDFs that are in the form of "jump" functions
        jumps = {}
        for w, cdf in [(w, d.cdf) for w, d in zip(weights, distributions) if len(d.cdf) == 2]:
            jumps[cdf.intervals[0].upper] = jumps.get(cdf.intervals[0].upper, Jump(cdf.intervals[0].upper, 1, 0))
            jumps.get(cdf.intervals[0].upper).weight += w

        # --------------------------------------------------------------------------------------------------------------

        lower = sorted(
            [
                (i.lower, f, w)
                for d, w in zip(distributions, weights)
                    for i, f in zip(d.cdf.intervals, d.cdf.functions)
                if not isinstance(f, ConstantFunction) and w > 0
            ] + [
                (j.knot, j, j.weight) for j in jumps.values() if j.weight > 0
            ],
            key=itemgetter(0)
        )
        upper = sorted(
            [
                (i.upper, f, w)
                for d, w in zip(distributions, weights)
                    for i, f in zip(d.cdf.intervals, d.cdf.functions)
                if not isinstance(f, ConstantFunction) and w > 0
            ],
            key=itemgetter(0)
        )

        # --------------------------------------------------------------------------------------------------------------

        m = 0
        c = 0

        while lower or upper:
            pivot = None
            m_ = m
            offset = 0

            # Process all function intervals whose lower bound is minimal and
            # smaller than the smallest upper interval bound
            while lower and (pivot is None and first(lower, first) <= first(upper, first, np.inf) or
                   pivot == first(lower, first, np.inf)):
                l, f, w = lower.pop(0)
                if isinstance(f, ConstantFunction) or l == -np.inf or isinstance(f, LinearFunction) and f.m == 0:
                    continue
                if isinstance(f, Jump):  # and isinstance(functions[-1], LinearFunction):
                    offset += w
                if isinstance(f, LinearFunction):
                    m_ += f.m * w
                pivot = l

            # Do the same for the upper bounds...
            while upper and (pivot is None and first(upper, first) <= first(lower, first, np.inf) or
                   pivot == first(upper, first, np.inf)):
                u, f, w = upper.pop(0)
                if isinstance(f, (ConstantFunction, Jump)) or u == np.inf or isinstance(f, LinearFunction) and f.m == 0:
                    continue
                m_ -= f.m * w
                pivot = u

            if pivot is None:
                continue

            y = m * pivot + c
            m = m_ if abs(m_) > 1e-8 else 0
            c = y - m * pivot + offset

            intervals[-1].upper = pivot
            if (c or m) and (m != functions[-1].m or c != functions[-1].c):
                # Split the last interval at the pivot point
                intervals.append(ContinuousSet(pivot, np.inf, INC, EXC))
                # Evaluate the old function at the new pivot point to get the intercept
                functions.append(LinearFunction(m, c) if abs(m) > 1e-8 else ConstantFunction(c))

        # If the merging ends with an "approximate" constant function
        # remove it. This may happen for numerical imprecision.
        while len(functions) > 1 and abs(functions[-1].m) <= 1e-08 and functions[-1].m:
            del intervals[-1]
            del functions[-1]

        cdf = PiecewiseFunction()
        cdf.functions = functions
        cdf.intervals = intervals

        cdf.ensure_right(ConstantFunction(1), cdf.intervals[-1].lower)

        distribution = QuantileDistribution()
        distribution.cdf = cdf

        return distribution

    def sample(self, n) -> np.ndarray:
        """
        Sample n samples from the underlying quantile distribution.
        :param n: the number of samples
        :return: 1D numpy array with samples.
        """

        if len(self.cdf.intervals) == 2:
            return np.full(n, self.cdf.intervals[0].upper)

        # create probability distribution with the probabilities of each function part
        interval_probabilities = np.array([function.eval(interval.upper) - function.eval(interval.lower)
                                           for function, interval in
                                           zip(self.cdf.functions[1:-1], self.cdf.intervals[1:-1])])

        # sample in which function part the samples will be
        sample_intervals = np.random.choice(list(range(0, len(interval_probabilities))), size=(n,),
                                            p=interval_probabilities)

        # initialize result
        result = np.zeros((n, ))

        # for every interval
        for idx, interval in enumerate(interval for interval in self.cdf.intervals[1:-1]):

            # get the indices where the samples are from this interval
            indices = np.where(sample_intervals == idx)[0]

            # sample as many points as needed
            samples = interval.sample(len(indices))

            # write them into the result
            result[indices] = samples

        return result

    def to_json(self):
        return {'epsilon': self.epsilon,
                'min_samples_mars': self.min_samples_mars,
                'cdf': self._cdf.to_json()}

    @staticmethod
    def from_json(data):
        q = QuantileDistribution(epsilon=data['epsilon'],
                                 min_samples_mars=data['min_samples_mars'])
        q.cdf = PiecewiseFunction.from_json(data['cdf'])
        return q


