import itertools
import math
import os
import random
from collections import defaultdict

from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import multivariate_normal

import dnutils
from dnutils import out
from ..constants import avalailable_colormaps, plotstyle, ddarkblue
from .gaussian import MultiVariateGaussian
from .example import BooleanFeature, SymbolicFeature, NumericFeature
from .intervals import Interval

logger = dnutils.getlogger(name='ProbsLogger', level=dnutils.ERROR)
style.use(plotstyle)


class GenericBayesFoo:  # aka Node-Distribution
    """Represents the multi-type distributions for a (Leaf-)Node in the matcalo.core.algorithms.StructRegTree.
    """

    def __init__(self, leaf=None):
        """This class contains a mapping from tuples of value combinations for the symbolic features to
        matcalo.core.algorithms.MultiVariateGaussian distributions over the numeric features (`training_dists`),
        as well as a mapping from tuples of value combinations for the symbolic features to an integer representing
        the count for the respective feature value combination (`training_cnts`).

        Example:
            training_dists = {  (True, 'Green'): N(µ_1, Σ_1),
                                (False, 'Green'): N(µ_2, Σ_2),
                                ...
                                (True, 'Yellow'): N(µ_n, Σ_n)}

            training_cnts = {   (True, 'Green'): 56,
                                (False, 'Green'): 78,
                                ...
                                (True, 'Yellow'): 130}

        :param leaf:    (int) the index of the Node these distributions belong to.
        """
        self.training_dists = defaultdict(MultiVariateGaussian)
        self.training_cnts = defaultdict(float)
        self.symbolics = None
        self.numerics = None
        self.leaf = leaf

    @property
    def counts(self):
        return {k: v/sum(self.training_cnts.values()) for k, v in self.training_cnts.items()}

    def add_trainingssample(self, example):
        # store names of features and targets, if not already done for later mapping
        self.symbolics = tuple(st.name for st in example.t if type(st) in (SymbolicFeature, BooleanFeature))
        self.numerics = [st.name for st in example.t if type(st) == NumericFeature]

        # if there is at least one numeric target, update distribution of respective symbolic target-value combination
        # with sample, i.e. update P(x1,...,xm|xm+1,...,xn), with x1,...,xm NumericFeature, xm+1,...,xn SymbolicFeature
        if [float(st.value) for st in example.t if type(st) == NumericFeature]:
            self.training_dists[tuple(st.value for st in example.t if type(st) in (SymbolicFeature, BooleanFeature))].update([float(st.value) for st in example.t if type(st) == NumericFeature])

        # update P(xm+1,...,xn)
        self.training_cnts[tuple(st.value for st in example.t if type(st) in (SymbolicFeature, BooleanFeature))] += 1

    def distribution(self):
        # return P(x1,...,xm|xm+1,...,xn) and P(xm+1,...,xn)
        # TODO: return P(x1,...,xm|xm+1,...,xn) * P(xm+1,...,xn) instead?
        return self.training_dists, self.training_cnts

    def cond_distribution(self, given):
        # example: generate P(x1,x2,x5|µ1<=x3<=µ2,x4=A) from P(x1,...,x5) with x3,x4 ∈ given
        pass

    def query(self, cats, nums):
        r"""Calculates the probability of :math:`P(x1,x2,x5|\mu_1 \leq x3 \leq \mu_2,x4=A)` from :math:`P(x1,...,x5)`
        where :math:`x3 \in \text{nums}, x4 \in \text{cats and } x1, x2, x5 \in \text{targets} \setminus \text{nums} \cup \text{cats}`.

        :param cats:    the allowed values of the categorical targets
        :type cats:     list of sets
        :param nums:    the constraints on the values of the numeric targets
        :type nums:     list of Intervals and/or callables
        :return:        the probability for the given query
        :rtype:         float
        """
        # self.training_dists[q].conditional(q)
        s = 0.

        # TODO: figure out what kind of query is required, then pass on to specialized function
        if all(isinstance(v, Interval) for v in nums):
            # CASE 1: query looks like cats = [{True},{True, False}, {A, B}], nums = [Interval(x, y), Interval(i, j)] (default)
            for c in itertools.product(*cats):
                if c in self.training_dists:
                    # sum up probabilities for categorical combinations
                    s += self.training_dists[c].cdf(nums)
        elif any(callable(v) for v in nums):
            # CASE 2: query looks like cats = [{True},{True, False}, {A, B}], nums = [Interval(x, y), min]
            # (at least one numerical value is a function denoting that the minimum/maximum of the respective
            # value is looked for)
            pass
        else:
            pass

        return s

    def pred_val(self, precision=2):
        """This representation is used for the graph plot of matcalo.core.algorithms.StructRegTree.
        """
        return ",<BR/>".join([f'{tuple([f"{sfname}={sfval}" for sfname, sfval in zip(self.symbolics, k)])}:<BR/>{v} ({round(v/sum(self.training_cnts.values())*100, precision)}%)' for k, v in self.training_cnts.items()])

    @property
    def name(self):
        return f"MVG-Leaf-{self.leaf}"

    def __str__(self):
        return ",\n".join([f'{tuple([f"{sfname}={sfval}" for sfname,sfval in zip(self.symbolics, k)])}: {str(v)}' for k, v in self.training_dists.items()])

    def graphstr(self):
        """This representation is used for the graph plot of matcalo.core.algorithms.StructRegTree"""
        return ",<BR/>".join([f'{tuple([f"{sfname}={sfval}" for sfname,sfval in zip(self.symbolics, k)])}:<BR/>{str(v)}' for k, v in self.training_dists.items()])

    def plot(self, directory='/tmp', pdf=False, view=False, contour=True, filled=False):
        """Generate r x c matrix of plots, depending on the number of distributions available.
        If r = 1 and c = 1, axs will be a single figure, if r = 1 and c > 1, axs will be a 1-dimensional array
        of subplots, and if r and c > 1, axs will be a 2-dimensional array of subplots. Therefore use
        (axs[int(i / r), i % c] if c > 1 and r > 1 else axs[i % c] if c > 1 else axs) to address the respective
        figure correctly. A 1-dim bell curve plot will be generated for 1-dimensional distributions, a contour or
        3d-plot using the first two dimensions of the underlying >1-dimensional distribution.

        :param directory:   (str) the directory to store the generated plot files
        :param pdf:         (bool) whether to store files as PDF. If false, a png is generated by default
        :param view:        (bool) whether to display generated plots, default False (only stores files)
        :param contour:     (bool) whether to draw a contour plot instead of a 3D-plot.
        :param filled:      (bool) whether to fill the contour plot or only draw the rings. Default is false. Only works
                            in combination with `contour=True`.
        :return:            None
        """
        # Only save figures, do not show
        if not view:
            plt.ioff()

        numdists = len(self.training_dists)
        r, c = [round(math.sqrt(numdists)), math.ceil(math.sqrt(numdists))]
        figuretitle = f'(Multivariate) Gaussian Distribution{"s" if numdists > 1 else ""} for Leaf #{self.leaf}'
        fig, axs = plt.subplots(r, c)
        fig.canvas.set_window_title(figuretitle)
        # fig.suptitle(figuretitle)
        for i, (symb, dist) in enumerate(self.training_dists.items()):
            # TODO remove. Randomly choose 3d or contour plot (filled or not) with random colormap
            filled = random.choice([True, False])
            contour = random.choice([True, False])
            contour = False
            plotcolormap = random.choice(avalailable_colormaps)
            plotcolormap = 'viridis'
            if logger.level < dnutils.ERROR:
                out(logger.level, dnutils.ERROR)
                out(f'Chosen settings for leaf {self.leaf}: filled={filled}, contour={contour}, randcol={plotcolormap}')

            plottitle = f'Distribution for \n{[f"{k}={v}" for k, v in zip(self.symbolics, symb)]}: N{dist.mean_, dist.cov_}'
            (axs[int(i / r), i % c] if c > 1 and r > 1 else axs[i % c] if c > 1 else axs).set_title(plottitle)
            try:
                ax = (axs[int(i / r), i % c] if c > 1 and r > 1 else axs[i % c] if c > 1 else axs)
                if dist.dim == 1:
                    # for 1-dimensional distributions
                    x = np.linspace(dist.mean[0] - 2 * dist.cov[0][0], dist.mean[0] + 2 * dist.cov[0][0], 500)
                    y = multivariate_normal.pdf(x, mean=dist.mean[0], cov=dist.cov[0][0])
                    ax.plot(x, y, ddarkblue)
                    ax.set_xlabel(self.numerics[0])
                    ax.set_ylabel('PDF')
                else:
                    # data is >=2-dimensional
                    out(f'OLD mean {dist.mean}, cov\n{np.array(dist.cov)}', dist)

                    # determine pair of features with maximum covariances in overall covariance matrix, to make a
                    # reasonable decision which dimension to plot
                    max_val = -1
                    max_pair = None
                    for n in range(len(dist.cov)):
                        for m in range(len(dist.cov)):
                            if m >= n: continue
                            if dist.cov[n][m] > max_val:
                                max_val = dist.cov[n][m]
                                max_pair = (n, m)

                    out('maxval, maxpair', max_val, max_pair)

                    # crossproduct extract: a[np.ix_([i, j], [i, j])] extracts [[a[i, i] a[i, j]], [a[j, i] a[j, j]]]
                    cov = np.array(dist.cov)[np.ix_([max_pair[0], max_pair[1]], [max_pair[0], max_pair[1]])]
                    mean = [dist.mean[max_pair[0]], dist.mean[max_pair[1]]]
                    out(f'NEW mean {mean}, cov\n{cov}')

                    if contour:
                        # for > 2-dimensional distributions
                        x, y = np.mgrid[mean[0]-2*cov[0][0]:mean[0]+2*cov[0][0]:.01, mean[1]-2*cov[1][1]:mean[1]+2*cov[1][1]:.01]
                        out('CONTOUR', x, y)
                        pos = np.dstack((x, y))
                        rv = multivariate_normal(mean, cov, allow_singular=True)
                        out(x, y, rv.pdf(pos))
                        if filled:
                            ax.contourf(x, y, rv.pdf(pos), cmap=plotcolormap)
                        else:
                            ax.contour(x, y, rv.pdf(pos), cmap=plotcolormap)
                        ax.set_xlabel(self.numerics[max_pair[0]])
                        ax.set_ylabel(self.numerics[max_pair[1]])
                    else:
                        # for multivariate distributions, only generate a 3D-plot of the first 2 dims
                        x = np.linspace(mean[0] - 2 * cov[0][0], mean[0] + 2 * cov[0][0], 500)
                        y = np.linspace(mean[1] - 2 * cov[1][1], mean[1] + 2 * cov[1][1], 500)
                        out('3D', x, y)
                        rv = multivariate_normal(mean, cov, allow_singular=True)

                        X, Y = np.meshgrid(x, y)
                        pos = np.empty(X.shape + (2,))
                        pos[:, :, 0] = X
                        pos[:, :, 1] = Y

                        # remove previously created axis, as it is not a 3d-projection, and create a new one
                        ax.remove()
                        ax = fig.add_subplot(r, c, i + 1, projection='3d')
                        ax.set_title(plottitle)
                        out('plot input:', x, y, rv.pdf(pos))
                        ax.plot_surface(X, Y, rv.pdf(pos), cmap=plotcolormap, linewidth=0)

                        # The heatmap
                        # levels = np.linspace(0, 1, 40)
                        # ax.contourf(X, Y, 0.1 * rv.pdf(pos), zdir='z', levels=0.1 * levels, alpha=0.9)
                        #
                        # # The wireframe
                        # ax.plot_wireframe(X, Y, rv.pdf(pos), rstride=5, cstride=5, color='k')

                        # # The scatter. Note that the altitude is defined based on the pdf of the
                        # # random variable
                        # sample = rv.rvs(500)
                        # ax.scatter(sample[:, 0], sample[:, 1], 1.05 * rv.pdf(sample), c='k')

                        ax.set_xlabel(self.numerics[max_pair[0]])
                        ax.set_ylabel(self.numerics[max_pair[1]])
                        ax.set_zlabel('PDF')
            except np.linalg.LinAlgError:
                logger.warning(f'Could not generate plot for distribution N{dist.mean_, dist.cov_} of leaf {self.leaf} (Singular Matrix)')

        # remove excess axes
        for p in range(r*c-numdists):
            (axs[-1, numdists % c + p - 1] if c > 1 and r > 1 else axs[numdists % c + p - 1]).remove()

        # save figure as PDF or PNG
        if pdf:
            with PdfPages(os.path.join(directory, f'{self.name}.pdf')) as pdf:
                pdf.savefig(fig)
        else:
            plt.savefig(os.path.join(directory, f'{self.name}.png'))

        if view:
            plt.show()
