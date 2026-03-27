import os
import tempfile
import unittest
from unittest import TestCase

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from jpt.distributions import SymbolicType
from jpt.distributions.univariate import IntegerType
from jpt.distributions.univariate.gaussian import Gaussian
from jpt.plotting.engines.matplotlib_engine import MatplotlibRendering

from test.testutils import gaussian_numeric


# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib not installed')
class TestMatplotlibMultinomial(TestCase):

    def setUp(self):
        self.engine = MatplotlibRendering()
        DistABC = SymbolicType('TestType', labels=['A', 'B', 'C'])
        self.dist = DistABC().set(params=[0.5, 0.3, 0.2])
        self.tmpdir = tempfile.mkdtemp()

    def test_plot_multinomial_vertical(self):
        """Verify vertical bar plot creates a file and returns None."""
        result = self.engine.plot_multinomial(
            self.dist,
            title='Test Multinomial',
            fname='test_multinomial',
            directory=self.tmpdir,
            view=False
        )
        self.assertIsNone(result)
        self.assertTrue(
            os.path.exists(os.path.join(self.tmpdir, 'test_multinomial.png'))
        )

    def test_plot_multinomial_horizontal(self):
        """Verify horizontal bar plot runs without error."""
        self.engine.plot_multinomial(
            self.dist,
            fname='test_horiz',
            directory=self.tmpdir,
            horizontal=True,
            view=False
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.tmpdir, 'test_horiz.png'))
        )

    def test_plot_multinomial_alphabet(self):
        """Verify alphabetical sorting runs without error."""
        self.engine.plot_multinomial(
            self.dist,
            fname='test_alpha',
            directory=self.tmpdir,
            alphabet=True,
            view=False
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.tmpdir, 'test_alpha.png'))
        )

    def test_plot_multinomial_max_values(self):
        """Verify max_values limits the number of bars."""
        self.engine.plot_multinomial(
            self.dist,
            fname='test_maxval',
            directory=self.tmpdir,
            max_values=2,
            view=False
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.tmpdir, 'test_maxval.png'))
        )

    def test_plot_multinomial_pdf_format(self):
        """Verify PDF output format."""
        self.engine.plot_multinomial(
            self.dist,
            fname='test_pdf',
            directory=self.tmpdir,
            pdf=True,
            view=False
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.tmpdir, 'test_pdf.pdf'))
        )


# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib not installed')
class TestMatplotlibNumeric(TestCase):

    def setUp(self):
        self.engine = MatplotlibRendering()
        self.dist = gaussian_numeric()
        self.tmpdir = tempfile.mkdtemp()

    def test_plot_numeric_returns_figure(self):
        """Verify plot_numeric returns a matplotlib Figure."""
        result = self.engine.plot_numeric(
            self.dist,
            title='Test Numeric',
            fname='test_numeric',
            directory=self.tmpdir,
            view=False
        )
        self.assertIsInstance(result, Figure)

    def test_plot_numeric_pdf_format(self):
        """Verify PDF output is created."""
        self.engine.plot_numeric(
            self.dist,
            fname='test_numeric_pdf',
            directory=self.tmpdir,
            pdf=True,
            view=False
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.tmpdir, 'test_numeric_pdf.pdf'))
        )

    def test_plot_numeric_directory_creation(self):
        """Verify that a non-existent directory is created."""
        subdir = os.path.join(self.tmpdir, 'subdir', 'nested')
        self.engine.plot_numeric(
            self.dist,
            fname='test_nested',
            directory=subdir,
            view=False
        )
        self.assertTrue(os.path.isdir(subdir))


# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib not installed')
class TestMatplotlibInteger(TestCase):

    def setUp(self):
        self.engine = MatplotlibRendering()
        Die = IntegerType('Dice', 1, 6)
        self.dist = Die()
        self.dist.set([1 / 6] * 6)
        self.tmpdir = tempfile.mkdtemp()

    def test_plot_integer_vertical(self):
        """Verify vertical integer plot returns a Figure."""
        result = self.engine.plot_integer(
            self.dist,
            title='Test Integer',
            fname='test_int',
            directory=self.tmpdir,
            view=False
        )
        self.assertIsInstance(result, Figure)

    def test_plot_integer_horizontal(self):
        """Verify horizontal integer plot returns a Figure."""
        result = self.engine.plot_integer(
            self.dist,
            fname='test_int_horiz',
            directory=self.tmpdir,
            horizontal=True,
            view=False
        )
        self.assertIsInstance(result, Figure)


# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib not installed')
class TestMatplotlibGaussian(TestCase):

    def setUp(self):
        self.engine = MatplotlibRendering()
        self.tmpdir = tempfile.mkdtemp()

    def test_plot_gaussian_1d(self):
        """Verify 1D Gaussian plot returns a Figure."""
        dist = Gaussian(mean=[0.], cov=[[1.]])
        result = self.engine.plot_gaussian(
            dist,
            title='Test Gaussian 1D',
            fname='test_gauss1d',
            directory=self.tmpdir,
            view=False
        )
        self.assertIsInstance(result, Figure)

    def test_plot_gaussian_2d(self):
        """Verify 2D Gaussian heatmap plot returns a Figure."""
        dist = Gaussian(
            mean=[0., 0.],
            cov=[[1., 0.], [0., 1.]]
        )
        result = self.engine.plot_gaussian(
            dist,
            title='Test Gaussian 2D',
            fname='test_gauss2d',
            directory=self.tmpdir,
            dim=2,
            view=False
        )
        self.assertIsInstance(result, Figure)
