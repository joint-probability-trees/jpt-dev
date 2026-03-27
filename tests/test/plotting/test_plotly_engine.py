import os
import tempfile
import unittest
from unittest import TestCase

import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from jpt.distributions import SymbolicType
from jpt.distributions.univariate import IntegerType
from jpt.distributions.univariate.gaussian import Gaussian
from jpt.plotting.engines.plotly_engine import PlotlyRendering

from test.testutils import gaussian_numeric


# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_PLOTLY, 'plotly not installed')
class TestPlotlyMultinomial(TestCase):

    def setUp(self):
        self.engine = PlotlyRendering()
        DistABC = SymbolicType('TestType', labels=['A', 'B', 'C'])
        self.dist = DistABC().set(params=[0.5, 0.3, 0.2])
        self.tmpdir = tempfile.mkdtemp()

    def test_plot_multinomial_returns_figure(self):
        """Verify plot_multinomial returns a plotly Figure."""
        result = self.engine.plot_multinomial(
            self.dist,
            title='Test Multinomial',
            view=False
        )
        self.assertIsInstance(result, go.Figure)

    def test_plot_multinomial_has_bar_trace(self):
        """Verify the figure contains a Bar trace."""
        result = self.engine.plot_multinomial(self.dist, view=False)
        self.assertEqual(1, len(result.data))
        self.assertIsInstance(result.data[0], go.Bar)

    def test_plot_multinomial_horizontal(self):
        """Verify horizontal orientation is set correctly."""
        result = self.engine.plot_multinomial(
            self.dist,
            horizontal=True,
            view=False
        )
        self.assertEqual('h', result.data[0].orientation)

    def test_plot_multinomial_vertical(self):
        """Verify vertical orientation is set correctly."""
        result = self.engine.plot_multinomial(
            self.dist,
            horizontal=False,
            view=False
        )
        self.assertEqual('v', result.data[0].orientation)

    def test_plot_multinomial_color_hex(self):
        """Verify hex color input is accepted."""
        result = self.engine.plot_multinomial(
            self.dist,
            color='#ff0000',
            view=False
        )
        self.assertIsInstance(result, go.Figure)

    def test_plot_multinomial_file_html(self):
        """Verify HTML file output is created."""
        self.engine.plot_multinomial(
            self.dist,
            fname='test.html',
            directory=self.tmpdir,
            view=False
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.tmpdir, 'test.html'))
        )

    def test_plot_multinomial_title_false(self):
        """Verify title=False suppresses the title."""
        result = self.engine.plot_multinomial(
            self.dist,
            title=False,
            view=False
        )
        self.assertIsNone(result.layout.title.text)

    def test_plot_multinomial_max_values(self):
        """Verify max_values limits the data points."""
        result = self.engine.plot_multinomial(
            self.dist,
            max_values=2,
            view=False
        )
        self.assertEqual(2, len(result.data[0].y))


# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_PLOTLY, 'plotly not installed')
class TestPlotlyNumeric(TestCase):

    def setUp(self):
        self.engine = PlotlyRendering()
        self.dist = gaussian_numeric()

    def test_plot_numeric_returns_figure(self):
        """Verify plot_numeric returns a plotly Figure."""
        result = self.engine.plot_numeric(
            self.dist,
            title='Test Numeric',
            view=False
        )
        self.assertIsInstance(result, go.Figure)

    def test_plot_numeric_has_two_traces(self):
        """Verify the figure has a line trace and a scatter trace."""
        result = self.engine.plot_numeric(self.dist, view=False)
        self.assertEqual(2, len(result.data))
        # First trace: dashed CDF line
        self.assertIsInstance(result.data[0], go.Scatter)
        self.assertEqual('lines', result.data[0].mode)
        # Second trace: scatter markers for breakpoints
        self.assertIsInstance(result.data[1], go.Scatter)
        self.assertEqual('markers', result.data[1].mode)

    def test_plot_numeric_fill(self):
        """Verify fill parameter is passed to the trace."""
        result = self.engine.plot_numeric(
            self.dist,
            fill='tozeroy',
            view=False
        )
        self.assertEqual('tozeroy', result.data[0].fill)

    def test_plot_numeric_color(self):
        """Verify custom color is accepted."""
        result = self.engine.plot_numeric(
            self.dist,
            color='rgba(255,0,0,0.5)',
            view=False
        )
        self.assertIsInstance(result, go.Figure)


# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_PLOTLY, 'plotly not installed')
class TestPlotlyInteger(TestCase):

    def setUp(self):
        self.engine = PlotlyRendering()
        Die = IntegerType('Dice', 1, 6)
        self.dist = Die()
        self.dist.set([1 / 6] * 6)

    def test_plot_integer_returns_figure(self):
        """Verify plot_integer returns a plotly Figure."""
        result = self.engine.plot_integer(
            self.dist,
            title='Test Integer',
            view=False
        )
        self.assertIsInstance(result, go.Figure)

    def test_plot_integer_has_bar_trace(self):
        """Verify the figure contains a Bar trace."""
        result = self.engine.plot_integer(self.dist, view=False)
        self.assertIsInstance(result.data[0], go.Bar)

    def test_plot_integer_horizontal(self):
        """Verify horizontal orientation."""
        result = self.engine.plot_integer(
            self.dist,
            horizontal=True,
            view=False
        )
        self.assertEqual('h', result.data[0].orientation)


# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_PLOTLY, 'plotly not installed')
class TestPlotlyGaussian(TestCase):

    def setUp(self):
        self.engine = PlotlyRendering()

    def test_plot_gaussian_1d(self):
        """Verify 1D Gaussian returns Figure with scatter traces."""
        dist = Gaussian(mean=[0.], cov=[[1.]])
        result = self.engine.plot_gaussian(
            dist,
            title='Test 1D',
            view=False
        )
        self.assertIsInstance(result, go.Figure)
        # 1D Gaussian produces 4 traces: curve, mu line, -sigma line, +sigma line
        self.assertEqual(4, len(result.data))
        for trace in result.data:
            self.assertIsInstance(trace, go.Scatter)

    def test_plot_gaussian_2d_heatmap(self):
        """Verify 2D Gaussian with dim=2 produces a Heatmap trace."""
        dist = Gaussian(
            mean=[0., 0.],
            cov=[[1., 0.], [0., 1.]]
        )
        result = self.engine.plot_gaussian(
            dist,
            title='Test 2D Heatmap',
            dim=2,
            view=False
        )
        self.assertIsInstance(result, go.Figure)
        self.assertEqual(1, len(result.data))
        self.assertIsInstance(result.data[0], go.Heatmap)

    def test_plot_gaussian_3d_surface(self):
        """Verify 2D Gaussian with dim=3 produces a Surface trace."""
        dist = Gaussian(
            mean=[0., 0.],
            cov=[[1., 0.], [0., 1.]]
        )
        result = self.engine.plot_gaussian(
            dist,
            title='Test 3D Surface',
            dim=3,
            view=False
        )
        self.assertIsInstance(result, go.Figure)
        self.assertEqual(1, len(result.data))
        self.assertIsInstance(result.data[0], go.Surface)


# ----------------------------------------------------------------------

@unittest.skipUnless(HAS_PLOTLY, 'plotly not installed')
class TestPlotlyMultivariate(TestCase):

    def test_plot_multivariate_returns_empty_figure(self):
        """Verify multivariate stub returns an empty Figure."""
        engine = PlotlyRendering()

        class MockDist:
            __qualname__ = 'MockDist'

        result = engine.plot_multivariate(
            MockDist(),
            view=False
        )
        self.assertIsInstance(result, go.Figure)
        self.assertEqual(0, len(result.data))
