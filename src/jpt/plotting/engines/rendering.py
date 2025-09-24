from typing import Any, Union


# engines
PLOTLY = 'plotly'
MATPLOTLIB = 'matplotlib'


class DistributionRendering:
    '''
    Abstract supertype of distribution rendering engines. Instantiating this class with the `engine` parameter will
    overriding this class's __new__ method and automatically create an instance of the respective subclass.
    '''

    @staticmethod
    def instantiate_engine(engine):

        if engine == PLOTLY:
            from .plotly_engine import PlotlyRendering
            return PlotlyRendering()
        elif engine == MATPLOTLIB:
            from .matplotlib_engine import MatplotlibRendering
            return MatplotlibRendering()
        elif isinstance(engine, DistributionRendering):
            return engine
        else:
            raise TypeError("Rendering engine must either be of type `DistributionRendering` or one of ['matplotlib', 'plotly'].")

    def plot_multinomial(
            self,
            dist: Any,
            title: str = None,
            fname: str = None,
            directory: str = '/tmp',
            view: bool = False,
            **kwargs
    ):
        raise NotImplementedError

    def plot_numeric(
            self,
            dist: Any,
            title: Union[str, bool] = None,
            fname: str = None,
            xlabel: str = 'value',
            directory: str = '/tmp',
            view: bool = False,
            **kwargs
    ):
        raise NotImplementedError

    def plot_integer(
            self,
            dist: Any,
            title: str = None,
            fname: str = None,
            directory: str = '/tmp',
            view: bool = False,
            **kwargs
    ):
        raise NotImplementedError
