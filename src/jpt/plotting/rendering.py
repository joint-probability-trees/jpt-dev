import os
from typing import Any, Union

import numpy as np
import plotly.graph_objects as go
from dnutils import ifnone, project, ifnot

from ..base.functions import ConstantFunction
from .helpers import color_to_rgb

# engines
PLOTLY = 'plotly'
MATPLOTLIB = 'matplotlib'


class DistributionRendering:
    '''
    Abstract supertype of distribution rendering engines. Instantiating this class with the `engine` parameter will
    overriding this class's __new__ method and automatically create an instance of the respective subclass.
    '''
    def __new__(
            cls,
            engine: str
    ):
        subclass_map = {subclass.engine: subclass for subclass in cls.__subclasses__()}
        subclass = subclass_map.get(engine, cls)
        instance = super(DistributionRendering, subclass).__new__(subclass)
        return instance

    def plot_multinomial(
            self,
            dist: Any,
            title: str = None,
            fname: str = None,
            directory: str = '/tmp',
            view: bool = False,
            horizontal: bool = False,
            max_values: int = None,
            alphabet: bool = False,
            color: str = 'rgb(15,21,110)',
            xvar: str = None,
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
            color: str = 'rgb(15,21,110)',
            fill: str = None,
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
            horizontal: bool = False,
            max_values: int = None,
            alphabet: bool = False,
            color: str = 'rgb(15,21,110)',
            **kwargs
    ):
        raise NotImplementedError


class PlotlyRendering(DistributionRendering):
    engine = PLOTLY

    def plot_multinomial(
            self,
            dist: Any,
            title: str = None,
            fname: str = None,
            directory: str = '/tmp',
            view: bool = False,
            horizontal: bool = False,
            max_values: int = None,
            alphabet: bool = False,
            color: str = 'rgb(15,21,110)',
            xvar: str = None,
            **kwargs
    ):
        '''Generates a ``horizontal`` (if set) otherwise `vertical` bar plot representing the variable's distribution.

        :param title:       the title of the plot. defaults to the type of this distribution, can be left
                            empty by passing `False`.
        :param fname:       the name of the file to be stored. Available file formats: png, svg, jpeg, webp, html
        :param directory:   the directory to store the generated plot files
        :param view:        whether to display generated plots, default False (only stores files)
        :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
        :param max_values:  maximum number of values to plot
        :param alphabet:    whether the bars are sorted in alphabetical order of the variable names. If False, the bars
                            are sorted by probability (descending); default is False
        :param color:       the color of the plot traces; accepts str of form:
                            * rgb(r,g,b) with r,g,b being int or float
                            * rgba(r,g,b,a) with r,g,b being int or float, a being float
                            * #f0c (as short form of #ff00cc) or #f0cf (as short form of #ff00ccff)
                            * #ff00cc
                            * #ff00ccff
        :return:            plotly.graph_objs.Figure
        '''

        # generate data
        max_values = min(ifnone(max_values, len(dist.labels)), len(dist.labels))

        # prepare prob-label pairs containing only the first `max_values` highest probability tuples
        pairs = sorted(
            [
                (dist._params[idx], lbl) for idx, lbl in enumerate(dist.labels.values())
            ],
            key=lambda x: x[0],
            reverse=True
        )[:max_values]

        if alphabet:
            # re-sort remaining values alphabetically
            pairs = sorted(pairs, key=lambda x: x[1])

        probs = project(pairs, 0)
        labels = project(pairs, 1)

        # extract rgb colors from given hex, rgb or rgba string
        # rgb, rgba = ("rgb(100,100,100)", "rgba(100,100,100,.4)")
        rgb, rgba = color_to_rgb(color)

        mainfig = go.Figure(
            [
                go.Bar(
                    x=probs if horizontal else labels,  # labels if horizontal else probs,
                    y=labels if horizontal else probs,  # probs if horizontal else x,
                    name=title or "Multinomial Distribution",
                    text=probs,
                    orientation='h' if horizontal else 'v',
                    marker=dict(
                        color=rgba,
                        line=dict(color=rgb, width=3)
                    )
                )
            ]
        )

        # determine variable name from class (qualname only works for Multinomial, empty on Bool,
        # therefore xvar would be required)
        xname = xvar if xvar is not None else "_".join(dist.__class__.__qualname__.split("_")[:-2]).lower()
        mainfig.update_layout(
            xaxis=dict(
                title=f'P({xname})' if horizontal else xname,
                range=[0, 1] if horizontal else None,
            ),
            yaxis=dict(
                title=xname if horizontal else f'P({xname})',
                range=None if horizontal else [0, 1]
            ),
            title=None if title is False else f'{title or f"Distribution of {dist._cl}"}',
            height=1000,
            width=1000
        )

        if fname is not None:

            if not os.path.exists(directory):
                os.makedirs(directory)

            fpath = os.path.join(directory, fname or dist.__class__.__name__)

            if fname.endswith('.html'):
                mainfig.write_html(
                    fpath,
                    config=dict(
                        displaylogo=False,
                        toImageButtonOptions=dict(
                            format='svg',  # one of png, svg, jpeg, webp
                            filename=fname or dist.__class__.__name__,
                            scale=1
                        )
                    ),
                    include_plotlyjs="cdn"
                )
                mainfig.write_json(fpath.replace(".html", ".json"))
            else:
                mainfig.write_image(
                    fpath,
                    scale=1
                )

        if view:
            mainfig.show(
                config=dict(
                    displaylogo=False,
                    toImageButtonOptions=dict(
                        format='svg',  # one of png, svg, jpeg, webp
                        filename=fname or dist.__class__.__name__,
                        scale=1
                    )
                )
            )

        return mainfig

    def plot_numeric(
            self,
            dist: Any,
            title: Union[str, bool] = None,
            fname: str = None,
            xlabel: str = 'value',
            directory: str = '/tmp',
            view: bool = False,
            color: str = 'rgb(15,21,110)',
            fill: str = None,
            **kwargs
    ):
        '''
        Generates a plot of the piecewise linear function representing
        the variable's cumulative distribution function

        :param title:       the title of the plot. defaults to the type of this distribution, can be left
                            empty by passing `False`.
        :param fname:       the name of the file to be stored. Available file formats: png, svg, jpeg, webp, html
        :param xlabel:      the label of the x-axis
        :param directory:   the directory to store the generated plot files
        :param view:        whether to display generated plots, default False (only stores files)
        :param color:       the color of the plot traces; accepts str of form:
                            * rgb(r,g,b) with r,g,b being int or float
                            * rgba(r,g,b,a) with r,g,b being int or float, a being float
                            * #f0c (as short form of #ff00cc) or #f0cf (as short form of #ff00ccff)
                            * #ff00cc
                            * #ff00ccff
        :return:            plotly.graph_objs.Figure
        '''

        # generate data
        if len(dist.cdf.intervals) == 2:
            std = abs(dist.cdf.intervals[0].upper) * .1
        else:
            std = ifnot(
                np.std([i.upper - i.lower for i in dist.cdf.intervals[1:-1]]),
                dist.cdf.intervals[1].upper - dist.cdf.intervals[1].lower
            ) * 2

        # add horizontal line before first interval of distribution
        X = np.array([dist.cdf.intervals[0].upper - std])

        for i, f in zip(dist.cdf.intervals[:-1], dist.cdf.functions[:-1]):
            if isinstance(f, ConstantFunction):
                X = np.append(X, [np.nextafter(i.upper, i.upper - 1), i.upper])
            else:
                X = np.append(X, i.upper)

        # add horizontal line after last interval of distribution
        X = np.append(X, dist.cdf.intervals[-1].lower + std)
        X_ = np.array([dist.labels[x] for x in X])
        Y = np.array(dist.cdf.multi_eval(X))

        # hacky workaround to clip very small numbers close to zero
        X_[abs(X_) < 5.e-306] = 0

        bounds = np.array([i.upper for i in dist.cdf.intervals[:-1]])
        bounds_ = np.array([dist.labels[b] for b in bounds])

        mainfig = go.Figure()

        # extract rgb colors from given hex, rgb or rgba string
        rgb, rgba = color_to_rgb(color)

        # plot dashed CDF
        mainfig.add_trace(
            go.Scatter(
                x=X_,
                y=Y,
                mode='lines',
                name='Piecewise linear CDF from bounds',
                line=dict(
                    color=rgb,
                    width=4,
                    dash='dash'
                ),
                fill=fill
            )
        )

        # scatter function limits
        mainfig.add_trace(
            go.Scatter(
                x=bounds_,
                y=np.asarray(dist.cdf.multi_eval(bounds)),
                marker=dict(
                    symbol='circle',
                    color=rgba,
                    size=15,
                    line=dict(
                        color=rgb,
                        width=2
                    ),
                ),
                mode='markers',
                name="Piecewise Function Limits",
            )
        )

        mainfig.update_layout(
            xaxis=dict(
                title=xlabel,
                side='bottom'
            ),
            yaxis=dict(
                title='%'
            ),
            title=None if title is False else f'{title or f"Distribution of {dist._cl}"}',
            height=1000,
            width=1200
        )

        if fname is not None:

            if not os.path.exists(directory):
                os.makedirs(directory)

            fpath = os.path.join(directory, fname or dist.__class__.__name__)

            if fname.endswith('html'):
                mainfig.write_html(
                    fpath,
                    config=dict(
                        displaylogo=False,
                        toImageButtonOptions=dict(
                            format='svg',  # one of png, svg, jpeg, webp
                            filename=fname or dist.__class__.__name__,
                            scale=1
                        )
                    ),
                    include_plotlyjs="cdn"
                )
                mainfig.write_json(fpath.replace(".html", ".json"))
            else:
                mainfig.write_image(
                    fpath,
                    scale=1
                )

        if view:
            mainfig.show(
                config=dict(
                    displaylogo=False,
                    toImageButtonOptions=dict(
                        format='svg',  # one of png, svg, jpeg, webp
                        filename=fname or dist.__class__.__name__,
                        scale=1
                    )
                )
            )

        return mainfig

    def plot_integer(
            self,
            dist: Any,
            title: str = None,
            fname: str = None,
            directory: str = '/tmp',
            view: bool = False,
            horizontal: bool = False,
            max_values: int = None,
            alphabet: bool = False,
            color: str = 'rgb(15,21,110)',
            **kwargs
    ):
        '''Generates a ``horizontal`` (if set) otherwise `vertical` bar plot representing the variable's distribution.

                :param title:       the name of the variable this distribution represents
                :param fname:       the name of the file to be stored. Available file formats: png, svg, jpeg, webp, html
                :param directory:   the directory to store the generated plot files
                :param view:        whether to display generated plots, default False (only stores files)
                :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
                :param max_values:  maximum number of values to plot
                :param alphabet:    whether the bars are sorted in alphabetical order of the variable names. If False, the bars
                                    are sorted by probability (descending); default is False
                :param color:       the color of the plot traces; accepts str of form:
                                    * rgb(r,g,b) with r,g,b being int or float
                                    * rgba(r,g,b,a) with r,g,b being int or float, a being float
                                    * #f0c (as short form of #ff00cc) or #f0cf (as short form of #ff00ccff)
                                    * #ff00cc
                                    * #ff00ccff
                :return:            plotly.graph_objs.Figure
                '''

        # generate data
        max_values = min(ifnone(max_values, len(dist.labels)), len(dist.labels))

        # prepare prob-label pairs containing only the first `max_values` highest probability tuples
        pairs = sorted(
            [
                (dist._params[idx], lbl) for idx, lbl in enumerate(dist.labels.values())
            ],
            key=lambda x: x[0],
            reverse=True
        )[:max_values]

        if alphabet:
            # re-sort remaining values alphabetically
            pairs = sorted(pairs, key=lambda x: x[1])

        probs = project(pairs, 0)
        labels = project(pairs, 1)

        # extract rgb colors from given hex, rgb or rgba string
        rgb, rgba = color_to_rgb(color)

        mainfig = go.Figure(
            [
                go.Bar(
                    x=probs if horizontal else labels,  # labels if horizontal else probs,
                    y=labels if horizontal else probs,  # probs if horizontal else x,
                    name="Integer Distribution",
                    text=probs,
                    orientation='h' if horizontal else 'v',
                    marker=dict(
                        color=rgba,
                        line=dict(color=rgb, width=3)
                    )
                )
            ]
        )
        mainfig.update_layout(
            height=1000,
            width=1000,
            xaxis=dict(
                title='$P(\\text{label})$' if horizontal else '$\\text{label}$',
                range=[0, 1] if horizontal else None
            ),
            yaxis=dict(
                title='$\\text{label}$' if horizontal else '$P(\\text{label})$',
                range=None if horizontal else [0, 1]
            ),
            title=f'{title or f"Distribution of {dist._cl}"}'
        )

        if fname is not None:

            if not os.path.exists(directory):
                os.makedirs(directory)

            fpath = os.path.join(directory, fname or dist.__class__.__name__)

            if fname.endswith('html'):
                mainfig.write_html(
                    fpath,
                    include_plotlyjs="cdn"
                )
            else:
                mainfig.write_image(
                    fpath,
                    scale=1
                )

        if view:
            mainfig.show(
                config=dict(
                    displaylogo=False,
                    toImageButtonOptions=dict(
                        format='svg',  # one of png, svg, jpeg, webp
                        filename=fname or dist.__class__.__name__,
                        scale=1
                    )
                )
            )

        return mainfig


class MatplotlibRendering(DistributionRendering):
    engine = MATPLOTLIB



if __name__ == '__main__':
    c1 = DistributionRendering(PLOTLY)
    c2 = DistributionRendering(MATPLOTLIB)
    c3 = DistributionRendering(None)
    print(type(c1))
    print(type(c2))
    print(type(c3))
    # c1.plot_multinomial()
    # c2.plot_multinomial()
