import itertools
from typing import Tuple

import numpy as np


default_config = dict(
    displaylogo=False,
    toImageButtonOptions=dict(
        format='svg',  # one of png, svg, jpeg, webp
        filename='jpt_plot.svg',
        scale=1  # Multiply title/legend/axis/canvas sizes by this factor
    )
)


def hex_to_rgb(
        col
) -> Tuple[int, ...]:
    '''
    Parse hexadecimal string to extract color/alpha information

    :param col:     color string of one of the following forms
                    * #f0c (as short form of #ff00cc)
                    * #f0cf (as short form of #ff00ccff)
                    * #ff00cc
                    * #ff00ccff
    :return:        a tuple reprsenting either RGB or RGBA values
    '''
    h = col.strip("#")

    # e.g. "#f0c" -> "#ff00cc" -> (255, 0, 204)
    # e.g. "#f0cf" -> "#ff00ccff" -> (255, 0, 204, 255)
    if len(h) <= 4:
        return hex_to_rgb(f'{"".join([v * 2 for v in h], )}')

    # e.g. "#2D6E0F" -> (45, 110, 15)
    if len(h) <= 6:
        return tuple(int(h[i:i + 2], 16) for i, _ in enumerate(h) if i%2 == 0)

    # e.g. "#2D6E0F33" -> (45, 110, 15, 51)
    return tuple(int(h[i:i + 2], 16) for i, _ in enumerate(h[:6]) if i%2 == 0) + (round(int(h[6:8], 16)/255, 2),)


def color_to_rgb(
        color,
        opacity=.6
) -> Tuple[str, str]:
    '''
    Extracts the color and alpha information of a given `color` string and reassembles it to an rgb and an rgba
    color.

    :param color:       the color to examine; accepts str of form:
                        * rgb(r,g,b) with r,g,b being int or float
                        * rgba(r,g,b,a) with r,g,b being int or float, a being float
                        * #f0c (as short form of #ff00cc) or #f0cf (as short form of #ff00ccff)
                        * #ff00cc
                        * #ff00ccff
    :param opacity:     the default opacity to assume if the given string contains only color information
    :return:            plotly.graph_objs.Figure
    '''
    if color.startswith('#'):
        color = hex_to_rgb(color)
        if len(color) == 4:
            opacity = color[-1]
            color = color[:-1]
    elif color.startswith('rgba'):
        color = tuple(map(float, color[5:-1].split(',')))
        opacity = color[-1]
        color = color[:-1]
    elif color.startswith('rgb'):
        color = tuple(map(float, color[4:-1].split(',')))
    return f'rgb{*color,}', f'rgba{*color + (opacity,),}'


def pdf_grid_3d(
        jpt,
        variable1,
        variable2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Create a mesh grid consistent of ``x``, ``y``, ``z`` coordinate values
    representing all distinct interval boundaries forming the joint probability
    density function a projection of ``variable1`` and ``variable2``
    represented the JPT passed the first argument.

    :param jpt:
    :param variable1:
    :param variable2:
    :return:
    '''

    var1_boundaries = set()
    var2_boundaries = set()

    for leaf in jpt.leaves.values():

        var1_boundaries.update(
            itertools.chain(
                *[(i.min, i.max) for i in leaf.distributions[variable1].pdf.intervals]
            )
        )
        var2_boundaries.update(
            itertools.chain(
                *[(i.min, i.max) for i in leaf.distributions[variable2].pdf.intervals]
            )
        )

    v1_values = list(sorted(filter(np.isfinite, var1_boundaries)))
    v2_values = list(sorted(filter(np.isfinite, var2_boundaries)))

    X, Y = np.meshgrid(v1_values, v2_values)

    Z = np.zeros(X.shape)
    for i, (row_x, row_y) in enumerate(zip(X, Y)):
        for j, (x, y) in enumerate(zip(row_x, row_y)):
            Z[i, j] = jpt.pdf(
                jpt.bind(
                    {
                        variable1: x,
                        variable2: y
                    }
                )
            )
    return X, Y, Z
