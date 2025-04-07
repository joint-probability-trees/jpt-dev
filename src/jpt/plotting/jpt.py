import html
import math
import os
import tempfile
import threading
from multiprocessing import Pool
from typing import Iterable, Dict, Tuple

import datetime as dt

from dnutils import ifnone
from graphviz import Digraph
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..trees import JPT
from ..base.constants import green, orange
from ..trees import DecisionNode
from ..variables import Variable


IMG_TEMPLATE = '''
{begin_row}
<TD><IMG SCALE="TRUE" SRC="{img_name}.png"/></TD>
{end_row}
'''

DISTIRBUTIONS_TEMPLATE = '''
<TR>
    <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2">
        <TABLE>
            {code_distributions}
        </TABLE>
    </TD>
</TR>
'''

TEMPLATE_LEAF_HEADER = '''
<TR>
    <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2">
        <B>{leaf_label}</B><BR/>
        {leaf_description}
    </TD>
</TR>
'''

TEMPLATE_LEAF_BODY = '''{code_leaf_header}{code_distributions}
<TR>
    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B># Samples:</B></TD>
    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{leaf_samples} ({leaf_prior:.3f} %)</TD>
</TR>
<TR>
    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>Expectation:</B></TD>
    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{code_expectations}</TD>
</TR>
<TR>
    <TD BORDER="1" ROWSPAN="{path_span}" ALIGN="CENTER" VALIGN="MIDDLE"><B>Path:</B></TD>
    <TD BORDER="1" ROWSPAN="{path_span}" ALIGN="CENTER" VALIGN="MIDDLE">{code_path}</TD>
</TR>
'''

TEMPLATE_LEAF = '''<<TABLE ALIGN="CENTER" VALIGN="MIDDLE" BORDER="0" CELLBORDER="0" CELLSPACING="0">
    {code_leaf_body}
</TABLE>>'''

_locals = threading.local()


def render_leaf(args) -> Tuple[Tuple, Dict]:
    leaf_idx, directory, plotvars, max_symb_values, alphabet, leaffill = args
    jpt = _locals.jpt
    imgs = ''

    leaf = jpt.leaves[leaf_idx]

    # plot and save distributions for later use in tree plot
    rc = math.ceil(
        math.sqrt(
            len(plotvars)
        )
    )
    code_distributions = ''

    for i, pvar in enumerate(plotvars):

        if type(pvar) is str:
            pvar = jpt.varnames[pvar]

        img_name = html.escape(f'./{pvar.name}-{leaf.idx}')

        params = {} if pvar.numeric else {
            'horizontal': True,
            'max_values': max_symb_values,
            'alphabet': alphabet
        }

        leaf.distributions[pvar].plot(
            title=html.escape(pvar.name),
            fname=img_name,
            directory=directory,
            view=False,
            **params
        )

        code_img = IMG_TEMPLATE.format(
            begin_row="<TR>" if i % rc == 0 else "",
            img_name=img_name,
            end_row="</TR>" if i % rc == rc - 1 or i == len(plotvars) - 1 else ""
        )

        code_distributions += code_img

        # close current figure to allow for other plots
        plt.close()

    if plotvars:
        code_distributions_table = DISTIRBUTIONS_TEMPLATE.format(
            code_distributions=code_distributions
        )
    else:
        code_distributions_table = ""

    br_land = '<BR/>\u2227 '

    # content for node labels
    leaf_label = 'Leaf #%s (p = %.4f)' % (leaf.idx, leaf.prior)

    code_leaf_header = TEMPLATE_LEAF_HEADER.format(
        leaf_label=leaf_label,
        leaf_description=html.escape(leaf.str_node)
    )

    expectations = []
    for v, dist in leaf.value.items():
        varstr = html.escape(v.name)
        if jpt.targets is not None and v in jpt.targets:
            varstr = f'<B>{varstr}</B>'
        if v.symbolic:
            valstr = str(dist.mode())
        else:
            valstr = f'{dist.expectation():.2f}'
        expectations.append(f'{varstr}={html.escape(valstr)}')

    code_expectations = '<BR/>'.join(expectations)

    code_leaf_body = TEMPLATE_LEAF_BODY.format(
        code_leaf_header=code_leaf_header,
        code_distributions=code_distributions_table,
        leaf_samples=leaf.samples,
        leaf_prior=leaf.prior * 100,
        code_expectations=code_expectations,
        code_path=br_land.join([html.escape(var.str(val, fmt='set')) for var, val in leaf.path.items()]),
        path_span=len(leaf.path)
    )

    # stitch together
    code_leaf = TEMPLATE_LEAF.format(
        code_leaf_body=code_leaf_body
    )

    return (
        (str(leaf.idx),),
        dict(
            label=code_leaf,
            shape='box',
            style='rounded,filled',
            fillcolor=leaffill or green
        )
    )


class JPTPlotter:
    '''
        :param title: title of the plot
        :param filename: the name of the JPT (will also be used as filename; extension will be added automatically)
        :param directory: the location to save the SVG file to
        :param plotvars: the variables to be plotted in the graph
        :param max_symb_values: limit the maximum number of symbolic values that are plotted to this number
        :param nodefill: the color of the inner nodes in the plot; accepted formats: RGB, RGBA, HSV, HSVA or color name
        :param leaffill: the color of the leaf nodes in the plot; accepted formats: RGB, RGBA, HSV, HSVA or color name
        :param alphabet: whether to plot symbolic variables in alphabetic order, if False, they are sorted by
        probability (descending); default is False
    '''

    def __init__(
            self,
            jpt: JPT,
            title: str = "unnamed",
            filename: str or None = None,
            directory: str = None,
            plotvars: Iterable[Variable] = None,
            max_symb_values: int = 10,
            nodefill: str = None,
            leaffill: str = None,
            alphabet: bool = False,
            verbose: bool = False
    ) -> None:
        self.jpt = jpt
        self.title = title
        self.filename = filename
        self.directory = directory
        self.plotvars = ifnone(plotvars, [], list)
        self.max_symb_values = max_symb_values
        self.nodefill = nodefill
        self.leaffill = leaffill
        self.alphabet = alphabet
        self.verbose = verbose

    def render_decision_node(self, node: DecisionNode) -> Tuple[Tuple, Dict]:
        return (
            (str(node.idx),),
            dict(
                label=node.str_node,
                shape='ellipse',
                style='rounded,filled',
                fillcolor=self.nodefill or orange
            )
        )

    def plot(
            self,
            view: bool = False
    ) -> str:
        """
        Generates an SVG representation of the generated regression tree.
        :param view: whether the generated SVG file will be opened automatically

        :return:   (str) the path under which the renderd image has been saved.
        """
        if self.directory is None:
            directory = tempfile.mkdtemp(
                prefix=f'jpt_{self.title}-{dt.datetime.now().strftime("%Y-%m-%d_%H-%M")}',
                dir=tempfile.gettempdir()
            )
        else:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            directory = self.directory

        dot = Digraph(
            format='svg',
            name=self.title,
            directory=directory,
            filename=f'{self.filename or self.title}'
        )

        _locals.jpt = self.jpt

        # create nodes
        pool = Pool()
        if self.verbose:
            progress = tqdm(total=len(self.jpt.leaves))

        for args, kwargs in pool.imap_unordered(
                render_leaf,
                iterable=[(
                    i,
                    directory,
                    [v.name if isinstance(v, Variable) else v for v in self.plotvars],
                    self.max_symb_values,
                    self.alphabet,
                    self.leaffill
                ) for i in self.jpt.leaves.keys()]
        ):
            dot.node(*args, **kwargs)
            if self.verbose:
                progress.update(1)
                progress.set_description(f'Rendering Leaf # {args[0]}')

        if self.verbose:
            progress.close()
        pool.close()
        pool.join()

        # for leaf in self.jpt.leaves.values():
        #     args, kwargs = self.render_leaf(leaf, directory)

        for node in self.jpt.innernodes.values():
            args, kwargs = self.render_decision_node(node)
            dot.node(*args, **kwargs)

        # create edges
        for idx, n in self.jpt.innernodes.items():
            for i, c in enumerate(n.children):
                if c is None:
                    continue
                dot.edge(str(n.idx), str(c.idx), label=html.escape(n.str_edge(i)))

        # show graph
        filepath = '%s.svg' % os.path.join(
            directory,
            ifnone(self.filename, self.title)
        )

        JPT.logger.info(f'Saving rendered image to {filepath}.')

        # improve aspect ratio of graph having many leaves or disconnected nodes
        dot = dot.unflatten(stagger=3)
        dot.render(view=view, cleanup=False)

        return filepath
