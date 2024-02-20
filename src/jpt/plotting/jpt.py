import html
import math
import os
import tempfile
from multiprocessing import Pool
from typing import Iterable, Dict, Tuple

import datetime as dt

from dnutils import ifnone
from graphviz import Digraph
from matplotlib import pyplot as plt
from tqdm import tqdm

from jpt import JPT
from jpt.base.constants import green, orange
from jpt.trees import Leaf, DecisionNode
from jpt.variables import Variable


LEAF_TEMPLATE = '''
'''


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

    def render_leaf(self, args) -> Tuple[Tuple, Dict]:
        leaf_idx, directory = args
        imgs = ''

        leaf = self.jpt.leaves[leaf_idx]

        # plot and save distributions for later use in tree plot
        rc = math.ceil(
            math.sqrt(
                len(self.plotvars)
            )
        )
        img = ''
        for i, pvar in enumerate(self.plotvars):
            if type(pvar) is str:
                pvar = self.jpt.varnames[pvar]
            img_name = html.escape(f'{pvar.name}-{leaf.idx}.png')

            params = {} if pvar.numeric else {
                'horizontal': True,
                'max_values': self.max_symb_values,
                'alphabet': self.alphabet
            }

            leaf.distributions[pvar].plot(
                title=html.escape(pvar.name),
                fname=img_name,
                directory=directory,
                view=False,
                **params
            )
            img += f'''
                {"<TR>" if i % rc == 0 else ""}
                    <TD><IMG SCALE="TRUE" SRC="{img_name}"/></TD>
                {"</TR>" if i % rc == rc - 1 or i == len(self.plotvars) - 1 else ""}
            '''

            # close current figure to allow for other plots
            plt.close()

        if self.plotvars:
            imgs = f'''
                <TR>
                    <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2">
                        <TABLE>
                            {img}
                        </TABLE>
                    </TD>
                </TR>
            '''

        land = '<BR/>\u2227 '
        element = ' \u2208 '

        # content for node labels
        leaf_label = 'Leaf #%s (p = %.4f)' % (leaf.idx, leaf.prior)
        nodelabel = f'''
            <TR>
                <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2"><B>{leaf_label}</B><BR/>{html.escape(leaf.str_node)}</TD>
            </TR>
        '''

        nodelabel = f'''{nodelabel}{imgs}
            <TR>
                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>#samples:</B></TD>
                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{leaf.samples} ({leaf.prior * 100:.3f}%)</TD>
            </TR>
            <TR>
                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>Expectation:</B></TD>
                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{',<BR/>'.join([f'{"<B>" + html.escape(v.name) + "</B>" if self.jpt.targets is not None and v in self.jpt.targets else html.escape(v.name)}=' + (f'{html.escape(str(dist.expectation()))!s}' if v.symbolic else f'{dist.expectation():.2f}') for v, dist in leaf.value.items()])}</TD>
            </TR>
            <TR>
                <TD BORDER="1" ROWSPAN="{len(leaf.path)}" ALIGN="CENTER" VALIGN="MIDDLE"><B>path:</B></TD>
                <TD BORDER="1" ROWSPAN="{len(leaf.path)}" ALIGN="CENTER" VALIGN="MIDDLE">{f"{land}".join([html.escape(var.str(val, fmt='set')) for var, val in leaf.path.items()])}</TD>
            </TR>
        '''

        # stitch together
        lbl = f'''<<TABLE ALIGN="CENTER" VALIGN="MIDDLE" BORDER="0" CELLBORDER="0" CELLSPACING="0">
            {nodelabel}
        </TABLE>>'''

        return (
            (str(leaf.idx),),
            dict(
                label=lbl,
                shape='box',
                style='rounded,filled',
                fillcolor=self.leaffill or green
            )
        )

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

        # create nodes
        # sep = ",<BR/>"
        pool = Pool()
        if self.verbose:
            progress = tqdm(total=len(self.jpt.leaves))
        for args, kwargs in pool.imap_unordered(
                self.render_leaf,
                iterable=[(i, directory) for i in self.jpt.leaves.keys()]
        ):
            dot.node(*args, **kwargs)
            if self.verbose:
                progress.update(1)

        if self.verbose:
            progress.close()

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
