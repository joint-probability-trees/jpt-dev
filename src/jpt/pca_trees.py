from typing import List

import jpt.trees
from jpt.variables import VariableMap, Variable
import numpy as np


class PCANode(jpt.trees.Node):
    """
    Superclass for Nodes in a PCAJPT
    """

    def __init__(self, idx, parent):
        """
        :param idx:             the identifier of a node
        :type idx:              int
        :param parent:          the parent node
        :type parent:           jpt.learning.trees.Node
        """
        super(PCANode, self).__init__(idx, parent)


# ----------------------------------------------------------------------------------------------------------------------

class PCADecisionNode(PCANode):
    """
    Decision Node in a PCAJPT where linear combinations of all variables are considered instead of just one variable
    """

    def __init__(self, idx: int, parent: PCANode, variables: List[Variable], weights: np.array = None):
        super(PCADecisionNode, self).__init__(idx, parent)


# ----------------------------------------------------------------------------------------------------------------------

class PCALeaf(PCANode):
    pass


# ----------------------------------------------------------------------------------------------------------------------

class PCAJPT(jpt.trees.JPT):
    """
    This class represents an extension to JPTs where the PCA is applied before each split in training
    and Leafs therefore obtain an additional rotation matrix.
    """
    def __init__(self):
        pass
