from jpt.base.pyximporter import pyx_import

pyx_import(
    '.impurity'
)

from .impurity import Impurity
