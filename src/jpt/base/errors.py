from operator import itemgetter
from typing import Optional


class Unsatisfiability(Exception):
    '''Error that is raised on logically unsatisfiable inferences.'''

    def __init__(self, msg: str = None, reasons: Optional['VariableMap'] = None):
        super().__init__(msg)
        self._reasons = reasons

    @property
    def reasons(self):
        yield from sorted(
            [(l, r) for r, l in self._reasons.items()],
            key=itemgetter(0),
            reverse=True
        )
