from operator import itemgetter
from typing import Optional, Dict, Iterable, Tuple


class Unsatisfiability(Exception):
    '''Error that is raised on logically unsatisfiable inferences.'''

    def __init__(self, msg: str = None, reasons: Optional[Dict] = None):
        super().__init__(msg)
        self._reasons = reasons

    @property
    def reasons(self) -> Iterable[Tuple]:
        yield from sorted(
            [(likelihood, assignment) for assignment, likelihood in self._reasons.items()],
            key=itemgetter(0),
            reverse=True
        )
