

class Unsatisfiability(Exception):
    '''Error that is raised on logically unsatisfiable inferences.'''

    def __init__(self, msg: str = None, reasons: 'VariableMap' = None):
        super().__init__(msg)
        self.reasons = reasons
