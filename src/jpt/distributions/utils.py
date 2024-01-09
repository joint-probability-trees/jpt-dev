from collections import OrderedDict


class HashableOrderedDict(OrderedDict):
    '''
    Ordered dict that can be hashed.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, arg):
        return self[arg]

    def __hash__(self):
        return hash((
            HashableOrderedDict,
            tuple(self.items())
        ))
