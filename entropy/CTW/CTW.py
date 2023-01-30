from ..entropy_factory import EntropyHandler
from .__finite_context_tree_weighted import CTWManagerFinite
from .__infinite_context_tree import CTWManagerInfinite


class CTW(EntropyHandler):
    def __init__(self, D: [int, None]=None):
        self.model = CTWManagerInfinite() if D is None else CTWManagerFinite(D, True)

    def insert_pattern(self, p):
        return self.model.update_by_sequence(p)

    def get_entropy(self,l,last_key:int=0):
        return self.model.get_entropy(l,last_key)

    def print_v_tree(self):
        self.model.print_v_tree()