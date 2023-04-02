from ..entropy_factory import EntropyHandler
from .__finite_context_tree_weighted import CTWManagerFinite
from .__infinite_context_tree import CTWManagerInfinite
import numpy as np
from .efficient_inf_ctw import UnboundProbabilityException
class CTW(EntropyHandler):
    def __init__(self, D: [int, None]=None,__model=None):
        self.model = (CTWManagerInfinite(__model) if D is None else CTWManagerFinite(D, True)) if __model is None else __model
        self.is_constant=False
    def insert_pattern(self, p):
        print('inserting patterns',flush=True)
        if isinstance(p,np.ndarray):
            p = p.astype(int).tolist()
        if sum(p)==0:
            self.is_constant=True
            return
        return self.model.update_by_sequence(p)

    def get_entropy(self,l,last_key:int=0):
        if self.is_constant:
            return 0.
        return self.model.get_entropy(l,last_key)

    def print_v_tree(self):
        self.model.print_v_tree()

    @property
    def tree(self):
        return self.model.tree

    @staticmethod
    def from_dict(data_dict):
        current = CTW(__model= CTWManagerInfinite.from_dict(data_dict))
        return current
    def to_dict(self):
        return self.model.to_dict()
