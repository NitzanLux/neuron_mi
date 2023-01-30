from . import __tree as te
from . import efficient_inf_ctw as eic
from typing import List
from tqdm import tqdm

# class CTWManagerInfinite(te.CTWManager):
#
#     def __init__(self):
#         super().__init__(te.Root())
#
#     def insert_pattern(self, p: List):
#         """
#         intsert single context pattern into the tree
#         :param p: pattern
#         """
#         cur_node = self.tree
#         for i in p[-2::-1] + [None]:  # reading the pattern backward in time.
#             # update counting
#             cur_node.data.update_key(p[-1])
#
#             # Travel to the child node
#             if not cur_node.has_child(i):
#                 cur_node.add_children(i)
#             cur_node = cur_node[i]
#
#         # for the last None node  ---> update counting
#         cur_node.data.update_key(p[-1])
#
#     def update_by_sequence(self, p: List):
#         """
#         insert all possible contexts of the sequence into the tree.
#         :param p: pattern
#         """
#         for i in tqdm(range(len(p))):
#             # cur_time = time.time()
#             self.insert_pattern(p[:i+1])
#         self.tree.create_representative()
#         self.tree.data.update_probs()
class CTWManagerInfinite(te.CTWManager):

    def __init__(self):
        super().__init__(eic.Node.generate_root())

    def insert_pattern(self, p: List):
        """
        intsert single context pattern into the tree
        :param p: pattern
        """
        self.tree.add_child(p)
    def update_by_sequence(self, p: List):
        """
        insert all possible contexts of the sequence into the tree.
        :param p: pattern
        """
        for i in tqdm(range(len(p)-1,-1,-1)):
            self.insert_pattern(p[:i+1])

        self.tree.update_w()
        print('finished up updating',flush=True)

    def get_entropy(self, l, last_key=0):
        return -self.tree.get_log_prob_w(last_key) / l