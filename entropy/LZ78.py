from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Iterable, Union,Tuple
import numpy as np

class BaseNode(ABC):
    def __init__(self, key: [str, int, None], letters=Iterable[Union[str, int]]):
        self.__key = key
        self._counter = 1
        self.__children: Dict[key, 'BaseNode'] = dict()
        self.letters = letters
        self.pointer_node = self

    @abstractmethod
    def add_children(self):
        pass

    def get_key(self):
        return self.__key

    def has_children(self):
        return len(self.__children) > 0

    def __iter__(self):
        for v in self.__children.values():
            yield v

    def __getitem__(self, keys: [str, List[str, int], int]) -> [BaseNode, None]:
        keys = self._parse_seq(keys)
        assert keys[0] in self.__children, f'{keys[0]} not found for parent {str(self)}'
        if self.__children[(keys[0])].has_children() and len(keys) > 1:
            return self.__children[keys[0]][keys[1:]]
        elif len(keys) == 1:
            return self.__children[keys[0]]
        return None

    def __setitem__(self, key, child):
        assert str(key) not in self.__children, f'{key} is already a child for parent {str(self)}'
        self.__children[key] = child

    def __contains__(self, seq: [str, List[str, int]]):
        seq = self._parse_seq(seq)
        if self.get_key() is None:
            return (seq is None or seq in self[seq[0]] and self[seq[0]].has_children())
        if seq[0] == self.get_key():
            return len(seq) == 1 or (
                        seq[1] in self.__children and self[seq[1]].has_children() and seq[1:] in self[seq[1]])
        return False

    @staticmethod
    def _parse_seq(seq: [int, str, List[str, int]]):
        if isinstance(seq, str):
            seq = list(seq)
        # elif isinstance(seq, int):
        #     seq = [seq]
        elif not isinstance(seq, list):
            seq = [seq]
        return seq

    @abstractmethod
    def __str__(self):
        pass

    def __show_tree(self, level=0):
        ret = "\t" * level + repr(self.get_key()) + "\n"
        for child in sorted(self, key=lambda x: x.get_key()):
            ret += child.__show_tree(level + 1)
        return ret

    def __repr__(self):
        return self.__show_tree()

    # def str(self):
    #     return '<tree node representation>'

    @abstractmethod
    def _update_counter(self):
        pass

    def _add_sequence(self, seq: [str, List[str, int]], base_update_node: [None, 'BaseNode'] = None) -> [None,
                                                                                                         'BaseNode']:
        seq = self._parse_seq(seq)
        if base_update_node is None:
            base_update_node = self

        if seq in self:
            return base_update_node

            # self.add_children()
        if len(seq) > 1:
            if not self[seq[0]].has_children():
                self[seq[0]].add_children()
            base_update_node = self[seq[0]]._add_sequence(seq[1:], base_update_node)
        else:
            self[seq[0]].add_children()
        return base_update_node


class Root(BaseNode):

    def __init__(self, letters: Iterable[str, int]):
        super().__init__(None, letters)

        self.add_children()

    def add_children(self):
        assert not self.has_children(), 'cannot adda children if they already have'
        for i in self.letters:
            self[i] = self._Node(i, self)
        self._counter = len(self.letters)

    def __str__(self):
        return ''

    def add_sequence(self, seq):
        if seq in self:
            return
        base_update_node = self._add_sequence(seq)
        if base_update_node is not None:
            base_update_node._update_counter()

    def increment_sequence(self, seq, increment_factor=1):
        item = self[seq]
        item._counter += increment_factor

    def get_next_node(self,cur_node:[BaseNode,None],letter)->Tuple['_Node',float]:
        if cur_node is not None and cur_node.has_children():
            return cur_node[letter],cur_node[letter].get_prob()
        return self[letter] ,self[letter].get_prob()

    def calculate_pob(self,seq):
        seq = self._parse_seq(seq)
        cur_node= self
        p=None
        for i in seq:
            cur_node,p=self.get_next_node(cur_node,i)
        return p
    def _update_counter(self):
        counter = 0
        for i in self:
            i._update_counter()
            counter += i._counter
        self._counter = counter

    class _Node(BaseNode):
        def __init__(self, key: [str, int], parent: [BaseNode]):
            super().__init__(key, parent.letters)
            self.__parent = parent
        def get_parent(self):
            return self.__parent
        def get_prob(self):
            return float(self._counter) / float(self.__parent._counter)

        @classmethod
        def __create_new_object(cls, key: [str, int], parent: [BaseNode]):
            return cls(key, parent)

        def add_children(self):
            assert not self.has_children(), 'cannot adda children if they already have'
            for i in self.letters:
                self[i] = self.__create_new_object(i, self)
            self._counter = len(self.letters)

        def __str__(self):
            return str(self.__parent) + str(self.get_key())

        def __create_nodes_by_layers(self) -> Iterable[BaseNode]:  # todo change name
            stack: List['BaseNode'] = [self]
            nodes_order: List['BaseNode'] = [self]
            while len(stack) > 0:
                node = stack.pop(0)
                nodes_order.append(node)
                stack.extend(node)
            return reversed(nodes_order)

        def _update_counter(self):
            current_counter = self._counter
            stack = self.__create_nodes_by_layers()

            # update children
            for node in stack:
                if not node.has_children():
                    continue
                node._counter = sum([i._counter for i in node])
            counter_diff = self._counter - current_counter
            # update parents
            cur_parent = self.__parent
            while not isinstance(cur_parent, Root):
                cur_parent._counter += counter_diff
                cur_parent = cur_parent.__parent
            cur_parent._counter += counter_diff

    # def find_cond_node=()

    def calculate_prob(self, c: [str, int], cond: [str, List[str, int]]):
        assert isinstance(c, int) or len(c) == 1, 'cannot assign probability for more then one character'
        cur_cond = self._parse_seq(cond)
        cur_node: [None, Root._Node] = None
        for i in cur_cond:
            if cur_node is None or not cur_node.has_children():
                cur_node = self[i]
            else:
                cur_node = cur_node[i]
        if cur_node.has_children():
            cur_node = cur_node[c]
        else:
            cur_node = self[c]
        return cur_node.get_prob() if cur_node is not None else None


def buildLZ78(seq, max_depth=None):
    sequence_set: set = set()
    start_counter = 0
    end_counter = 1
    tree = Root(set(seq))
    while start_counter < len(seq) and end_counter < len(seq) + 1:
        if ''.join([str(i) for i in seq[start_counter:end_counter]]) in sequence_set or (
                max_depth is not None and (end_counter - start_counter) > max_depth):
            if max_depth is not None and end_counter - start_counter > max_depth:
                tree.increment_sequence(seq[start_counter:end_counter])
                start_counter, end_counter = end_counter, end_counter + 1
            else:
                end_counter += 1

        else:
            tree.add_sequence(seq[start_counter:end_counter])
            sequence_set.add(''.join([str(i) for i in seq[start_counter:end_counter]]))
            start_counter, end_counter = end_counter, end_counter + 1
    return tree

# def buildLZ76(seq):
#     start_counter = 0
#     end_counter = 1
#     tree = Root(set(seq))
#     while start_counter < len(seq) and end_counter < len(seq) + 1:
#         if seq[start_counter:end_counter] in seq[:start_counter]:
#             end_counter+=1
#         else:
#             tree.add_sequence(seq[start_counter:end_counter])
#             start_counter,end_counter=end_counter,end_counter+1
#     return tree


def LZ78(seq:[List[int,str],str],over_letters=None):
    tree = buildLZ78(seq)
    cur_node =tree
    entropy = 0.
    joint_dist = 1.
    ent_arr=[]
    counter=0
    prob_sum=1
    for i,s in enumerate(seq):

        cur_node,prob = tree.get_next_node(cur_node,s)
        if cur_node.get_parent()==tree:
            prob_sum=1
        # prob = tree.calculate_pob(seq)
        # if over_letters is not None and i in over_letters:
        # prob_sum *= prob
        prob_sum*=prob

        # joint_dist *= prob
        # if cur_node==tree:
        #     entropy+=joint_dist
        #     joint_dist = 1.
        # cur_node=tree
            # counter+=1
    return -np.log(prob_sum)

#%%
def test():
    import matplotlib.pyplot as plt
    # from entropy.entropy import DSampEn as Dsamp,CTW
    from scipy.stats import poisson

    poisson_ent = lambda x:poisson.entropy(x)
    def create_spike_trains(l,size):
        d = np.zeros((size,))
        e = np.random.poisson(l,size)
        e = np.cumsum(e)
        e=e[e<d.shape[0]]
        d[e]=1
        d = d.astype(int)
        return d.tolist()
    x =  np.linspace(0.001,10000,1000)
    y=[]
    for i in x:
        y.append(poisson_ent(i))

    plt.plot(x,y)
    plt.show()

    lambdas = np.arange(10,200,10)
    lambdas = np.repeat(lambdas,5,axis=0)
    data=[]
    spikes=[]
    numeric_ent=[]
    for i in lambdas:
        st = create_spike_trains(i,1000)

        spikes.append(sum(st))
        d = LZ78(st,{1})
        data.append(d)
        numeric_ent.append(poisson_ent(i))
        # numeric_ent_idx.append(i)
    plt.scatter(lambdas,data)
    # numeric_ent_idx = sorted(numeric_ent_idx)
    plt.scatter(lambdas,numeric_ent,color='red')
    plt.xlabel("$\lambda$")
    plt.ylabel("entropy")
    plt.show()
    plt.scatter(lambdas,spikes)
    plt.xlabel("$\lambda$")
    plt.ylabel("spikes")
    plt.show()
    data= np.array(data)
    numeric_ent = np.array(numeric_ent)
    r = data/numeric_ent
    plt.scatter(lambdas,r)
    plt.show()
# test()
#%% LZ76 calculation
# seq="01011010001101110010"
# tree = buildLZ76(seq)