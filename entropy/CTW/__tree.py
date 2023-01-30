from __future__ import annotations
from fractions import Fraction
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Iterable, Union, Tuple
import numpy as np
from binarytree import Node
from .__print_tree import Node as s_node
from .__print_tree import drawTree2
from typing import Dict, List, Iterable, Union, Tuple
from mpmath import mp, mpf
import mpmath
from itertools import product
from .__fractional_precision import FractionPrecition as fp
import math
from copy import deepcopy

USE_FRACTIONS = False

MAX_DENOMINATOR = 10000000000
mp.dps = 100


def log(a):
    return mpmath.log(a)


def logaddexp(a, b):
    return log(mpmath.exp(mpf(a)) + mpmath.exp(mpf(b)))


class BaseNode(ABC):
    def __init__(self, key: [str, int, None], use_fractions=USE_FRACTIONS):
        self.__key = key
        self.__children: Dict[key, 'BaseNode'] = dict()
        self.data: [WeightedData,None] = WeightedData(self, use_fractions=use_fractions)
        # self.data: [WeightedData, None] = data

    @abstractmethod
    def add_children(self, keys: [Iterable[int, None], None, int]):
        pass

    def get_child_keys(self):
        return set(self.__children.keys())

    def get_key(self):
        return self.__key

    def has_children(self):
        return len(self.__children) > 0

    def has_child(self, node: ['BaseNode', None, int]):
        if isinstance(node, BaseNode):
            node = node.get_key()
        if isinstance(node, int):
            node = node
        return node in self.__children  # and self.__children[node.get_key()]==node

    def __len__(self):
        return len(self.__children)

    def __iter__(self):
        for v in self.__children.values():
            yield v

    def __getitem__(self, keys: [List[None, int], int]) -> [BaseNode, None]:
        keys = self._parse_seq(keys)
        assert keys[0] in self.__children, f'{keys[0]} not found for parent {str(self)}'
        if self.__children[(keys[0])].has_children() and len(keys) > 1:
            return self.__children[keys[0]][keys[1:]]
        elif len(keys) == 1:
            return self.__children[keys[0]]
        return None

    def __setitem__(self, key, child):
        assert key not in self.__children, f'{key} is already a child for parent {str(self)}'
        self.__children[key] = child

    def __contains__(self, seq: [None, int, List[None, int]]):
        seq = self._parse_seq(seq)
        if self.get_key() is None:
            return (seq is None or (
                    self.has_child(seq[0]) and (seq in self[seq[0]] and self[seq[0]].has_child(seq[1]))))
        if self.has_child(seq[0]):
            return (self[seq[0]].has_children() and seq[1:] in self[seq[0]]) and (
                    len(seq) > 1 and seq[1] in self[seq[0]])
        return False

    @staticmethod
    def _parse_seq(seq: List[None, int]):

        if isinstance(seq, np.ndarray):
            seq = seq.tolist()
        elif not isinstance(seq, list):
            seq = [seq]
        return seq

    @abstractmethod
    def __str__(self):
        pass

    def __build_v_tree_node(self):
        text = repr(self.get_key()) + f"[{repr(self.data)}]"
        children = []
        if 1 in self.__children:
            children.append(self[1].__build_v_tree_node())
        if None in self.__children:
            children.append(self[None].__build_v_tree_node())
        if 0 in self.__children:
            children.append(self[0].__build_v_tree_node())
        out = s_node(text)(children)
        return out

    def print_v_tree(self):
        out = drawTree2(False)(False)(self.__build_v_tree_node()).encode('utf-8-sig')
        print(out.decode('utf-8-sig'))

    def _add_sequence(self, seq: [str, List[str, int]], base_update_node: [None, 'BaseNode'] = None) -> [None,
                                                                                                         'BaseNode']:
        seq = self._parse_seq(seq)
        if base_update_node is None:
            base_update_node = self

        if seq in self:
            return base_update_node

        if not self.has_child(seq[0]):
            self.add_children(seq[0])
        if len(seq) > 1:
            base_update_node = self[seq[0]]._add_sequence(seq[1:], base_update_node)

        return base_update_node

    @abstractmethod
    def get_current_sequence(self):
        pass

    def __isequal(self, other):
        if type(self) != type(other):
            return False
        return self.__dict__ == other.__dict__

    def __eq__(self, other):
        seq_comparison = False
        if isinstance(other, Iterable):
            data_list = self.get_current_sequence()
            seq_comparison = (data_list == list(other))
        return self.__isequal(other) or seq_comparison


class Root(BaseNode):

    def __init__(self):
        super().__init__(None)

    def add_children(self, keys: [Iterable[str], Iterable[int], str, int]):
        if not isinstance(keys, Iterable):
            keys = [keys]
        for i in keys:
            assert not self.has_child(i), f'child {i} already exists.'
            self[i] = self._Node(i, self)



    def __str__(self):
        return '-> ' + str(self.get_child_keys()) + ''

    def get_current_sequence(self):
        return []

    def add_sequence(self, seq):
        if seq in self:
            return
        base_update_node = self._add_sequence(seq)

    def create_representative(self):
        stack = [self]
        while len(stack) > 0:
            cur_node = stack.pop(0)
            if len(cur_node) == 1:
                next_node = list(cur_node)[0]
                next_node.data = cur_node.data
                next_node.data.representative_nodes.append(cur_node)
            stack.extend(cur_node)

    def get_next_node(self, letter, cur_node: [BaseNode, None] = None) -> Tuple['_Node', float]:
        if cur_node is not None and cur_node.has_children():
            return cur_node[letter]
        return self[letter]

    def calculate_pob(self, seq):
        seq = self._parse_seq(seq)
        cur_node = self
        p = None
        for i in seq:
            cur_node, p = self.get_next_node(i, cur_node)
        return p

    class _Node(BaseNode):
        def __init__(self, key: [str, int, None], parent: [BaseNode]):
            super().__init__(key)
            self.__parent = parent

        def get_parent(self):
            return self.__parent

        def update_data(self):
            if self.data is None:
                self.data = WeightedData(self, use_fractions=USE_FRACTIONS)
            elif self.__parent.data == self.data:
                self.data = deepcopy(self.__parent.data)


        @classmethod
        def __create_new_object(cls, key: [str, int], parent: [BaseNode]):
            return cls(key, parent)

        def add_children(self, keys: [Iterable[str], Iterable[int], str, int]):
            if not isinstance(keys, Iterable):
                keys = [keys]
            for i in keys:
                assert not self.has_child(i), f'child {i} already exists.'
                self[i] = self.__create_new_object(i, self)

        def __str__(self):
            return str(self.__parent) + (" @ " + str(self.get_key()) if len(self.__parent) > 1 else '') + " -> " + str(
                self.get_child_keys())

        def __create_nodes_by_layers(self) -> Iterable[BaseNode]:  # todo change name
            stack: List['BaseNode'] = [self]
            nodes_order: List['BaseNode'] = [self]
            while len(stack) > 0:
                node = stack.pop(0)
                nodes_order.append(node)
                stack.extend(node)
            return reversed(nodes_order)

        def get_current_sequence(self):
            data_list = []
            cur_node = self
            while isinstance(cur_node, type(self)):
                data_list.append(cur_node.get_key())
                cur_node = cur_node.get_parent()
            return list(reversed(data_list))


#
# class DataNode(ABC):
#
#     def __init__(self, node):
#         self.node = node
#
#     @abstractmethod
#     def __repr__(self):
#         pass
#
#     def get_key(self):
#         return self.node.key
#
#     def update_probs(self):
#         pass
#
#     def update_key(self, c):
#         pass


class WeightedData():
    def __init__(self, node, use_fractions=USE_FRACTIONS):
        # super().__init__(node)
        self.node = node
        self.__b = 0
        self.__a = 0
        self.prob_w = 0. if use_fractions else mpf(0.)
        self.prob_e = 0. if use_fractions else mpf(0.)
        self._use_fractions = use_fractions
        self.representative_nodes = [node]
        # self.update_needed_e = False

    def update_key(self, key: [int, None]):
        assert key in {0, None, 1}, "not a valid key"
        self.__update_probs_e(key)
        # print(key, sep='')
        if key == 1:
            self.__b += 1
        elif key == 0:
            self.__a += 1

        # self.update_needed_e = True

    def get_key(self):
        return self.node.key

    def get_ab(self):
        return (self.__a, self.__b)

    def get_next_datas(self):
        cur_data = self
        cur_node = self.node
        while (len(cur_node) == 1 and list(cur_node)[0].data == cur_data):
            cur_node = list(cur_node)[0]
            # assert cur_data==cur_node.data, "data are not compressed"
        if len(cur_node) == 0:
            return []
        else:
            return [i.data for i in cur_node]

    def get_log_prob_w(self, key: [int]):
        if self._use_fractions:
            return log(self.get_prob_w(key)) / mpmath.log(2)
        return self.get_prob_w(key) / mpmath.log(2)

    def get_prob_w(self, key: [int]):
        assert key in {0, 1}, "not a valid key"

        if key == 1:
            return self.prob_w
        return self.prob_w

    def get_prob_e(self, key: [int]):
        assert key in {0, 1}, "not a valid key"

        if key == 1:
            return self.prob_e
        return 1 - self.prob_e

    def update_probs(self):
        # self.__update_probs_e()
        self.__update_all_children_probs_w()
        # self.__update_probs_w()

    def __update_probs_e(self, updated_key: int):
        if self._use_fractions:
            numerator = self.__a + fp(0.5)
            new_pe = abs(updated_key - fp(numerator) / fp(self.__a + self.__b + 1))
            self.prob_e = new_pe * self.prob_e
        else:
            if updated_key == 0:
                numerator = self.__a + 0.5
            else:
                numerator = self.__b + 0.5
            new_pe = log(numerator) - log(self.__a + self.__b + 1.)
            self.prob_e = new_pe + self.prob_e
        # self.update_needed_e = False

    def __update_from_leaf(self):
        stack = [self.node]
        data_leaf = []
        while len(stack) > 0:
            cur_node = stack.pop(0)
            if len(cur_node) == 0:
                data_leaf.append(cur_node.data)
            else:
                stack.extend(cur_node)

    def __update_w_by_data(self):
        self.__update_probs_w()

    def __update_all_children_probs_w(self):
        stack = [self]
        counter=0
        while len(stack) >counter:
            cur_node = stack[counter]
            counter+=1
            # print(counter,'c')
            stack.extend(cur_node.get_next_datas())
            # print('len ',len(cur_node.get_next_datas()))
        for i,n in enumerate(reversed(stack)):
            # print(i)
            n.__update_probs_w()

    def __update_probs_w(self):
        if self._use_fractions:
            probs = fp(1) if self.node.has_children() else fp(0)
            for n in self.get_next_datas():
                # n.data.update_probs()
                probs = probs * n.data.get_prob_w(1)
            if len(self.get_next_datas()) == 0:
                self.prob_w = 0.5
            else:
                two_to_pow = mp.power(2, len(self.representative_nodes))
                factor = fp(f'1/{str(two_to_pow)}')
                self.prob_w = (1 - factor) * self.get_prob_e(1) + factor * probs
                # self.prob_w = (self.get_prob_e(1) + (probs)) * (fp(0.5 + 0.5 * (not self.node.has_children())))
        else:
            probs = 0.
            for n in self.node:
                # n.data.update_probs()
                probs = probs + n.data.get_prob_w(1)
            if len(list(self.node)) == 0:
                self.prob_w = self.prob_e
            else:
                two_to_pow = mp.power(2, len(self.representative_nodes))
                factor = mpf(1.) / two_to_pow
                self.prob_w = logaddexp(self.prob_e + log(1 - factor), probs + log(factor))

    def __repr__(self):
        l = lambda x: x.limit_denominator(MAX_DENOMINATOR)
        if not self._use_fractions:
            l = lambda x: fp(math.exp(x)).limit_denominator(MAX_DENOMINATOR)
        return f"(a = {self.__a},b = {self.__b})  pe = {l(self.prob_e)} pw = {l(self.prob_w)}"

    def validate_ab(self):
        assert self.__a == sum(
            [i.data.__a for i in self.node]), f"{self.node}   a: {self.__a} {[i.data.__a for i in self.node]}"
        assert self.__b == sum(
            [i.data.__b for i in self.node]), f"{self.node}   b: {self.__b} {[i.data.__b for i in self.node]}"


class CTWManager(ABC):
    def __init__(self, tree):
        self.tree = tree

    def print_v_tree(self):
        self.tree.print_v_tree()

    @abstractmethod
    def update_by_sequence(self, sequence):
        pass

    def __str__(self):
        return self.tree.__str__()

    def get_entropy(self, l, last_key=0):
        return -self.tree.data.get_log_prob_w(last_key) / l

        # return
