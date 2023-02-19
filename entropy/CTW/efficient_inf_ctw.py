from __future__ import annotations

from typing import List, Tuple
import sys


import mpmath
from mpmath import mp, mpf
from .__fractional_precision import FractionPrecition as fp
from .__print_tree import drawTree2
from .__print_tree import Node as s_node
import pickle
MAX_DENOMINATOR = 10000000
mp.dps = 50
LETTERS = {0, 1, None}


def log(a):
    return mpmath.log(a)


def logaddexp(a, b):
    return log(mpmath.exp(mpf(a)) + mpmath.exp(mpf(b)))


# def validate_pattern(function):
#     def wrapper(*args, **kwargs):
#         for i in kwargs.keys():
#             if 'pattern' in i:
#                 assert kwargs[i][0] is None and (len(kwargs[i]) > 1 and all(
#                     [isinstance(i, int) for i in kwargs[i][1:]])), f'pattern is not valid method{function},key{i}'
#         return function(*args, **kwargs)
#
#     return wrapper


class Node():

    def __init__(self, pattern: List[None, int], __children=None, __occurrences=None, __parent=None, ):
        self.__context_pattern: List[None, int] = pattern
        self.__children = dict() if __children is None else __children
        self.__occurrences = 1 if __occurrences is None else __occurrences
        self.__parent = None if __parent is None else __parent
        # self.a=0
        # self.b=0
        # self.__prob_w =mpf(0.)
        # self.__prob_e =mpf(0.)
        # self.__child_prob_w=0
        # self.update_predictions(pattern[0])

    def __build_params(self, a: None | int = None, b: None | int = None, prob_e: [None, mpf] = None,
                       __prob_w: [None, mpf] = None):
        conditions = [a is None, b is None, prob_e is None]
        assert all(conditions) or not any(conditions), "a,b and e supposed to be all none or all with value"

        self.a = 0 if a is None else a
        self.b = 0 if b is None else b
        self.__prob_e = mpf(0) if prob_e is None else prob_e
        if __prob_w is None:
            self.__prob_w = mpf(0.)
            self.__update_prob_w()
        else:
            self.__prob_w = __prob_w
        # self.update_predictions(pattern[0])

    def __getitem__(self, item):
        if isinstance(item, list):
            item = item[-1]
        return self.__children[item]

    def to_dict(self):
        children_dict = dict()
        for k, v in self.__children.items():
            children_dict[k] = v.to_dict()
        return dict(children = children_dict,
        constractor = dict(__context_pattern=self.__context_pattern, __occurrences=self.__occurrences,
             __parent=self.__parent),
        parameters = dict(a=self.a, b=self.b, prob_e=self.prob_e, __prob_w=self.__prob_w))
    @staticmethod
    def from_dict(data_dict):
        __children=dict()
        for k,v in data_dict['children']:
            __children[k]=Node.from_dict(v)
        data_dict['constractor'][__children]=__children
        obj = Node(**data_dict['constractor'])
        obj.__build_params(**data_dict['parameters'])
        return obj

    @property
    def parent(self):
        return self.__parent

    @property
    def children(self):
        return self.__children

    @property
    def context_pattern(self):
        return self.__context_pattern

    @property
    def prob_w(self):
        return self.__prob_w

    def __update_prob_w(self):
        if len(self.__children) > 0:
            two_to_pow = mp.power(2, len(self.context_pattern))
            factor = mpf(1.) / two_to_pow
            child_prob_w = mpf(0.)
            for i in self.children.values():
                child_prob_w += i.prob_w
            self.__prob_w = logaddexp(self.prob_e + log(1 - factor), child_prob_w + log(factor))
        else:
            self.__prob_w = log(mpf(0.5))

    @property
    def prob_e(self):
        return self.__prob_e

    @property
    def occurrences(self):
        return self.__occurrences

    def __update_probs_e(self, updated_key: int | None):
        numerator = None
        if updated_key == 0:
            numerator = self.a + 0.5
        elif updated_key == 1:
            numerator = self.b + 0.5
        if numerator is not None:
            new_pe = log(numerator) - log(self.a + self.b + 1.)
            self.__prob_e = new_pe + self.__prob_e
        else:
            assert self.a == 0 and self.b == 0, f'cannot assigne value by none if a and b are non zero (a: {self.a},b: {self.b})'
            self.__prob_e = log(0.5)
            del self.a
            del self.b

    def update_w(self):
        stack = [self]
        counter = 0
        while len(stack) > counter:
            cur_node = stack[counter]
            counter += 1
            # print(counter,'c')
            stack.extend(cur_node.children.values())
            # print('len ',len(cur_node.get_next_datas()))
        for i, n in enumerate(stack[::-1]):
            # print(i)
            n.__update_prob_w()
        self.__update_prob_w()

    def __update_predictions(self, x: None | int):
        assert x in LETTERS, f"letter {x} is not valid ,can only append {LETTERS}"
        self.__update_probs_e(x)
        if x == 0:
            self.a += 1
        elif x == 1:
            self.b += 1
        if self.parent is not None:
            self.parent.__update_predictions(x)

    def __travel_to_position(self, pattern: List[None, int]) -> Tuple['Node', int, List[None, int]]:
        """
        position means the point in the tree where the patterns supposed to end
        :param pattern: current_pattern
        :return: the parent node and the reminder of pattern
        """
        if self.context_pattern == pattern:  # if equal
            return self, len(pattern), []
        if len(self.context_pattern) > len(pattern):  # if pattern is equal to the current node pattern(or even differ)

            i = -1
            for o, n in zip(self.context_pattern[::-1], pattern[::-1]):
                i += 1
                if o == n:
                    continue
                return self, i, pattern[:len(pattern) - i]
            assert False, f"Edge case!!!!!!!!!!!!!!!!!!!!!!!!!!! pattern:{pattern}\n, existing pattern:{self.context_pattern}"

        else:
            if self.context_pattern == pattern[len(pattern) - len(self.context_pattern):]:
                if pattern[len(pattern) - len(self.context_pattern) - 1] in self.__children:
                    return self.__children[pattern[len(pattern) - len(self.context_pattern) - 1]].__travel_to_position(
                        pattern=pattern[:len(pattern) - len(self.context_pattern)])
                else:
                    return self, len(self.context_pattern), pattern[:len(pattern) - len(self.context_pattern)]
            else:
                i = -1
                for o, n in zip(self.context_pattern[::-1], pattern[::-1]):
                    i += 1
                    if o == n:
                        continue
                    return self, i, pattern[:len(pattern) - i]
                assert False, f"Edge case!!!!!!!!!!!!!!!!!!!!!!!!!!! pattern:{pattern}\n, existing pattern:{self.context_pattern}"
            # assert False, f"Edge case!!!!!!!!!!!!!!!!!!!!!!!!!!! pattern:{pattern}\n, existing pattern:{self.context_pattern}"

    def add_child(self, pattern):
        if pattern[0] is not None:
            pattern = [None] + pattern
        # assert pattern[-len(self.context_pattern):] == pattern,'' todo cheack that the pattern are different.
        cur_node, index, reminder = self.__travel_to_position(pattern=pattern[:-1])
        if len(reminder) == 0:
            cur_node.__occurrences += 1
            return

        prediction = pattern[-1]
        if len(cur_node.context_pattern) == index:
            cur_child = cur_node.__create_new_object(pattern=pattern[:-1])
            cur_node.__children[reminder[-1]] = cur_child

        elif len(cur_node.context_pattern) > index:
            cur_node.__split(len(cur_node.context_pattern) - index - 1)
            assert reminder[-1] not in cur_node.__children
            cur_child = cur_node.__create_new_object(pattern=pattern[:-1])
            cur_node.__children[reminder[-1]] = cur_child
        else:
            assert False, "Edge case!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        cur_child.__update_predictions(prediction)

    def __create_new_object(self, pattern: List[None, int]) -> 'Node':
        # assert pattern[len(pattern) - len(self.context_pattern):] == self.context_pattern, "pattern are not match"
        context_pattern: List[None, int] = pattern[:len(pattern) - len(self.context_pattern)]
        obj = Node(context_pattern)
        obj.__build_params()
        obj.__parent = self
        return obj

    def __split(self, index: int) -> 'Node':
        context_pattern: List[None, int] = self.context_pattern[:index + 1]
        obj = Node(context_pattern)
        obj.__build_params(self.a, self.b, self.__prob_e)
        self.__context_pattern = self.context_pattern[index + 1:]
        self.__pass_children(obj)
        return obj

    def __pass_children(self, new_child):
        new_child.__children = self.__children

        for i in new_child.__children.values():  # pass parent
            i.__parent = new_child

        self.__children = dict()
        self.__children[new_child.context_pattern[-1]] = new_child
        new_child.__parent = self

    def __build_v_tree_node(self):
        text = repr(self.context_pattern) + f"[({self.a},{self.b}),pe {float(self.prob_e)},pw {float(self.prob_w)}]"
        children = []
        if 1 in self.__children:
            children.append(self[1].__build_v_tree_node())
        if None in self.__children:
            children.append(self[None].__build_v_tree_node())
        if 0 in self.__children:
            children.append(self[0].__build_v_tree_node())
        out = s_node(text)(children)
        return out

    @staticmethod
    def generate_root():
        base_node = Node([])
        base_node.__build_params()
        return base_node

    def print_v_tree(self):
        out = drawTree2(False)(False)(self.__build_v_tree_node()).encode('utf-8-sig')
        print(out.decode('utf-8-sig'))

    def get_log_prob_w(self, *kwars):
        return self.prob_w / mpmath.log(2.)
