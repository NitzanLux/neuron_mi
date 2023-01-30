from . import __tree as te

from abc import abstractmethod

LETTERS = (0, 1)




class CTWManagerFinite(te.CTWManager):

    def __init__(self, D: [int], is_contextless=False):
        super().__init__(te.Root())
        self.D = D
        self.build_tree(list(LETTERS) + ([None] if is_contextless else []))


    def update_by_sequence(self, raw_sequence, cond_sequence=None):

        cur_node = self.tree
        if cond_sequence is not None:
            cond_sequence = cond_sequence[len(cond_sequence) - self.D:]
        else:
            cond_sequence = [None] * self.D
        sequence = cond_sequence + raw_sequence
        for i in range(len(sequence)):
            cur_node = self.tree
            if self.D + 1 + i > len(sequence):
                break
            # print('@$@$---------------', sequence[i:i + min(self.D + 1, len(sequence))], '<----->', i)
            cur_seq = sequence[i:i + self.D][:]
            for j, c in enumerate(cur_seq[::-1] + [sequence[i + self.D]]):
                # print('**-------', cur_node.get_current_sequence(), c)

                cur_node.data.update_key(sequence[i + self.D])
                # print(sequence[i+self.D],i+self.D)
                # print(sequence[i:i + min(self.D + 1, len(sequence))], cur_node.data.get_ab(),
                #       cur_node.get_current_sequence())
                cur_node = self.tree.get_next_node(c, cur_node)

            # cur_node = self.tree
            # for j,c in enumerate(sequence[i:i+min(self.D+1,len(sequence))]):
            #     cur_node.data.validate_ab()
            #     cur_node = self.tree.get_next_node(c, cur_node)

        self.tree.data.update_probs()

    def build_tree(self, letters=LETTERS):
        stack = [self.tree]
        while len(stack) > 0:
            cur_node = stack.pop(0)
            cur_node.data = te.WeightedData(cur_node)
            if cur_node.get_depth() >= self.D:
                continue
            cur_node.add_children(letters)
            stack.extend(cur_node)



#%%tests
#
# a = CTWManagerFinite(2, True)
# a.update_by_sequence([0, 1, 1, 0, 1, 0, 0])
# a.print_v_tree()

#%
# a = CTWManagerFinite(3)
# a.update_by_sequence([0, 1, 1, 0, 1, 0, 0],[0,1,0])
# a.print_v_tree()
