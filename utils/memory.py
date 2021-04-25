import random
import numpy as np


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.probs = []
        self.rewards = []
        self.next_states = []
        self.is_terminals = []
    
    def reset_memory(self):
        self.actions = []
        self.states = []
        self.probs = []
        self.rewards = []
        self.next_states = []
        self.is_terminals = []

    def split(self, batch_size):
        split_res = []
        length = len(self.actions)
        if batch_size is None:
            split_res.append((0, length-1))
        for idx in range(0, length, batch_size):
            split_res.append((idx, idx+batch_size))
        return split_res

    def sample(self, batch_size):
        samples = set(random.sample(list(range(len(self.actions))), batch_size))
        ret = [
            [aa for ix, aa in enumerate(self.actions) if ix in samples],
            [ss for ix, ss in enumerate(self.states) if ix in samples],
            [pp for ix, pp in enumerate(self.probs) if ix in samples],
            [rr for ix, rr in enumerate(self.rewards) if ix in samples],
            [nn for ix, nn in enumerate(self.next_states) if ix in samples],
            [ii for ix, ii in enumerate(self.is_terminals) if ix in samples],
        ]
        return ret
