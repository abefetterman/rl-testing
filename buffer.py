import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, len):
        self.buf = []
        self.len = len
    def __len__(self):
        return len(self.buf)
    def add(self, new):
        if len(self.buf) >= self.len:
            self.buf.pop(0)
        self.buf.append(new)
    def sample(self, count):
        s = random.sample(self.buf, count)
        return [x for x in zip(*s)]

class PriorityBuffer(object):
    def __init__(self, len, alpha=1, beta=1):
        self.buf = []
        self.priorities = []
        self.priorities_max = 1
        self.alpha = alpha
        self.beta = beta
        self.len = len
    def __len__(self):
        return len(self.buf)
    def add(self, new):
        if len(self.buf) >= self.len:
            self.buf.pop(0)
            self.priorities.pop(0)
        self.buf.append(new)
        self.priorities.append(self.priorities_max)
    def sample(self, count):
        buffer_size = len(self.buf)
        p = np.array(self.priorities) ** self.alpha
        p = p / np.sum(p)
        idxs = np.random.choice(buffer_size, count, p=p)
        p_choice = p[idxs]
        buf_choice = [self.buf[x] for x in idxs]
        is_weights = (count * p_choice) ** ( - self.beta)
        is_weights = is_weights / np.max(is_weights)
        sample = [x for x in zip(*buf_choice)]
        return sample + [idxs, is_weights]
    def update_priorities(self, idxs, td_error):
        for i,e in zip(idxs,td_error):
            self.priorities[i] = e
        self.priorities_max = max(self.priorities)
