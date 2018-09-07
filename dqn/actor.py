import numpy as np

class RandomActor:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def get(self, state):
        return np.random.randint(self.n_actions)
