import tensorflow as tf
import numpy as np

class RecentStateBuffer(object):
    def __init__(self, bufferSize, preprocess=None, postprocess=None):
        self.bufferSize=bufferSize
        self.preprocess=preprocess
        self.postprocess=postprocess
        self.reset()
    def reset(self):
        self.buffer=[]
    def add(self,state):
        if self.preprocess is not None:
            state=self.preprocess(state)
        self.buffer.append(state)
        if len(self.buffer)>self.bufferSize:
            self.buffer.pop(0)
        return self.get()
    def get(self):
        if self.postprocess is None:
            return self.buffer[:]
        return self.postprocess(self.buffer)

class ReplayBuffer(object):
    def __init__(self, bufferSize, imageSize=[84,84,4]):
        self.bufferSize = bufferSize
        self.imageSize = imageSize
        self.reset()
    def reset(self):
        self.this_state_buffer = np.zeros([self.bufferSize]+self.imageSize)
        self.next_state_buffer = np.zeros([self.bufferSize]+self.imageSize)
        self.action_buffer = np.zeros([self.bufferSize,1])
        self.reward_buffer = np.zeros([self.bufferSize,1])
        self.done_buffer = np.zeros([self.bufferSize,1])
        self.writeIndex = 0
        self.bufferCount = 0
    def add(self, this_state, action, reward, next_state, done):
        if this_state.shape[-1]<4: return
        i = self.writeIndex
        self.this_state_buffer[i] = this_state
        self.next_state_buffer[i] = next_state
        self.action_buffer[i] = action
        self.reward_buffer[i] = reward
        self.done_buffer[i] = done
        self.writeIndex = (i + 1) % self.bufferSize
        self.bufferCount = min(self.bufferSize, self.bufferCount + 1)
    def sample(self, n_samples):
        if (n_samples < self.bufferCount):
            idxes = np.random.choice(self.bufferCount, n_samples, replace=False)
        else:
            idxes = np.arange(self.bufferCount)
        return self.this_state_buffer[idxes], self.action_buffer[idxes], \
                self.reward_buffer[idxes], self.next_state_buffer[idxes], \
                self.done_buffer[idxes]
