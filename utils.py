import tensorflow as tf

class RecentStateBuffer(object):
    def __init__(self, bufferSize):
        self.bufferSize=bufferSize
        self.reset()
    def reset(self):
        self.buffer=[]
    def add(self,state):
        self.buffer.append(state)
        if len(self.buffer)>self.bufferSize:
            self.buffer.pop(0)
    def get(self):
        return self.buffer[:]

class ReplayBuffer(object):
    def __init__(self, bufferSize, imageSize=(84,84,4)):
        self.bufferSize = bufferSize
        self.imageSize = imageSize
        self.transitionSize = imageSize[0] * imageSize[1] * imageSize[2] * 2 + 3
    def reset(self):
        self.buffer = tf.zeros([self.bufferSize,self.transitionSize])
        self.writeIndex = 0
