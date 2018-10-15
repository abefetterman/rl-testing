import tensorflow as tf

class MLP(object):
    def __init__(self, hidden_dim=32, layers=1, actions=6, name='mlp'):
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.actions = actions
        self.name = name

    def preprocess(self, x):
        return x

    def apply(self, x):
        with tf.variable_scope(self.name):
            net = x
            for size in [self.hidden_dim]*self.layers:
                net = tf.layers.dense(net, units=self.hidden_dim, activation=tf.nn.relu)
            output = tf.layers.dense(net, units=self.actions, activation=None)
        return output
