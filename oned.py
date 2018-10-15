from mlp import MLP
import tensorflow as tf
import gym

class MLP(object):
    def __init__(self, hidden_dim=32, layers=1, actions=6, name='mlp'):
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.actions = actions
        self.name = name

    def apply(self, x):
        with tf.variable_scope(self.name):
            net = x
            for size in [self.hidden_dim]*self.layers:
                net = tf.layers.dense(net, units=self.hidden_dim, activation=tf.nn.relu)
            output = tf.layers.dense(net, units=self.actions, activation=None)
        return output


class OneDProblem(object):
    def __init__(self, env_name='CartPole-v0', hidden_dim=32, n_layers=1):
        self.env_name = env_name
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def make_env(self):
        return gym.make(self.env_name)

    def make_net(self, env, name='mlp'):
        n_acts = env.action_space.n
        return MLP(self.hidden_dim, self.n_layers, n_acts, name=name)

    def make_obs_ph(self, env):
        obs_dim = env.observation_space.shape
        return tf.placeholder(shape=(None, *obs_dim), dtype=tf.float32)

    def preprocess_obs(self, x):
        return x.reshape(1,-1)
