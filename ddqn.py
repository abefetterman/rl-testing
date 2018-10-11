import tensorflow as tf
import numpy as np
import gym

class DQNModel(object):
    def __init__(self, hidden_dim=32, n_layers=1, n_acts=3, name='model'):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_acts = n_acts
        self.name = name
    def apply(self, obs_ph):
        with tf.variable_scope(self.name):
            net = obs_ph
            for size in [self.hidden_dim]*self.n_layers:
                net = tf.layers.dense(net, units=self.hidden_dim, activation=tf.nn.relu)
            value = tf.layers.dense(net, units=self.n_acts, activation=None)
        return value
    def get_trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    def copy_from(self, other_network):
        my_vars = self.get_trainable_vars()
        other_vars = other_network.get_trainable_vars()

        # y <- x
        ops = [y.assign(x) for x,y in zip(other_vars,my_vars)]

        return ops

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



def train(sess=None, env_name='CartPole-v0', hidden_dim=32, n_layers=1,
          lr=1e-2, gamma=0.99, n_iters=50, batch_size=5000, eps=0.1,
          n_samples=100, buffer_size = 10000, update_period = 10,
          target_replace_period=100,
          ):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    dqn_train = DQNModel(hidden_dim, n_layers, n_acts, name='train_model')
    dqn_target = DQNModel(hidden_dim, n_layers, n_acts, name='target_model')

    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    predicted_values_target = dqn_target.apply(obs_ph)
    greedy_action = tf.argmax(predicted_values_target, 1)
    greedy_value = tf.reduce_max(predicted_values_target, 1)

    buffer = PriorityBuffer(buffer_size)

    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    done_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    reward_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    weight_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    next_state_value_ph = tf.placeholder(shape=(None,), dtype=tf.float32)

    predicted_values_train = dqn_train.apply(obs_ph)
    discounted_reward = reward_ph + gamma * (1 - done_ph) * next_state_value_ph
    action_one_hots = tf.one_hot(act_ph, n_acts)
    estimated_reward = tf.reduce_sum(predicted_values_train * action_one_hots, 1)

    td_error = discounted_reward - estimated_reward
    loss = tf.reduce_sum(td_error * td_error * weight_ph)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    copy_op = dqn_target.copy_from(dqn_train)

    if (sess==None):
        sess=tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    sess.run(copy_op)

    for i in range(n_iters):
        batch_rets, batch_lens = [], []

        obs, rew, done, ep_rews = env.reset(), 0, False, []
        batch_loss, batch_steps = 0, 0
        while True:
            batch_steps += 1
            act = sess.run(greedy_action, {obs_ph: obs.reshape(1,-1)})[0]
            if random.random() < eps:
                act = random.randrange(n_acts)
            next_obs, rew, done, _ = env.step(act)
            buffer.add((obs.reshape(1,-1), act, rew, done, next_obs.reshape(1,-1)))
            obs = next_obs
            ep_rews.append(rew)

            if len(buffer) > n_samples and (batch_steps % update_period) == 0:
                obss, acts, rews, dones, next_obss, idxs, weights = buffer.sample(n_samples)
                next_values = sess.run(greedy_value, {obs_ph: np.vstack(next_obss)})
                # print(next_values)
                new_priorities, step_loss, _ = sess.run([td_error, loss, train_op], feed_dict={
                                              act_ph: np.array(acts),
                                              reward_ph: np.array(rews),
                                              done_ph: np.array(dones),
                                              obs_ph: np.vstack(obss),
                                              next_state_value_ph: np.array(next_values),
                                              weight_ph: weights,
                                          })
                buffer.update_priorities(idxs, np.abs(new_priorities))
                batch_loss += step_loss / batch_size
            if batch_steps > 0 and (batch_steps % target_replace_period) == 0:
                sess.run(copy_op)

            if done:
                batch_rets.append(sum(ep_rews))
                batch_lens.append(len(ep_rews))
                obs, rew, done, ep_rews = env.reset(), 0, False, []
                if batch_steps > batch_size:
                    break

        print('itr: %d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    try:
        with tf.Session() as sess:
            train(sess)
    except KeyboardInterrupt:
        print('bye')
