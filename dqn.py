import tensorflow as tf
import numpy as np
import gym
from oned import OneDProblem
from atari import AtariProblem

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

def train(problem, sess=None,
          lr=1e-4, gamma=0.99, n_iters=50, batch_size=5000, eps=0.1,
          n_samples=100, buffer_size = 10000, update_period = 10,
          ):
    env = problem.make_env()
    n_acts = env.action_space.n

    dqn = problem.make_net(env)

    obs_ph = problem.make_obs_ph(env)
    predicted_values = dqn.apply(obs_ph)
    greedy_action = tf.argmax(predicted_values, 1)
    greedy_value = tf.reduce_max(predicted_values, 1)

    buffer = PriorityBuffer(buffer_size)

    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    done_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    reward_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    weight_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    next_state_value_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    discounted_reward = reward_ph + gamma * (1 - done_ph) * next_state_value_ph
    action_one_hots = tf.one_hot(act_ph, n_acts)
    estimated_reward = tf.reduce_sum(predicted_values * action_one_hots, 1)

    td_error = discounted_reward - estimated_reward
    loss = tf.reduce_sum(td_error * td_error * weight_ph)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    if (sess==None):
        sess=tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())

    for i in range(n_iters):
        batch_rets, batch_lens = [], []

        obs_raw, rew, done, ep_rews = env.reset(), 0, False, []
        obs = problem.preprocess_obs(obs_raw)
        batch_loss, batch_steps = 0, 0
        while True:
            env.render()
            batch_steps += 1
            act = sess.run(greedy_action, {obs_ph: obs})[0]
            if random.random() < eps:
                act = random.randrange(n_acts)
            next_obs_raw, rew, done, _ = env.step(act)
            next_obs = problem.preprocess_obs(next_obs_raw)
            buffer.add((obs, act, rew, done, next_obs))
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
                print(step_loss/batch_size)


            if done:
                batch_rets.append(sum(ep_rews))
                batch_lens.append(len(ep_rews))
                print('{}, {} avg'.format(sum(ep_rews),len(ep_rews)))
                obs, rew, done, ep_rews = env.reset(), 0, False, []
                if batch_steps > batch_size:
                    break

        print('itr: %d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    try:
        problem = AtariProblem()
        with tf.Session() as sess:
            train(problem, sess)
    except KeyboardInterrupt:
        print('bye')
