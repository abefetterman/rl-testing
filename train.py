from gym.envs.atari import AtariEnv
import numpy as np
from utils import *
from dqn import DQN

model=DQN()
env = AtariEnv(game='pong', obs_type='image', frameskip=4)
action_space=6
episodes=1
done=False
reward=0
states = RecentStateBuffer(4)
with tf.Session() as sess:
    for i in range(episodes):
        states.reset()
        state = env.reset()
        states.add(state)
        last_states=states.get()
        action=0
        while len(last_states)<4:
            action = np.random.randint(action_space)
            state, reward, done, _ = env.step(action)
            states.add(state)
            new_states=states.get()
            #DQN.add_transition(last_states,action,reward,new_states,done)
            last_states=new_states
        out=model.eval(last_states)
        sess.run(out)
