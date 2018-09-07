import tensorflow as tf
from dqn import DQN
import numpy as np
from utils import *
from buffers import *
from actor import RandomActor

# model
model=DQN()
images,result=model.build()
actor = RandomActor(6)

# buffers
replay = ReplayBuffer(100)
postprocess = lambda x: np.stack(x, axis=-1)
states = RecentStateBuffer(4, preprocess=preprocess_image, postprocess=postprocess)

def rand_state():
    return (255*np.random.rand(200,160)).astype(np.uint8)
# boilerplate
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    states.reset()
    this_state = states.add(rand_state())
    for i in range(6):
        action = actor.get(this_state)
        image,reward,done = rand_state(),0,False

        next_state = states.add(image)
        replay.add(this_state, action, reward, next_state, done)
        this_state = next_state
        minibatch = replay.sample(5)

        print(minibatch[0].shape)
    #print(sess.run(result, feed_dict={images:states.get()}))
