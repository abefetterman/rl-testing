import tensorflow as tf
from dqn import DQN
import numpy as np

model=DQN()
result=model.build()
with tf.Session() as sess:
    rand_array=np.random.rand(1,84,84,4)
    print(sess.run(result, feed_dict={images:rand_array}))
