import tensorflow as tf

def _log_activations(x):
    tf.summary.histogram(x.op.name+'/activations', x)
    tf.summary.scalar(x.op.name+'/sparsity', tf.nn.zero_fraction(x))

class ConvNet(object):
    def __init__(self, actions=6, name='conv_net'):
        self.actions = actions

    def preprocess(self, images):
        new_images = tf.convert_to_tensor(images)
        new_images = tf.image.rgb_to_grayscale(new_images)
        new_images = tf.image.resize_image_with_crop_or_pad(new_images, 160, 160)
        new_size = tf.constant([84,84])
        new_images = tf.image.resize_images(new_images,new_size)
        new_images = tf.squeeze(new_images,axis=3)
        new_images = tf.transpose(new_images,perm=[1,2,0])
        return new_images

    def apply(self, images):
        with tf.variable_scope(self.name):
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=5e-2, dtype=tf.float32)
            conv1 = tf.layers.conv2d(
              inputs=images,
              filters=32,
              kernel_size=[8, 8],
              strides=(4,4),
              padding="same",
              kernel_initializer=kernel_initializer,
              activation=tf.nn.relu)
            _log_activations(conv1)

            conv2 = tf.layers.conv2d(
              inputs=conv1,
              filters=64,
              kernel_size=[4, 4],
              strides=(2,2),
              padding="same",
              kernel_initializer=kernel_initializer,
              activation=tf.nn.relu)
            _log_activations(conv2)

            conv3 = tf.layers.conv2d(
              inputs=conv2,
              filters=64,
              kernel_size=[3, 3],
              padding="same",
              kernel_initializer=kernel_initializer,
              activation=tf.nn.relu)
            _log_activations(conv3)

            conv3_flat = tf.reshape(conv3, [-1, 11 * 11 * 64])
            dense = tf.layers.dense(inputs=conv3_flat, units=512, activation=tf.nn.relu)
            _log_activations(dense)

            logit = tf.layers.dense(inputs=dense, units=self.actions, activation=None)
            _log_activations(logit)

        return logit
