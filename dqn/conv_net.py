import tensorflow as tf

def preprocess(images):
    new_images = tf.convert_to_tensor(images)
    new_images = tf.image.rgb_to_grayscale(new_images)
    new_images = tf.image.resize_image_with_crop_or_pad(new_images, 160, 160)
    new_size = tf.constant([84,84])
    new_images = tf.image.resize_images(new_images,new_size)
    new_images = tf.squeeze(new_images,axis=3)
    new_images = tf.transpose(new_images,perm=[1,2,0])
    return new_images

def _init_variable(name, shape, mean=0.0, stddev=0.0, device='/cpu:0'):
    with tf.device(device):
        initializer=tf.constant_initializer(mean)
        if stddev>1e-8:
            initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def _log_activations(x):
    tf.summary.histogram(x.op.name+'/activations', x)
    tf.summary.scalar(x.op.name+'/sparsity', tf.nn.zero_fraction(x))

def build(images, actions=6, shape=[None,84,84,4]):
    with tf.variable_scope('conv1') as scope:
        kernel = _init_variable('weights',
                                shape=[8,8,4,32],
                                stddev=5e-2)
        # HW stride 4, in NHWC order:
        conv = tf.nn.conv2d(images, kernel, strides=[1,4,4,1],padding='SAME')
        biases = _init_variable('biases', shape=[32])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _log_activations(conv1)

    with tf.variable_scope('conv2') as scope:
        kernel = _init_variable('weights',
                                shape=[4,4,32,64],
                                stddev=5e-2)
        # HW stride 2, in NHWC order:
        conv = tf.nn.conv2d(conv1, kernel, strides=[1,2,2,1],padding='SAME')
        biases = _init_variable('biases', shape=[64])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _log_activations(conv2)

    with tf.variable_scope('conv3') as scope:
        kernel = _init_variable('weights',
                                shape=[3,3,64,64],
                                stddev=5e-2)
        # HW stride 1, in NHWC order:
        conv = tf.nn.conv2d(conv2, kernel, strides=[1,1,1,1],padding='SAME')
        biases = _init_variable('biases', shape=[64])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _log_activations(conv3)

    with tf.variable_scope('dense4') as scope:
        new_dims = tf.constant([-1, 7744], dtype=tf.int32)
        reshape = tf.reshape(conv3, new_dims)
        dim = reshape.get_shape().as_list()[1]
        weights = _init_variable('weights', shape=[dim, 512], stddev=4e-2)
        biases = _init_variable('biases', [512], mean=0.1)
        dense4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _log_activations(dense4)

    with tf.variable_scope('linear5') as scope:
        weights = _init_variable('weights', shape=[512, self.actions], stddev=4e-2)
        biases = _init_variable('biases', shape=[self.actions])
        output = tf.add(tf.matmul(dense4, weights), biases, name=scope.name)
        _log_activations(output)

    return output
