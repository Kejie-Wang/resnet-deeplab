import tensorflow as tf


def conv2d_same(inputs, num_outputs, kernel_size, stride, biased=False):
    depth_in = inputs.shape[-1].value
    kernel = tf.get_variable(
        'weights', shape=[kernel_size, kernel_size, depth_in, num_outputs])
    outputs = tf.nn.conv2d(
        inputs, kernel, strides=[1, stride, stride, 1], padding='SAME')
    if biased:
        bias = tf.get_variable('biases', shape=[num_outputs])
        outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def atrous_conv2d_same(inputs, num_outputs, kernel_size, rate=1, biased=False):
    depth_in = inputs.shape[-1].value
    kernel = tf.get_variable(
        'weights', shape=[kernel_size, kernel_size, depth_in, num_outputs])
    outputs = tf.nn.atrous_conv2d(inputs, kernel, rate=rate, padding='SAME')
    if biased:
        bias = tf.get_variable('biases', shape=[num_outputs])
        outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def max_pooling(inputs, kernel_size, stride):
    return tf.nn.max_pool(
        inputs,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME')


def batch_norm(inputs, activation_fn=None, is_training=False):
    depth_in = inputs.shape[-1].value
    mean = tf.get_variable(
        'moving_mean',
        shape=[depth_in],
        dtype=tf.float32,
        trainable=is_training)
    variance = tf.get_variable(
        'moving_variance',
        shape=[depth_in],
        dtype=tf.float32,
        trainable=is_training)
    beta = tf.get_variable('beta', shape=[depth_in], dtype=tf.float32)
    gamma = tf.get_variable('gamma', shape=[depth_in], dtype=tf.float32)
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, variance_epsilon=0.001)
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs
