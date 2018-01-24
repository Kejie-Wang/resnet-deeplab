import tensorflow as tf

import models.resnet_utils as resnet_utils


def bottleneck(inputs,
               depth,
               stride,
               rate=1,
               scope=None,
               use_bounded_activations=False):
    with tf.variable_scope(scope, "bottleneck"):
        depth_in = inputs.shape[-1].value
        with tf.variable_scope("shortcut"):
            # increase the dimension and decrese size if needed.
            if depth != depth_in or stride > 1:
                shortcut = resnet_utils.conv2d_same(inputs, depth, 1, stride)
                shortcut = resnet_utils.batch_norm(
                    shortcut, is_training=False, activation_fn=None)
            else:
                shortcut = inputs
        with tf.variable_scope('conv1'):
            residual = resnet_utils.conv2d_same(
                inputs, num_outputs=depth / 4, kernel_size=1, stride=stride)
        with tf.variable_scope('batch_norm1'):
            residual = resnet_utils.batch_norm(
                residual, activation_fn=tf.nn.relu, is_training=False)
        with tf.variable_scope('conv2'):
            if rate == 1:
                residual = resnet_utils.conv2d_same(
                    residual, num_outputs=depth / 4, kernel_size=3, stride=1)
            else:  # rate is not equal 1, use astrous convolution.
                residual = resnet_utils.atrous_conv2d_same(
                    residual, num_outputs=depth / 4, kernel_size=3, rate=rate)
        with tf.variable_scope('batch_norm2'):
            residual = resnet_utils.batch_norm(
                residual, activation_fn=tf.nn.relu, is_training=False)
        with tf.variable_scope('conv3'):
            residual = resnet_utils.conv2d_same(
                residual, num_outputs=depth, kernel_size=1, stride=1)
        with tf.variable_scope('batch_norm3'):
            residual = resnet_utils.batch_norm(
                residual, activation_fn=tf.nn.relu, is_training=False)
        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)
    return output


def resnet_deeplab(inputs,
                   num_units,
                   is_training=False,
                   scope=None,
                   reuse=None):
    assert len(num_units) == 4
    net = inputs
    with tf.variable_scope(
            scope, default_name='reset_deeplab', reuse=reuse) as sc:
        with tf.variable_scope('root_block'):
            with tf.variable_scope('conv1'):
                net = resnet_utils.conv2d_same(
                    net, num_outputs=64, kernel_size=7, stride=2)
            with tf.variable_scope('batch_norm1'):
                net = resnet_utils.batch_norm(net, activation_fn=tf.nn.relu)
            with tf.variable_scope('pool1'):
                net = resnet_utils.max_pooling(net, kernel_size=3, stride=2)
        with tf.variable_scope('block1'):
            for i in range(num_units[0]):
                net = bottleneck(net, 256, 1, scope='unit%d' % (i + 1))
        with tf.variable_scope('block2'):
            net = bottleneck(net, 512, 2, scope='unit1')
            for i in range(1, num_units[1]):
                net = bottleneck(net, 512, 1, scope='unit%d' % (i + 1))
        with tf.variable_scope('block3'):
            for i in range(num_units[2]):
                net = bottleneck(
                    net, 1024, 1, rate=2, scope='unit%d' % (i + 1))
        with tf.variable_scope('block4'):
            for i in range(num_units[3]):
                net = bottleneck(
                    net, 2048, 1, rate=4, scope='unit%d' % (i + 1))
    return net


def resnet50_deeplab(inputs):
    resnet_deeplab(inputs, num_units=[3, 4, 6, 3], scope="resnet50_deeplab")


def resnet101_deeplab(inputs):
    resnet_deeplab(inputs, num_units=[3, 4, 23, 3], scope="resnet101_deeplab")


def resnet152_deeplab(inputs):
    resnet_deeplab(inputs, num_units=[3, 8, 36, 3], scope="resnet152_deeplab")
