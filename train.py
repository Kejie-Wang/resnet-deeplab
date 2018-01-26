from __future__ import print_function

import argparse
import math
import os
import time

import numpy as np
import tensorflow as tf

import dataset.reader as reader
import models.resnet_deeplab as resnet_deeplab
from utils.util import INFO, WARN, FAIL


def get_arguments():
    """
    define all configurable params
    """
    parser = argparse.ArgumentParser(description="Resnet-deeplab train model.")

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='The batch size of each iteration.')
    parser.add_argument(
        '--epoch_size', type=int, default=50, help='The epoch size of train.')
    parser.add_argument(
        '--print_step', type=int, default=50, help='The number of print step.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset',
        help='The directory of dataset')
    parser.add_argument(
        '--pretrain_model_path',
        type=str,
        default=os.path.join('pretrain_model', 'model.ckpt'),
        help='The path of pretrained model.')
    parser.add_argument(
        '--saved_model_dir',
        type=str,
        default=os.path.join('saved_model'),
        help='The path of saved model.')
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='Directory of log.')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=150,
        help='The number of class in the dataset.')
    parser.add_argument(
        '--input_size',
        type=str,
        default='512x512',
        help='The size of input image.')
    parser.add_argument(
        '--is_training',
        action='store_true',
        help=
        'Whether to update the mean and variance in batch normalization layer.'
    )
    parser.add_argument(
        '--not_restore_fc',
        action='store_true',
        help='Whether to restore the last fully connected layer.')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default='0.0005',
        help='Regularisation parameter for L2-loss.')
    parser.add_argument(
        '--lr', type=float, default='1e-4', help='The base learning rate.')
    parser.add_argument(
        '--power', type=float, default='0.8', help='Decay for learning rate.')
    parser.add_argument(
        '--momentum',
        type=float,
        default='0.9',
        help='Momentum component of the optimiser.')

    args = parser.parse_args()

    return args


def print_arguments(args):
    INFO('*' * 15, 'arguments', '*' * 15)
    for key, value in vars(args).items():
        INFO(key, ":", value)
    INFO('*' * 40)


def main():
    args = get_arguments()
    print_arguments(args)

    height, width = map(int, args.input_size.split('x'))

    with tf.device('/cpu:0'):
        with tf.name_scope('image_reader'):
            dataset = reader.Dataset(args.data_dir, subset='train')
            dataset.make_batch(
                args.batch_size, input_size=(height, width), shuffle=True)

            images, labels = dataset.next_batch()

    model = resnet_deeplab.Resnet101Deeplab(images, args.num_classes,
                                            args.is_training)

    predictions = model.prediction

    restore_var = [
        v for v in tf.global_variables()
        if 'fc' not in v.name or not args.not_restore_fc
    ]
    all_trainable = [
        v for v in tf.global_variables()
        if 'beta' not in v.name or 'gamma' not in v.name
    ]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable
                      if 'fc' not in v.name]  # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable
                      if 'weights' in v.name]  # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'bias' in v.name]  # lr *20.0

    with tf.name_scope('loss'):
        prediction_size = tf.stack(predictions.shape[1:3])
        labels = tf.image.resize_nearest_neighbor(labels, prediction_size)
        labels = tf.squeeze(
            labels, squeeze_dims=[3])  # squeeze the last (color) channel.
        labels = tf.reshape(labels, [-1])
        # filter the indices which labels exceeds than num class.
        boolean_mask = tf.less(labels, args.num_classes)
        labels = tf.boolean_mask(labels, boolean_mask)
        labels = tf.cast(labels, tf.int32)
        predictions = tf.reshape(predictions, [-1, args.num_classes])
        predictions = tf.boolean_mask(predictions, boolean_mask)
        entropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=predictions, labels=labels))
        l2_loss = tf.add_n([
            args.weight_decay * tf.nn.l2_loss(v)
            for v in tf.trainable_variables()
            if 'weights' in v.name or 'biases' in v.name
        ])
        loss = entropy_loss + l2_loss

        tf.summary.scalar('entropy_loss', entropy_loss)
        tf.summary.scalar('l2_loss', l2_loss)
        tf.summary.scalar('loss', loss)

    num_steps = math.ceil(
        dataset.num_examples / args.batch_size) * args.epoch_size

    with tf.name_scope('optimization'):
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        lr = tf.scalar_mul(args.lr,
                           tf.pow((1 - step_ph / num_steps), args.power))
        opt_conv = tf.train.MomentumOptimizer(lr, args.momentum)
        opt_fc_w = tf.train.MomentumOptimizer(lr * 10.0, args.momentum)
        opt_fc_b = tf.train.MomentumOptimizer(lr * 20.0, args.momentum)

        grads = tf.gradients(loss,
                             conv_trainable + fc_w_trainable + fc_b_trainable)
        grads_conv = grads[:len(conv_trainable)]
        grads_fc_w = grads[len(conv_trainable):(
            len(conv_trainable) + len(fc_w_trainable))]
        grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

        train_op_conv = opt_conv.apply_gradients(
            zip(grads_conv, conv_trainable))
        train_op_fc_w = opt_fc_w.apply_gradients(
            zip(grads_fc_w, fc_w_trainable))
        train_op_fc_b = opt_fc_b.apply_gradients(
            zip(grads_fc_b, fc_b_trainable))

        optimization = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

    if tf.gfile.Exists(args.log_dir):
        tf.gfile.DeleteRecursively(args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)

    # saver for saving and restoring model.
    loader = tf.train.Saver(var_list=restore_var)
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        loader.restore(sess, args.pretrain_model_path)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(args.log_dir + '/train',
                                             sess.graph)

        start_time = time.time()
        for i in range(num_steps):
            summary, _ = sess.run(
                [merged, optimization], feed_dict={
                    step_ph: i
                })
            train_writer.add_summary(summary, i)
            if i % args.print_step == 0 and i > 0:
                pred, entropy_loss_val, l2_loss_val = sess.run(
                    [predictions, entropy_loss, l2_loss])
                duration = (time.time() - start_time) / i
                INFO(
                    'step {:d} entropy_loss = {:.3f} l2_loss= {:.3f} total_loss= {:.3f} ({:.3f} sec/step)'.
                    format(i, entropy_loss_val, l2_loss_val,
                           entropy_loss_val + l2_loss_val, duration))
                if np.isnan(entropy_loss_val + l2_loss_val):
                    for var in all_trainable:
                        val = sess.run(var)
                        if np.sum(np.isnan(val)) > 0:
                            FAIL(var)
                            FAIL(val)
            if i % 1000 == 0:
                saver.save(
                    sess,
                    os.path.join(args.saved_model_dir, "model.ckpt"),
                    global_step=i)
        train_writer.close()


if __name__ == '__main__':
    main()
