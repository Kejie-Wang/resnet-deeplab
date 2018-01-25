from __future__ import print_function

import argparse
import math
import os
import time

import numpy as np
import tensorflow as tf

import dataset.reader as reader
import models.resnet_deeplab as resnet_deeplab


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
        default=os.path.join('saved_model', 'model.ckpt'),
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
        default='500x500',
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
        '--lr', type=float, default='1e-6', help='The learning rate.')

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()

    height, width = map(int, args.input_size.split('x'))

    with tf.device('/cpu:0'):
        with tf.name_scope('image_reader'):
            dataset = reader.Dataset(args.data_dir, subset='train')
            dataset.make_batch((height, width), args.batch_size, shuffle=True)

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
        lables = tf.image.resize_nearest_neighbor(labels, prediction_size)
        labels = tf.squeeze(
            lables, squeeze_dims=[3])  # squeeze the last (color) channel.
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

    with tf.name_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        optimization = optimizer.minimize(
            loss, var_list=tf.trainable_variables())

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

        train_writer = tf.summary.FileWriter(args.log_dir + '/train',
                                             sess.graph)

        num_steps = math.ceil(
            dataset.num_examples / args.batch_size) * args.epoch_size
        start_time = time.time()
        for i in range(num_steps):
            sess.run(optimization)
            if i % args.print_step == 0 and i > 0:
                pred, loss_val = sess.run([predictions, loss])
                duration = (time.time() - start_time) / i
                print('step {:d} loss = {:.3f}, ({:.3f} sec/step)'.format(
                    i, loss_val, duration))

        saver.save(sess, os.path.join(args.saved_model_dir, "model.ckpt"))


if __name__ == '__main__':
    main()
