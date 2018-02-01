from __future__ import print_function

import argparse
import os
import tensorflow as tf

import models.resnet_deeplab as resnet_deeplab
import dataset.reader as reader
from utils.util import INFO, WARN, FAIL


def get_arguments():
    parser = argparse.ArgumentParser(description="Resnet-deeplab train model.")

    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset',
        help='The directory of dataset')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=150,
        help='The number of class in the dataset.')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join('saved_model', 'model.ckpt'),
        help='The path of saved model.')

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()

    dataset = reader.Dataset(args.data_dir, subset='validation')
    dataset.make_batch(batch_size=1, epoch_size=1, shuffle=False)
    image_batch, labels_batch = dataset.next_batch()

    model = resnet_deeplab.Resnet101Deeplab(image_batch, args.num_classes)

    restore_var = tf.global_variables()

    # upsampling the predictions with the same size with labels
    predictions = tf.image.resize_bilinear(model.prediction,
                                           tf.shape(image_batch)[1:3])
    predictions = tf.argmax(predictions, axis=3)
    # flatten the predictions and labels.
    predictions = tf.reshape(predictions, [-1])
    labels = tf.reshape(labels_batch, [-1])

    boolean_mask = tf.less(labels, args.num_classes)
    predictions = tf.boolean_mask(predictions, boolean_mask)
    labels = tf.boolean_mask(labels, boolean_mask)

    predictions = tf.cast(predictions, tf.int32)
    labels = tf.cast(labels, tf.int32)
    corr_pred_pixel = tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels), tf.int32))
    wrong_pred_pixel = tf.reduce_sum(
        tf.cast(tf.not_equal(predictions, labels), tf.int32))

    # weights = tf.cast(tf.less(labels, args.num_classes), tf.float32)
    mean_iou, update_op = tf.metrics.mean_iou(
        labels, predictions, num_classes=args.num_classes)

    with tf.name_scope('loss'):
        prediction_size = tf.shape(model.prediction)[1:3]
        labels = tf.image.resize_nearest_neighbor(labels_batch,
                                                  prediction_size)
        labels = tf.squeeze(
            labels, squeeze_dims=[3])  # squeeze the last (color) channel.
        labels = tf.reshape(labels, [-1])
        # filter the indices which labels exceeds than num class.
        boolean_mask = tf.less(labels, args.num_classes)
        labels = tf.boolean_mask(labels, boolean_mask)
        labels = tf.cast(labels, tf.int32)
        predictions = tf.reshape(model.prediction, [-1, args.num_classes])
        predictions = tf.boolean_mask(predictions, boolean_mask)
        entropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=predictions, labels=labels))
        l2_loss = tf.add_n([
            0.005 * tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if 'weights' in v.name or 'biases' in v.name
        ])
        loss = entropy_loss + l2_loss

    all_corr_pred = 0
    all_wrong_pred = 0

    loader = tf.train.Saver(var_list=restore_var)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        path = tf.train.latest_checkpoint('saved_model')
        loader.restore(sess, path)

        for i in range(dataset.num_examples):
            _, corr, wrong = sess.run([update_op, corr_pred_pixel, wrong_pred_pixel])
            all_corr_pred += corr
            all_wrong_pred += wrong
            # print(sess.run([entropy_loss, l2_loss, loss]))
            if i % 100 == 0:
                INFO(i / dataset.num_examples)
                INFO('Mean IoU: {:.3f}'.format(mean_iou.eval(session=sess)))
                INFO('correct:', all_corr_pred, 'all:',
                     all_corr_pred + all_wrong_pred)

        INFO('Mean IoU: {:.3f}'.format(mean_iou.eval(session=sess)))
        INFO('Pixel accuracy: {:.3f}'.format(all_corr_pred /
                                             (all_corr_pred + all_wrong_pred)))


if __name__ == '__main__':
    main()
