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
    image, label = dataset.next_batch()

    model = resnet_deeplab.Resnet101Deeplab(image, args.num_classes)

    restore_var = tf.global_variables()

    prediction = model.prediction
    # upsampling the prediction with the same size with label
    prediction = tf.image.resize_bilinear(prediction, tf.shape(image)[1:3])
    prediction = tf.argmax(prediction, axis=3)
    # flatten the prediction and label.
    prediction = tf.reshape(prediction, [-1])
    label = tf.reshape(label, [-1])

    boolean_mask = tf.less(label, args.num_classes)
    prediction = tf.boolean_mask(prediction, boolean_mask)
    label = tf.boolean_mask(label, boolean_mask)

    # weights = tf.cast(tf.less(label, args.num_classes), tf.float32)
    mean_iou, update_op = tf.metrics.mean_iou(
        label, prediction, num_classes=args.num_classes)

    loader = tf.train.Saver(var_list=restore_var)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        loader.restore(sess, args.model_path)

        for i in range(dataset.num_examples):
            sess.run(update_op)
            if i % 10 == 0:
                INFO(i / dataset.num_examples)
                INFO('Mean IoU: {:.3f}'.format(mean_iou.eval(session=sess)))

        INFO('Mean IoU: {:.3f}'.format(mean_iou.eval(session=sess)))


if __name__ == '__main__':
    main()
