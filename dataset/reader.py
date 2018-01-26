import cv2
import numpy as np
import os
import tensorflow as tf

IMG_MEAN = np.array(
    (104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


# Reference: https://github.com/DrSleep/tensorflow-deeplab-resnet/blob/master/deeplab_resnet/image_reader.py
def read_image_and_mask(image_filename, mask_filename):
    image_contents = tf.read_file(image_filename)
    mask_contents = tf.read_file(mask_filename)

    image = tf.image.decode_jpeg(image_contents, channels=3)
    image_r, image_g, image_b = tf.split(
        axis=2, num_or_size_splits=3, value=image)
    image = tf.cast(
        tf.concat(axis=2, values=[image_b, image_g, image_r]),
        dtype=tf.float32)
    mask = tf.image.decode_png(mask_contents, channels=1)

    # subtract the image mean.
    image -= IMG_MEAN

    return image, mask


def read_labeled_image_list(data_dir, filename):
    images = []
    masks = []
    with open(filename, 'r') as f:
        for line in f:
            image_filename, mask_filename = line.strip('\n').split()
            images.append(os.path.join(data_dir, image_filename))
            masks.append(os.path.join(data_dir, mask_filename))

    return images, masks


def image_scaling(image, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.
    Args:
      image: Training image to scale.
      label: Segmentation mask to scale.
    """

    scale = tf.random_uniform(
        [1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    image = tf.image.resize_images(image, new_shape)
    label = tf.image.resize_nearest_neighbor(
        tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return image, label


def image_mirroring(image, label):
    """
    Randomly mirrors the images.
    Args:
      image: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    distort_left_right_random = tf.random_uniform(
        [1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    image = tf.reverse(image, mirror)
    label = tf.reverse(label, mirror)

    return image, label


def random_crop_and_pad_image_and_labels(image,
                                         label,
                                         crop_h,
                                         crop_w,
                                         ignore_label=255):
    """
    Randomly crop and pads the input images.
    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0,
                                                tf.maximum(
                                                    crop_h, image_shape[0]),
                                                tf.maximum(
                                                    crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4])
    image_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    image_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))

    return image_crop, label_crop


class Dataset(object):
    """
    Deeplab dataset reader
    """

    def __init__(self, data_dir, subset='train'):
        self.data_dir = data_dir
        self.subset = subset

        data_filename = self.get_filename()
        self.images, self.masks = read_labeled_image_list(
            data_dir, data_filename)

    def get_filename(self):
        if self.subset in ['train', 'validation']:
            return os.path.join(self.data_dir, self.subset + '.txt')
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def make_batch(self,
                   batch_size=None,
                   input_size=None,
                   epoch_size=None,
                   shuffle=False):

        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.masks))
        dataset =dataset.map(read_image_and_mask)
        
        # if the input size if not none
        # then do data argumentation with
        # random image scale and mirroring
        if input_size is not None:
            height, width = input_size
            def preprocess(image, mask):
                image, mask = image_scaling(image, mask)
                image, mask = image_mirroring(image, mask)
                image, mask = random_crop_and_pad_image_and_labels(
                    image, mask, height, width)

                return image, mask

            dataset = dataset.map(preprocess)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(epoch_size)

        self.dataset = dataset
        self.iter = dataset.make_one_shot_iterator()

    def next_batch(self):
        return self.iter.get_next()

    @property
    def num_examples(self):
        return len(self.images)
