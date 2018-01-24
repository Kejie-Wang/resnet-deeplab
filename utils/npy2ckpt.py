import argparse
import numpy as np
import os
import tensorflow as tf

from models.resnet_deeplab import resnet101_deeplab

FLAGS = None
params = None


def char_add(char, inc=1):
    return chr(ord(char) + inc)


def load_npy(npy_path):
    global params
    params = np.load(npy_path, encoding='latin1').item()


def extract_number(prefix, postfix, string):
    beg = string.find(prefix)
    end = -1
    if beg != -1:
        end = string.find(postfix, beg)
    if beg == -1 or end == -1:
        return "-1"
    return string[beg + len(prefix):end]


def retrieve_data(name):
    key = None
    data = None
    if name.find('root_block') != -1:
        if name.find('conv') != -1:
            key = 'conv1'
        elif name.find('batch_norm') != -1:
            key = 'bn_conv1'
    else:
        block_num = int(extract_number("block", '/', name))
        unit_num = int(extract_number('unit', '/', name))

        block_name = str(block_num + 1) + ('a' if unit_num == 1 else 'b')
        if unit_num != 1 and (block_num == 2 or block_num == 3):
            block_name += str(unit_num - 1)
        if block_num != -1 and unit_num != -1:
            # a shortcut variable
            if name.find('shortcut') != -1:
                if name.find('weight') != -1:
                    key = 'res' + block_name + '_branch1'
                else:
                    key = 'bn' + block_name + '_branch1'
            elif name.find('conv') != -1:
                conv_num = int(extract_number('conv', '/', name))
                key = 'res' + block_name + '_branch2' + char_add(
                    'a', conv_num - 1)
            elif name.find('batch_norm') != -1:
                batch_norm_num = int(extract_number('batch_norm', '/', name))
                key = 'bn' + block_name + '_branch2' + char_add(
                    'a', batch_norm_num - 1)

    # retrieve data
    if params.get(key) is not None:
        var_name_list = [
            'weights', 'moving_mean', 'moving_variance', 'beta', 'gamma'
        ]
        for var in var_name_list:
            if name.find(var) != -1:
                data = params.get(key).get(var)
                break
    return data


def main():
    if not os.path.exists(FLAGS.npy_path):
        print("npy model path does NOT exist!")
        return
    load_npy(FLAGS.npy_path)

    images = tf.constant(0, tf.float32, shape=[1, 512, 512, 3])
    model = resnet101_deeplab(images)

    var_list = tf.global_variables()

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for var in var_list:
            data = retrieve_data(var.name)
            if data is None:
                print("Can NOT find", var.name, 'in the npy model.')
            else:
                sess.run(var.assign(data))
            print(var.name)

        saver = tf.train.Saver(var_list=var_list)
        if not os.path.exists(FLAGS.ckpt_path):
            os.makedirs(FLAGS.ckpt_path)
        model_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')
        saver.save(sess, model_path, write_meta_graph=False)
        print("Save the model into", model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='npy to ckpt converter.')

    parser.add_argument('npy_path', type=str, help='The path of npy model.')
    parser.add_argument(
        '--ckpt_path',
        default=os.path.join('.', 'pretrain_model'),
        type=str,
        help='The path of output ckpt model.')

    FLAGS, unparsed = parser.parse_known_args()
    main()
