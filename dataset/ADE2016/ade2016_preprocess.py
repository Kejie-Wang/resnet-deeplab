import argparse
import os


def is_an_image(filename):
    suffix = ['.jpg', '.png']
    for suf in suffix:
        if filename.endswith(suf):
            return True
    return False


def list_all_images(data_dir):
    return sorted([
        filename for filename in os.listdir(data_dir) if is_an_image(filename)
    ])


def remove_suffix(filename):
    suffix_index = filename.rfind('.')
    if suffix_index != -1:
        filename = filename[0:suffix_index]
    return filename


def train_val_process(data_dir, out_dir):
    train_image_dir = os.path.join(data_dir, 'images', 'training')
    val_image_dir = os.path.join(data_dir, 'images', 'validation')
    train_label_dir = os.path.join(data_dir, 'annotations', 'training')
    val_label_dir = os.path.join(data_dir, 'annotations', 'validation')

    train_image = list_all_images(train_image_dir)
    val_image = list_all_images(val_image_dir)
    train_label = list_all_images(train_label_dir)
    val_label = list_all_images(val_label_dir)

    train_data = list(zip(train_image, train_label))
    val_data = list(zip(val_image, val_label))

    for image, label in train_data:
        assert remove_suffix(image) == remove_suffix(label)

    for image, label in val_data:
        assert remove_suffix(image) == remove_suffix(label)

    with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
        for image, label in train_data:
            image = os.path.join(train_image_dir, image)
            label = os.path.join(train_label_dir, label)
            f.write(image + ' ' + label + '\n')

    with open(os.path.join(out_dir, 'validation.txt'), 'w') as f:
        for image, label in val_data:
            image = os.path.join(val_image_dir, image)
            label = os.path.join(val_label_dir, label)
            f.write(image + ' ' + label + '\n')


def test_process(data_dir, out_dir):
    test_image_dir = os.path.join(data_dir, 'testing')

    test_image = list_all_images(test_image_dir)

    with open(os.path.join(out_dir, 'test.txt'), 'w') as f:
        for image in test_image:
            image = os.path.join(data_dir, image)
            f.write(image + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='ADE2016 dataset preprocessor.')

    parser.add_argument(
        '--data_path', help='The directory of train and validation data.')
    parser.add_argument(
        '--test_path', help='The directory of release testing data.')
    parser.add_argument(
        '--out_path', default='.', help='The Directory of output file.')

    args = parser.parse_args()
    if args.data_path is not None:
        train_val_process(args.data_path, args.out_path)
    if args.test_path is not None:
        test_process(args.test_path, args.out_path)


if __name__ == '__main__':
    main()
