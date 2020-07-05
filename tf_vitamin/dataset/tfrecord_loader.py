import re
import tensorflow as tf
import numpy as np

from tf_vitamin.entity.image_config import ImageConfig


AUTO = tf.data.experimental.AUTOTUNE


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


class ImageDatasetLoader:
    def __init__(self, image_config: ImageConfig):
        self.orig_image_shape = image_config.orig_shape
        self.reshape_size = image_config.reshape_size

    def load_tfrecord(self, filename, labeled=True, ordered=False):
        ignore_order = tf.data.Options()
        if not ordered:
            # disable order, increase speed.
            ignore_order.experimental_deterministic = False

        dataset = tf.data.TFRecordDataset(
            filename, num_parallel_reads=AUTO)

        # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.with_options(ignore_order)
        if labeled:
            dataset = dataset.map(self.parse_labeled_example,
                                  num_parallel_calls=AUTO)
        else:
            dataset = dataset.map(self.parse_unlabeled_example,
                                  num_parallel_calls=AUTO)

        return dataset

    def parse_labeled_example(self, example):
        record_format = {
            # tf.string means byte string
            "image": tf.io.FixedLenFeature([], tf.string),
            # shape [] means single element
            "target": tf.io.FixedLenFeature([], tf.int64),
        }

        example = tf.io.parse_single_example(example, record_format)
        image = self.decode_image(example['image'])
        label = tf.cast(example['target'], tf.int32)
        return image, label  # returns a dataset of (image, label) pairs

    def parse_unlabeled_example(self, example):
        record_format = {
            # tf.string means byte string
            "image": tf.io.FixedLenFeature([], tf.string),
            # shape [] means single element
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, record_format)
        image = self.decode_image(example['image'])
        image_name = example['image_name']
        return image, image_name

    def decode_image(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)

        # convert image to floats in [0, 1] range
        image = tf.cast(image, tf.float32) / 255.0

        # explicit size needed for TPU
        image = tf.reshape(image, [*self.orig_image_shape])
        image = tf.image.resize(image, [*self.reshape_size])

        return image
