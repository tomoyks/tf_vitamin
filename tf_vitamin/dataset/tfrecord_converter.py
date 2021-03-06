import re
from abc import abstractmethod, ABCMeta
from enum import Enum
import tensorflow as tf
import numpy as np

AUTO = tf.data.experimental.AUTOTUNE


class ConvertMode(Enum):
    TRAIN = 'train'
    TEST = 'test'


class BaseTfrecordConverter(metaclass=ABCMeta):
    def __init__(self, filename, record_format, img_size):
        self.filename = filename
        self.record_format = record_format
        self.img_size = img_size

    def convert2dataset(self, mode, ordered=False):
        dataset = tf.data.TFRecordDataset(
            self.filename, num_parallel_reads=AUTO)

        ignore_order = tf.data.Options()
        if not ordered:
            # disable order, increase speed.
            ignore_order.experimental_deterministic = False
            # uses data as soon as it streams in, rather than in its original order
            dataset = dataset.with_options(ignore_order)

        if mode == ConvertMode.TRAIN or mode == ConvertMode.TRAIN.value:
            dataset = dataset.map(self.parse_train_example,
                                  num_parallel_calls=AUTO)
        elif mode == ConvertMode.TEST or mode == ConvertMode.TEST.value:
            dataset = dataset.map(self.parse_test_example,
                                  num_parallel_calls=AUTO)

        return dataset

    @abstractmethod
    def parse_train_example(self, example):
        pass

    @abstractmethod
    def parse_test_example(self, example):
        pass

    def decode_image(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)

        # convert image to floats in [0, 1] range
        image = tf.cast(image, tf.float32) / 255.0

        image = tf.reshape(image, [self.img_size, self.img_size, 3])
        image = tf.image.resize(image, [self.img_size, self.img_size])

        return image


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
