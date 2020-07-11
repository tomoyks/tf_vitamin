from abc import abstractmethod, ABCMeta
from enum import Enum
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


class ConvertMode(Enum):
    TRAIN = 'train'
    TEST = 'test'


class TfrecordConverter(metaclass=ABCMeta):
    def __init__(self, filename, record_format):
        self.filename = filename
        self.record_format = record_format

    def convert2dataset(self, mode, ordered=False):
        dataset = tf.data.TFRecordDataset(
            self.filename, num_parallel_reads=AUTO)

        ignore_order = tf.data.Options()
        if not ordered:
            # disable order, increase speed.
            ignore_order.experimental_deterministic = False
            # uses data as soon as it streams in, rather than in its original order
            dataset = dataset.with_options(ignore_order)

        if mode == ConvertMode.TRAIN:
            dataset = dataset.map(self.parse_train_example,
                                  num_parallel_calls=AUTO)
        elif mode == ConvertMode.TEST:
            dataset = dataset.map(self.parse_test_example,
                                  num_parallel_calls=AUTO)

        return dataset

    @abstractmethod
    def parse_train_example(self, example):
        pass

    @abstractmethod
    def parse_test_example(self, example):
        pass

    @staticmethod
    def decode_image(image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)

        # convert image to floats in [0, 1] range
        image = tf.cast(image, tf.float32) / 255.0

        # explicit size needed for TPU
        # image = tf.reshape(image, [*self.orig_image_shape])
        # image = tf.image.resize(image, [*self.reshape_size])

        return image
