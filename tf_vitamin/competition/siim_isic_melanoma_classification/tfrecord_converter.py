import tensorflow as tf
from tf_vitamin.dataset.tfrecord_converter import BaseTfrecordConverter


class TfrecordConverter(BaseTfrecordConverter):
    def __init__(self, filename, record_format, img_size):
        super().__init__(filename, record_format, img_size)

    def parse_train_example(self, example):
        example = tf.io.parse_single_example(example, self.record_format)
        image = self.decode_image(example['image'])
        label = tf.cast(example['target'], tf.float32)
        return image, label

    def parse_test_example(self, example):
        example = tf.io.parse_single_example(example, self.record_format)
        image = self.decode_image(example['image'])
        return image
