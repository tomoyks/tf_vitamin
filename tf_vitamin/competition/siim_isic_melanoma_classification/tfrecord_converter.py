import tensorflow as tf
from tf_vitamin.dataset.tfrecord_converter import TfrecordConverter


class TfrecordConverter(TfrecordConverter):
    def __init__(self, filename, record_format):
        super().__init__(filename, record_format)

    def parse_train_example(self, example):
        image = self.decode_image(example['image'])
        label = tf.cast(example['target'], tf.int32)
        return image, label

    def parse_test_example(self, example):
        image = self.decode_image(example['image'])
        image_name = example['image_name']
        return image, image_name
