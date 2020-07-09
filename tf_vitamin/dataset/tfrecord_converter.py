import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


class TfrecordConverter:
    def __init__(self, filename, record_format):
        self.filename = filename
        self.record_format = record_format

    def convert2dataset(self, ordered=False):
        dataset = tf.data.TFRecordDataset(
            self.filename, num_parallel_reads=AUTO)

        ignore_order = tf.data.Options()
        if not ordered:
            # disable order, increase speed.
            ignore_order.experimental_deterministic = False
            # uses data as soon as it streams in, rather than in its original order
            dataset = dataset.with_options(ignore_order)

        dataset = dataset.map(self.parse_example,
                              num_parallel_calls=AUTO)
        return dataset

    def parse_example(self, example):
        example = tf.io.parse_single_example(example, self.record_format)
        image = self.decode_image(example['image'])
        image_name = example['image_name']
        return image, image_name

    def decode_image(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)

        # convert image to floats in [0, 1] range
        image = tf.cast(image, tf.float32) / 255.0

        # explicit size needed for TPU
        # image = tf.reshape(image, [*self.orig_image_shape])
        # image = tf.image.resize(image, [*self.reshape_size])

        return image
