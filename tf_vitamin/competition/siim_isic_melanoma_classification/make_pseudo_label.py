import os
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# 特徴量を記述するディクショナリを作成
from tf_vitamin.dataset.tfrecord_converter import TfrecordConverter, count_data_items

train_record_feature_desc = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'patient_id': tf.io.FixedLenFeature([], tf.int64),
    'sex': tf.io.FixedLenFeature([], tf.int64),
    'age_approx': tf.io.FixedLenFeature([], tf.int64),
    'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
    'source': tf.io.FixedLenFeature([], tf.int64),
    'target': tf.io.FixedLenFeature([], tf.int64)
}

test_record_feature_desc = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_name': tf.io.FixedLenFeature([], tf.string),
}


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class TestTfrecordConverter(TfrecordConverter):
    def __init__(self, filename, record_format, img_size):
        super(TestTfrecordConverter, self).__init__(filename, record_format, img_size)

    def parse_train_example(self, example):
        example = tf.io.parse_single_example(example, self.record_format)
        image = self.decode_image(example['image'])
        label = tf.cast(example['target'], tf.float32)
        return image, label

    def parse_test_example(self, example):
        example = tf.io.parse_single_example(example, self.record_format)
        image = self.decode_image(example['image'])
        return image


def create_tf_record(data_train, filename_train, data_test, filename_test):
    # 学習データをTFRecordに書き込む
    with tf.io.TFRecordWriter(filename_train) as writer:
        images, targets = data_train
        for image, target in zip(images, targets):
            ex = record2example(image, target)
            writer.write(ex.SerializeToString())

    # 評価データをTFRecordに書き込む
    with tf.io.TFRecordWriter(filename_test) as writer:
        images, targets = data_test
        for image, target in zip(images, targets):
            ex = record2example(image, target)
            writer.write(ex.SerializeToString())


def record2example(image, target):
    return tf.train.Example(features=tf.train.Features(feature={
        "image": _bytes_feature(image.tobytes()),
        "target": _float_feature(target)
    }))


AUTO = tf.data.experimental.AUTOTUNE


def run(tfrec_path, pseudo_label_path, output_filename):
    test_tfrecords = tf.io.gfile.glob(tfrec_path)
    print(f'test dataset size: {count_data_items(test_tfrecords)}')

    # ファイルの読み込み
    unlabeled_dataset = TestTfrecordConverter(test_tfrecords, test_record_feature_desc, 1024).convert2dataset(
        mode='test')

    pseudo_label = pd.read_csv(pseudo_label_path).target
    pseudo_label_dataset = tf.data.Dataset.from_tensor_slices(pseudo_label)
    unlabeled_dataset = unlabeled_dataset.zip((unlabeled_dataset, pseudo_label_dataset))

    # 学習データをTFRecordに書き込む
    with tf.io.TFRecordWriter(output_filename) as writer:
        for idx, data in enumerate(tfds.as_numpy(unlabeled_dataset)):
            image = data[0]
            target = data[1]

            example = record2example(image, target)
            writer.write(example.SerializeToString())

            if (idx + 1) % 1000:
                print(f'completed {idx + 1}.')


if __name__ == '__main__':
    data_dir = pathlib.Path(os.getcwd()) / 'data' / 'siim_isic_melanoma_classification'
    if not data_dir.exists():
        raise FileNotFoundError(f'Directory not found. {data_dir}')

    # KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
    tfrec_path = 'gs://kds-b0df39ea391018abfc80c95d79f90b2d362a510da49c600fae2b3501/tfrecords/test*.tfrec'
    pseudo_label_path = data_dir / 'sub_0-948.csv'
    output_path = str(data_dir / 'pseudo_dataset.tfrec')
    run(tfrec_path, pseudo_label_path, output_path)
