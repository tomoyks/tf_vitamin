import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


class DatasetLoader:
    def __init__(self, dataset_loader, seed=2048):
        self.dataset_loader = dataset_loader
        self.seed = seed

    def get_training_dataset(self, file_names, data_augment, batch_size, ):
        dataset = self.dataset_loader.load_tfrecord(file_names, labeled=True)
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        dataset = dataset.repeat()  # the training dataset must repeat for several epochs
        dataset = dataset.shuffle(self.seed)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def get_test_dataset(self, file_names, batch_size, ordered=False):
        dataset = self.dataset_loader.load_tfrecord(file_names, labeled=False, ordered=ordered)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
        return dataset
