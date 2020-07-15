import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


class DatasetCoordinator:
    def __init__(self, seed=2048):
        self.seed = seed

    def prepare_dataset(self, dataset, batch_size, repeat=False, shuffle=False, data_augment=None):
        if repeat:
            dataset = dataset.repeat()

        if shuffle:
            dataset = dataset.shuffle(self.seed)
            opt = tf.data.Options()
            opt.experimental_deterministic = False
            dataset = dataset.with_options(opt)

        if data_augment is not None:
            dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTO)

        return dataset

    def prepare_train_dataset(self, dataset, batch_size, data_augment=None):
        if data_augment is not None:
            dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.seed)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTO)
        return dataset

    def prepare_test_dataset(self, dataset, batch_size, repeat=False):
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTO)
        return dataset

    def concat_pseudo_label(self, train_dataset, test_dataset, pseudo_label):
        pseudo_label_dataset = tf.data.Dataset.from_tensor_slices(pseudo_label)
        test_dataset = test_dataset.zip((test_dataset, pseudo_label_dataset))
        dataset = train_dataset.concatenate(test_dataset)
        return dataset
