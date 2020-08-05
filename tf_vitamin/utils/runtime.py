import tensorflow as tf
import numpy as np


def get_distribution_strategy():
    """
    Detect hardware, return appropriate distribution strategy.
    """

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        
        if tf.__version__ >= '2.3.0':
            strategy = tf.distribute.TPUStrategy(tpu)
        else:
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow.
        # Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)

    return strategy, tpu


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
