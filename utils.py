# -------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------
import keras
from keras.datasets import mnist

MNIST_NUM_CLASSES = 10


def get_mnist_data(sample_idx=0):
    """
    :param sample_idx: index of sample to return

    :return: data_train, label_train, data_test, label_test, data_sample, label_sample
    """
    (data_train, label_train), (data_test, label_test) = mnist.load_data()
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], data_train.shape[2], 1)
    data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], data_train.shape[2], 1)

    data_train = data_train.astype('float32')
    data_test = data_test.astype('float32')

    data_train /= 255
    data_test /= 255

    label_train = keras.utils.to_categorical(label_train, MNIST_NUM_CLASSES)
    label_test = keras.utils.to_categorical(label_test, MNIST_NUM_CLASSES)

    # a Single sample to test prediction on
    data_sample = data_test[sample_idx].reshape(
        1, data_test[sample_idx].shape[0], data_test[sample_idx].shape[1], data_test[sample_idx].shape[2])
    label_sample = label_test[sample_idx]

    return data_train, label_train, data_test, label_test, data_sample, label_sample
