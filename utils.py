# -------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------
import keras
from keras.datasets import mnist
import keras.backend as K
import matplotlib.pyplot as plt

import numpy as np

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


def display_layer_activations(model, layer_idx, data_sample):
    """
    Display the activation volume of the specified layer. Each feature map is displayed in a separate subplot.
    Expected format of layers is [b, r, c, ch]

    :param model:
    :param layer_idx:
    :param data_sample:
    :return:
    """

    # Define a function to get the activation volume
    get_layer_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_idx].output]
    )

    # Get the activations in a usable format
    act_volume = np.asarray(get_layer_output(
        [data_sample, 0],  # second input specifies the learning phase 0=output, 1=training
    ))

    # Reshape the activations, the casting above adds another dimension
    act_volume = act_volume.reshape(
        act_volume.shape[1],
        act_volume.shape[2],
        act_volume.shape[3],
        act_volume.shape[4]
    )

    max_ch = np.int(np.round(np.sqrt(act_volume.shape[-1])))

    f = plt.figure()

    for ch_idx in range(act_volume.shape[-1]):
        f.add_subplot(max_ch, max_ch, ch_idx + 1)
        plt.imshow(act_volume[0, :, :, ch_idx], cmap='Greys')

    f.suptitle("Feature maps of layer @ idx %d: %s" % (layer_idx, model.layers[layer_idx].name))
