# -------------------------------------------------------------------------------------------------
#  Various utility function used by many Files
#
# Author: Salman Khan
# Date  : 03/09/17
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

import keras.backend as K


def get_activation_cb(model, layer_idx):
    """
    Return a callback that returns the output activation of the specified layer
    The callback expects input cb([input_image, 0])

    :param model:
    :param layer_idx:
    :return: callback function.
    """
    get_layer_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_idx].output]
    )

    return get_layer_output


def plot_activations(img, l1_act_cb, l2_act_cb, tgt_filt_idx):
    """
    Plot 2 figures:
    [1] the test image,
    [2] l1_activations, l2_activations and the difference between the activations

    :param img:
    :param l1_act_cb:
    :param l2_act_cb:
    :param tgt_filt_idx:

    :return: Handles of the two images created
    """

    f1 = plt.figure()
    plt.imshow(img)

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    l1_act = np.array(l1_act_cb([img, 0]))
    l2_act = np.array(l2_act_cb([img, 0]))

    l1_act = np.squeeze(l1_act, axis=0)
    l2_act = np.squeeze(l2_act, axis=0)

    l1_act = l1_act[0, tgt_filt_idx, :, :]
    l2_act = l2_act[0, tgt_filt_idx, :, :]

    min_l2_act = l1_act.min()
    max_l2_act = l2_act.max()

    f2 = plt.figure()
    f2.add_subplot(1, 3, 1)
    plt.imshow(l1_act, cmap='seismic', vmin=min_l2_act, vmax=max_l2_act)
    plt.title('L1 Conv Layer Activation @ idx %d' % tgt_filt_idx)
    plt.colorbar(orientation='horizontal')
    plt.grid()

    f2.add_subplot(1, 3, 2)
    plt.imshow(l2_act, cmap='seismic', vmin=min_l2_act, vmax=max_l2_act)
    plt.title('L2 Contour Integration Layer Activation @ idx %d' % tgt_filt_idx)
    plt.colorbar(orientation='horizontal')
    plt.grid()

    f2.add_subplot(1, 4, 4)
    plt.imshow(l2_act - l1_act, cmap='seismic')
    plt.colorbar(orientation='horizontal')
    plt.title("Difference")
    plt.grid()

    return f1, f2
