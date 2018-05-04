# -------------------------------------------------------------------------------------------------
# Use the model learned from learned_lateral_weights.py for further exploration
#
# Author: Salman Khan
# Date  : 29/06/17
# TODO: Current not working, needs updating
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model

from contour_integration_models.mnist_cnn_classifer import learned_lateral_weights
import utils
reload(utils)  # Force Reload utils to pick up latest changes
reload(learned_lateral_weights)


if __name__ == "__main__":

    plt.ion()

    # 1. Get MNIST Data
    # --------------------------------------------
    x_train, y_train, x_test, y_test, x_sample, y_sample = utils.get_mnist_data()

    # 2. Load the Model
    # -------------------------------------------
    contour_integration_model = load_model(
        "learned_lateral_weights_3x3_overlap.hf",
        custom_objects={'ContourIntegrationLayer': learned_lateral_weights.ContourIntegrationLayer()}
    )
    # contour_integration_model.summary()

    # 3. Show activation volumes before and after the lateral connections
    # -----------------------------------------------------------------------------------
    ff_volume = utils.display_layer_activations(contour_integration_model, 0, x_sample)
    lat_volume = utils.display_layer_activations(contour_integration_model, 1, x_sample)

    # How the lateral connections are changing the activation volume
    diff = lat_volume - ff_volume

    max_ch = np.int(np.round(np.sqrt(diff.shape[-1])))
    f = plt.figure()

    for ch_idx in range(diff.shape[-1]):
        f.add_subplot(max_ch, max_ch, ch_idx + 1)
        plt.imshow(diff[0, :, :, ch_idx], cmap='Greys')

    f.suptitle("How the lateral connections are changing the feed-forward input")

    # 4. Show Feed-forward and lateral Filters
    # --------------------------------------------------------------------
    weights_ff = contour_integration_model.layers[0].get_weights()[0]
    weights_l = contour_integration_model.layers[1].get_weights()[0]

    # Normalize the display
    max_filt_value = np.max([np.max(weights_ff), np.max(weights_l)])
    min_filt_value = np.min([np.min(weights_ff), np.min(weights_l)])

    fig_dim = np.int(np.round(np.sqrt(weights_ff.shape[-1])))

    # Plot the feed forward Filters
    fig = plt.figure()
    fig.suptitle("FF convolution Filters")
    for ch_idx in range(weights_ff.shape[-1]):
        fig.add_subplot(fig_dim, fig_dim, ch_idx + 1)
        plt.imshow(weights_ff[:, :, 0, ch_idx], cmap='Greys', vmin=min_filt_value, vmax=max_filt_value)

    # Plot the lateral connections filters
    fig = plt.figure()
    fig.suptitle("Lateral Connections Filters")

    for ch_idx in range(weights_l.shape[-1]):
        fig.add_subplot(fig_dim, fig_dim, ch_idx + 1)
        plt.imshow(weights_l[:, :, ch_idx], cmap='Greys', vmin=min_filt_value, vmax=max_filt_value)
