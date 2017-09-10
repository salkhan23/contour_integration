# -------------------------------------------------------------------------------------------------
#  First attempt at diagonal contour completion. Needs cleanup
# Author: Salman Khan
# Date  : 04/09/17
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K

import alex_net_cont_int_models as cont_int_models
import alex_net_utils

reload(cont_int_models)
reload(alex_net_utils)


if __name__ == "__main__":
    plt.ion()
    K.clear_session()

    # 1. Load/Make the model
    # ----------------------
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # Gaussian Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "gaussian_multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        weights_type='enhance_and_suppress',
        n=25,
        sigma=6.0
    )
    # contour_integration_model.summary()

    # Define callback functions to get activations of L1 convolutional layer &
    # L2 contour integration layer
    l1_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    l2_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)

    # # Fragment
    # fragment = np.zeros((11, 11, 3))
    # fragment[:, (0, 3, 4, 5, 9, 10), :] = 255.0
    # alex_net_utils.plot_l1_and_l2_kernel_and_contour_fragment(contour_integration_model, 10, fragment)

    # 2. Meat
    # ----------------------

    tgt_filter_index = 54

    # Fragment is the target filter
    conv1_weights = K.eval(contour_integration_model.layers[1].weights[0])
    fragment = conv1_weights[:, :, :, tgt_filter_index]
    # Scale the filter to lie withing [0, 255]
    fragment = (fragment - fragment.min())*(255 / (fragment.max() - fragment.min()))
    use_smoothing = True

    img = np.zeros((227, 227, 3))
    loc_x = np.array([86,  97, 108, 119, 130])
    loc_y = np.array([100, 104, 108, 112, 116])

    new_img = alex_net_utils.tile_image(img, fragment, (loc_x, loc_y), rotate=False, gaussian_smoothing=False)

    plt.figure()
    plt.imshow(new_img / 255.0)

    new_img = new_img / 255.0
    alex_net_utils.plot_l1_and_l2_activations(new_img, l1_activations_cb, l2_activations_cb, tgt_filter_index)

    alex_net_utils.plot_l1_and_l2_kernel_and_contour_fragment(contour_integration_model, tgt_filter_index, fragment)



