# -------------------------------------------------------------------------------------------------
# The basic idea of this script is to match the gain from contour enhancement as see in
# [Li, Piech and Gilbert - 2006 - Contour Saliency in Primary Visual Cortex] in the Gaussian
# Multiplicative model.
#
# First a cost function the compares the mean square error between the expected contour
# enhancement gain and from the model is defined. Gradient descent is used to find the
# best fit alpha and bias terms that minimizes the loss. A stack of images containing contours of
# various lengths and spacing are jointly optimized.
#
# TODO: General for other target filters.
# TODO: Include contour separation figures as well
#
# Author: Salman Khan
# Date  : 27/08/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras.backend as K

from contour_integration_models.alex_net import masked_models as cont_int_models
import alex_net_utils

reload(cont_int_models)
reload(alex_net_utils)

np.random.seed(7)  # Set the random seed for reproducibility

if __name__ == "__main__":

    plt.ion()
    K.clear_session()

    # 1. Build the model
    # ---------------------------------------------------------------------
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # Gaussian Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "gaussian_multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        weights_type='enhance',
        n=25,
        sigma=6.0
    )
    # contour_integration_model.summary()

    # Define callback functions to get activations of L1 convolutional layer &
    # L2 contour integration layer
    l1_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    l2_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)

    # 2. Extract the neural data we would like to match
    # ---------------------------------------------------------------------
    with open('.//data//neuro_data//Li2006.pickle', 'rb') as handle:
        data = pickle.load(handle)

    expected_gains = data['contour_len_avg_gain']

    # 2. Create a bank of images - a batch of multiple contour lengths
    # ---------------------------------------------------------------------
    tgt_filter_idx = 10
    tgt_neuron_loc = 27

    # Fragment
    fragment = np.zeros((11, 11, 3))
    fragment[:, (0, 3, 4, 5, 9, 10), :] = 255.0

    contour_len_arr = np.arange(1, 11, 2)

    images_list = []
    for c_len in contour_len_arr:
        # print("Creating image with contour of length %d" % c_len)
        test_image = np.zeros((227, 227, 3))

        contour_tile_locations = alex_net_utils.vertical_contour_generator(
            fragment.shape[0],
            bw_tile_spacing=0,
            cont_len=c_len,
            cont_start_loc=tgt_neuron_loc * 4
        )
        contour_tile_locations = np.array(contour_tile_locations)

        test_image = alex_net_utils.tile_image(
            test_image,
            fragment,
            contour_tile_locations.T,
            rotate=False,
            gaussian_smoothing=False
        )

        # Image preprocessing
        test_image = test_image / 255.0  # Bring test_image back to the [0, 1] range.
        test_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format
        images_list.append(test_image)

    images_arr = np.stack(images_list, axis=0)

    # # For sanity check the generated images:
    # for image_idx in range(images_arr.shape[0]):
    #     print("display image at index %d" % image_idx)
    #     display_img = np.transpose(images_arr[image_idx, :, :, :], (1, 2, 0))
    #     plt.figure()
    #     plt.imshow(display_img)

    # 3. Build a cost function that minimizes the distance between the expected gain and actual gain
    # ----------------------------------------------------------------------------------------------
    l1_output_cb = contour_integration_model.layers[1].output
    l2_output_cb = contour_integration_model.layers[2].output
    input_cb = contour_integration_model.input

    # Mean Square Error Loss
    current_gain = l2_output_cb[:, tgt_filter_idx, tgt_neuron_loc, tgt_neuron_loc] / \
        (l1_output_cb[:, tgt_filter_idx, tgt_neuron_loc, tgt_neuron_loc] + 1e-5)
    loss = (expected_gains - current_gain) ** 2 / expected_gains.shape[0]

    alpha_cb = contour_integration_model.layers[2].alpha
    bias_cb = contour_integration_model.layers[2].bias

    # Gradients of alpha and bias wrt to the loss function
    grads = K.gradients(loss, [alpha_cb, bias_cb])
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  # Normalize gradients, prevents too large or small gradients

    iterate = K.function([input_cb], [loss, grads])

    # 4. Now iterate to find the best fit alpha and bias
    # --------------------------------------------------
    step = 0.025
    old_loss = 10000000

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel('Loss')

    l1_act = 0
    l2_act = 0 + 1e-5

    for r_idx in range(1000):
        loss_value, grad_value = iterate([images_arr])
        print("%d: loss %s" % (r_idx, loss_value))

        alpha, bias = contour_integration_model.layers[2].get_weights()

        if old_loss < loss_value.mean():
            step /= 2.0
            print("Lowering step value to %0.4f" % step)
        else:
            new_alpha = alpha - grad_value[0] * step
            new_bias = bias - grad_value[1] * step

            print("New alpha=%0.4f, New bias =%0.4f" % (alpha[10, :, :], bias[10, :, :]))
            contour_integration_model.layers[2].set_weights([new_alpha, new_bias])

            # Print Contour Enhancement gains
            l1_act = np.array(l1_activations_cb([images_arr, 0]))
            l1_act = np.squeeze(np.array(l1_act), axis=0)
            l2_act = np.array(l2_activations_cb([images_arr, 0]))
            l2_act = np.squeeze(np.array(l2_act), axis=0)
            print("Contour Enhancement Gain %s" % (l2_act[:, 10, 27, 27] / l1_act[:, 10, 27, 27]))

        old_loss = loss_value.mean()

        ax.plot(r_idx, loss_value.mean(), marker='.', color='blue')
        plt.show()

    alpha, bias = contour_integration_model.layers[2].get_weights()
    print("Best values of alpha %0.4f,and bias=%0.4f" % (alpha[10, :, :], bias[10, :, :]))

    # 5. Plot the gain curve and compare with neuroData
    # --------------------------------------------------
    plt.figure()
    plt.plot(data['contour_len_avg_len'], expected_gains, label='From Ref')
    plt.plot(data['contour_len_avg_len'], l2_act[:, 10, 27, 27] / l1_act[:, 10, 27, 27], label='model')
    plt.legend()
    plt.xlabel("Contour Length")
    plt.ylabel("Contour Enhancement Gain")
