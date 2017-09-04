# -------------------------------------------------------------------------------------------------
# The basic idea of this script is to match the gain from contour enhancement as see in
# [Li, Piech and Gilbert - 2006 - Contour Saliency in Primary Visual Cortex] in the Multiplicative
# model.
#
# First a cost function the compares the mean square error between the expected contour
# enhancement gain and from the model is defined. Gradient descent is used to find the
# best fit weights and bias terms that minimizes the loss. A stack of images containing contours of
# various lengths and spacing are jointly optimized.
#
# TODO: General for other target filters.
# TODO: Include contour separation images as well
#
# Author: Salman Khan
# Date  : 27/08/17
# -------------------------------------------------------------------------------------------------

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras.backend as K

import alex_net_cont_int_models as cont_int_models
import alex_net_utils

reload(cont_int_models)
reload(alex_net_utils)

np.random.seed(7)  # Set the random seed for reproducibility


if __name__ == "__main__":
    plt.ion()

    # 1. Build the model
    # ---------------------------------------------------------------------
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # Gaussian Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        weights_type='enhance',
        n=25,
    )
    # contour_integration_model.summary()

    # Define callback functions to get activations of L1 convolutional layer &
    # L2 contour integration layer
    l1_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    l2_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)

    # 2. Extract the neural data we would like to match
    # ---------------------------------------------------------------------
    with open('.//neuro_data//Li2006.pickle', 'rb') as handle:
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

        start_x, start_y = alex_net_utils.vertical_contour_generator(
            fragment.shape[0],
            bw_tile_spacing=0,
            cont_len=c_len,
            cont_start_loc=tgt_neuron_loc * 4
        )

        test_image = alex_net_utils.tile_image(
            test_image,
            fragment,
            (start_x, start_y),
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

    weights_cb = contour_integration_model.layers[2].raw_kernel
    bias_cb = contour_integration_model.layers[2].bias

    # Gradients of weights and bias wrt to the loss function
    grads = K.gradients(loss, [weights_cb, bias_cb])
    grads = [gradient / (K.sqrt(K.mean(K.square(gradient))) + 1e-5) for gradient in grads]

    iterate = K.function([input_cb], [loss, grads[0], grads[1]])

    # 4. Now iterate to find the best fit alpha and bias
    # --------------------------------------------------
    step = 0.0025
    old_loss = 10000000

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel('Loss')

    l1_act = 0
    l2_act = 0 + 1e-5

    for r_idx in range(200):
        loss_value, grad_w, grad_b = iterate([images_arr])
        grad_value = [grad_w, grad_b]
        print("%d: loss %s" % (r_idx, loss_value))

        weights, bias = contour_integration_model.layers[2].get_weights()

        if old_loss < loss_value.mean():
            step /= 2.0
            print("Lowering step value to %0.4f" % step)
        else:
            new_weights = weights - grad_value[0] * step
            new_bias = bias - grad_value[1] * step

            # Print Contour Enhancement Gains
            l1_act = np.array(l1_activations_cb([images_arr, 0]))
            l1_act = np.squeeze(np.array(l1_act), axis=0)
            l2_act = np.array(l2_activations_cb([images_arr, 0]))
            l2_act = np.squeeze(np.array(l2_act), axis=0)
            print("Contour Enhancement Gain %s" % (l2_act[:, 10, 27, 27] / l1_act[:, 10, 27, 27]))

            contour_integration_model.layers[2].set_weights([new_weights, new_bias])

        old_loss = loss_value.mean()

        ax.plot(r_idx, loss_value.mean(), marker='.', color='blue')

    plt.figure()
    weights, bias = contour_integration_model.layers[2].get_weights()

    plt.imshow(weights[10, :, :])
    plt.title("Best weights, with bias=%0.4f" % bias[10])

    # 5. Plot the gain curve and compare with neuroData
    # --------------------------------------------------
    plt.figure()
    plt.plot(data['contour_len_avg_len'], expected_gains, label='From Ref')
    plt.plot(data['contour_len_avg_len'], l2_act[:, 10, 27, 27] / l1_act[:, 10, 27, 27], label='model')
    plt.legend()
    plt.xlabel("Contour Length")
    plt.ylabel("Contour Enhancement Gain")
