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
# Finds weights for  Multiplicative model
#
# TODO: General for other target filters.
# TODO: Include contour separation images as well
#
# Author: Salman Khan
# Date  : 10/09/17
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


def optimize_contour_enhancement_layer_weights(
        model, tgt_filt_idx, frag, contour_generator_cb, n_runs, learning_rate=0.00025):
    """

    :param model:
    :param tgt_filt_idx:
    :param frag:
    :param contour_generator_cb:
    :param learning_rate: THe learning rate (the size of the step in the gradient direction)
    :param n_runs: Number of loops to iterate over

    :return:
    """
    tgt_n_loc = 27  # neuron looking @ center of RF
    tgt_n_visual_rf_start = tgt_n_loc * 4

    # 1. Extract the neural data to match
    # -----------------------------------
    with open('.//neuro_data//Li2006.pickle', 'rb') as handle:
        data = pickle.load(handle)

    expected_gains = data['contour_len_avg_gain']

    # 2. Setup the optimization problem
    # ------------------------------------------------------
    l1_output_cb = model.layers[1].output
    l2_output_cb = model.layers[2].output
    input_cb = model.input

    # Mean Square Error Loss
    current_gain = l2_output_cb[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] / \
        (l1_output_cb[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] + 1e-5)

    loss = (expected_gains - current_gain) ** 2 / expected_gains.shape[0]

    # Callbacks for the weights
    w_cb = model.layers[2].raw_kernel
    b_cb = model.layers[2].bias

    # Gradients of weights and bias wrt to the loss function
    grads = K.gradients(loss, [w_cb, b_cb])
    grads = [gradient / (K.sqrt(K.mean(K.square(gradient))) + 1e-5) for gradient in grads]

    iterate = K.function([input_cb], [loss, grads[0], grads[1], l1_output_cb, l2_output_cb])

    # 3. Loop to get optimized weights
    # --------------------------------
    smooth_edges = True
    frag_len = frag.shape[0]

    # Loop Initialization
    old_loss = 10000000
    losses = []
    # ADAM Optimization starting parameters
    m_w = 0
    v_w = 0

    m_b = 0
    v_b = 0

    # Main Loop
    for run_idx in range(n_runs):

        # Create test set of images (new set for each run)
        contour_len_arr = np.arange(1, 11, 2)
        images = []

        for c_len in contour_len_arr:

            test_image = np.zeros((227, 227, 3))
            n_tiles = test_image.shape[0] // frag_len

            # Place randomly oriented fragments in the image
            start_x = range(
                tgt_n_visual_rf_start - (n_tiles / 2) * frag_len,
                tgt_n_visual_rf_start + (n_tiles / 2 + 1) * frag_len,
                frag_len,
            )
            start_y = np.copy(start_x)

            start_x = np.repeat(start_x, len(start_x))
            start_y = np.tile(start_y, len(start_y))

            test_image = alex_net_utils.tile_image(
                test_image,
                frag,
                (start_x, start_y),
                rotate=True,
                gaussian_smoothing=smooth_edges
            )

            # Place contour in image
            start_x, start_y = contour_generator_cb(
                frag_len,
                bw_tile_spacing=0,
                cont_len=c_len,
                cont_start_loc=tgt_n_visual_rf_start
            )

            test_image = alex_net_utils.tile_image(
                test_image,
                fragment,
                (start_x, start_y),
                rotate=False,
                gaussian_smoothing=smooth_edges
            )

            # Image preprocessing
            test_image = test_image / 255.0  # Bring test_image back to the [0, 1] range.
            test_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format
            images.append(test_image)

        images = np.stack(images, axis=0)

        # # Plot the generated images
        # f = plt.figure()
        # for img_idx, img in enumerate(images):
        #     display_img = np.transpose(img, (1, 2, 0))
        #     f.add_subplot(2, 3, img_idx + 1)
        #     plt.imshow(display_img)

        # now iterate
        loss_value, grad_w, grad_b, l1_out, l2_out = iterate([images])
        print("%d: loss %s" % (run_idx, loss_value.mean()))

        w, b = model.layers[2].get_weights()

        if loss_value.mean() > old_loss:
            # step /= 2.0
            # print("Lowering step value to %f" % step)
            pass
        else:
            m_w = 0.9 * m_w + (1 - 0.9) * grad_w
            v_w = 0.999 * v_w + (1 - 0.999) * grad_w**2

            new_w = w - learning_rate * m_w / (np.sqrt(v_w) + 1e-8)

            m_b = 0.9 * m_b + (1 - 0.9) * grad_b
            v_b = 0.999 * v_b + (1 - 0.999) * grad_b**2

            new_b = b - learning_rate * m_b / (np.sqrt(v_b) + 1e-8)

            # Print Contour Enhancement Gains
            print("Contour Enhancement Gain %s" %
                  (l2_out[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] /
                   l1_out[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc]))

            model.layers[2].set_weights([new_w, new_b])

        old_loss = loss_value.mean()
        losses.append(loss_value.mean())

    # At the end of simulation plot loss vs iteration
    plt.figure()
    plt.plot(range(n_runs), losses)


def plot_optimized_weights(model, tgt_filt_idx, start_w, start_b):
    """

    :param model:
    :param tgt_filt_idx:
    :param start_w:
    :param start_b:
    :return:
    """
    mask = K.eval(model.layers[2].mask)  # mask does not change
    opt_w, opt_b = model.layers[2].get_weights()

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(start_w[tgt_filt_idx, :, :] * mask[tgt_filt_idx, :, :])
    plt.title("Start weights & bias=%0.4f" % start_b[tgt_filt_idx])

    f.add_subplot(1, 2, 2)
    plt.imshow(mask[tgt_filt_idx, :, :] * opt_w[tgt_filt_idx, :, :])
    plt.title("Best weights & bias=%0.4f" % opt_b[tgt_filt_idx])


if __name__ == "__main__":
    plt.ion()
    K.clear_session()

    # 1. Build the model
    # ---------------------------------------------------------------------
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        n=25,
        activation='relu'
    )
    # contour_integration_model.summary()

    # Store the start weights & bias for comparison later
    start_weights, start_bias = contour_integration_model.layers[2].get_weights()

    # 2. Vertical Contours
    # ----------------------------------------------------------------------
    tgt_filter_idx = 10
    fragment = np.zeros((11, 11, 3))
    fragment[:, (0, 3, 4, 5, 9, 10), :] = 255.0

    optimize_contour_enhancement_layer_weights(
        contour_integration_model,
        tgt_filter_idx,
        fragment,
        alex_net_utils.vertical_contour_generator,
        200,
    )

    plot_optimized_weights(contour_integration_model, tgt_filter_idx, start_weights, start_bias)

    #  Horizontal Contours
    # ------------------------------------------------------------------------
    tgt_filter_idx = 5
    fragment = np.zeros((11, 11, 3))
    fragment[0:6, :, :] = 255.0

    optimize_contour_enhancement_layer_weights(
        contour_integration_model,
        tgt_filter_idx,
        fragment,
        alex_net_utils.horizontal_contour_generator,
        200,
    )

    plot_optimized_weights(contour_integration_model, tgt_filter_idx, start_weights, start_bias)











    #
    # # 5 Plot the gain curve and compare with neuroData
    # # # -------------------------------------------------
    # # plt.figure()
    # # plt.plot(data['contour_len_avg_len'], expected_gains, label='From Ref')
    # # plt.plot(data['contour_len_avg_len'], l2_act[:, 10, 27, 27] / l1_act[:, 10, 27, 27], label='model')
    # # plt.legend()
    # # plt.xlabel("Contour Length")
    # # plt.ylabel("Contour Enhancement Gain")
