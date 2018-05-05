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

from contour_integration_models.alex_net import masked_models as cont_int_models
import alex_net_utils
import li_2006_routines

reload(cont_int_models)
reload(alex_net_utils)
reload(li_2006_routines)

np.random.seed(7)  # Set the random seed for reproducibility


def optimize_contour_enhancement_layer_weights(
        model, tgt_filt_idx, frag, contour_generator_cb,
        n_runs, learning_rate=0.00025, offset=0, optimize_type='both', axis=None):
    """
    Optimize the l2 kernel (contour integration) weights corresponding to the specified target L1 kernel.
    A loss function is defined that compares the model l2 gain (L2 activation /L1 activation) and compares with the
    expected contour integration gain from Li-2006.

    :param model: contour integration model
    :param tgt_filt_idx:
    :param frag:
    :param contour_generator_cb:
    :param n_runs: Number of loops to iterate over. If n_runs is < 5, input images for each run are shown.
    :param learning_rate: THe learning rate (the size of the step in the gradient direction)
    :param offset: the pixel offset by which each row should be shifted by as it moves away from the center row.
         Used by diagonal contour optimization.
    :param optimize_type: Optimize over [length, spacing, or both(Default)]
    :param axis: Figure axis on which the loss function over time should be plotted. If None, a new figure
         is created. Default = None.

    :return: None
    """

    # Validate input parameters
    valid_optimize_type = ['length', 'spacing', 'both']
    if optimize_type.lower() not in valid_optimize_type:
        raise Exception("Invalid optimization type specified. Valid = [length, spacing, or both(Default)")
    optimize_type = optimize_type.lower()

    # Some Initialization
    tgt_n_loc = 27  # neuron looking @ center of RF
    tgt_n_visual_rf_start = tgt_n_loc * 4  # stride length of L1 conv Layer

    # 1. Extract the neural data to match
    # -----------------------------------
    with open('.//data//neuro_data//Li2006.pickle', 'rb') as handle:
        data = pickle.load(handle)

    expected_gains = []
    contour_len_arr = []
    relative_colinear_dist_arr = []

    if optimize_type == 'length' or optimize_type == 'both':
        expected_gains = np.append(expected_gains, data['contour_len_avg_gain'])
        contour_len_arr = data["contour_len_avg_len"]

    if optimize_type == 'spacing' or optimize_type == 'both':
        expected_gains = np.append(expected_gains, data['contour_separation_avg_gain'])
        relative_colinear_dist_arr = np.array(data["contour_separation_avg_rcd"])

    # 2. Setup the optimization problem
    # ------------------------------------------------------
    l1_output_cb = model.layers[1].output
    l2_output_cb = model.layers[2].output
    input_cb = model.input

    # Mean Square Error Loss
    current_gains = l2_output_cb[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] / \
        (l1_output_cb[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] + 1e-8)

    loss = (expected_gains - current_gains) ** 2 / expected_gains.shape[0]

    # Callbacks for the weights (learnable parameters)
    w_cb = model.layers[2].raw_kernel
    b_cb = model.layers[2].bias

    # Gradients of weights and bias wrt to the loss function
    grads = K.gradients(loss, [w_cb, b_cb])
    grads = [gradient / (K.sqrt(K.mean(K.square(gradient))) + 1e-8) for gradient in grads]

    iterate = K.function([input_cb], [loss, grads[0], grads[1], l1_output_cb, l2_output_cb])

    # 3. Initializations
    # -------------------
    smooth_edges = True
    frag_len = frag.shape[0]

    contour_spacing_contour_len = 7  # Reference uses length 7 for relative spacing part

    # 4. Main Loop
    # ---------------------------
    old_loss = 10000000
    losses = []
    # ADAM Optimization starting parameters
    m_w = 0
    v_w = 0

    m_b = 0
    v_b = 0

    for run_idx in range(n_runs):

        # Create test set of images (new set for each run)
        # ------------------------------------------------
        images = []

        # Contour Lengths
        if optimize_type == 'length' or optimize_type == 'both':

            for c_len in contour_len_arr:

                test_image = np.zeros((227, 227, 3))
                test_image_len = test_image.shape[0]

                # Background Tiles
                bg_tile_locations = alex_net_utils.get_background_tiles_locations(
                    frag_len, test_image_len, offset, 0, tgt_n_visual_rf_start)

                test_image = alex_net_utils.tile_image(
                    test_image,
                    frag,
                    bg_tile_locations,
                    rotate=True,
                    gaussian_smoothing=smooth_edges
                )

                # Place contour in image
                contour_tile_locations = contour_generator_cb(
                    frag_len,
                    bw_tile_spacing=0,
                    cont_len=c_len,
                    cont_start_loc=tgt_n_visual_rf_start,
                    row_offset=offset
                )
                contour_tile_locations = np.array(contour_tile_locations)

                test_image = alex_net_utils.tile_image(
                    test_image,
                    frag,
                    contour_tile_locations.T,
                    rotate=False,
                    gaussian_smoothing=smooth_edges
                )

                # Image preprocessing
                # -------------------
                # # zero_mean and 1 standard deviation
                # test_image -= np.mean(test_image, axis=0)
                # test_image /= np.std(test_image, axis=0)

                # Normalize pixels to be in the range [0, 1]
                test_image = test_image / 255.0

                # Theano back-end expects channel first format
                test_image = np.transpose(test_image, (2, 0, 1))
                images.append(test_image)

        # Contour Spacing
        if optimize_type == 'spacing' or optimize_type == 'both':

            spacing_bw_tiles_arr = np.floor(relative_colinear_dist_arr * frag_len) - frag_len

            for spacing in spacing_bw_tiles_arr:

                spacing = int(spacing)

                test_image = np.zeros((227, 227, 3))
                test_image_len = test_image.shape[0]

                # Background Tiles
                bg_tile_locations = alex_net_utils.get_background_tiles_locations(
                    frag_len, test_image_len, offset, spacing, tgt_n_visual_rf_start)

                test_image = alex_net_utils.tile_image(
                    test_image,
                    frag,
                    bg_tile_locations,
                    rotate=True,
                    gaussian_smoothing=smooth_edges
                )

                # Place contour in image
                contour_tile_locations = contour_generator_cb(
                    frag_len,
                    bw_tile_spacing=spacing,
                    cont_len=contour_spacing_contour_len,
                    cont_start_loc=tgt_n_visual_rf_start,
                    row_offset=offset
                )
                contour_tile_locations = np.array(contour_tile_locations)

                test_image = alex_net_utils.tile_image(
                    test_image,
                    frag,
                    contour_tile_locations.T,
                    rotate=False,
                    gaussian_smoothing=smooth_edges
                )

                # Image preprocessing
                test_image = test_image / 255.0  # Bring test_image back to the [0, 1] range.
                test_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format

                images.append(test_image)

        images = np.stack(images, axis=0)

        # Plot the generated images
        # -------------------------
        if n_runs <= 5:
            f = plt.figure()
            num_images_per_row = 5

            if optimize_type == 'both':
                num_rows = 2
            else:
                num_rows = 1

            for img_idx, img in enumerate(images):
                display_img = np.transpose(img, (1, 2, 0))
                f.add_subplot(num_rows, num_images_per_row, img_idx + 1)
                plt.imshow(display_img)

        # Now iterate
        loss_value, grad_w, grad_b, l1_out, l2_out = iterate([images])
        print("%d: loss %s" % (run_idx, loss_value.mean()))

        w, b = model.layers[2].get_weights()

        if loss_value.mean() > old_loss:
            # step /= 2.0
            # print("Lowering step value to %f" % step)
            pass
        else:
            m_w = 0.9 * m_w + (1 - 0.9) * grad_w
            v_w = 0.999 * v_w + (1 - 0.999) * grad_w ** 2

            new_w = w - learning_rate * m_w / (np.sqrt(v_w) + 1e-8)

            m_b = 0.9 * m_b + (1 - 0.9) * grad_b
            v_b = 0.999 * v_b + (1 - 0.999) * grad_b ** 2

            new_b = b - learning_rate * m_b / (np.sqrt(v_b) + 1e-8)

            # Print Contour Enhancement Gains
            print("Contour Enhancement Gain %s" %
                  (l2_out[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] /
                   l1_out[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc]))

            model.layers[2].set_weights([new_w, new_b])

        old_loss = loss_value.mean()
        losses.append(loss_value.mean())

    # At the end of simulation plot loss vs iteration
    if axis is None:
        f, axis = plt.subplots()
    axis.plot(range(n_runs), losses, label='learning rate = %0.8f' % learning_rate)
    font_size = 20
    axis.set_xlabel("Iteration", fontsize=font_size)
    axis.set_ylabel("Loss", fontsize=font_size)
    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)


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

    # Use the same scale for plotting the kernel
    max_v_opt = max(opt_w.max(), abs(opt_w.min()))
    max_v_start = max(start_w.max(), abs(start_w.min()))

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(start_w[tgt_filt_idx, :, :] * mask[tgt_filt_idx, :, :], vmin=-max_v_start, vmax=max_v_start)
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.tick_params(labelsize=20)
    # plt.title("Start weights & bias=%0.4f" % start_b[tgt_filt_idx])

    f.add_subplot(1, 2, 2)
    plt.imshow(mask[tgt_filt_idx, :, :] * opt_w[tgt_filt_idx, :, :], vmin=-max_v_opt, vmax=max_v_opt)
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.tick_params(labelsize=20)
    # plt.title("Best weights & bias=%0.4f" % opt_b[tgt_filt_idx])


if __name__ == "__main__":
    plt.ion()
    K.clear_session()

    # ------------------------------------------------------------------------------------
    # 1. Build the model
    # -----------------------------------------------------------------------------------
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

    # Define callback functions to get activations of L1 convolutional layer &
    # L2 contour integration layer
    l1_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    l2_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)

    # Store the start weights & bias for comparison later
    start_weights, start_bias = contour_integration_model.layers[2].get_weights()

    # # ----------------------------------------------------------------------------------
    # # 2. Vertical Contours
    # # ----------------------------------------------------------------------------------
    # print('#'*25, ' Vertical Contours ', '#'*25)
    # tgt_filter_idx = 10
    # fragment = np.zeros((11, 11, 3))
    # fragment[:, (0, 3, 4, 5, 9, 10), :] = 255.0
    #
    # # Note: optimize type both no longer works with updated tensorflow/keras library
    # optimize_contour_enhancement_layer_weights(
    #     contour_integration_model,
    #     tgt_filter_idx,
    #     fragment,
    #     alex_net_utils.vertical_contour_generator,
    #     n_runs=1000,
    #     offset=0,
    #     optimize_type='length',
    #     learning_rate=0.00025
    # )
    #
    # plot_optimized_weights(contour_integration_model, tgt_filter_idx, start_weights, start_bias)
    #
    # # Plot Gain vs Contour Length after Optimization
    # li_2006_routines.main_contour_length_routine(
    #     fragment,
    #     l1_activations_cb,
    #     l2_activations_cb,
    #     alex_net_utils.vertical_contour_generator,
    #     tgt_filter_idx,
    #     smoothing=True,
    #     row_offset=0,
    #     n_runs=100,
    # )
    #
    # # Plot Gain vs Contour Spacing after Optimization
    # li_2006_routines.main_contour_spacing_routine(
    #     fragment,
    #     l1_activations_cb,
    #     l2_activations_cb,
    #     alex_net_utils.vertical_contour_generator,
    #     tgt_filter_idx,
    #     smoothing=True,
    #     row_offset=0,
    #     n_runs=100)

    # --------------------------------------------------------------------------------------------
    # 3. Horizontal Contours
    # --------------------------------------------------------------------------------------------
    print('#'*25, ' Horizontal Contours ', '#'*25)
    tgt_filter_idx = 5
    fragment = np.zeros((11, 11, 3))
    fragment[0:6, :, :] = 255.0

    optimize_contour_enhancement_layer_weights(
        contour_integration_model,
        tgt_filter_idx,
        fragment,
        alex_net_utils.horizontal_contour_generator,
        n_runs=1000,
        offset=0,
        optimize_type='length',
        learning_rate=0.00025
    )

    plot_optimized_weights(contour_integration_model, tgt_filter_idx, start_weights, start_bias)

    # Plot Gain vs Contour Length after Optimization
    li_2006_routines.main_contour_length_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.horizontal_contour_generator,
        tgt_filter_idx,
        smoothing=True,
        row_offset=0,
        n_runs=100,
    )

    # Plot Gain vs Contour Spacing after Optimization
    li_2006_routines.main_contour_spacing_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.horizontal_contour_generator,
        tgt_filter_idx,
        smoothing=True,
        row_offset=0,
        n_runs=100)
