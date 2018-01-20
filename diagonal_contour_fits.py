from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras.backend as K

import alex_net_cont_int_models as cont_int_models
import gabor_fits
import alex_net_utils
import alex_net_hyper_param_search_multiplicative as multiplicative_model
import alex_net_cont_int_complex_bg as complex_bg

reload(cont_int_models)
reload(alex_net_utils)
reload(multiplicative_model)
reload(complex_bg)

np.random.seed(10)  # Set the random seed for reproducibility


def get_l1_filter_orientation_and_offset(tgt_filt, tgt_filt_idx, show_plots=True):
    """
    Given a Target AlexNet L1 Convolutional Filter, fit to a 2D spatial Gabor. Use this to as
    the orientation of the filter and calculate the row shift offset to use when tiling fragments.
    This offset represents the shift in pixels to use for each row  as you move away from the
    center row. Thereby allowing contours for the target filter to be generated.

    Raises an exception if no best fit parameters are found for any of the channels of the target
    filter.

    :param show_plots:
    :param tgt_filt_idx:
    :param tgt_filt:

    :return: optimal orientation, row offset.
    """
    tgt_filt_len = tgt_filt.shape[0]

    best_fit_params_list = gabor_fits.find_best_fit_2d_gabor(tgt_filt)

    # Plot the best fit params
    if show_plots:
        gabor_fits.plot_kernel_and_best_fit_gabors(tgt_filt, tgt_filt_idx, best_fit_params_list)

    # Remove all empty entries
    best_fit_params_list = [params for params in best_fit_params_list if params is not None]
    if not best_fit_params_list:
        # raise Exception("Optimal Params could not be found")
        return np.NaN, np.NaN

    # Find channel with highest energy (Amplitude) and use its preferred orientation
    # Best fit parameters: x0, y0, theta, amp, sigma, lambda1, psi, gamma
    best_fit_params_list = np.array(best_fit_params_list)
    amp_arr = best_fit_params_list[:, 3]
    amp_arr = np.abs(amp_arr)
    max_amp_idx = np.argmax(amp_arr)

    theta_opt = best_fit_params_list[max_amp_idx, 2]

    # TODO: Fix me - Explain why this is working
    # TODO: How to handle horizontal (90) angles
    # # Convert the orientation angle into a y-offset to be used when tiling fragments
    contour_angle = theta_opt + 90.0  # orientation is of the Gaussian envelope with is orthogonal to
    # # sinusoid carrier we are interested in.
    # contour_angle = np.mod(contour_angle, 180.0)

    # if contour_angle >= 89:
    #     contour_angle -= 180  # within the defined range of tan

    # contour_angle = contour_angle * np.pi / 180.0
    # offset = np.int(np.ceil(tgt_filter_len / np.tan(contour_angle)))
    row_offset = np.int(np.ceil(tgt_filt_len / np.tan(np.pi - contour_angle * np.pi / 180.0)))

    # print("L1 kernel %d, optimal orientation %0.2f(degrees), vertical offset of tiles %d"
    #       % (tgt_filt_idx, theta_opt, row_offset))

    return theta_opt, row_offset


def diagonal_contour_generator(frag_len, row_offset, bw_tile_spacing, cont_len, cont_start_loc):
    """
    Generate the start co-ordinates of fragment squares that form a diagonal contour of
    the specified length & row_offset at the specified location

    :param frag_len:
    :param row_offset: row_offset to complete contours, found from the orientation of the fragment.
        See get_l1_filter_orientation_and_offset.
    :param bw_tile_spacing:
    :param cont_len:
    :param cont_start_loc: Start visual RF location of center neuron.

    :return: start_x_arr, start_y_arr locations of fragments that form the contour
    """
    frag_spacing = frag_len + bw_tile_spacing

    # 1st dimension stays the same
    start_x = range(
        cont_start_loc - (cont_len / 2) * frag_spacing,
        cont_start_loc + ((cont_len / 2) + 1) * frag_spacing,
        frag_spacing
    )

    # If there is nonzero spacing between tiles, the offset needs to be updated
    if bw_tile_spacing:
        row_offset = np.int(frag_spacing / np.float(frag_len) * row_offset)

    # 2nd dimension shifts with distance from the center row
    if row_offset is not 0:
        start_y = range(
            cont_start_loc - (cont_len / 2) * row_offset,
            cont_start_loc + ((cont_len / 2) + 1) * row_offset,
            row_offset
        )
    else:
        start_y = np.copy(start_x)

    return start_x, start_y


def plot_visual_rf(location, img, axis=None):
    """

    :param location:
    :param img:
    :param axis:

    :return: the extracted component of visual field
    """
    if axis is None:
        f, axis = plt.subplots()

    stride = 4
    tgt_filt_len = 11
    neuron_visual_rf = img[
        location[0] * stride: location[0] * stride + tgt_filt_len,
        location[1] * stride: location[1] * stride + tgt_filt_len, :
    ]

    axis.imshow(neuron_visual_rf)

    return neuron_visual_rf


if __name__ == "__main__":
    plt.ion()
    K.clear_session()

    # 1. Build the model
    # ---------------------------------------------------------------------
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # # Masked Multiplicative Model
    # contour_integration_model = cont_int_models.build_contour_integration_model(
    #     "masked_multiplicative",
    #     "trained_models/AlexNet/alexnet_weights.h5",
    #     weights_type='enhance',
    #     n=25,
    #     activation='relu'
    # )
    # contour_integration_model.summary()

    # Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        n=25,
        activation='relu'
    )
    # contour_integration_model.summary()

    l1_weights = K.eval(contour_integration_model.layers[1].weights[0])

    # Define callback functions to get activations of L1 convolutional layer &
    # L2 contour integration layer
    l1_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    l2_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)

    # Store the start weights & bias for comparison later
    start_weights, start_bias = contour_integration_model.layers[2].get_weights()

    # 2. L1 Kernel to Target
    # ------------------------------------------------------------------------
    tgt_filter_idx = 54

    # Some Initialization
    # ------------------------------------------------------------------------
    test_image = np.zeros((227, 227, 3))
    test_image_len = test_image.shape[0]

    tgt_filter = l1_weights[:, :, :, tgt_filter_idx]
    tgt_filter_len = tgt_filter.shape[0]

    tgt_neuron_loc = (27, 27)
    target_neuron_rf_start = tgt_neuron_loc[0] * 4  # Visual RF start of center L2 neuron
    fragment_spacing = 0

    # Define the fragment
    fragment = np.copy(tgt_filter)
    fragment = fragment.sum(axis=2)
    fragment[fragment > 0] = 1
    fragment[fragment <= 0] = 0
    fragment = fragment * 255
    fragment = np.repeat(fragment[:, :, np.newaxis], 3, axis=2)

    fragment2 = np.zeros((11, 11, 3))  # Dimensions of the L1 convolutional layer of alexnet
    fragment2[:, (0, 3, 4, 5, 9, 10), :] = 255

    # 3. Get the Orientation of the Target Filter
    # --------------------------------------------------------------------------
    theta, offset = get_l1_filter_orientation_and_offset(tgt_filter, tgt_filter_idx, show_plots=False)

    # 4. Plot the visual receptive field of L2 neuron (parts of the images the neighbors see)
    # ---------------------------------------------------------------------------
    test_image = np.ones((227, 227, 3)) * 128

    # Create a test image with a contour in it
    contour_len = 9

    start_x_arr, start_y_arr = diagonal_contour_generator(
        tgt_filter_len,
        offset,
        fragment_spacing,
        contour_len,
        target_neuron_rf_start,
    )

    test_image = alex_net_utils.tile_image(
        test_image,
        fragment,
        (start_x_arr, start_y_arr),
        rotate=False,
        gaussian_smoothing=False
    )

    l2_masks = K.eval(contour_integration_model.layers[2].mask)
    tgt_l2_masks = l2_masks[tgt_filter_idx, :, :]

    test_image = test_image / 255.0
    plt.figure()
    plt.imshow(test_image)
    alex_net_utils.plot_l2_visual_field(tgt_neuron_loc, tgt_l2_masks, test_image)

    # # 6. Find Best Fit L2 weights
    # # ---------------------------------------------------------------------------------
    # # learning_rate_array = [0.0025, 0.001, 0.00025, 0.0001, 0.00005]
    # learning_rate_array = [0.0025]
    #
    # fig, ax = plt.subplots()
    #
    # for learning_rate in learning_rate_array:
    #
    #     print("############ Processing Learning Rate %0.8f ###########" % learning_rate)
    #
    #     contour_integration_model.layers[2].set_weights((start_weights, start_bias))
    #
    #     images = multiplicative_model.optimize_contour_enhancement_layer_weights(
    #         contour_integration_model,
    #         tgt_filter_idx,
    #         fragment,
    #         diagonal_contour_generator,
    #         n_runs=1000,
    #         offset=offset,
    #         optimize_type='both',
    #         learning_rate=learning_rate,
    #         axis=ax
    #     )
    #     ax.legend()
    #
    #     multiplicative_model.plot_optimized_weights(
    #         contour_integration_model,
    #         tgt_filter_idx,
    #         start_weights,
    #         start_bias
    #     )
    #     fig = plt.gcf()
    #     fig.suptitle("learning rate = %f" % learning_rate)
    #
    #     # Plot Gain vs Contour Length after Optimization
    #     complex_bg.main_contour_length_routine(
    #         fragment,
    #         l1_activations_cb,
    #         l2_activations_cb,
    #         diagonal_contour_generator,
    #         tgt_filter_idx,
    #         smoothing=True,
    #         row_offset=offset,
    #         n_runs=100,
    #     )
    #
    #     # Plot Gain vs Contour Spacing after Optimization
    #     complex_bg.main_contour_spacing_routine(
    #         fragment,
    #         l1_activations_cb,
    #         l2_activations_cb,
    #         diagonal_contour_generator,
    #         tgt_filter_idx,
    #         smoothing=True,
    #         row_offset=offset,
    #         n_runs=100
    #     )
