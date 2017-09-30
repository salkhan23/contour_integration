# -------------------------------------------------------------------------------------------------
#  Search for optimal weights for a diagonal contour using the multiplicative model
#
# Author: Salman Khan
# Date  : 27/09/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras.backend as K

import alex_net_cont_int_models as cont_int_models
import gabor_fits
import alex_net_utils

reload(cont_int_models)
reload(alex_net_utils)

np.random.seed(10)  # Set the random seed for reproducibility


def get_l1_filter_orientation_and_offset(tgt_filt, tgt_filt_idx):
    """
    Given a Target AlexNet L1 Convolutional Filter, fit to a 2D spatial Gabor. Use this to as
    the orientation of the filter and calculate the row shift offset to use when tiling fragments.
    This offset represents the shift in pixels to use for each row  as you move away from the
    center row. Thereby allowing contours for the target filter to be generated.

    Raises an exception if no best fit parameters are found for any of the channels of the target
    filter.

    :param tgt_filt_idx:
    :param tgt_filt:

    :return: optimal orientation, row offset.
    """
    tgt_filt_len = tgt_filt.shape[0]

    best_fit_params_list = gabor_fits.find_best_fit_2d_gabor(tgt_filt)

    # Plot the best fit params
    gabor_fits.plot_kernel_and_best_fit_gabors(tgt_filt, tgt_filt_idx, best_fit_params_list)

    # Remove all empty entries
    best_fit_params_list = [params for params in best_fit_params_list if params is not None]
    if not best_fit_params_list:
        raise Exception("Optimal Params could not be found")

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

    print("L1 kernel %d, optimal orientation %0.2f(degrees), vertical offset of tiles %d"
          % (tgt_filt_idx, theta_opt, row_offset))

    return theta_opt, row_offset


def diagonal_contour_generator(frag_len, row_offset, cont_len, space_bw_tiles, cont_start_loc):
    """

    Generate the start co-ordinates of fragment squares that form a diagonal contour of
    the specified length at the specified location

    :param space_bw_tiles:
    :param frag_len:
    :param row_offset: row_offset to complete contours, found from the orientation of the fragment.
            See get_l1_filter_orientation_and_offset.
    :param cont_len:
    :param cont_start_loc: Start visual RF location of center neuron.

    :return: start_x, start_y locations of fragments that form the contour
    """
    frag_spacing = frag_len + space_bw_tiles

    # 1st dimension stays the same
    start_x = range(
        cont_start_loc - (cont_len / 2) * frag_spacing,
        cont_start_loc + ((cont_len / 2) + 1) * frag_spacing,
        frag_spacing
    )

    # If there is nonzero spacing between tiles, the offset needs to be updated
    if space_bw_tiles:
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

    l1_weights = K.eval(contour_integration_model.layers[1].weights[0])

    # Store the start weights & bias for comparison later
    start_weights, start_bias = contour_integration_model.layers[2].get_weights()

    # 2. L1 Kernel to Target
    # ------------------------------------------------------------------------
    tgt_filter_idx = 54

    # 3. Some Initialization
    # ------------------------------------------------------------------------
    test_image = np.zeros((227, 227, 3))
    test_image_len = test_image.shape[0]

    tgt_filter = l1_weights[:, :, :, tgt_filter_idx]
    tgt_filter_len = tgt_filter.shape[0]

    target_neuron_rf_start = 27 * 4  # Visual RF start of center L2 neuron
    fragment_spacing = 0

    # 4. Get the Orientation of the Target Filter
    # --------------------------------------------------------------------------
    theta, offset = get_l1_filter_orientation_and_offset(tgt_filter, tgt_filter_idx)

    # After finding the offset, scale the target filter to range [0, 255]
    tgt_filter = (tgt_filter - tgt_filter.min()) / (tgt_filter.max() - tgt_filter.min()) * 255

    # 5. Validation step: Generate a test image with background tiles with row offset
    # --------------------------------------------------------------------------
    start_x_arr, start_y_arr = alex_net_utils.get_background_tiles_locations(
        tgt_filter_len, test_image_len, offset, fragment_spacing, target_neuron_rf_start)

    test_image = alex_net_utils.tile_image(
        test_image, tgt_filter, (start_x_arr, start_y_arr), rotate=True, gaussian_smoothing=True)

    # plt.figure()
    # display_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    # plt.imshow(display_image)
    #
    # 6. Validation step: Generate a test image with a contour of the specified length
    # --------------------------------------------------------------------------
    contour_len = 7
    start_x_arr, start_y_arr = diagonal_contour_generator(
        tgt_filter_len, offset, contour_len, fragment_spacing, target_neuron_rf_start)

    test_image = alex_net_utils.tile_image(
        test_image, tgt_filter, (start_x_arr, start_y_arr), rotate=False, gaussian_smoothing=True)

    plt.figure()
    display_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    plt.imshow(display_image)
