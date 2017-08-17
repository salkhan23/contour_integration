# -------------------------------------------------------------------------------------------------
# This is a demonstration of contour integration in a sea of similar but randomly oriented
# fragments that forms a complex background over which the contour should be enhanced.
#
# Ref: Li, Piech & Gilbert - 2006 - Contour Saliency in Primary Visual Cortex
#
# Author: Salman Khan
# Date  : 11/08/17
# -------------------------------------------------------------------------------------------------

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imrotate

import keras.backend as K

import base_alex_net as alex_net
import learned_lateral_weights
import utils
import alex_net_add_cont_int as linear_cont_int_model
import alex_net_mult_cont_int as nonlinear_cont_int_model
reload(utils)
reload(learned_lateral_weights)
reload(alex_net)
reload(linear_cont_int_model)
reload(nonlinear_cont_int_model)

np.random.seed(7)  # Set the random seed for reproducibility


def get_2d_gaussian_kernel(shape, sigma=1.0):
    """
    Returns a 2d (unnormalized) Gaussian kernel of the specified shape.

    :param shape: x,y dimensions of the gaussian
    :param sigma: standard deviation of generated Gaussian
    :return:
    """
    ax = np.arange(-shape[0] // 2 + 1, shape[0] // 2 + 1)
    ay = np.arange(-shape[1] // 2 + 1, shape[1] // 2 + 1)

    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    return kernel


def get_randomly_rotated_tile(tile, delta_rotation=45.0):
    """
    randomly rotate tile by 360/delta_rotation permutations

    :param delta_rotation:
    :param tile:
    :return:
    """
    num_possible_rotations = int(360 / delta_rotation)
    return imrotate(tile, angle=(np.random.randint(0, num_possible_rotations) * delta_rotation))


def tile_image(img, fragment, insert_locs, rotate=True):
    """
    Place tile 'fragments' at the specified starting positions (x, y) in the image.
    Each Fragment is multiplied with a Gaussian smoothing function to prevent tile edges becoming part of
    stimuli

    :param insert_locs: array of (x,y) positions to insert tiles at.
    :param img: image where tiles will be placed
    :param tile: tile or fragment to be inserted
    :param rotate: If true each tile is randomly rotated before insertion. Currently 8 possible orientations

    :return: tiled image
    """
    img_len = img.shape[0]
    tile_len = fragment.shape[0]

    g_kernel = get_2d_gaussian_kernel((tile_len, tile_len), sigma=5.0)
    g_kernel = np.reshape(g_kernel, (g_kernel.shape[0], g_kernel.shape[1], 1))
    g_kernel = np.repeat(g_kernel, 3, axis=2)

    for x in insert_locs[0]:
        for y in insert_locs[1]:
            print("Placing Fragment at location x=(%d, %d), y =(%d, %d)"
                  % (x, x + tile_len, y, y + tile_len))

            start_x_loc = max(x, 0)
            stop_x_loc = min(x + tile_len, img_len)

            start_y_loc = max(y, 0)
            stop_y_loc = min(y + tile_len, img_len)

            if rotate:
                tile = get_randomly_rotated_tile(fragment, 45)
            else:
                tile = fragment

            # multiply the file with the gaussian smoothing filter
            # The edges between the tiles will lie within the stimuli of some neurons.
            # to prevent these prom being interpreted as stimuli, gradually decrease them.
            tile = tile * g_kernel

            img[
                start_x_loc: stop_x_loc,
                start_y_loc: stop_y_loc,
                :
            ] = tile[0: stop_x_loc - start_x_loc, 0: stop_y_loc - start_y_loc, :]

    return img


def main(model, tgt_filt_idx, contour_len):
    """
    This is the main routine of this script.
    For the specified target filter (of convolutional layer 1 of Alexnet), it creates a sea, of similar but randomly
    oriented fragments, then inserts a contour of the specifed length into the image, and plots the l1 & l2
    activations of the of alexnet

    :param contour_len:
    :param tgt_filt_idx:
    :param model:

    :return: l2 activations
    """

    conv1_weights = K.eval(model.layers[1].weights[0])
    tgt_filter = conv1_weights[:, :, :, tgt_filt_idx]

    # Simpler but similar fragment
    fragment = np.zeros_like(tgt_filter)
    fragment[:, (0, 3, 4, 5, 9, 10), :] = 255
    fragment_len = fragment.shape[0]

    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # display_filt = (tgt_filter - tgt_filter.min()) * 1 / (tgt_filter.max() - tgt_filter.min())
    # plt.imshow(display_filt)
    # plt.title("Target Filter")
    # f.add_subplot(1, 2, 2)
    # plt.imshow(fragment / 255.0)
    # plt.title("Contour Fragment")

    # 3. Make Test Image Sea of similar but randomly oriented fragments
    # ------------------------------------------------------------------
    # Rather than tiling starting from location (0, 0). Start at the center of the RF of a central neuron.
    # This ensures that this central neuron is looking directly at the neuron and has a maximal response and
    # is similar to the Ref, where the central contour fragment was in the center of the RF of the neuron
    # being monitored and the origin.
    test_image = np.ones((227, 227, 3)) * 128.0
    image_len = test_image.shape[0]

    # Output dimensions of the first convolutional layer of Alexnet [55x55x96] and a stride=4 was used
    center_neuron_loc = 27 * 4
    num_tiles = image_len // fragment_len

    start_x = range(
        center_neuron_loc - (num_tiles / 2) * fragment_len,
        center_neuron_loc + (num_tiles / 2 + 1) * fragment_len,
        fragment_len
    )
    start_y = np.copy(start_x)

    test_image = tile_image(test_image, fragment, (start_x, start_y), rotate=True)

    # Insert Contour at the central Location
    # -----------------------------------------
    start_x = range(
        center_neuron_loc - (contour_len / 2) * fragment_len,
        center_neuron_loc + (contour_len / 2 + 1) * fragment_len,
        fragment_len
    )

    start_y = np.ones_like(start_x) * center_neuron_loc

    test_image = tile_image(test_image, fragment, (start_x, start_y), rotate=False)

    # Bring it back to the 0-1 range
    # ------------------------------------
    test_image = test_image / 255.0

    plt.figure()
    plt.imshow(test_image)
    plt.title("sea of randomly oriented fragments, Contour of length %d" % contour_len)

    # 5. Pass the test image through the model
    # -------------------------------------------
    _, l2_activations = linear_cont_int_model.plot_tgt_filters_activations(
        model,
        test_image,
        tgt_filt_idx,
    )
    f = plt.gcf()
    f.suptitle("Contour Length %d" % contour_len)

    return l2_activations


if __name__ == "__main__":

    plt.ion()

    # 1. Load the model
    # --------------------------------------------------
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # m_type = 'enhance'
    # m_type = 'suppress'
    # m_type = 'enhance_n_suppress'
    # m_type = 'enhance_n_suppress_5'
    # m_type = 'enhance_n_suppress_non_overlap'
    m_type = 'non_overlap_full'
    contour_integration_model = nonlinear_cont_int_model.build_model(
        "trained_models/AlexNet/alexnet_weights.h5", model_type=m_type)
    # alex_net_cont_int_model.summary()

    # 1. Contour of Length 3
    # -----------------------
    act = []
    target_filter_index = 10

    len1_l2_activations = main(contour_integration_model, 10, 1)
    act.append(len1_l2_activations[0, target_filter_index, 27, 27])

    len3_l2_activations = main(contour_integration_model, 10, 3)
    act.append(len3_l2_activations[0, target_filter_index, 27, 27])

    len5_l2_activations = main(contour_integration_model, 10, 5)
    act.append(len5_l2_activations[0, target_filter_index, 27, 27])

    len7_l2_activations = main(contour_integration_model, 10, 7)
    act.append(len7_l2_activations[0, target_filter_index, 27, 27])

    plt.figure()
    plt.plot(range(1, 9, 2), act)
    plt.xlabel("Contour length")
    plt.ylabel("Amplitude")

    # # 2. Select a Target Kernel and construct a contour fragment similar to the ones used in Ref
    # # ------------------------------------------------------------------------------------------
    # tgt_filt_idx = 10  # Vertical Filter
    #
    # conv1_weights = K.eval(contour_integration_model.layers[1].weights[0])
    # tgt_filter = conv1_weights[:, :, :, tgt_filt_idx]
    # tgt_filter_len = tgt_filter[0]
    #
    # # Simpler but similar fragment
    # fragment = np.zeros_like(tgt_filter)
    # fragment[:, (0, 3, 4, 5, 9, 10), :] = 255
    # fragment_len = fragment.shape[0]
    #
    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # display_filt = (tgt_filter - tgt_filter.min()) * 1/(tgt_filter.max() - tgt_filter.min())
    # plt.imshow(display_filt)
    # plt.title("Target Filter")
    # f.add_subplot(1, 2, 2)
    # plt.imshow(fragment / 255.0)
    # plt.title("Contour Fragment")
    #
    # # 3. Make Test Image Sea of similar but randomly oriented fragments
    # # ------------------------------------------------------------------
    # # Rather than tiling starting from location (0, 0). Start at the center of the RF of a central neuron.
    # # This ensures that this central neuron is looking directly at the neuron and has a maximal response and
    # # is similar to the Ref, where the central contour fragment was in the center of the RF of the neuron
    # # being monitored and the origin.
    # test_image = np.ones((227, 227, 3)) * 128.0
    # image_len = test_image.shape[0]
    #
    # # Output dimensions of the first convolutional layer of Alexnet [55x55x96] and a stride=4 was used
    # center_neuron_loc = 27 * 4
    # num_tiles = image_len // fragment_len
    #
    # start_x = range(
    #     center_neuron_loc - (num_tiles / 2) * fragment_len,
    #     center_neuron_loc + (num_tiles / 2 + 1) * fragment_len,
    #     fragment_len
    # )
    # start_y = np.copy(start_x)
    #
    # test_image = tile_image(test_image, fragment, (start_x, start_y), rotate=True)
    #
    # # Insert Contour at the central Location
    # # -----------------------------------------
    # contour_len = 9
    # start_x = range(
    #     center_neuron_loc - (contour_len / 2) * fragment_len,
    #     center_neuron_loc + (contour_len / 2 + 1) * fragment_len,
    #     fragment_len
    # )
    #
    # start_y = np.ones_like(start_x) * center_neuron_loc
    #
    # test_image = tile_image(test_image, fragment, (start_x, start_y), rotate=False)
    #
    # # Bring it back to the 0-1 range
    # # ------------------------------------
    # test_image = test_image / 255.0
    #
    # plt.figure()
    # plt.imshow(test_image)
    # plt.title("sea of randomly oriented fragments")
    #
    # # 5. Pass the test image through the model
    # # -------------------------------------------
    # linear_cont_int_model.plot_tgt_filters_activations(
    #     contour_integration_model,
    #     test_image,
    #     tgt_filt_idx,
    # )
