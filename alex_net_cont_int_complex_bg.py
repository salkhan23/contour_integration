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


def get_randomly_rotated_tile(tile, delta_rotation=45.0):
    """
    randomly rotate tile by 360/delta_rotation permutations

    :param delta_rotation:
    :param tile:
    :return:
    """
    num_possible_rotations = int(360 / delta_rotation)
    return imrotate(tile, angle=(np.random.randint(0, num_possible_rotations) * delta_rotation))


def tile_image(img, frag, insert_locs, rotate=True, gaussian_smoothing=True):
    """
    Place tile 'fragments' at the specified starting positions (x, y) in the image.

    :param frag: contour fragment to be inserted
    :param insert_locs: array of (x,y) positions to insert tiles at.
    :param img: image where tiles will be placed
    :param rotate: If true each tile is randomly rotated before insertion. Currently 8 possible orientations
    :param gaussian_smoothing: If True, each fragment is multiplied with a Gaussian smoothing function to prevent
            tile edges becoming part of stimuli, as they will lie in the center of the RF of my neurons.

    :return: tiled image
    """
    img_len = img.shape[0]
    tile_len = frag.shape[0]

    g_kernel = linear_cont_int_model.get_2d_gaussian_kernel((tile_len, tile_len), sigma=5.0)
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
                tile = get_randomly_rotated_tile(frag, 45)
            else:
                tile = frag

            # multiply the file with the gaussian smoothing filter
            # The edges between the tiles will lie within the stimuli of some neurons.
            # to prevent these prom being interpreted as stimuli, gradually decrease them.
            if gaussian_smoothing:
                tile = tile * g_kernel

            img[
                start_x_loc: stop_x_loc,
                start_y_loc: stop_y_loc,
                :
            ] = tile[0: stop_x_loc - start_x_loc, 0: stop_y_loc - start_y_loc, :]

    return img


def vertical_contour(model, tgt_filt_idx, frag, contour_len):
    """
    Creates a test image of a sea of "contour fragments" by tiling random rotations of the contour
    fragment (frag), then inserts a vertical contour of the specified length into the center of
    the image Plot the l1 & l2 activations of the contour integration alexnet model.

    :param model:
    :param tgt_filt_idx: target neuron activation
    :param frag: Contour fragment to be tiled.
    :param contour_len:

    :return: l2 activations
    """

    frag_len = frag.shape[0]

    # Sea of similar but randomly oriented fragments
    # -----------------------------------------------
    # Rather than tiling starting from location (0, 0). Start at the center of the RF of a central neuron.
    # This ensures that this central neuron is looking directly at the fragment and has a maximal response.
    # This is similar to the Ref, where the center contour fragment was in the center of the RF of the neuron
    # being monitored and the origin of the visual field
    test_image = np.ones((227, 227, 3)) * 128.0
    image_len = test_image.shape[0]

    # Output dimensions of the first convolutional layer of Alexnet [55x55x96] and a stride=4 was used
    center_neuron_loc = 27 * 4  # Starting Visual Field Location of neuron @ location (27,27)
    num_tiles = image_len // frag_len

    start_x = range(
        center_neuron_loc - (num_tiles / 2) * frag_len,
        center_neuron_loc + (num_tiles / 2 + 1) * frag_len,
        frag_len
    )
    start_y = np.copy(start_x)

    test_image = tile_image(test_image, frag, (start_x, start_y), rotate=True, gaussian_smoothing=True)

    # Insert Contour
    # --------------
    start_x = range(
        center_neuron_loc - (contour_len / 2) * frag_len,
        center_neuron_loc + (contour_len / 2 + 1) * frag_len,
        frag_len
    )
    start_y = np.ones_like(start_x) * center_neuron_loc

    test_image = tile_image(test_image, frag, (start_x, start_y), rotate=False, gaussian_smoothing=True)

    # Bring it back to the [0,1] range (rotation fcn scales pixels to [0, 255])
    test_image = test_image / 255.0

    plt.figure()
    plt.imshow(test_image)
    plt.title("Vertical contour of length %d in a sea of fragments" % contour_len)

    # Pass the test image through the model
    # -------------------------------------------
    _, l2_output = linear_cont_int_model.plot_tgt_filters_activations(
        model,
        test_image,
        tgt_filt_idx,
    )
    f = plt.gcf()
    f.suptitle("Vertical Contour of Length %d" % contour_len)

    return l2_output


if __name__ == "__main__":

    plt.ion()

    # 1. Load the model
    # ------------------
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # m_type = 'enhance_n_suppress_non_overlap'
    m_type = 'non_overlap_full'
    contour_integration_model = nonlinear_cont_int_model.build_model(
        "trained_models/AlexNet/alexnet_weights.h5", model_type=m_type)
    # alex_net_cont_int_model.summary()

    # 2. Vertical Contours of various lengths
    # ----------------------------------------
    tgt_filter_index = 10

    # 2(a) Contour fragment to use
    # -------------------------------
    # Fragment is the target filter
    # conv1_weights = K.eval(contour_integration_model.layers[1].weights[0])
    # fragment = conv1_weights[:, :, :, tgt_filter_index]

    # Simpler 'target filter' like contour fragment
    # fragment = np.zeros((11, 11, 3))  # Dimensions of the L1 convolutional layer of alexnet
    # fragment[:, (0, 3, 4, 5, 9, 10), :] = 255

    # Fragment similar to used in Ref
    # Average RF size of neuron = 0.6 degrees. Alex Net Conv L1 RF size 11x11
    # Contour Fragments placed in squares of size = 0.4 degrees. 0.4/0.6 * 11 = 7.3 = 8
    # Within each square, contour of size 0.2 x 0.05 degrees. 0.2/0.6*11, 0.05/0.6*11 = (3.6, 0.92) = (4, 1)
    # However, the make RF (square) increase the width of fragment to 2.
    fragment = np.zeros((8, 8, 3))
    fragment[(2, 3, 4, 5), 3, :] = 255
    fragment[(2, 3, 4, 5), 4, :] = 255

    # 2(b) Display the target Filter and the contour fragment
    # --------------------------------------------------
    conv1_weights = K.eval(contour_integration_model.layers[1].weights[0])
    tgt_filter = conv1_weights[:, :, :, tgt_filter_index]

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    display_filt = (tgt_filter - tgt_filter.min()) * 1 / (tgt_filter.max() - tgt_filter.min())  # normalize to [0, 1]
    plt.imshow(display_filt)
    plt.title("Target Filter")
    fig.add_subplot(1, 2, 2)
    plt.imshow(fragment / 255.0)
    plt.title("Contour Fragment")

    # 2(c) Response to contours of various lengths
    # --------------------------------------------------
    contour_lengths_arr = range(1, 11, 2)
    tgt_neuron_l2_act = []
    tgt_neuron_loc = (27, 27)

    for c_len in contour_lengths_arr:
        l2_activations = vertical_contour(
            contour_integration_model,
            tgt_filter_index,
            fragment,
            c_len,
        )

        tgt_neuron_l2_act.append(
            l2_activations[0, tgt_filter_index, tgt_neuron_loc[0], tgt_neuron_loc[1]])

    plt.figure()
    plt.plot(contour_lengths_arr, tgt_neuron_l2_act)
    plt.xlabel("Contour Length")
    plt.ylabel("Activation")
    plt.title("Neuron @loc (%d, %d) with kernel Index %d, L2 (contour Enhanced) activation"
              % (tgt_neuron_loc[0], tgt_neuron_loc[1], tgt_filter_index))