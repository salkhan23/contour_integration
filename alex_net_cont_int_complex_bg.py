# -------------------------------------------------------------------------------------------------
# This is a demonstration of contour integration in a 'sea' of similar but randomly oriented
# fragments that forms a complex background over which the contour should be enhanced.
#
# REF: Li, Piech & Gilbert - 2006 - Contour Saliency in Primary Visual Cortex
#
# Two results from the paper are replicated, figure 2 B3 & B4. Firing rates of V1 neurons
# increases as (B3) the number of fragments making up the contour increase and (B4) decreases as
# the spacing between contours increases.
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


def get_activation_cb(model, layer_idx):
    """
    Return a callback to return the output activation of the specified layer
    The callback expects input cb([input_image, 0])
    :param model:
    :param layer_idx:
    :return: callback function.
    """
    get_layer_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_idx].output]
    )

    return get_layer_output


def get_randomly_rotated_tile(tile, delta_rotation=45.0):
    """
    randomly rotate tile by 360/delta_rotation permutations

    :param delta_rotation:
    :param tile:
    :return:
    """
    num_possible_rotations = 360 // delta_rotation
    return imrotate(tile, angle=(np.random.randint(0, num_possible_rotations) * delta_rotation))


def tile_image(img, frag, insert_locs, rotate=True, gaussian_smoothing=True):
    """
    Place tile 'fragments' at the specified starting positions (x, y) in the image.

    :param frag: contour fragment to be inserted
    :param insert_locs: array of (x,y) starting positions of where tiles will be inserted
    :param img: image where tiles will be placed
    :param rotate: If true each tile is randomly rotated before insertion. Currently 8 possible orientations
    :param gaussian_smoothing: If True, each fragment is multiplied with a Gaussian smoothing function to prevent
            tile edges becoming part of stimuli, as they will lie in the center of the RF of my neurons.

    :return: tiled image
    """
    img_len = img.shape[0]
    tile_len = frag.shape[0]

    g_kernel = linear_cont_int_model.get_2d_gaussian_kernel((tile_len, tile_len), sigma=4.0)
    g_kernel = np.reshape(g_kernel, (g_kernel.shape[0], g_kernel.shape[1], 1))
    g_kernel = np.repeat(g_kernel, 3, axis=2)

    for x in insert_locs[0]:
        for y in insert_locs[1]:
            # print("Placing Fragment at location x=(%d, %d), y =(%d, %d)"
            #       % (x, x + tile_len, y, y + tile_len))

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


def get_contour_responses(l1_act_cb, l2_act_cb, tgt_filt_idx, frag, contour_len,
                          cont_gen_cb, space_bw_tiles=0, smooth_tiles=True):
    """
    Creates a test image of a sea of "contour fragments" by tiling random rotations of the contour
    fragment (frag), then inserts a vertical contour of the specified length into the center of
    the image Plot the l1 & l2 activations of the contour integration alexnet model.

    :param l1_act_cb: Callback function to get activations of L1 contour integration layer
    :param l2_act_cb: Callback function to get activations of L2 contour integration layer
    :param tgt_filt_idx: target neuron activation
    :param frag: Contour fragment (square) to use for creating the larger tiled image
    :param contour_len: length of contour to generate
    :param cont_gen_cb: Contour generating callback. Generates contours of the specified length
                        and spacing between fragments. For format see vertical_contour_generator
    :param space_bw_tiles: Amount of spacing (in pixels) between inserted tiles
    :param smooth_tiles: Use gaussian smoothing on tiles to prevent tile edges from becoming part
                         of the stimulus

    :return: l1 activation, l2 activation (of target filter) & generated image
    """

    frag_len = frag.shape[0] + space_bw_tiles

    # Sea of similar but randomly oriented fragments
    # -----------------------------------------------
    # Rather than tiling starting from location (0, 0). Start at the center of the RF of a central neuron.
    # This ensures that this central neuron is looking directly at the fragment and has a maximal response.
    # This is similar to the Ref, where the center contour fragment was in the center of the RF of the neuron
    # being monitored and the origin of the visual field
    test_image = np.zeros((227, 227, 3))
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

    test_image = tile_image(
        test_image,
        frag,
        (start_x, start_y),
        rotate=True,
        gaussian_smoothing=smooth_tiles
    )

    # Insert Contour
    # --------------
    cont_coordinates = cont_gen_cb(frag.shape[0], space_bw_tiles, contour_len, center_neuron_loc)

    test_image = tile_image(
        test_image,
        frag,
        cont_coordinates,
        rotate=False,
        gaussian_smoothing=smooth_tiles
    )

    # Bring it back to the [0, 1] range (rotation fcn scales pixels to [0, 255])
    test_image = test_image / 255.0

    # Get the activations of the first convolutional and second contour integration layer
    # -----------------------------------------------------------------------------------
    test_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format
    test_image = np.reshape(test_image, [1, test_image.shape[0], test_image.shape[1], test_image.shape[2]])
    # batch size is expected as the first dimension

    l1_act = l1_act_cb([test_image, 0])
    l1_act = np.squeeze(np.array(l1_act), axis=0)
    l2_act = l2_act_cb([test_image, 0])
    l2_act = np.squeeze(np.array(l2_act), axis=0)

    tgt_l1_act = l1_act[0, tgt_filt_idx, :, :]
    tgt_l2_act = l2_act[0, tgt_filt_idx, :, :]

    # return the test image back to its original format
    test_image = test_image[0, :, :, :]
    test_image = np.transpose(test_image, (1, 2, 0))

    return tgt_l1_act, tgt_l2_act, test_image


def vertical_contour_generator(frag_len, bw_tile_spacing, cont_len, cont_start_loc):
    """
    Generate the start co-ordinates of fragment squares that form a contour of the specified length
    at the specified location

    :param frag_len:
    :param bw_tile_spacing: Between fragment square spacing in pixels
    :param cont_len: length of fragment in units of fragment squares
    :param cont_start_loc: start starting location where the contour should be places
    :return:
    """
    mod_frag_len = frag_len + bw_tile_spacing

    start_x = range(
        cont_start_loc - (cont_len / 2) * mod_frag_len,
        cont_start_loc + (cont_len / 2 + 1) * mod_frag_len,
        mod_frag_len
    )
    start_y = np.ones_like(start_x) * cont_start_loc

    return start_x, start_y


def horizontal_contour_generator(frag_len, bw_tile_spacing, cont_len, cont_start_loc):
    """

    :param frag_len:
    :param bw_tile_spacing:
    :param cont_len:
    :param cont_start_loc:
    :return:
    """
    mod_frag_len = frag_len + bw_tile_spacing

    start_y = range(
        cont_start_loc - (cont_len / 2) * mod_frag_len,
        cont_start_loc + (cont_len / 2 + 1) * mod_frag_len,
        mod_frag_len
    )

    start_x = np.ones_like(start_y) * cont_start_loc

    return start_x, start_y


def plot_activations(img, l1_act, l2_act, tgt_filt_idx):
    """
    plot the test image, l1_activations, l2_activations and the difference between the activations

    :param img:
    :param l1_act:
    :param l2_act:
    :param tgt_filt_idx:
    :return:
    """

    plt.figure()
    plt.imshow(img)

    min_l2_act = l1_act.min()
    max_l2_act = l2_act.max()

    f = plt.figure()

    f.add_subplot(1, 3, 1)
    plt.imshow(l1_act, cmap='seismic', vmin=min_l2_act, vmax=max_l2_act)
    plt.title('L1 Conv Layer Activation @ idx %d' % tgt_filt_idx)
    plt.colorbar(orientation='horizontal')
    plt.grid()

    f.add_subplot(1, 3, 2)
    plt.imshow(l2_act, cmap='seismic', vmin=min_l2_act, vmax=max_l2_act)
    plt.title('L2 Contour Integration Layer Activation @ idx %d' % tgt_filter_index)
    plt.colorbar(orientation='horizontal')
    plt.grid()

    f.add_subplot(1, 4, 4)
    plt.imshow(l2_act - l1_act, cmap='seismic')
    plt.title("Difference")
    plt.grid()


def plot_l1_filter_and_contour_fragment(model, frag, tgt_filt_idx):
    """

    :param model:
    :param frag:
    :param tgt_filt_idx:
    :return:
    """
    conv1_weights = K.eval(model.layers[1].weights[0])
    tgt_filt = conv1_weights[:, :, :, tgt_filt_idx]

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    display_filt = (tgt_filt - tgt_filt.min()) * 1 / (tgt_filt.max() - tgt_filt.min())
    plt.imshow(display_filt)  # normalized to [0, 1]
    plt.title("Target Filter")
    f.add_subplot(1, 2, 2)
    plt.imshow(frag / 255.0)
    plt.title("Contour Fragment")


def main_contour_length_routine(frag, l1_act_cb, l2_act_cb, cont_gen_cb, tgt_filt_idx, smoothing, n_runs=1):
    """
    Contours of various lengths - Figure 2, B3

    :param smoothing:
    :param cont_gen_cb:
    :param tgt_filt_idx:
    :param l2_act_cb:
    :param l1_act_cb:
    :param n_runs:
    :param frag:

    :return:
    """
    contour_lengths_arr = range(1, 11, 2)

    tgt_neuron_loc = (27, 27)  # Neuron focused in the center of the image.
    tgt_neuron_l2_act = np.zeros((n_runs, len(contour_lengths_arr)))

    for run_idx in range(n_runs):
        for c_idx, c_len in enumerate(contour_lengths_arr):
            print("Run %d, Processing contour of length %d" % (run_idx, c_len))

            tgt_l1_activation, tgt_l2_activation, test_img = get_contour_responses(
                l1_act_cb,
                l2_act_cb,
                tgt_filt_idx,
                frag,
                c_len,
                cont_gen_cb,
                smooth_tiles=smoothing
            )

            tgt_neuron_l2_act[run_idx, c_idx] = tgt_l2_activation[tgt_neuron_loc[0], tgt_neuron_loc[1]]

            # plot_activations(test_img, tgt_l1_activation, tgt_l2_activation, tgt_filt_idx)
            # plt.suptitle("Vertical contour Length=%d in a sea of fragments" % c_len)

    # Plot the activation of the target neuron as contour length increases
    tgt_neuron_mean = tgt_neuron_l2_act.mean(axis=0)
    tgt_neuron_std = tgt_neuron_l2_act.std(axis=0)

    plt.figure()
    plt.errorbar(contour_lengths_arr, tgt_neuron_mean, tgt_neuron_std, marker='o')
    plt.xlabel("Contour Length")
    plt.ylabel("Activation")
    plt.title("L2 (contour enhanced) activation, Neuron @ (%d, %d, %d)"
              % (tgt_neuron_loc[0], tgt_neuron_loc[1], tgt_filt_idx))


def main_contour_spacing_routine(frag, l1_act_cb, l2_act_cb, cont_gen_cb, tgt_filt_idx, smoothing, n_runs=1):
    """
    Various Contour Spacing - Figure 2, B4

    :param frag:
    :param l1_act_cb:
    :param l2_act_cb:
    :param cont_gen_cb:
    :param tgt_filt_idx:
    :param smoothing:
    :param n_runs:

    :return:
    """
    c_len = 7  # Ref uses a contour of length 7
    frag_len = frag.shape[0]

    # Contour spacing was specified as relative colinear spacing = the distance between the centers
    # of adjacent fragments / length of contour fragments (fixed at 0.4 visual degrees)
    # Relative colinear distance of [1, 1.2, 1.4, 1.6, 1.8, 1.9] were use.
    # We simulate similar relative colinear distance by inserting spacing between the contour squares.
    relative_colinear_dist_arr = np.array([1, 1.2, 1.4, 1.6, 1.8, 1.9])
    spacing_bw_tiles = np.floor(relative_colinear_dist_arr * frag_len) - frag_len

    tgt_neuron_loc = (27, 27)  # Neuron focused in the center of the image.
    tgt_neuron_l2_act = np.zeros((n_runs, len(spacing_bw_tiles)))

    for run_idx in range(n_runs):
        for s_idx, spacing in enumerate(spacing_bw_tiles):
            relative_colinear_dist = (spacing + frag_len) / np.float(frag_len)
            print("Run %d, Processing relative colinear distance of %0.2f (spacing of %d)"
                  % (run_idx, relative_colinear_dist, spacing))

            tgt_l1_activation, tgt_l2_activation, test_img = get_contour_responses(
                l1_act_cb,
                l2_act_cb,
                tgt_filt_idx,
                frag,
                c_len,
                cont_gen_cb,
                space_bw_tiles=int(spacing),
                smooth_tiles=smoothing
            )

            tgt_neuron_l2_act[run_idx, s_idx] = tgt_l2_activation[tgt_neuron_loc[0], tgt_neuron_loc[1]]

            # plot_activations(test_img, tgt_l1_activation, tgt_l2_activation, tgt_filt_idx)
            # plt.suptitle("Vertical contour Length=%d, Rel. colinear dist=%0.2f in a sea of fragments"
            #              % (c_len, (frag_len + spacing) / np.float(frag_len)))

    # Plot the activation of the target neuron as contour spacing changes
    plt.figure()
    tgt_neuron_mean = tgt_neuron_l2_act.mean(axis=0)
    tgt_neuron_std = tgt_neuron_l2_act.std(axis=0)
    rcd = (spacing_bw_tiles + frag_len) / np.float(frag_len)

    plt.errorbar(rcd, tgt_neuron_mean, tgt_neuron_std, marker='o')
    plt.xlabel("Contour Distance")
    plt.ylabel("Activation")
    plt.title("L2 (contour Enhanced) activation. Neuron @ (%d, %d, %d)"
              % (tgt_neuron_loc[0], tgt_neuron_loc[1], tgt_filt_idx))

if __name__ == "__main__":

    plt.ion()

    # 1. Load the model
    # ------------------
    K.clear_session()
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # m_type = 'enhance_n_suppress_non_overlap'
    m_type = 'non_overlap_full'
    contour_integration_model = nonlinear_cont_int_model.build_model(
        "trained_models/AlexNet/alexnet_weights.h5", model_type=m_type)
    # alex_net_cont_int_model.summary()

    # Define callback functions to get activations of L1 convolutional layer &
    # L2 contour integration layer
    l1_activations_cb = get_activation_cb(contour_integration_model, 1)
    l2_activations_cb = get_activation_cb(contour_integration_model, 2)

    # --------------------------------------------------------------------------------------------
    #  Vertical Contours
    # --------------------------------------------------------------------------------------------
    tgt_filter_index = 10

    # # Fragment is the target filter
    # conv1_weights = K.eval(contour_integration_model.layers[1].weights[0])
    # fragment = conv1_weights[:, :, :, tgt_filter_index]
    # Scale the filter to lie withing [0, 255]
    # fragment = (fragment - fragment.min())*(255 / (fragment.max() - fragment.min()))
    # use_smoothing = True

    # Simpler 'target filter' like contour fragment
    fragment = np.zeros((11, 11, 3))  # Dimensions of the L1 convolutional layer of alexnet
    fragment[:, (0, 3, 4, 5, 9, 10), :] = 255
    use_smoothing = True

    # # Fragment from the Reference
    # # Average RF size of neuron = 0.6 degrees. Alex Net Conv L1 RF size 11x11
    # # Contour fragments placed in squares of size = 0.4 degrees. 0.4/0.6 * 11 = 7.3 = 8
    # # Within each square, contour of size 0.2 x 0.05 degrees.( 0.2, 0.5)/0.6 * 11 = (3.6, 0.92) = (4, 1)
    # # However, to make the fragment symmetric, increase the width of fragment to 2.
    # fragment = np.zeros((8, 8, 3))
    # fragment[(2, 3, 4, 5), 3, :] = 255
    # fragment[(2, 3, 4, 5), 4, :] = 255
    # use_smoothing = False

    plot_l1_filter_and_contour_fragment(contour_integration_model, fragment, tgt_filter_index)

    main_contour_length_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        vertical_contour_generator,
        tgt_filter_index,
        use_smoothing,
        n_runs=50
    )

    main_contour_spacing_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        vertical_contour_generator,
        tgt_filter_index,
        use_smoothing,
        n_runs=50
    )

    # --------------------------------------------------------------------------------------------
    #  Horizontal Contours
    # --------------------------------------------------------------------------------------------
    tgt_filter_index = 5

    # # Fragment is the target filter
    # conv1_weights = K.eval(contour_integration_model.layers[1].weights[0])
    # fragment = conv1_weights[:, :, :, tgt_filter_index]
    # # Scale the filter to lie withing [0, 255]
    # fragment = (fragment - fragment.min())*(255 / (fragment.max() - fragment.min()))
    # use_smoothing = True

    # # Simpler 'target filter' like contour fragment
    fragment = np.zeros((11, 11, 3))  # Dimensions of the L1 convolutional layer of alexnet
    fragment[0:6, :, :] = 255
    use_smoothing = True

    # # Fragment from the Reference
    # fragment = np.zeros((8, 8, 3))
    # fragment[3, (2, 3, 4, 5), :] = 255
    # fragment[4, (2, 3, 4, 5), :] = 255
    # use_smoothing = False

    main_contour_length_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        horizontal_contour_generator,
        tgt_filter_index,
        use_smoothing,
        n_runs=50
    )

    main_contour_spacing_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        horizontal_contour_generator,
        tgt_filter_index,
        use_smoothing,
        n_runs=50
    )
