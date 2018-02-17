# -------------------------------------------------------------------------------------------------
# This is a demonstration of contour integration in a 'sea' of similar but randomly oriented
# fragments that forms a complex background over which the contour should be enhanced.
#
# REF: Li, Piech & Gilbert - 2006 - Contour Saliency in Primary Visual Cortex
#
# Two results from the paper are replicated, figure 2 C3 & C4. Firing rates of V1 neurons
# increases as (B3) the number of fragments making up the contour increase and (B4) decreases as
# the spacing between contours increases.
#
# Author: Salman Khan
# Date  : 11/08/17
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


def get_contour_responses(l1_act_cb, l2_act_cb, tgt_filt_idx, frag, contour_len,
                          cont_gen_cb, space_bw_tiles=0, offset=0, smooth_tiles=True):
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
    :param offset: the offset by which a tile row should be shifted by as it moves away from center row
    :param smooth_tiles: Use gaussian smoothing on tiles to prevent tile edges from becoming part
                         of the stimulus

    :return: l1 activation, l2 activation (of target filter) & generated image
    """

    test_image = np.zeros((227, 227, 3))
    test_image_len = test_image.shape[0]

    frag_len = frag.shape[0]

    # Output dimensions of the first convolutional layer of Alexnet [55x55x96] and a stride=4 was used
    center_neuron_loc = 108  # RF size of 11 at center of l1 activation

    # Sea of similar but randomly oriented fragments
    # -----------------------------------------------
    # Rather than tiling starting from location (0, 0). Start at the center of the RF of a central neuron.
    # This ensures that this central neuron is looking directly at the fragment and has a maximal response.
    # This is similar to the Ref, where the center contour fragment was in the center of the RF of the neuron
    # being monitored and the origin of the visual field
    start_x, start_y = alex_net_utils.get_background_tiles_locations(
        frag_len, test_image_len, offset, space_bw_tiles, center_neuron_loc)

    test_image = alex_net_utils.tile_image(
        test_image,
        frag,
        (start_x, start_y),
        rotate=True,
        gaussian_smoothing=smooth_tiles
    )

    # Insert Contour
    # --------------
    cont_coordinates = cont_gen_cb(
        frag_len,
        bw_tile_spacing=space_bw_tiles,
        cont_len=contour_len,
        cont_start_loc=center_neuron_loc,
        row_offset=offset
    )

    test_image = alex_net_utils.tile_image(
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


def main_contour_length_routine(frag, l1_act_cb, l2_act_cb, cont_gen_cb, tgt_filt_idx,
                                smoothing, row_offset=0, n_runs=1, tgt_neuron_loc=(27, 27)):
    """
    Contours of various lengths - Figure 2, B3

    :param tgt_neuron_loc:
    :param frag:
    :param l1_act_cb:
    :param l2_act_cb:
    :param cont_gen_cb:
    :param tgt_filt_idx:
    :param smoothing:
    :param row_offset:
    :param n_runs:

    :return: handle of figure where contour length vs gain is plotted
    """
    contour_lengths_arr = range(1, 11, 2)

    tgt_neuron_l2_act = np.zeros((n_runs, len(contour_lengths_arr)))
    tgt_neuron_l1_act = np.zeros((n_runs, len(contour_lengths_arr)))

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
                offset=row_offset,
                smooth_tiles=smoothing
            )

            tgt_neuron_l1_act[run_idx, c_idx] = tgt_l1_activation[tgt_neuron_loc[0], tgt_neuron_loc[1]]
            tgt_neuron_l2_act[run_idx, c_idx] = tgt_l2_activation[tgt_neuron_loc[0], tgt_neuron_loc[1]]

            if n_runs == 1:
                fig1, fig2 = alex_net_utils.plot_l1_and_l2_activations(test_img, l1_act_cb, l2_act_cb, tgt_filt_idx)
                title = "Vertical contour Length=%d in a sea of fragments" % c_len
                fig1.suptitle(title)
                fig2.suptitle(title)

    # Plot the Contour Integration Gain of the target neuron as contour length increases
    tgt_neuron_gain = tgt_neuron_l2_act / (tgt_neuron_l1_act + 1e-8)
    tgt_neuron_gain_mean = tgt_neuron_gain.mean(axis=0)
    tgt_neuron_gain_std = tgt_neuron_gain.std(axis=0)

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.errorbar(contour_lengths_arr, tgt_neuron_gain_mean, tgt_neuron_gain_std, marker='o', label='model', color='b')
    ax.set_xlabel("Contour Length")
    ax.set_ylabel("Contour Integration Gain")
    ax.set_title("L2 (contour enhanced) gain as a function of contour length for Neuron @ (%d, %d, %d)"
                 % (tgt_neuron_loc[0], tgt_neuron_loc[1], tgt_filt_idx))

    # Plot Neurophysiological data Data from [Li-2006]
    with open('.//neuro_data//Li2006.pickle', 'rb') as handle:
        data = pickle.load(handle)

    ax.plot(data['contour_len_ma_len'], data['contour_len_ma_gain'],
            label='ma', color='r', marker='d', linestyle='dashed')
    ax.plot(data['contour_len_mb_len'], data['contour_len_mb_gain'],
            label='mb', color='g', marker='s', linestyle='dashed')

    ax.legend(loc='best')

    return f


def main_contour_spacing_routine(frag, l1_act_cb, l2_act_cb, cont_gen_cb, tgt_filt_idx,
                                 smoothing, row_offset=0, n_runs=1, tgt_neuron_loc=(27, 27)):
    """
    Various Contour Spacing - Figure 2, B4

    :param tgt_neuron_loc:
    :param frag:
    :param l1_act_cb:
    :param l2_act_cb:
    :param cont_gen_cb:
    :param tgt_filt_idx:
    :param smoothing:
    :param row_offset:
    :param n_runs:

    :return: handle of figure where contour spacing vs gain is plotted
    """
    c_len = 7  # Ref uses a contour of length 7
    frag_len = frag.shape[0]

    # Contour spacing was specified as relative colinear spacing = the distance between the centers
    # of adjacent fragments / length of contour fragments (fixed at 0.4 visual degrees)
    # Relative colinear distance of [1, 1.2, 1.4, 1.6, 1.8, 1.9] were use.
    # We simulate similar relative colinear distance by inserting spacing between the contour squares.
    relative_colinear_dist_arr = np.array([1, 1.2, 1.4, 1.6, 1.8, 1.9])
    spacing_bw_tiles = np.floor(relative_colinear_dist_arr * frag_len) - frag_len

    tgt_neuron_l1_act = np.zeros((n_runs, len(spacing_bw_tiles)))
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
                offset=row_offset,
                smooth_tiles=smoothing
            )

            tgt_neuron_l1_act[run_idx, s_idx] = tgt_l1_activation[tgt_neuron_loc[0], tgt_neuron_loc[1]]
            tgt_neuron_l2_act[run_idx, s_idx] = tgt_l2_activation[tgt_neuron_loc[0], tgt_neuron_loc[1]]

            if n_runs == 1:
                fig1, fig2 = alex_net_utils.plot_l1_and_l2_activations(test_img, l1_act_cb, l2_act_cb, tgt_filt_idx)
                title = "Vertical contour Length=%d, Rel. colinear dist=%0.2f in a sea of fragments" \
                        % (c_len, (frag_len + spacing) / np.float(frag_len))
                fig1.suptitle(title)
                fig2.suptitle(title)

    # Plot the Contour Integration Gain of the target neuron as between contour spacing increases
    tgt_neuron_gain = tgt_neuron_l2_act / (tgt_neuron_l1_act + 1e-5)
    tgt_neuron_gain_mean = tgt_neuron_gain.mean(axis=0)
    tgt_neuron_gain_std = tgt_neuron_gain.std(axis=0)
    rcd = (spacing_bw_tiles + frag_len) / np.float(frag_len)

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)

    ax.errorbar(rcd, tgt_neuron_gain_mean, tgt_neuron_gain_std, marker='o')
    ax.set_xlabel("Contour Distance")
    ax.set_ylabel("Contour Integration Gain")
    ax.set_title("L2 (contour enhanced) gain as a function of contour length for Neuron @ (%d, %d, %d)"
                 % (tgt_neuron_loc[0], tgt_neuron_loc[1], tgt_filt_idx))

    # Plot Neurophysiological data Data from [Li-2006]
    with open('.//neuro_data//Li2006.pickle', 'rb') as handle:
        data = pickle.load(handle)

    ax.plot(data['contour_separation_ma_rcd'], data['contour_separation_ma_gain'],
            label='ma', color='r', marker='d', linestyle='dashed')
    ax.plot(data['contour_separation_mb_rcd'], data['contour_separation_mb_gain'],
            label='mb', color='g', marker='s', linestyle='dashed')

    ax.legend(loc='best')

    return f


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

    alex_net_utils.plot_l1_and_l2_kernel_and_contour_fragment(contour_integration_model, tgt_filter_index, fragment)

    main_contour_length_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.vertical_contour_generator,
        tgt_filter_index,
        use_smoothing,
        row_offset=0,
        n_runs=50
    )

    main_contour_spacing_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.vertical_contour_generator,
        tgt_filter_index,
        use_smoothing,
        row_offset=0,
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

    alex_net_utils.plot_l1_and_l2_kernel_and_contour_fragment(contour_integration_model, tgt_filter_index, fragment)

    main_contour_length_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.horizontal_contour_generator,
        tgt_filter_index,
        use_smoothing,
        row_offset=0,
        n_runs=50
    )

    main_contour_spacing_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.horizontal_contour_generator,
        tgt_filter_index,
        use_smoothing,
        row_offset=0,
        n_runs=50
    )

    # TODO: Complete Linear but diagonal contours
    # # --------------------------------------------------------------------------------------------
    # #  Diagonal Filters
    # # --------------------------------------------------------------------------------------------
    # tgt_filter_index = 54
    #
    # # Fragment is the target filter
    # conv1_weights = K.eval(contour_integration_model.layers[1].weights[0])
    # fragment = conv1_weights[:, :, :, tgt_filter_index]
    # # Scale the filter to lie withing [0, 255]
    # fragment = (fragment - fragment.min())*(255 / (fragment.max() - fragment.min()))
    # use_smoothing = True
    #
    # # max_active = alex_net_utils.find_most_active_l1_kernel_index(fragment, no_overlap_l1_act_cb, (27, 27))
    # # alex_net_utils.plot_l1_and_l2_kernel_and_contour_fragment(contour_integration_model, max_active, fragment)
    # # alex_net_utils.plot_l1_and_l2_kernel_and_contour_fragment(contour_integration_model, 54, fragment)
    #
    # img = np.zeros((227, 227, 3))
    # loc_x = np.array([86,  97, 108, 119, 130])
    # loc_y = np.array([98, 103, 108, 113, 118])

    # new_img = tile_image(img, fragment, (loc_x, loc_y), rotate=False, gaussian_smoothing=False)
    #
    # plt.figure()
    # plt.imshow(new_img / 255.0)
