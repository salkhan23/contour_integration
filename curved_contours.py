# -------------------------------------------------------------------------------------------------
#  Curve Contours Image Generation Based on
#   "Field, Hayes and Hess -1993 - Contour Integration by the Human Visual System: Evidence for a
#    local Association Field"
#
# Author: Salman Khan
# Date  : 29/03/18
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imrotate

import keras.backend as K

import base_alex_net
import gabor_fits
import alex_net_utils

reload(alex_net_utils)

IMAGE_SIZE = np.array([227, 227, 3])


def get_stimulus_tile(tgt_filt, tgt_size):
    """
    Gabor Fit the provided target filter and generate a stimulus fragment of the provided size
    using the fitted parameters

    :param tgt_filt:
    :param tgt_size: [x,y,z]. z = depth
    :return:
    """
    gabor_params = gabor_fits.find_best_fit_2d_gabor(tgt_filt)
    # print("Best Fit Gabor Parameters for target filter {0}".format(gabor_params))

    x = np.arange(-0.5, 0.5, 1.0 / tgt_size[0])
    y = np.arange(-0.5, 0.5, 1.0 / tgt_size[1])
    xx, yy = np.meshgrid(x, y)

    # find best fit gabors for each channel
    frag = np.zeros((tgt_size[0], tgt_size[1], tgt_size[2]))

    for idx in range(tgt_size[2]):
        x0, y0, theta, amp, sigma, lambda1, psi, gamma = gabor_params[idx]
        frag_slice = gabor_fits.gabor_2d((xx, yy), x0, y0, theta, amp, sigma, lambda1, psi, gamma)
        frag_slice = frag_slice.reshape(tgt_size[0:2])

        frag[:, :, idx] = frag_slice

    return frag


def normalize_tile(tile):
    return (tile - tile.min()) / (tile.max() - tile.min())


def do_tiles_overlap(l1, r1, l2, r2):
    """
    Rectangles are specified by two points, the (x,y) coordinates of the top left corner (l1)
    and bottom right corner

    Two rectangles do not overlap if one of the following conditions is true.
    1) One rectangle is above top edge of other rectangle.
    2) One rectangle is on left side of left edge of other rectangle.

    Ref:  https://www.geeksforgeeks.org/find-two-rectangles-overlap/
    Different from the reference the y-axis is in the opposite direction. As you go down, y value increases
    This changes the direction of the y axis comparisons

    :param l1: top left corner of tile 1
    :param r1: bottom right corner of tile 1
    :param l2:
    :param r2:

    :return:  True of the input tiles overlap, false otherwise
    """
    # Does one square lie to the Left of the other
    if l1[0] > r2[0] or l2[0] > r1[0]:
        return False

    # Does one square lie above the other
    if l1[1] > r2[1] or l2[1] > r1[1]:
        return False

    return True


if __name__ == '__main__':

    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # ------------------------------------------------------------------------------------
    # 1. Get the target feature extracting kernel
    # ------------------------------------------------------------------------------------
    tgt_filter_idx = 5

    alex_net_model = base_alex_net.alex_net("trained_models/AlexNet/alexnet_weights.h5")

    feature_extract_kernels = K.eval(alex_net_model.layers[1].weights[0])

    tgt_filter = feature_extract_kernels[:, :, :, tgt_filter_idx]

    # # Display the target filter
    # display_filter = (tgt_filter - tgt_filter.min()) / (tgt_filter.max() - tgt_filter.min())
    # plt.figure()
    # plt.imshow(display_filter)
    # plt.title( "Target Filter @ index {0}".format(tgt_filter_idx))

    # -----------------------------------------------------------------------------------
    #  2. Get the contour fragment to use
    # -----------------------------------------------------------------------------------
    stim_tile_size = np.array([9, 9, 3])
    full_tile_size = np.array([15, 15, 3])

    # stim_tile = get_stimulus_tile(tgt_filter, stim_tile_size)
    # stim_tile = normalize_tile(stim_tile)

    # Manually Created simpler contour fragment
    stim_tile = np.zeros((stim_tile_size[0], stim_tile_size[1], 3))
    stim_tile[0:6, :, :] = 255.0

    # Smooth the tile so it goes to zero @ edges
    g_kernel = alex_net_utils.get_2d_gaussian_kernel(stim_tile_size[0:2], sigma=0.5)
    smooth_stim_tile = stim_tile * np.expand_dims(g_kernel, axis=2)

    # Normalize to [0, 255] - imrotate returns fragments with pixel values in the range [0, 255]
    smooth_stim_tile = normalize_tile(smooth_stim_tile)
    #
    # # Display the base fragment and the fragment smooth by a gaussian
    # f, ax_arr = plt.subplots(1, 3)
    #
    # ax_arr[0].imshow(stim_tile)
    # ax_arr[0].set_title("original fragment")
    #
    # ax_arr[1].imshow(np.squeeze(g_kernel))
    # ax_arr[1].set_title("smoothing kernel")
    #
    # ax_arr[2].imshow(smooth_stim_tile)
    # ax_arr[2].set_title("Smooth Fragment")

    # -----------------------------------------------------------------------------------
    # Check Rotations of the fragment look okay
    # -----------------------------------------------------------------------------------
    smooth_stim_tile = imrotate(smooth_stim_tile, 0)
    rot_ang_arr = np.arange(0, 180, 15)

    n_rows = np.int(np.floor(np.sqrt(rot_ang_arr.shape[0])))
    n_cols = np.int(np.ceil(rot_ang_arr.shape[0] / n_rows))

    fig, ax_arr = plt.subplots(n_rows, n_cols)
    fig.suptitle("Rotations")

    for idx, rot_ang in enumerate(rot_ang_arr):

        rot_fragment = imrotate(smooth_stim_tile, rot_ang)

        row_idx = np.int(idx/n_cols)
        col_idx = idx - row_idx * n_cols
        # print(row_idx, col_idx)

        ax_arr[row_idx][col_idx].imshow(rot_fragment)
        ax_arr[row_idx][col_idx].set_title("Angle = {}".format(rot_ang))

    # --------------------------------------------------------------------------------------
    # 3. Initializations
    # --------------------------------------------------------------------------------------
    test_image = np.zeros(IMAGE_SIZE, dtype='uint8')

    center_tile_start = (IMAGE_SIZE[0:2] // 2) - (full_tile_size[0:2] // 2)  # top left corner of central tile

    full_tile_start_loc_arr = alex_net_utils.get_background_tiles_locations(
        frag_len=full_tile_size[0],
        img_len=IMAGE_SIZE[0],  # TODO: this should take (x,y)not only x
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=center_tile_start[0]  # TODO: this should take (x,y) not only x
    )

    tile_start_loc_arr = np.array(full_tile_start_loc_arr)

    beta = 15
    contour_len = 7

    path_fragment_dist = full_tile_size[0]

    dist_delta = path_fragment_dist // 4  # In case tile overlaps with another path tile

    # --------------------------------------------------------------------------------------
    # 4. Add the contour path
    # --------------------------------------------------------------------------------------
    # 4.a Central tile
    central_stim_tile_center = (IMAGE_SIZE[0:2] // 2)

    test_image = alex_net_utils.tile_image(
        test_image,
        smooth_stim_tile,
        central_stim_tile_center - (stim_tile_size[0:2] // 2),
        rotate=False,
        gaussian_smoothing=False
    )

    path_tile_centers = [central_stim_tile_center]

    # 4.b Right hand size of contour
    acc_angle = 0
    tile_offset = np.zeros((2,))
    prev_tile_center = central_stim_tile_center

    for i in range(contour_len // 2):

        acc_angle += beta
        rotated_fragment = imrotate(smooth_stim_tile, angle=acc_angle)

        tile_offset[0] = path_fragment_dist * np.cos(acc_angle / 180.0 * np.pi)
        tile_offset[1] = -path_fragment_dist * np.sin(acc_angle / 180.0 * np.pi)  # TODO: check sign

        curr_tile_center = np.floor(prev_tile_center + tile_offset)
        curr_tile_center = np.array([int(loc) for loc in curr_tile_center])

        # check if the current tile overlaps with the previous tile
        # TODO: Check if current tile overlaps with ANY previous one.
        curr_tile_l = curr_tile_center - (path_fragment_dist // 2)
        curr_tile_r = curr_tile_center + (path_fragment_dist // 2)

        prev_tile_l = prev_tile_center - (path_fragment_dist // 2)
        prev_tile_r = prev_tile_center + (path_fragment_dist // 2)

        overlapping_rectangles = do_tiles_overlap(curr_tile_l, curr_tile_r, prev_tile_l, prev_tile_r)

        if overlapping_rectangles:
            print("Next Tile @ {0},{1} overlaps with tile at location {2},{3}".format(
                curr_tile_center[0],
                curr_tile_center[1],
                prev_tile_center[0],
                prev_tile_center[1]
            ))

            tile_offset[0] += (dist_delta * np.cos(acc_angle / 180.0 * np.pi))
            tile_offset[0] -= (dist_delta * np.sin(acc_angle / 180.0 * np.pi))

            curr_tile_center = np.floor(prev_tile_center + tile_offset)
            curr_tile_center = np.array([int(loc) for loc in curr_tile_center])

            print("Updated Tile Center {0}".format(curr_tile_center))

        test_image = alex_net_utils.tile_image(
            test_image,
            rotated_fragment,
            np.flip(curr_tile_center, 0) - (stim_tile_size[0:2] // 2),
            rotate=False,
            gaussian_smoothing=False
        )

        prev_tile_center = curr_tile_center
        path_tile_centers.append(curr_tile_center)

    # 4.c Left hand size of contour
    acc_angle = 0
    tile_offset = np.zeros((2,))
    prev_tile_center = central_stim_tile_center

    for i in range(contour_len // 2):

        acc_angle += beta
        rotated_fragment = imrotate(smooth_stim_tile, angle=acc_angle)

        tile_offset[0] = -path_fragment_dist * np.cos(acc_angle / 180.0 * np.pi)
        tile_offset[1] = +path_fragment_dist * np.sin(acc_angle / 180.0 * np.pi)  # TODO: check sign

        curr_tile_center = np.floor(prev_tile_center + tile_offset)
        curr_tile_center = np.array([int(loc) for loc in curr_tile_center])

        # check if the current tile overlaps with the previous tile
        # TODO: Check if current tile overlaps with ANY previous one.
        curr_tile_l = curr_tile_center - (path_fragment_dist // 2)
        curr_tile_r = curr_tile_center + (path_fragment_dist // 2)

        prev_tile_l = prev_tile_center - (path_fragment_dist // 2)
        prev_tile_r = prev_tile_center + (path_fragment_dist // 2)

        overlapping_rectangles = do_tiles_overlap(curr_tile_l, curr_tile_r, prev_tile_l, prev_tile_r)

        if overlapping_rectangles:
            print("Next Tile @ {0},{1} overlaps with tile at location {2},{3}".format(
                curr_tile_center[0],
                curr_tile_center[1],
                prev_tile_center[0],
                prev_tile_center[1]
            ))

            tile_offset[0] -= (dist_delta * np.cos(acc_angle / 180.0 * np.pi))
            tile_offset[0] += (dist_delta * np.sin(acc_angle / 180.0 * np.pi))

            curr_tile_center = np.floor(prev_tile_center + tile_offset)
            curr_tile_center = np.array([int(loc) for loc in curr_tile_center])

            print("Updated Tile Center {0}".format(curr_tile_center))

        test_image = alex_net_utils.tile_image(
            test_image,
            rotated_fragment,
            np.flip(curr_tile_center, 0) - (stim_tile_size[0:2] // 2),
            rotate=False,
            gaussian_smoothing=False
        )

        prev_tile_center = curr_tile_center
        path_tile_centers.append(curr_tile_center)

    path_tile_centers = np.array(path_tile_centers)

    # Display the contour path
    plt.figure()
    plt.imshow(test_image)
    plt.title("Contour Path without any background")

    # --------------------------------------------------------------------------------------
    # 5. Place randomly distorted fragments in the background in unoccupied locations
    # --------------------------------------------------------------------------------------
    for path_tile_center in path_tile_centers:

        path_tile_start_loc = path_tile_center - (full_tile_size[0: 2] // 2)
        path_tile_start_loc = np.expand_dims(path_tile_start_loc, axis=1)

        dist_to_bg_tiles = np.linalg.norm(path_tile_start_loc - tile_start_loc_arr, axis=0)

        overlapping_bg_tile_idx = np.argmin(dist_to_bg_tiles)

        # print("Path Tile center {}".format(path_tile_start_loc))
        # print("Bg tile that overlaps index {}".format(overlapping_bg_tile_idx))
        # print("Bg tile that overlaps location {}".format(tile_start_loc_arr[:, overlapping_bg_tile_idx]))
        # raw_input()

        tile_start_loc_arr = np.delete(tile_start_loc_arr, overlapping_bg_tile_idx, axis=1)

    # Add an random offset to the tile star location
    bg_stim_tile_start_loc_arr = tile_start_loc_arr + \
                                 np.random.randint(0, full_tile_size[0] - stim_tile_size[0], tile_start_loc_arr.shape)

    test_image = alex_net_utils.tile_image(
        test_image,
        smooth_stim_tile,
        np.array([bg_stim_tile_start_loc_arr[1, :], bg_stim_tile_start_loc_arr[0, :]]),
        rotate=True,
        gaussian_smoothing=False
    )

    plt.figure()
    plt.imshow(test_image)
