# -------------------------------------------------------------------------------------------------
# Generate (curved contours embedded in a sea of distractors) stimuli similar to those used in
#    "Field, Hayes and Hess - 1993 - Contour Integration by the Human Visual System: Evidence for a
#     local association field"
#
# Author: Salman Khan
# Date  : 11/04/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imrotate

import keras.backend as K

import base_alex_net
import gabor_fits
import alex_net_utils

reload(alex_net_utils)
reload(base_alex_net)
reload(gabor_fits)

IMAGE_SIZE = np.array([227, 227, 3])


def get_target_feature_extract_kernel(tgt_filt_idx):
    """
    Return the target kernel @ the specified index from the first feature extracting
    covolutional layer of alex_net

    :param tgt_filt_idx: Index of Filter to retrieve
    :return: target feature extracting filter

    """
    alex_net_model = base_alex_net.alex_net("trained_models/AlexNet/alexnet_weights.h5")
    feat_extract_kernels = K.eval(alex_net_model.layers[1].weights[0])

    return feat_extract_kernels[:, :, :, tgt_filt_idx]


def normalize_tile(tile):
    return (tile - tile.min()) / (tile.max() - tile.min())


def get_fragment_from_feature_extract_kernel(tgt_filt, tgt_size):
    """
    Gabor Fit the provided target filter and generate a stimulus fragment of the
    provided size using the fitted parameters

    :param tgt_filt: Feature extracting kernel
    :param tgt_size: [x,y,z]. z = depth

    :return:
    """
    gabor_params = gabor_fits.find_best_fit_2d_gabor(tgt_filt)
    # print("Best Fit Gabor Parameters for target filter {0}".format(gabor_params))

    x = np.arange(-0.5, 0.5, 1.0 / tgt_size[0])
    y = np.arange(-0.5, 0.5, 1.0 / tgt_size[1])
    xx, yy = np.meshgrid(x, y)

    frag = np.zeros((tgt_size[0], tgt_size[1], tgt_size[2]))

    for idx in range(tgt_size[2]):
        x0, y0, theta, amp, sigma, lambda1, psi, gamma = gabor_params[idx]
        frag_slice = gabor_fits.gabor_2d((xx, yy), x0, y0, theta, amp, sigma, lambda1, psi, gamma)
        frag_slice = frag_slice.reshape(tgt_size[0:2])

        frag[:, :, idx] = frag_slice

    return frag


def plot_fragment_rotations(frag, delta_rot=15):
    """
    Plot all possible rotations (multiples of  delta_rot) of the specified fragment

    :param frag:
    :param delta_rot:

    :return: None
    """
    rot_ang_arr = np.arange(0, 180, delta_rot)

    n_rows = np.int(np.floor(np.sqrt(rot_ang_arr.shape[0])))
    n_cols = np.int(np.ceil(rot_ang_arr.shape[0] / n_rows))

    fig, ax_arr = plt.subplots(n_rows, n_cols)
    fig.suptitle("Rotations")

    for idx, rot_ang in enumerate(rot_ang_arr):
        rot_frag = imrotate(frag, rot_ang)

        row_idx = np.int(idx / n_cols)
        col_idx = idx - row_idx * n_cols
        # print(row_idx, col_idx)

        ax_arr[row_idx][col_idx].imshow(rot_frag)
        ax_arr[row_idx][col_idx].set_title("Angle = {}".format(rot_ang))


def do_tiles_overlap(l1, r1, l2, r2):
    """
    Rectangles are specified by two points, the (x,y) coordinates of the top left corner (l1)
    and bottom right corner

    Two rectangles do not overlap if one of the following conditions is true.
    1) One rectangle is above top edge of other rectangle.
    2) One rectangle is on left side of left edge of other rectangle.

    Ref:  https://www.geeksforgeeks.org/find-two-rectangles-overlap/

    Different from Ref, and more aligned with the coordinates system in the rest of the file, x
    controls vertical while y controls horizontal

    :param l1: top left corner of tile 1
    :param r1: bottom right corner of tile 1
    :param l2:
    :param r2:

    :return:  True of the input tiles overlap, false otherwise
    """

    # Does one square lie to the Left of the other
    if l1[1] > r2[1] or l2[1] > r1[1]:
        return False

    # Does one square lie above the other
    if l1[0] > r2[0] or l2[0] > r1[0]:
        return False

    return True


if __name__ == '__main__':

    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    # 1. Get the target feature extracting kernel
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 5
    tgt_filter = get_target_feature_extract_kernel(tgt_filter_idx)

    # # Display the target filter
    # plt.figure()
    # plt.imshow(normalize_tile(tgt_filter))
    # plt.title( "Target Filter @ index {0}".format(tgt_filter_idx))

    # -----------------------------------------------------------------------------------
    #  2. Get the contour fragment to use
    # -----------------------------------------------------------------------------------
    # In [Fields - 1993], each tile (full tile is 32x32). However, only about a forth
    # of the tile is occupied by the stimulus. These two regions are represented
    # separately as full_tile and stim_tile respectively.
    # TODO: Match stimulus sizes to those of ref.
    stim_tile_size = np.array([9, 9, 3])
    full_tile_size = np.array([15, 15, 3])

    # # Stimuli Generated From the feature extracting kernel
    # stim_tile_raw = get_fragment_from_feature_extract_kernel(tgt_filter, stim_tile_size)
    # stim_tile_raw = normalize_tile(stim_tile_raw)

    # # # Manually specified simpler contour fragment
    stim_tile_raw = np.zeros((stim_tile_size[0], stim_tile_size[1], 3))
    stim_tile_raw[0:6, :, :] = 255.0
    stim_tile_raw = normalize_tile(stim_tile_raw)

    # Smooth the tile so it goes to zero @ edges
    g_kernel = alex_net_utils.get_2d_gaussian_kernel(stim_tile_size[0:2], sigma=0.75)
    stim_tile = stim_tile_raw * np.expand_dims(g_kernel, axis=2)

    # # Display the base fragment and the fragment smooth by a gaussian
    # f, ax_arr = plt.subplots(1, 3)
    #
    # ax_arr[0].imshow(stim_tile_raw)
    # ax_arr[0].set_title("original fragment")
    # ax_arr[1].imshow(np.squeeze(g_kernel))
    # ax_arr[1].set_title("smoothing kernel")
    # ax_arr[2].imshow(stim_tile)
    # ax_arr[2].set_title("Smooth Fragment")

    stim_tile = 255 * normalize_tile(stim_tile)

    # -----------------------------------------------------------------------------------
    # Check Rotations of the fragment look okay
    # -----------------------------------------------------------------------------------
    # plot_fragment_rotations(stim_tile, delta_rot=45)

    # -----------------------------------------------------------------------------------
    # 3. Initializations
    # -----------------------------------------------------------------------------------
    test_image = np.ones(IMAGE_SIZE, dtype='uint8') * 0

    # top left corner of central tile
    center_tile_center = (IMAGE_SIZE[0:2] // 2)
    center_stim_tile_start = center_tile_center - (stim_tile_size[0:2] // 2)
    center_full_tile_start = center_tile_center - (full_tile_size[0:2] // 2)

    full_tile_start_loc_arr = alex_net_utils.get_background_tiles_locations(
        frag_len=full_tile_size[0],
        img_len=IMAGE_SIZE[0],
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=center_full_tile_start[0]
    )

    beta = 15
    contour_len = 7

    path_fragment_dist = full_tile_size[0]
    dist_delta = path_fragment_dist // 4  # If tile overlaps with existing path tile

    # -----------------------------------------------------------------------------------
    # 4. Add the contour path
    # -----------------------------------------------------------------------------------
    # 4.a Central Tile
    test_image = alex_net_utils.tile_image(
        test_image,
        stim_tile,
        center_stim_tile_start,
        rotate=False,
        gaussian_smoothing=False
    )

    path_tile_centers = [center_tile_center]

    # # 4.b Right hand size of contour
    acc_angle = 0
    tile_offset = np.zeros((2,), dtype=np.int)
    prev_tile_center = center_tile_center

    for i in range(contour_len // 2):

        acc_angle += (np.random.choice((-1, 1), size=1) * beta)

        rotated_fragment = imrotate(stim_tile, angle=acc_angle)

        # Different from conventional (x, y) co-ordinates, the origin of the displayed
        # array starts in the top left corner. x increases in the vertically down
        # direction while y increases in the horizontally right direction.
        tile_offset[0] = -path_fragment_dist * np.sin(acc_angle / 180.0 * np.pi)
        tile_offset[1] = path_fragment_dist * np.cos(acc_angle / 180.0 * np.pi)

        curr_tile_center = prev_tile_center + tile_offset
        # print("Current tile center {0}. (offsets {1}, previous {2})".format(
        #     curr_tile_center,
        #     tile_offset,
        #     prev_tile_center
        # ))

        # check if the current tile overlaps with the previous tile
        # TODO: Check if current tile overlaps with ANY previous one.
        curr_tile_l = curr_tile_center - (path_fragment_dist // 2)
        curr_tile_r = curr_tile_center + (path_fragment_dist // 2)

        prev_tile_l = prev_tile_center - (path_fragment_dist // 2)
        prev_tile_r = prev_tile_center + (path_fragment_dist // 2)

        is_overlapping = do_tiles_overlap(curr_tile_l, curr_tile_r, prev_tile_l, prev_tile_r)

        if is_overlapping:
            print("Next Tile @ {0},{1} overlaps with tile at location {2},{3}".format(
                curr_tile_center[0],
                curr_tile_center[1],
                prev_tile_center[0],
                prev_tile_center[1]
            ))

            tile_offset[0] -= dist_delta * np.sin(acc_angle / 180.0 * np.pi)
            tile_offset[1] += dist_delta * np.cos(acc_angle / 180.0 * np.pi)

            curr_tile_center = prev_tile_center + tile_offset
            # print("Updated Current tile center {0}. (offsets {1}, previous {2})".format(
            #     curr_tile_center,
            #     tile_offset,
            #     prev_tile_center
            # ))

        test_image = alex_net_utils.tile_image(
            test_image,
            rotated_fragment,
            curr_tile_center - stim_tile_size[0:2] // 2,
            rotate=False,
            gaussian_smoothing=False
        )

        prev_tile_center = curr_tile_center
        path_tile_centers.append(curr_tile_center)

    # 4.c Left hand size of contour
    acc_angle = 0
    tile_offset = np.zeros((2,), dtype=np.int)
    prev_tile_center = center_tile_center

    for i in range(contour_len // 2):

        acc_angle += (np.random.choice((-1, 1), size=1) * beta)

        rotated_fragment = imrotate(stim_tile, angle=acc_angle)

        # Different from conventional (x, y) co-ordinates, the origin of the displayed
        # array starts in the top left corner. x increases in the vertically down
        # direction while y increases in the horizontally right direction.
        tile_offset[0] = +path_fragment_dist * np.sin(acc_angle / 180.0 * np.pi)
        tile_offset[1] = -path_fragment_dist * np.cos(acc_angle / 180.0 * np.pi)

        curr_tile_center = prev_tile_center + tile_offset
        # print("Current tile center {0}. (offsets {1}, previous {2})".format(
        #     curr_tile_center,
        #     tile_offset,
        #     prev_tile_center
        # ))

        # check if the current tile overlaps with the previous tile
        # TODO: Check if current tile overlaps with ANY previous one.
        curr_tile_l = curr_tile_center - (path_fragment_dist // 2)
        curr_tile_r = curr_tile_center + (path_fragment_dist // 2)

        prev_tile_l = prev_tile_center - (path_fragment_dist // 2)
        prev_tile_r = prev_tile_center + (path_fragment_dist // 2)

        is_overlapping = do_tiles_overlap(curr_tile_l, curr_tile_r, prev_tile_l, prev_tile_r)

        if is_overlapping:
            print("Next Tile @ {0},{1} overlaps with tile at location {2},{3}".format(
                curr_tile_center[0],
                curr_tile_center[1],
                prev_tile_center[0],
                prev_tile_center[1]
            ))

            tile_offset[0] += dist_delta * np.sin(acc_angle / 180.0 * np.pi)
            tile_offset[1] -= dist_delta * np.cos(acc_angle / 180.0 * np.pi)

            curr_tile_center = prev_tile_center + tile_offset
            # print("Updated Current tile center {0}. (offsets {1}, previous {2})".format(
            #     curr_tile_center,
            #     tile_offset,
            #     prev_tile_center
            # ))

        test_image = alex_net_utils.tile_image(
            test_image,
            rotated_fragment,
            curr_tile_center - stim_tile_size[0:2] // 2,
            rotate=False,
            gaussian_smoothing=False
        )

        prev_tile_center = curr_tile_center
        path_tile_centers.append(curr_tile_center)

    path_tile_centers = np.array(path_tile_centers)

    # Display the contour
    # plt.figure()
    # plt.imshow(test_image)
    # plt.title("Contour Image without any background")

    # -----------------------------------------------------------------------------------
    # 5. Place randomly distorted fragments in the background in unoccupied locations
    # -----------------------------------------------------------------------------------
    max_offset_full_and_stim_tiles = full_tile_size[0] - stim_tile_size[0]

    # Uniformly distribute the position of the stim tile within the full tile
    bg_stim_tile_start_loc_arr = full_tile_start_loc_arr + \
        np.random.randint(0, max_offset_full_and_stim_tiles, full_tile_start_loc_arr.shape)

    # Remove BG tile that overlap with the contour
    replaced_tile_loc_arr = []

    for path_tile_center in path_tile_centers:

        path_tile_center = np.expand_dims(path_tile_center, axis=0)
        path_tile_start = path_tile_center - (stim_tile_size[0: 2] // 2)

        dist_to_bg_tiles = np.linalg.norm(path_tile_start - bg_stim_tile_start_loc_arr, axis=1)

        overlapping_tile_idx_arr = np.argwhere(dist_to_bg_tiles <= stim_tile_size[0])

        # for idx, dist in enumerate(dist_to_bg_tiles):
        #     print("{0}: {1}".format(idx, dist))
        #
        # print("path tile {0} overlaps with bg tiles at index {1}".format(
        #     path_tile_center, overlapping_tile_idx_arr
        # ))
        # raw_input()

        for idx in overlapping_tile_idx_arr:
            replaced_tile_loc_arr.append(bg_stim_tile_start_loc_arr[idx, :], )

        bg_stim_tile_start_loc_arr = \
            np.delete(bg_stim_tile_start_loc_arr, overlapping_tile_idx_arr, axis=0)

    replaced_tile_loc_arr = np.array(replaced_tile_loc_arr)
    replaced_tile_loc_arr = np.squeeze(replaced_tile_loc_arr, axis=1)

    # Add BG tiles
    test_image = alex_net_utils.tile_image(
        test_image,
        stim_tile,
        bg_stim_tile_start_loc_arr,
        rotate=True,
        delta_rotation=beta,
        gaussian_smoothing=False
    )

    # Display the Final Image
    plt.figure()
    plt.imshow(test_image)
    plt.title("Final Image")

    # -----------------------------------------------------------------------------------
    # Debugging Plots
    # -----------------------------------------------------------------------------------
    # Highlight Contour Tiles
    contour_highlighted_image = alex_net_utils.highlight_tiles(
        test_image,
        stim_tile_size[0:2],
        path_tile_centers - (stim_tile_size[0:2] // 2)
    )

    plt.figure()
    plt.imshow(contour_highlighted_image)
    plt.title("Contour fragments Highlighted")

    # Highlight Background Full Tiles
    bg_tiles_image = alex_net_utils.highlight_tiles(
        contour_highlighted_image,
        full_tile_size[0:2],
        full_tile_start_loc_arr,
        edge_color=[0, 255, 0]
    )

    plt.figure()
    plt.imshow(bg_tiles_image)
    plt.title("Background Tiles highlighted")

    # Highlight Removed background Tiles
    removed_bg_tiles_image = alex_net_utils.highlight_tiles(
        bg_tiles_image,
        stim_tile_size[0:2],
        np.array(replaced_tile_loc_arr),
        edge_color=[0, 0, 255]
    )

    plt.figure()
    plt.imshow(removed_bg_tiles_image)
    plt.title("Removed Bg Tiles highlighted")
