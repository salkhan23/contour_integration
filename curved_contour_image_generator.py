# -------------------------------------------------------------------------------------------------
#  Generate curved contours embedded in a sea of distractors, similar to the stimuli used in
#  " Field, Hayes & Hess - 1993 - Contour Integration by the Human Visual System: Evidence for a
#   local association field "
#
# Author: Salman Khan
# Date  : 11/04/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

import keras.backend as K

import base_alex_net
import gabor_fits
import alex_net_utils

reload(base_alex_net)
reload(gabor_fits)
reload(alex_net_utils)


def normalize_fragment(frag):
    """
    Normalize fragment to the 0, 1 range
    :param frag:

    :return: Normalized fragment
    """
    return (frag - frag.min()) / (frag.max() - frag.min())


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
    if l1[1] >= r2[1] or l2[1] >= r1[1]:
        return False

        # Does one square lie above the other
    if l1[0] >= r2[0] or l2[0] >= r1[0]:
        return False

    return True


def _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, d, d_delta, frag_size, direction,
        random_frag_direction=False):

    # Orientation of the Gabor is wrt to y axis, change it so it is with respect to x-axis
    # as acc_angle (next location) is wrt to the x-axis.
    acc_angle = frag_params["theta_deg"] - 90

    tile_offset = np.zeros((2,), dtype=np.int)
    prev_tile_start = center_frag_start

    tile_starts = []

    if direction == 'rhs':
        d = d
    elif direction == 'lhs':
        d = -d
    else:
        raise Exception("Invalid Direction")

    for i in range(c_len // 2):

        if random_frag_direction:
            frag_direction = np.random.choice((-1, 1), size=1)
        else:
            frag_direction = 1

        acc_angle += beta * frag_direction
        # acc_angle = np.mod(acc_angle, 360)

        rotated_frag_params = frag_params.copy()
        rotated_frag_params['theta_deg'] = (acc_angle + 90)

        rotated_frag = gabor_fits.get_gabor_fragment(rotated_frag_params, frag.shape[0:2])

        # Different from conventional (x, y) co-ordinates, the origin of the displayed
        # array starts in the top left corner. x increases in the vertically down
        # direction while y increases in the horizontally right direction.

        tile_offset[0] = d * np.sin(acc_angle / 180.0 * np.pi)
        tile_offset[1] = -d * np.cos(acc_angle / 180.0 * np.pi)

        curr_tile_start = prev_tile_start + tile_offset
        # print("Current tile start {0}. (offsets {1}, previous {2}, acc_angle={3})".format(
        #     curr_tile_start, tile_offset, prev_tile_start, acc_angle))

        # check if the current tile overlaps with the previous tile
        # TODO: Check if current tile overlaps with ANY previous one.
        l1 = curr_tile_start
        r1 = l1 + frag_size
        l2 = prev_tile_start
        r2 = l2 + frag_size
        is_overlapping = do_tiles_overlap(l1, r1, l2, r2)

        if is_overlapping:
            print("Next Tile @ {0} overlaps with tile at location {1}".format(
                curr_tile_start, prev_tile_start))

            tile_offset[0] += d_delta * np.sin(acc_angle / 180.0 * np.pi)
            tile_offset[1] -= d_delta * np.cos(acc_angle / 180.0 * np.pi)

            curr_tile_start = prev_tile_start + tile_offset
            # print("Current tile start {0}. (offsets {1}, previous {2}, acc_angle={3})".format(
            #     curr_tile_start, tile_offset, prev_tile_start, acc_angle))

        img = alex_net_utils.tile_image(
            img,
            rotated_frag,
            curr_tile_start,
            rotate=False,
            gaussian_smoothing=False
        )

        prev_tile_start = curr_tile_start
        tile_starts.append(curr_tile_start)

    return img, tile_starts


def add_contour_path_constant_separation(img, frag, frag_params, c_len, beta, d):
    """
    Add curved contours to the test image as added in the ref. a constant separation (d)
    is projected from the previous tile to find the location of the next tile.

    If the tile overlaps, fragment separation is increased by a factor of d // 4.

    :param img:
    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param d:
    :return:
    """
    img_size = np.array(img.shape[0:2])
    img_center = img_size // 2

    frag_size = np.array(frag.shape[0:2])
    center_frag_start = img_center - (frag_size // 2)

    d_delta = d // 4

    img = alex_net_utils.tile_image(
        img,
        frag,
        center_frag_start,
        rotate=False,
        gaussian_smoothing=False,
    )
    c_tile_starts = [center_frag_start]

    img, tiles = _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, d, d_delta, frag_size, 'rhs',
        random_frag_direction=True)
    c_tile_starts.extend(tiles)

    img, tiles = _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, d, d_delta, frag_size, 'lhs',
        random_frag_direction=True)
    c_tile_starts.extend(tiles)

    # ---------------------------
    c_tile_starts = np.array(c_tile_starts)

    return img, c_tile_starts


def _add_single_side_of_contour_closest_nonoverlap(
        img, center_frag_start, frag, frag_params, c_len, beta, d, direction, random_frag_direction=False):

    # Orientation of the Gabor is wrt to y axis, change it so it is with respect to x-axis
    # as acc_angle (next location) is wrt to the x-axis.
    acc_angle = frag_params["theta_deg"] - 90

    tile_offset = np.zeros((2,), dtype=np.int)
    prev_tile_start = center_frag_start

    tile_starts = []

    if direction == 'rhs':
        d = d
    elif direction == 'lhs':
        d = -d
    else:
        raise Exception("Invalid Direction")

    for i in range(c_len // 2):

        if random_frag_direction:
            frag_direction = np.random.choice((-1, 1), size=1)
        else:
            frag_direction = 1

        acc_angle += (beta * frag_direction)
        acc_angle = np.mod(acc_angle, 360)

        rotated_frag_params = frag_params.copy()
        rotated_frag_params['theta_deg'] = (acc_angle + 90)

        rotated_frag = gabor_fits.get_gabor_fragment(rotated_frag_params, frag.shape[0:2])

        # Different from conventional (x, y) co-ordinates, the origin of the displayed
        # array starts in the top left corner. x increases in the vertically down
        # direction while y increases in the horizontally right direction.
        if acc_angle <= 45 or 135 <= acc_angle <= 235 or acc_angle >= 315:
            # print("Keep x constant")
            x_dir = np.sign(np.cos(acc_angle * np.pi / 180.0))

            tile_offset[1] = x_dir * d
            tile_offset[0] = -tile_offset[1] * np.tan(acc_angle * np.pi / 180.0)

        else:
            # print("Keep y constant")
            y_dir = np.sign(np.sin(acc_angle * np.pi / 180.0))

            tile_offset[0] = -y_dir * d
            tile_offset[1] = -tile_offset[0] / np.tan(acc_angle * np.pi / 180.0)

        curr_tile_start = prev_tile_start + tile_offset
        print("Current tile start {0}. (offsets {1}, previous {2}, acc_angle={3})".format(
            curr_tile_start, tile_offset, prev_tile_start, acc_angle))

        img = alex_net_utils.tile_image(
            img,
            rotated_frag,
            curr_tile_start,
            rotate=False,
            gaussian_smoothing=False
        )

        prev_tile_start = curr_tile_start
        tile_starts.append(curr_tile_start)

    return img, tile_starts


def add_contour_path_closest_nonoverlap(img, frag, frag_params, c_len, beta, d):
    """
    Add curved contours to the test image.Different from add_contour_path_constant_separation,
    either x or y direction is held constant rather than the diagonal. This ensures the tile
    does not overlap with the previous tile

    :param img:
    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param d:
    :return:
    """
    img_size = np.array(img.shape[0:2])
    img_center = img_size // 2

    frag_size = np.array(frag.shape[0:2])
    center_frag_start = img_center - (frag_size // 2)

    img = alex_net_utils.tile_image(
        img,
        frag,
        center_frag_start,
        rotate=False,
        gaussian_smoothing=False,
    )
    c_tile_starts = [center_frag_start]

    img, tiles = _add_single_side_of_contour_closest_nonoverlap(
        img, center_frag_start, frag, frag_params, c_len, beta, d, 'rhs', random_frag_direction=True)
    c_tile_starts.extend(tiles)

    img, tiles = _add_single_side_of_contour_closest_nonoverlap(
        img, center_frag_start, frag, frag_params, c_len, beta, d, 'lhs', random_frag_direction=True)
    c_tile_starts.extend(tiles)

    # ---------------------------
    c_tile_starts = np.array(c_tile_starts)

    return img, c_tile_starts


def get_nonoverlapping_bg_fragment(f_tile_start, c_tile_starts, c_tile_size, max_offset):
    """

    :param f_tile_start:
    :param c_tile_starts:
    :param c_tile_size:
    :param max_offset:
    :return:
    """
    for r_idx in range(max_offset):
        for c_idx in range(max_offset):

            l1 = f_tile_start + np.array([r_idx, c_idx])
            r1 = l1 + c_tile_size

            is_overlapping = False

            for c_tile in c_tile_starts:

                l2 = c_tile
                r2 = c_tile + c_tile_size

                # print("checking bg tile @ start {0} with contour tile @ start {1}".format(l1, l2))

                if do_tiles_overlap(l1, r1, l2, r2):
                    # print('overlaps!')
                    is_overlapping = True

            if not is_overlapping:
                return l1

    return None


def add_background_fragments(img, frag, c_frag_starts, f_tile_size, beta, frag_params):
    """

    :param img:
    :param frag:
    :param c_frag_starts:
    :param f_tile_size:
    :param beta:
    :param frag_params:

    :return: (1) image with background tiles added
             (2) array of bg fragment tiles
             (3) array of bg fragment tiles removed
             (4) array of bg fragment tiles that were relocated
    """
    img_size = np.array(img.shape[0:2])
    img_center = img_size // 2

    center_full_tile_start = img_center - f_tile_size // 2

    # Get start locations of all full tiles
    f_tile_starts = alex_net_utils.get_background_tiles_locations(
        frag_len=f_tile_size[0],
        img_len=img_size[0],
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=center_full_tile_start[0]
    )

    # Displace the stimulus fragment in each full tile
    max_displace = f_tile_size[0] - frag.shape[0]

    bg_frag_starts = f_tile_starts + \
        np.random.randint(0, max_displace, f_tile_starts.shape)

    # Remove or replace all tiles that overlap with contour path fragments
    # --------------------------------------------------------------------
    removed_bg_frag_starts = []
    relocate_bg_frag_starts = []

    for c_frag_start in c_frag_starts:

        c_frag_start = np.expand_dims(c_frag_start, axis=0)

        # Find overlapping background fragments
        dist_to_c_frag = np.linalg.norm(c_frag_start - bg_frag_starts, axis=1)
        # for ii, dist in enumerate(dist_to_c_frag):
        #     print("{0}: {1}".format(ii, dist))

        ovlp_bg_frag_idx_arr = np.argwhere(dist_to_c_frag <= frag.shape[0])
        # for idx in ovlp_bg_frag_idx_arr:
        #     print("contour fragment @ {0}, overlaps with bg fragment @ index {1} and location {2}".format(
        #         c_frag_start, idx, bg_frag_starts[idx, :]))

        ovlp_bg_frag_idx_to_remove = []

        for ii, bg_frag_idx in enumerate(ovlp_bg_frag_idx_arr):

            f_tile_start = f_tile_starts[bg_frag_idx, :]

            # Is relocation possible?
            novlp_bg_frag = get_nonoverlapping_bg_fragment(
                np.squeeze(f_tile_start, axis=0),
                c_frag_starts,
                frag.shape[0:2],
                max_displace
            )

            if novlp_bg_frag is not None:
                # print("Relocating tile @ {0} to {1}".format(bg_frag_starts[bg_frag_idx, :], novlp_bg_frag))

                bg_frag_starts[bg_frag_idx, :] = np.expand_dims(novlp_bg_frag, axis=0)
                relocate_bg_frag_starts.append(novlp_bg_frag)

            else:
                # print("Remove bg fragment at index {0}, location {1}".format(
                #     bg_frag_idx, bg_frag_starts[bg_frag_idx, :]))

                removed_bg_frag_starts.append(bg_frag_starts[bg_frag_idx, :])
                ovlp_bg_frag_idx_to_remove.append(bg_frag_idx)

        # Remove the tiles that cannot be replaced from bg_frag and bg_full lists
        bg_frag_starts = \
            np.delete(bg_frag_starts, ovlp_bg_frag_idx_to_remove, axis=0)

        f_tile_starts = \
            np.delete(f_tile_starts, ovlp_bg_frag_idx_to_remove, axis=0)

    removed_bg_frag_starts = np.array(removed_bg_frag_starts)
    removed_bg_frag_starts = np.squeeze(removed_bg_frag_starts, axis=1)

    relocate_bg_frag_starts = np.array(relocate_bg_frag_starts)

    # Now add the background fragment tiles
    # -------------------------------------

    rotated_frag_params = frag_params.copy()
    num_possible_rotations = 360 // beta

    for start in bg_frag_starts:

        rotated_frag_params['theta_deg'] = (np.random.randint(0, np.int(num_possible_rotations)) * beta)
        rotated_frag = gabor_fits.get_gabor_fragment(rotated_frag_params, frag.shape[0:2])

        img = alex_net_utils.tile_image(
            img,
            rotated_frag,
            start,
            rotate=False,
            gaussian_smoothing=False
        )

    return img, bg_frag_starts, removed_bg_frag_starts, relocate_bg_frag_starts


def generate_contour_images(
        n_images, frag, frag_params, c_len, beta, f_tile_size, destination, image_format='JPEG'):
    """

    # In the Ref, a visible stimulus of a small size is placed inside a large tile
    # Here, full tile refers to the large tile & fragment tile refers to the visible stimulus

    :param n_images:
    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param f_tile_size
    :param destination:
    :param image_format:

    :return:
    """
    img_size = np.array([227, 227, 3])

    print("Generating {0} images for fragment [orientation {1}, contour length {2},"
          "inter fragment rotation {3}]".format(n_images, frag_params['theta_deg'], c_len, beta))

    for img_idx in range(n_images):

        print("Image {}".format(img_idx))

        bg = np.int(np.mean(frag))
        img = np.ones(img_size, dtype=np.uint8) * bg

        img, c_frag_starts = add_contour_path_constant_separation(
            img, frag, frag_params, c_len, beta, f_tile_size[0])

        img, bg_frag_starts, removed_tiles, relocated_tiles = add_background_fragments(
            img, frag, c_frag_starts, f_tile_size, beta, frag_params)

        # # Highlight Contour tiles
        # img = alex_net_utils.highlight_tiles(img, fragment.shape[0:2], c_frag_starts)
        #
        # # Highlight Background Fragment tiles
        # img = alex_net_utils.highlight_tiles(img, fragment.shape[0:2], bg_frag_starts, edge_color=(0, 255, 0))
        #
        # # Highlight Removed tiles
        # img = alex_net_utils.highlight_tiles(img, fragment.shape[0:2], removed_tiles, edge_color=(0, 0, 255))
        #
        # # Highlight Relocated tiles
        # img = alex_net_utils.highlight_tiles(img, fragment.shape[0:2], relocated_tiles, edge_color=(0, 255, 255))
        #
        # # highlight full tiles
        # f_tile_starts = alex_net_utils.get_background_tiles_locations(
        #     frag_len=f_tile_size[0],
        #     img_len=img_size[0],
        #     row_offset=0,
        #     space_bw_tiles=0,
        #     tgt_n_visual_rf_start=img_size[0] // 2 - (f_tile_size[0] // 2)
        # )
        #
        # img = alex_net_utils.highlight_tiles(
        #     img, f_tile_size, f_tile_starts, edge_color=(255, 255, 0))

        # ------------------------------------------------------------
        filename = "orient_{0}_clen_{1}_beta_{2}__{3}".format(
            frag_params["theta_deg"], c_len, beta, img_idx)

        plt.imsave(os.path.join(destination, filename + '.jpg'), img, format=image_format)


def plot_fragment_rotations(frag, frag_params, delta_rot=15):
    """
    Plot all possible rotations (multiples of  delta_rot) of the specified fragment

    :param frag:
    :param frag_params
    :param delta_rot:
    :return: None
    """
    rot_ang_arr = np.arange(0, 180, delta_rot)

    n_rows = np.int(np.floor(np.sqrt(rot_ang_arr.shape[0])))
    n_cols = np.int(np.ceil(rot_ang_arr.shape[0] / n_rows))

    fig, ax_arr = plt.subplots(n_rows, n_cols)
    fig.suptitle("Rotations")

    rot_frag_params = frag_params.copy()

    for idx, rot_ang in enumerate(rot_ang_arr):

        rot_frag_params["theta_deg"] = rot_ang + frag_params['theta_deg']

        rot_frag = gabor_fits.get_gabor_fragment(rot_frag_params, frag.shape[0:2])

        row_idx = np.int(idx / n_cols)
        col_idx = idx - row_idx * n_cols
        # print(row_idx, col_idx)

        ax_arr[row_idx][col_idx].imshow(rot_frag)
        ax_arr[row_idx][col_idx].set_title("Angle = {}".format(rot_ang))


def get_gabor_from_target_filter(tgt_filt, match=None):
    """
    Get best Fit gabor from feature extracting kernel
    Best fit parameters for the channel with the highest absolute amplitude are used

    :param tgt_filt: target filter to match
    :param match: list of params names to match. Default is to match orientation (theta_deg) only

    Valid entries for the list are
    ['x0', y0', 'theta_deg', 'amp', 'sigma', 'lambda1', 'psi', 'gamma' ]

    :return: dictionary of best fit parameters
    """

    if match is None:
        match = ['theta_deg']

    params = {
        'x0': 0,
        'y0': 0,
        'theta_deg': 0,
        'amp': 1,
        'sigma': 2.5,
        'lambda1': 8,
        'psi': 0,
        'gamma': 1
    }

    best_fit_params_list = gabor_fits.find_best_fit_2d_gabor(tgt_filt)

    amp_max_value = 0
    amp_max_idx = 0

    for idx, fitted_params in enumerate(best_fit_params_list):

        if abs(fitted_params[3]) > amp_max_value:
            amp_max_value = abs(fitted_params[3])
            amp_max_idx = idx

    fit_params = {
        'x0': best_fit_params_list[amp_max_idx][0],
        'y0': best_fit_params_list[amp_max_idx][1],
        'theta_deg': best_fit_params_list[amp_max_idx][2],
        'amp': best_fit_params_list[amp_max_idx][3],
        'sigma': 3,
        'lambda1': best_fit_params_list[amp_max_idx][5],
        'psi': best_fit_params_list[amp_max_idx][6],
        'gamma': best_fit_params_list[amp_max_idx][7]
    }

    for key in match:
        if key in params:
            params[key] = fit_params[key]
            print("Matching {0}={1}".format(key, params[key]))

    return params


def get_mean_pixel_value_at_boundary(frag, width=1):
    """

    :return:
    """
    x_top = frag[0:width, :, :]
    x_bottom = frag[-width:, :, :]

    y_top = frag[:, 0:width, :]
    y_bottom = frag[:, -width:, :]

    y_top = np.transpose(y_top, axes=(1, 0, 2))
    y_bottom = np.transpose(y_bottom, axes=(1, 0, 2))

    border_points = np.array([x_top, x_bottom, y_top, y_bottom])

    print x_top.shape
    print(border_points.shape)



    mean_border_value = np.mean(border_points, axis=(0, 1, 2))



    mean_border_value = [np.uint8(ch) for ch in mean_border_value]

    print("Mean border value {}".format(mean_border_value))

    return mean_border_value


if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    #  Target Feature and its orientation
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 10
    tgt_filter = base_alex_net.get_target_feature_extracting_kernel(tgt_filter_idx)

    # # Display the target filter
    # plt.figure()
    # plt.imshow(normalize_fragment(tgt_filter))
    # plt.title('Target Filter')

    # -----------------------------------------------------------------------------------
    #  Contour Fragment
    # -----------------------------------------------------------------------------------
    fragment_gabor_params = get_gabor_from_target_filter(
        tgt_filter,
        # match=[ 'x0', 'y0', 'theta_deg', 'amp', 'sigma', 'lambda1', 'psi', 'gamma']
        # match=['x0', 'y0', 'theta_deg', 'amp', 'psi', 'gamma']
        match=[ 'theta_deg']
    )

    fragment_gabor_params['theta_deg'] = np.int(fragment_gabor_params['theta_deg'])

    fragment = gabor_fits.get_gabor_fragment(
        fragment_gabor_params, tgt_filter.shape[0:2])

    # Display the contour fragment
    plt.figure()
    plt.imshow(fragment)
    plt.title("Contour Fragment")

    # # Plot rotations of the fragment
    # plot_fragment_rotations(fragment, fragment_gabor_params, delta_rot=15)

    # -----------------------------------------------------------------------------------
    #  Initializations
    # -----------------------------------------------------------------------------------
    image_size = np.array([227, 227, 3])

    bg_value = np.mean(fragment, axis=(0, 1))
    bg_value = [np.uint8(chan) for chan in bg_value]

    bg_value = get_mean_pixel_value_at_boundary(fragment)

    test_image = np.ones(image_size, dtype=np.uint8) * bg_value


    beta_rotation = 15
    contour_len = 9

    # In the Ref, the visible portion of the fragment moves around inside large tiles.
    # Here, full tile refers to the large tile & fragment tile refers to the visible stimulus
    fragment_size = np.array(fragment.shape[0:2])
    full_tile_size = np.array([17, 17])

    # -----------------------------------------------------------------------------------
    #  Add the Contour Path
    # -----------------------------------------------------------------------------------
    test_image, path_fragment_starts = add_contour_path_constant_separation(
        test_image,
        fragment,
        fragment_gabor_params,
        contour_len,
        beta_rotation,
        full_tile_size[0],
    )

    # test_image, path_fragment_starts = add_contour_path_closest_nonoverlap(
    #     test_image,
    #     fragment,
    #     fragment_gabor_params,
    #     contour_len,
    #     beta_rotation,
    #     full_tile_size[0] - 2
    # )

    plt.figure()
    plt.imshow(test_image)
    plt.title("Contour Fragments")

    # ----------------------------------------------------------------------------------
    #  Add background Fragments
    # ----------------------------------------------------------------------------------
    test_image, bg_tiles, bg_removed_tiles, bg_relocated_tiles = add_background_fragments(
        test_image,
        fragment,
        path_fragment_starts,
        full_tile_size,
        beta_rotation,
        fragment_gabor_params
    )

    plt.figure()
    plt.imshow(test_image)
    plt.title("Contour of length {0} embedded in a sea of distractors".format(contour_len))

    # -----------------------------------------------------------------------------------
    # Debugging Plots
    # -----------------------------------------------------------------------------------
    # Highlight Contour Fragments
    test_image = alex_net_utils.highlight_tiles(
        test_image, fragment.shape[0:2], path_fragment_starts)

    # # Highlight background fragment Tiles
    # test_image = alex_net_utils.highlight_tiles(
    #     test_image, fragment.shape[0:2], bg_tiles, edge_color=[0, 255, 0])

    # # Highlight Removed tiles
    # test_image = alex_net_utils.highlight_tiles(
    #     test_image, fragment.shape[0:2], bg_removed_tiles, edge_color=[0, 0, 255])

    # # Highlight Relocated tiles
    # test_image = alex_net_utils.highlight_tiles(
    #     test_image, fragment.shape[0:2], bg_relocated_tiles, edge_color=[255, 0, 255])

    # Highlight Full Tiles
    bg_tile_starts = alex_net_utils.get_background_tiles_locations(
        frag_len=full_tile_size[0],
        img_len=image_size[0],
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=image_size[0] // 2 - (full_tile_size[0] // 2)
    )

    test_image = alex_net_utils.highlight_tiles(
        test_image, full_tile_size, bg_tile_starts, edge_color=(255, 255, 0))

    plt.figure()
    plt.imshow(test_image)
    plt.title('Debugging Image')
