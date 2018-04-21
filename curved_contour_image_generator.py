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
from scipy.misc import imrotate

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
        img, center_frag_start, frag, frag_orientation, c_len, beta, d, d_delta, frag_size, direction,
        random_frag_direction=False):

    acc_angle = frag_orientation
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
        acc_angle = np.mod(acc_angle, 360)

        rotated_frag = imrotate(frag, angle=acc_angle - frag_orientation)

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


def add_contour_path_constant_separation(img, frag, frag_orientation, c_len, beta, d):
    """
    Add curved contours to the test image as added in the ref. a constant separation (d)
    is projected from the previous tile to find the location of the next tile.

    If the tile overlaps, fragment separation is increased by a factor of d // 4.

    :param img:
    :param frag:
    :param frag_orientation:
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
        img, center_frag_start, frag, frag_orientation, c_len, beta, d, d_delta, frag_size, 'rhs',
        random_frag_direction=True)
    c_tile_starts.extend(tiles)

    img, tiles = _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_orientation, c_len, beta, d, d_delta, frag_size, 'lhs',
        random_frag_direction=True)
    c_tile_starts.extend(tiles)

    # ---------------------------
    c_tile_starts = np.array(c_tile_starts)

    return img, c_tile_starts


def _add_single_side_of_contour_closest_nonoverlap(
        img, center_frag_start, frag, frag_orientation, c_len, beta, d, direction, random_frag_direction=False):

    acc_angle = frag_orientation
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

        rotated_frag = imrotate(frag, angle=acc_angle - frag_orientation)

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


def add_contour_path_closest_nonoverlap(img, frag, frag_orientation, c_len, beta, d):
    """
    Add curved contours to the test image.Different from add_contour_path_constant_separation,
    either x or y direction is held constant rather than the diagonal. This ensures the tile
    does not overlap with the previous tile

    :param img:
    :param frag:
    :param frag_orientation:
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
        img, center_frag_start, frag, frag_orientation, c_len, beta, d, 'rhs', random_frag_direction=True)
    c_tile_starts.extend(tiles)

    img, tiles = _add_single_side_of_contour_closest_nonoverlap(
        img, center_frag_start, frag, frag_orientation, c_len, beta, d, 'lhs', random_frag_direction=True)
    c_tile_starts.extend(tiles)

    # ---------------------------
    c_tile_starts = np.array(c_tile_starts)

    return img, c_tile_starts


def get_nonoverlapping_bg_fragment(f_tile_start, c_tile_start, c_tile_size, max_offset):
    """

    :param f_tile_start:
    :param c_tile_start:
    :param c_tile_size:
    :param max_offset:
    :return:
    """

    l1 = c_tile_start
    r1 = c_tile_start + c_tile_size

    for r_idx in range(max_offset):
        for c_idx in range(max_offset):

            l2 = f_tile_start + np.array([r_idx, c_idx])
            r2 = l2 + c_tile_size

            # print("checking tile @ start {0}. contour tile @ start {1}".format(l2, l1))

            if not do_tiles_overlap(l1, r1, l2, r2):
                return l2

            # print("Overlaps!")

    return None


def add_background_fragments(img, frag, c_frag_starts, f_tile_size, beta):
    """

    :param img:
    :param frag:
    :param c_frag_starts:
    :param f_tile_size:
    :param beta:

    :return: (1) image with background tiles added
             (2) array of bg frag tiles
             (3) array of bg frag tiles removed
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

    for c_frag_start in c_frag_starts:

        c_frag_start = np.expand_dims(c_frag_start, axis=0)

        dist_to_c_frag = np.linalg.norm(c_frag_start - bg_frag_starts, axis=1)
        ovlp_bg_frag_idx_arr = np.argwhere(dist_to_c_frag <= frag.shape[0])

        # for ii, dist in enumerate(dist_to_c_frag):
        #     print("{0}: {1}".format(ii, dist))
        # print("path tile {0} overlaps with bg tiles at index {1}".format(
        #     c_frag_start, ovlp_bg_frag_idx_arr))

        # First see if the overlapping bg fragment can be replaced with a non-overlapping one
        ovlp_bg_frag_idx_to_remove = []

        for ii, bg_frag_idx in enumerate(ovlp_bg_frag_idx_arr):

            f_tile_start = f_tile_starts[bg_frag_idx, :]

            novlp_bg_frag = get_nonoverlapping_bg_fragment(
                np.squeeze(f_tile_start, axis=0),
                np.squeeze(c_frag_start, axis=0),
                frag.shape[0:2],
                max_displace
            )
            # print("Non_overlapping Tile {0}".format(novlp_bg_frag))

            if novlp_bg_frag is not None:
                # print("Replacing tile @ {0} with tile @".format(
                #     bg_frag_starts[bg_frag_idx, :], novlp_bg_frag))

                bg_frag_starts[bg_frag_idx, :] = np.expand_dims(novlp_bg_frag, axis=0)

            else:
                removed_bg_frag_starts.append(bg_frag_starts[bg_frag_idx, :])
                ovlp_bg_frag_idx_to_remove.append(bg_frag_idx)

        # Remove the tiles that cannot be replaced from bg_frag and bg_full lists
        bg_frag_starts = \
            np.delete(bg_frag_starts, ovlp_bg_frag_idx_to_remove, axis=0)

        f_tile_starts = \
            np.delete(f_tile_starts, ovlp_bg_frag_idx_to_remove, axis=0)

    removed_bg_frag_starts = np.array(removed_bg_frag_starts)
    removed_bg_frag_starts = np.squeeze(removed_bg_frag_starts, axis=1)

    # Now add the background fragment tiles
    # -------------------------------------
    img = alex_net_utils.tile_image(
        img,
        frag,
        bg_frag_starts,
        rotate=True,
        delta_rotation=beta,
        gaussian_smoothing=False
    )

    return img, bg_frag_starts, removed_bg_frag_starts


def generate_contour_images(n_images, tgt_filt_idx, c_len, beta, destination, image_format='JPEG'):
    """

    :param n_images:
    :param tgt_filt_idx:
    :param c_len:
    :param beta:
    :param destination:
    :param image_format:

    :return:
    """
    tgt_filt = base_alex_net.get_target_feature_extracting_kernel(tgt_filt_idx)

    # Best fits angle is wrt the y axis (theta = 0), change it to be  wrt to the x axis
    tgt_filt_orientation = np.int(gabor_fits.get_filter_orientation(tgt_filt, o_type='average'))
    tgt_filt_orientation = np.int(np.floor(90 + tgt_filt_orientation))

    print("Target Filter Index {0}, orientation {1}".format(tgt_filt_idx, tgt_filt_orientation))

    # Fragment
    x = np.linspace(-1, 1, tgt_filt.shape[0])
    y = np.copy(x)
    x_mesh, y_mesh = np.meshgrid(x, y)

    frag = gabor_fits.gabor_2d(
        (x_mesh, y_mesh),
        x0=0,
        y0=0,
        theta_deg=tgt_filt_orientation - 90,
        amp=1,
        sigma=0.6,
        lambda1=3,
        psi=0,
        gamma=1
    )
    frag = frag.reshape((x.shape[0], y.shape[0]))
    frag = np.stack((frag, frag, frag), axis=2)

    frag = normalize_fragment(frag)
    frag = imrotate(frag, 0)

    img_size = np.array([227, 227, 3])

    # In the Ref, the visible portion of the fragment moves around inside large tiles.
    # Here, full tile refers to the large tile & fragment tile refers to the visible stimulus
    f_tile_size = np.array([17, 17])

    for img_idx in range(n_images):
        img = np.zeros(img_size, dtype=np.uint8)

        img, c_frag_starts = add_contour_path_constant_separation(
            img, frag, tgt_filt_orientation, c_len, beta, f_tile_size[0])

        img, bg_frag_starts, _ = add_background_fragments(
            img, frag, c_frag_starts, f_tile_size, beta)

        filename = "img_{0}_filt_orient_{1}_clen_{2}_beta_{3}".format(
            img_idx, tgt_filt_orientation, c_len, beta)

        print os.path.join(destination, filename + '.jpg')

        plt.imsave(os.path.join(destination, filename + '.jpg'), img, format=image_format)


if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    #  Target Feature and its orientation
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 10
    tgt_filter = base_alex_net.get_target_feature_extracting_kernel(tgt_filter_idx)

    # Best fits angle is wrt the y axis (theta = 0), change it to be  wrt to the x axis
    tgt_filter_orientation = np.int(gabor_fits.get_filter_orientation(tgt_filter, o_type='average'))
    tgt_filter_orientation = np.int(np.floor(90 + tgt_filter_orientation))

    title = "Target Filter Index {0}, orientation {1}".format(tgt_filter_idx, tgt_filter_orientation)
    print(title)

    # # Display the target filter
    # plt.figure()
    # plt.imshow(normalize_fragment(tgt_filter))
    # plt.title(title)

    # -----------------------------------------------------------------------------------
    #  Contour Fragment
    # -----------------------------------------------------------------------------------

    # # A. Derived from the target filter
    # # ---------------------------------
    # Blend in the edges of the fragment @ the edges
    # g_kernel = alex_net_utils.get_2d_gaussian_kernel(tgt_filter.shape[0:2], sigma=0.6)
    # g_kernel = np.expand_dims(g_kernel, axis=2)
    # fragment = tgt_filter * g_kernel

    # B. Generated Directly from a Gabor
    # -------------------------------------
    x_arr = np.linspace(-1, 1, tgt_filter.shape[0])
    y_arr = np.copy(x_arr)
    xx, yy = np.meshgrid(x_arr, y_arr)

    fragment = gabor_fits.gabor_2d(
        (xx, yy),
        x0=0,
        y0=0,
        theta_deg=tgt_filter_orientation - 90,
        amp=1,
        sigma=0.6,
        lambda1=3,
        psi=0,
        gamma=1
    )
    fragment = fragment.reshape((x_arr.shape[0], y_arr.shape[0]))
    fragment = np.stack((fragment, fragment, fragment), axis=2)

    # -------------------------------------
    fragment = normalize_fragment(fragment)
    fragment = imrotate(fragment, 0)

    # # Display the contour fragment
    # plt.figure()
    # plt.imshow(fragment)
    # plt.title("Contour Fragment")

    # -----------------------------------------------------------------------------------
    #  Initializations
    # -----------------------------------------------------------------------------------
    image_size = np.array([227, 227, 3])
    test_image = np.zeros(image_size, dtype=np.uint8)

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
        tgt_filter_orientation,
        contour_len,
        beta_rotation,
        full_tile_size[0]
    )

    # test_image, path_fragment_starts = add_contour_path_closest_nonoverlap(
    #     test_image,
    #     fragment,
    #     tgt_filter_orientation,
    #     contour_len,
    #     beta_rotation,
    #     full_tile_size[0] - 2
    # )

    # plt.figure()
    # plt.imshow(test_image)
    # plt.title("Contour Fragments")

    # ----------------------------------------------------------------------------------
    #  Add background Fragments
    # ----------------------------------------------------------------------------------
    test_image, bg_tiles, removed_bg_tiles, = add_background_fragments(
        test_image,
        fragment,
        path_fragment_starts,
        full_tile_size,
        beta_rotation
    )

    plt.figure()
    plt.imshow(test_image)
    plt.title("Contour of length {0} embedded in a sea of distractors".format(contour_len))

    # -----------------------------------------------------------------------------------
    # Debugging Plots
    # -----------------------------------------------------------------------------------
    contour_highlighted_image = alex_net_utils.highlight_tiles(
        test_image,
        fragment.shape[0:2],
        path_fragment_starts
    )

    plt.figure()
    plt.imshow(contour_highlighted_image)
    plt.title("Contour fragments Highlighted")

    # Display Full tiles
    bg_tile_starts = alex_net_utils.get_background_tiles_locations(
        frag_len=full_tile_size[0],
        img_len=image_size[0],
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=image_size[0] // 2 - (full_tile_size[0] // 2)
    )

    bg_frags_highlighted_image = alex_net_utils.highlight_tiles(
        contour_highlighted_image,
        full_tile_size,
        bg_tile_starts,
        edge_color=(0, 255, 0)
    )

    plt.figure()
    plt.imshow(bg_frags_highlighted_image)
    plt.title("Full Tiles Highlighted")
