# -------------------------------------------------------------------------------------------------
#  Generate Curve Contour Test Images
#
# Author: Salman Khan
# Date  : 19/04/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imrotate

import keras.backend as K

import base_alex_net
import gabor_fits
import alex_net_utils

reload(gabor_fits)
reload(alex_net_utils)
reload(base_alex_net)


def get_target_feature_extracting_kernel(tgt_filt_idx):
    """
    Return the target kernel @ the specified index from the first feature extracting
    covolutional layer of alex_net

    :param tgt_filt_idx:
    :return:  target Feature extracting kernel
    """
    alex_net_model = base_alex_net.alex_net("trained_models/AlexNet/alexnet_weights.h5")
    feat_extract_kernels = K.eval(alex_net_model.layers[1].weights[0])

    return feat_extract_kernels[:, :, :, tgt_filt_idx]


def normalize_frag(frag):
    """
    Normalize fragment to the 0,1 range
    :param frag:

    :return: Normalized fragment
    """
    return (frag - frag.min()) / (frag.max() - frag.min())


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


def add_curved_contour_path(img, frag, frag_orientation, c_len, beta):
    """
    Add curved contours to the test image, fragments are added based on the closest non
    overlapping x, or y co-ordinate

    :param img:
    :param frag:
    :param frag_orientation:
    :param c_len:
    :param beta:

    :return: (1) image with a curved contour added,
             (2) starting locations of all contour path fragments
    """
    img_size = np.array(img.shape[0:2])
    img_center = img_size // 2

    frag_size = np.array(frag.shape[0:2])
    center_frag_start = img_center - (frag_size // 2)

    min_frag_spacing = frag_size[0]

    # A. Add the central tile
    # -----------------------
    img = alex_net_utils.tile_image(
        img,
        frag,
        center_frag_start,
        rotate=False,
        gaussian_smoothing=False,
    )
    c_tile_starts = [center_frag_start]

    # B. Add the right hand side of the contour. Rotations are counter clockwise from the
    # orientation of the fragment.
    # -----------------------------------------------------------------------------------
    acc_angle = frag_orientation

    tile_offset = np.zeros((2,), dtype=np.int)
    prev_tile_start = center_frag_start

    for i in range(c_len // 2):

        acc_angle += beta #* np.random.choice((-1, 1), size=1)
        acc_angle = np.mod(acc_angle, 360)

        rotated_frag = imrotate(frag, angle=acc_angle - frag_orientation)

        # Different from conventional (x, y) co-ordinates, the origin of the displayed
        # array starts in the top left corner. x increases in the vertically down
        # direction while y increases in the horizontally right direction.
        if acc_angle <= 45 or 135 <= acc_angle <= 235 or acc_angle >= 315:
            # print("Keep x constant")
            x_dir = np.sign(np.cos(acc_angle * np.pi / 180.0))

            tile_offset[1] = x_dir * min_frag_spacing
            tile_offset[0] = -tile_offset[1] * np.tan(acc_angle * np.pi / 180.0)

        else:
            # print("Keep y constant")
            y_dir = np.sign(np.sin(acc_angle * np.pi / 180.0))

            tile_offset[0] = -y_dir * min_frag_spacing
            tile_offset[1] = -tile_offset[0] / np.tan(acc_angle * np.pi / 180.0)

        curr_tile_start = prev_tile_start + tile_offset
        # print("Current tile start {0}. (offsets {1}, previous {2}, acc_angle={3})".format(
        #     curr_tile_start,
        #     tile_offset,
        #     prev_tile_start,
        #     acc_angle
        # ))

        img = alex_net_utils.tile_image(
            img,
            rotated_frag,
            curr_tile_start,
            rotate=False,
            gaussian_smoothing=False
        )

        prev_tile_start = curr_tile_start
        c_tile_starts.append(curr_tile_start)

    # C. Add the left hand side of the contour.
    # -----------------------------------------------------------------------------------
    acc_angle = frag_orientation

    tile_offset = np.zeros((2,), dtype=np.int)
    prev_tile_start = center_frag_start

    for i in range(c_len // 2):

        acc_angle += beta  # * np.random.choice((-1, 1), size=1)
        acc_angle = np.mod(acc_angle, 360)

        rotated_frag = imrotate(frag, angle=acc_angle - frag_orientation)

        # Different from conventional (x, y) co-ordinates, the origin of the displayed
        # array starts in the top left corner. x increases in the vertically down
        # direction while y increases in the horizontally right direction.
        if acc_angle <= 45 or 135 <= acc_angle <= 235 or acc_angle >= 315:
            # print("Keep x constant")
            x_dir = np.sign(np.cos(acc_angle * np.pi / 180.0))

            tile_offset[1] = -x_dir * min_frag_spacing
            tile_offset[0] = -tile_offset[1] * np.tan(acc_angle * np.pi / 180.0)

        else:
            # print("Keep y constant")
            y_dir = np.sign(np.sin(acc_angle * np.pi / 180.0))

            tile_offset[0] = +y_dir * min_frag_spacing
            tile_offset[1] = -tile_offset[0] / np.tan(acc_angle * np.pi / 180.0)

        curr_tile_start = prev_tile_start + tile_offset
        # print("Current tile start {0}. (offsets {1}, previous {2}, acc_angle={3})".format(
        #     curr_tile_start,
        #     tile_offset,
        #     prev_tile_start,
        #     acc_angle
        # ))

        img = alex_net_utils.tile_image(
            img,
            rotated_frag,
            curr_tile_start,
            rotate=False,
            gaussian_smoothing=False
        )

        prev_tile_start = curr_tile_start
        c_tile_starts.append(curr_tile_start)

    c_tile_starts = np.array(c_tile_starts)

    return img, c_tile_starts


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
    if l1[0, 1] > r2[0, 1] or l2[0, 1] > r1[0, 1]:
        return False

        # Does one square lie above the other
    if l1[0, 0] > r2[0, 0] or l2[0, 0] > r1[0, 0]:
        return False

    return True


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


def add_background_fragments(img, frag, c_frag_starts, full_tile_size, beta):
    """

    :param img:
    :param frag:
    :param c_frag_starts:
    :param full_tile_size:
    :param beta:

    :return: (1) image with background tiles added
             (2) array of bg frag tiles
             (3) array of bg frag tiles removed
    """

    img_size = np.array(img.shape[0:2])
    img_center = img_size // 2

    center_bg_full_tile_start = img_center - full_tile_size // 2

    # Get start locations of all full tiles
    bg_full_tile_starts = alex_net_utils.get_background_tiles_locations(
        frag_len=full_tile_size[0],
        img_len=img_size[0],
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=center_bg_full_tile_start[0]
    )

    # displace the background fragment in each full tile
    max_displace = full_tile_size[0] - frag.shape[0]

    bg_frag_tile_starts = bg_full_tile_starts + \
        np.random.randint(0, max_displace, bg_full_tile_starts.shape)

    # Remove or replace all tiles that overlap with contour path fragments
    # --------------------------------------------------------------------
    removed_frag_starts = []

    for c_frag_start in c_frag_starts:

        c_frag_start = np.expand_dims(c_frag_start, axis=0)

        dist_to_c_frag = np.linalg.norm(c_frag_start - bg_frag_tile_starts, axis=1)
        ovlp_bg_frag_idx_arr = np.argwhere(dist_to_c_frag <= frag.shape[0])

        # for ii, dist in enumerate(dist_to_c_frag):
        #     print("{0}: {1}".format(ii, dist))
        print("path tile {0} overlaps with bg tiles at {1}".format(
            c_frag_start, ovlp_bg_frag_idx_arr))

        # First see if the overlapping bg fragment can be replaced with a non-overlapping one
        for ii, idx in enumerate(ovlp_bg_frag_idx_arr):

            f_tile_start = bg_full_tile_starts[idx, :]

            novlp_bg_frag = get_nonoverlapping_bg_fragment(
                 f_tile_start,
                 c_frag_start,
                 frag.shape[0:2],
                 max_displace
            )
            # print("Non_overlapping Tile {0}".format(novlp_bg_frag))

            if novlp_bg_frag is not None:

                print("Replacing tile @ {0} with tile @".format(
                    bg_frag_tile_starts[idx, :]))

                bg_frag_tile_starts[idx, :] = novlp_bg_frag
                ovlp_bg_frag_idx_arr = np.delete(ovlp_bg_frag_idx_arr, ii)
            else:
                removed_frag_starts.append(bg_frag_tile_starts[idx, :])

        # for idx in ovlp_bg_frag_idx_arr:
        #     removed_frag_starts.append(bg_frag_tile_starts[idx, :])

        bg_frag_tile_starts = \
            np.delete(bg_frag_tile_starts, ovlp_bg_frag_idx_arr, axis=0)

        bg_full_tile_starts = \
            np.delete(bg_full_tile_starts, ovlp_bg_frag_idx_arr, axis=0)

    removed_frag_starts = np.array(removed_frag_starts)
    removed_frag_starts = np.squeeze(removed_frag_starts, axis=1)

    # Add BG tiles
    img = alex_net_utils.tile_image(
        img,
        frag,
        bg_frag_tile_starts,
        rotate=True,
        delta_rotation=beta,
        gaussian_smoothing=False
    )

    return img, bg_frag_tile_starts, removed_frag_starts


if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    #  Target Feature and its orientation
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 10
    tgt_filter = get_target_feature_extracting_kernel(tgt_filter_idx)

    # Get the orientation of the feature extracting kernel
    tgt_filter_orientation, _ = gabor_fits.get_l1_filter_orientation_and_offset(
        tgt_filter, tgt_filter_idx, show_plots=False)

    # Bets fit angle is wrt the y axis (theta = 0), change it to be  wrt to the x axis
    tgt_filter_orientation = np.int(np.floor(90 + tgt_filter_orientation))
    print("Fragment orientation {0}".format(tgt_filter_orientation))

    # # Display the target filter
    # plt.figure()
    # plt.imshow(normalize_frag(tgt_filter))
    # plt.title( "Target Filter @ index {0}, Orientation {1}".format(
    #     tgt_filter_idx, tgt_filter_orientation))

    # -----------------------------------------------------------------------------------
    #  Contour fragment to use
    # -----------------------------------------------------------------------------------
    fragment = normalize_frag(tgt_filter)

    # blend in the edges of the fragment @ the edges
    g_kernel = alex_net_utils.get_2d_gaussian_kernel(fragment.shape[0:2], sigma=0.75)
    fragment = fragment * np.expand_dims(g_kernel, axis=2)
    fragment = imrotate(fragment, 0)

    # # Display the contour fragment
    # plt.figure()
    # plt.imshow(fragment)
    # plt.title("Contour Fragment")

    # Display rotations of the fragment
    # plot_fragment_rotations(fragment)

    # -----------------------------------------------------------------------------------
    #  Initializations
    # -----------------------------------------------------------------------------------
    image_size = np.array([227, 227, 3])
    test_image = np.zeros(image_size, dtype=np.uint8)

    beta_rotation = 15
    contour_len = 9

    # -----------------------------------------------------------------------------------
    #  Add the contour fragments
    # -----------------------------------------------------------------------------------
    test_image, path_fragments_starts = add_curved_contour_path(
        test_image,
        fragment,
        tgt_filter_orientation,
        contour_len,
        beta_rotation
    )

    # # Display the curved contour
    # plt.figure()
    # plt.imshow(test_image)
    # plt.title("Curved Contour")

    # -----------------------------------------------------------------------------------
    #  Add background fragments
    # -----------------------------------------------------------------------------------

    # The contour paths constructed as above can be at most
    # sqrt(fragment_size[0]**2 +fragment_size[1]**2) from each other
    max_inter_fragment_dist = np.int(np.floor(np.sqrt(2) * fragment.shape[0]))
    full_bg_tile_size = \
        np.array([max_inter_fragment_dist, max_inter_fragment_dist])

    test_image, bg_tiles, removed_bg_tiles, = add_background_fragments(
        test_image,
        fragment,
        path_fragments_starts,
        full_bg_tile_size,
        beta_rotation
    )

    plt.figure()
    plt.imshow(test_image)
    plt.title("Contour of length {0}, embedded in a sea of distractors".format(contour_len))

    # -----------------------------------------------------------------------------------
    # Debugging Plots
    # -----------------------------------------------------------------------------------
    contour_highlighted_image = alex_net_utils.highlight_tiles(
        test_image,
        fragment.shape[0:2],
        path_fragments_starts
    )

    plt.figure()
    plt.imshow(contour_highlighted_image)
    plt.title("Contour fragments Highlighted")
