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
import copy

import keras.backend as keras_backend
import keras
from keras.preprocessing.image import load_img

from base_models import alex_net
import gabor_fits
import alex_net_utils


reload(alex_net)
reload(gabor_fits)
reload(alex_net_utils)


def normalize_fragment(frag):
    """
    Normalize fragment to the 0, 1 range
    :param frag:

    :return: Normalized fragment
    """
    return (frag - frag.min()) / (frag.max() - frag.min())


def do_tiles_overlap(l1, r1, l2, r2, border_can_overlap=True):
    """
    Rectangles are specified by two points, the (x,y) coordinates of the top left corner (l1)
    and bottom right corner

    Two rectangles do not overlap if one of the following conditions is true.
    1) One rectangle is above top edge of other rectangle.
    2) One rectangle is on left side of left edge of other rectangle.

    Ref:  https://www.geeksforgeeks.org/find-two-rectangles-overlap/

    Different from Ref, and more aligned with the coordinates system in the rest of the file, x
    controls vertical while y controls horizontal

    :param border_can_overlap:
    :param l1: top left corner of tile 1
    :param r1: bottom right corner of tile 1
    :param l2:
    :param r2:

    :return:  True of the input tiles overlap, false otherwise
    """
    # Does one square lie to the Left of the other
    if border_can_overlap:
        if l1[1] >= r2[1] or l2[1] >= r1[1]:
            return False
    else:
        if l1[1] > r2[1] or l2[1] > r1[1]:
            return False

    # Does one square lie above the other
    if border_can_overlap:
        if l1[0] >= r2[0] or l2[0] >= r1[0]:
            return False
    else:
        if l1[0] > r2[0] or l2[0] > r1[0]:
            return False

    return True


def _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, alpha, d, d_delta, frag_size,
        random_beta_rot=False, random_alpha_rot=True):
    """

    :param img:
    :param center_frag_start:
    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param alpha:
    :param d:
    :param d_delta:
    :param frag_size:
    :param random_beta_rot: [False]
    :param random_alpha_rot:

    :return:
    """
    if type(frag_params) is not list:
        frag_params = [frag_params]

    tile_offset = np.zeros((2,), dtype=np.int)
    prev_tile_start = center_frag_start

    tile_starts = []

    acc_angle = 0

    for i in range(c_len // 2):

        if random_beta_rot:
            beta = np.random.choice((-1, 1), size=1) * beta

        acc_angle += beta
        # acc_angle = np.mod(acc_angle, 360)
        # print("fragment idx {} acc_angle {}".format(i, acc_angle))

        rotated_frag_params_list = copy.deepcopy(frag_params)

        if random_alpha_rot:
            alpha = np.random.choice((-1, 1), size=1) * alpha

        # Rotate the next fragment
        # ------------------------
        for c_params in rotated_frag_params_list:
            c_params["theta_deg"] += (acc_angle + alpha)

        rotated_frag = gabor_fits.get_gabor_fragment(rotated_frag_params_list, frag.shape[0:2])

        # Find the location of the next fragment
        # --------------------------------------
        # TODO: Should this be gabor params of the chan with the highest amplitude
        loc_angle = rotated_frag_params_list[0]['theta_deg'] - alpha

        # Note
        # [1] Origin of (x, y) top left corner
        # [2] Dim 0 increases downward direction, Dim 1 increases in the right direction
        # [3] Gabor angles are specified wrt y-axis i.e. 0 orientation is vertical. For position
        #     we need the angles to be relative to the x-axis.
        tile_offset[0] = d * np.cos(loc_angle / 180.0 * np.pi)
        tile_offset[1] = d * np.sin(loc_angle / 180.0 * np.pi)

        curr_tile_start = prev_tile_start + tile_offset
        # print("Current tile start {0}. (offsets {1}, previous {2}, loc_angle={3})".format(
        #     curr_tile_start, tile_offset, prev_tile_start, loc_angle))

        # check if the current tile overlaps with the previous tile
        # TODO: Check if current tile overlaps with ANY previous one.
        l1 = curr_tile_start
        r1 = l1 + frag_size
        l2 = prev_tile_start
        r2 = l2 + frag_size
        is_overlapping = do_tiles_overlap(l1, r1, l2, r2)

        while is_overlapping:
            print("Tile {0} overlaps with tile at location {1}".format(curr_tile_start, prev_tile_start))
            tile_offset[0] += d_delta * np.cos(loc_angle / 180.0 * np.pi)
            tile_offset[1] += d_delta * np.sin(loc_angle / 180.0 * np.pi)

            curr_tile_start = prev_tile_start + tile_offset
            print("Current tile relocated to {0}. (offsets {1})".format(curr_tile_start, tile_offset))

            l1 = curr_tile_start
            r1 = l1 + frag_size
            is_overlapping = do_tiles_overlap(l1, r1, l2, r2)

        img = alex_net_utils.tile_image(
            img,
            rotated_frag,
            curr_tile_start,
            rotate=False,
            gaussian_smoothing=False
        )

        prev_tile_start = curr_tile_start
        tile_starts.append(curr_tile_start)

        # plt.figure()
        # plt.imshow(img)
        # raw_input()

    return img, tile_starts


def add_contour_path_constant_separation(
        img, frag, frag_params, c_len, beta, alpha, d, center_frag_start=None,
        rand_inter_frag_direction_change=True, random_alpha_rot=True, base_contour='random'):
    """
    Add curved contours to the test image as added in the ref. a constant separation (d)
    is projected from the previous tile to find the location of the next tile.

    If the tile overlaps, fragment separation is increased by a factor of d // 4.

    :param img:
    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param alpha:
    :param d:
    :param center_frag_start:
    :param rand_inter_frag_direction_change:
    :param random_alpha_rot:[True]
    :param base_contour: this determines the shape of the base contour. If set to sigmoid (default), the
        generated contour (2 calls to this function with d and -d) are symmetric about the origin, if set to
        circle, they are mirror symmetric about the vertical axis. This is for the case random_frag_direction
        is set to false.
    :return:
    """

    if base_contour.lower() not in ['sigmoid', 'circle', 'random']:
        raise Exception("Invalid base contour. Should be [sigmoid or circle]")

    frag_size = np.array(frag.shape[0:2])

    if center_frag_start is None:
        img_size = np.array(img.shape[0:2])
        img_center = img_size // 2
        center_frag_start = img_center - (frag_size // 2)

    d_delta = d // 4

    # Add center fragment
    if alpha == 0:
        frag_from_contour_rot = 0
    else:
        if random_alpha_rot:
            frag_from_contour_rot = np.random.choice((-alpha, alpha), size=1)
        else:
            frag_from_contour_rot = alpha

    first_frag_params_list = copy.deepcopy(frag_params)

    for c_params in first_frag_params_list:
        c_params["theta_deg"] = c_params["theta_deg"] + frag_from_contour_rot

    first_frag = gabor_fits.get_gabor_fragment(first_frag_params_list, frag.shape[0:2])

    img = alex_net_utils.tile_image(
        img,
        first_frag,
        center_frag_start,
        rotate=False,
        gaussian_smoothing=False,
    )
    c_tile_starts = [center_frag_start]

    img, tiles = _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, alpha, d, d_delta, frag_size,
        random_beta_rot=rand_inter_frag_direction_change,
        random_alpha_rot=random_alpha_rot)
    c_tile_starts.extend(tiles)

    if base_contour == 'circle':
        beta = -beta
    elif base_contour == 'random':
        beta = np.random.choice((-1, 1), size=1) * beta

    img, tiles = _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, alpha, -d, -d_delta, frag_size,
        random_beta_rot=rand_inter_frag_direction_change,
        random_alpha_rot=random_alpha_rot)
    c_tile_starts.extend(tiles)

    # ---------------------------
    c_tile_starts = np.array(c_tile_starts)

    return img, c_tile_starts


def get_nonoverlapping_bg_fragment(f_tile_start, f_tile_size, c_tile_starts, c_tile_size, max_offset):
    """

    :param f_tile_size:
    :param f_tile_start:
    :param c_tile_starts:
    :param c_tile_size:
    :param max_offset:
    :return:
    """
    # print("get_nonoverlapping_bg_fragment: full tile start: {}".format(f_tile_start))

    for r_idx in range(max_offset):
        for c_idx in range(max_offset):

            l1 = f_tile_start + np.array([r_idx, c_idx])  # top left corner of new bg tile
            r1 = l1 + c_tile_size   # lower right corner of new bg tile

            is_overlapping = False

            # print("Checking start location {}".format(l1))

            if (r1[0] > f_tile_start[0] + f_tile_size[0]) or (r1[1] > f_tile_start[1] + f_tile_size[1]):
                # new bg tile is outside the full tile
                continue

            for c_tile in c_tile_starts:

                l2 = c_tile  # contour tile top left corner
                r2 = c_tile + c_tile_size  # bottom right corner of new bg tile
                # print("checking bg tile @ start {0} with contour tile @ start {1}".format(l1, l2))

                if do_tiles_overlap(l1, r1, l2, r2, border_can_overlap=False):
                    # print('overlaps!')
                    is_overlapping = True
                    break

            if not is_overlapping:
                # print("Found non-overlapping")
                return l1
    return None


def add_background_fragments(img, frag, c_frag_starts, f_tile_size, delta_rotation, frag_params,
                             relocate_allowed=True):
    """

    :param img:
    :param frag:
    :param c_frag_starts:
    :param f_tile_size:
    :param delta_rotation:
    :param frag_params:
    :param relocate_allowed: If a bg frag overlaps with a contour fragment, try to
        relocate fragment, so it can fit in the tile without overlapping with the
        contour fragment

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
    bg_frag_starts = np.copy(f_tile_starts)

    if max_displace != 0:
        bg_frag_starts += np.random.randint(0, max_displace, f_tile_starts.shape)

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

        ovlp_bg_frag_idx_arr = np.argwhere(dist_to_c_frag <= np.sqrt(2)*frag.shape[0])
        # for idx in ovlp_bg_frag_idx_arr:
        #     print("contour fragment @ {0}, overlaps with bg fragment @ index {1} and location {2}".format(
        #         c_frag_start, idx, bg_frag_starts[idx, :]))

        ovlp_bg_frag_idx_to_remove = []

        for ii, bg_frag_idx in enumerate(ovlp_bg_frag_idx_arr):

            f_tile_start = f_tile_starts[bg_frag_idx, :]

            novlp_bg_frag = None
            if relocate_allowed:
                # Is relocation possible?
                novlp_bg_frag = get_nonoverlapping_bg_fragment(
                    f_tile_start=np.squeeze(f_tile_start, axis=0),
                    f_tile_size=f_tile_size,
                    c_tile_starts=c_frag_starts,
                    c_tile_size=frag.shape[0:2],
                    max_offset=max_displace,
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

    if removed_bg_frag_starts.size > 0:
        removed_bg_frag_starts = np.squeeze(removed_bg_frag_starts, axis=1)

    relocate_bg_frag_starts = np.array(relocate_bg_frag_starts)

    # Now add the background fragment tiles
    # -------------------------------------
    if type(frag_params) is not list:
        frag_params = [frag_params]
    rotated_frag_params_list = copy.deepcopy(frag_params)

    num_possible_rotations = 360 // delta_rotation

    for start in bg_frag_starts:

        random_rotation = np.random.randint(0, np.int(num_possible_rotations)) * delta_rotation
        for c_params in rotated_frag_params_list:
            c_params['theta_deg'] = c_params['theta_deg'] + random_rotation

        rotated_frag = gabor_fits.get_gabor_fragment(rotated_frag_params_list, frag.shape[0:2])

        img = alex_net_utils.tile_image(
            img,
            rotated_frag,
            start,
            rotate=False,
            gaussian_smoothing=False
        )

    return img, bg_frag_starts, removed_bg_frag_starts, relocate_bg_frag_starts


def generate_contour_images(
        n_images, frag, frag_params, c_len, beta, alpha, f_tile_size, img_size=None, bg_frag_relocate=True,
        rand_inter_frag_direction_change=True, random_alpha_rot=True, center_frag_start=None, base_contour='random'):
    """
    Generate n_images with the specified fragment parameters.

    In the Ref, a visible stimulus of a small size is placed inside a large tile
    Here, full tile refers to the large tile & fragment tile refers to the visible stimulus

    :param n_images:
    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param alpha:
    :param f_tile_size
    :param img_size: [Default = (227, 227, 3)]
        :param center_frag_start:
    :param bg_frag_relocate: If True, for a full tile that contains a background fragment, try to
             relocate bg fragment within the full tile to see if it can fit.
    :param rand_inter_frag_direction_change: [True]
    :param random_alpha_rot: [True]
    :param base_contour:

    :return: array of generated images [n_images, r, c, ch]
    """
    if img_size is None:
        img_size = np.array([227, 227, 3])

    if center_frag_start is None:
        img_center = img_size[0:2] // 2
        frag_size = np.array(frag.shape[0:2])
        center_frag_start = img_center - (frag_size // 2)

    # print("Generating {0} images for fragment [ contour length {1}, inter fragment rotation {2}]".format(
    #     n_images, c_len, beta))

    # bg = np.mean(fragment, axis=(0, 1))
    # bg = [np.uint8(chan) for chan in bg_value]
    bg = get_mean_pixel_value_at_boundary(frag)

    images = np.zeros((n_images, img_size[0], img_size[1], img_size[2]), dtype='uint8')

    for img_idx in range(n_images):

        # print("Image {}".format(img_idx))

        img = np.ones(img_size, dtype=np.uint8) * bg

        c_frag_starts = np.array([])
        if (c_len > 1) or (c_len == 1 and beta == 0):
            img, c_frag_starts = add_contour_path_constant_separation(
                img, frag, frag_params, c_len, beta, alpha, f_tile_size[0],
                center_frag_start=center_frag_start,
                rand_inter_frag_direction_change=rand_inter_frag_direction_change,
                random_alpha_rot=random_alpha_rot,
                base_contour=base_contour
            )
        # If c_len == 1 and beta != 0, only background fragments are added. In this case the enhancement gain
        # should be 1 (no enhancement) and serves as example when not to enhance the fragment

        img, bg_frag_starts, removed_tiles, relocated_tiles = add_background_fragments(
            img, frag, c_frag_starts, f_tile_size, 10, frag_params, bg_frag_relocate)

        images[img_idx, ] = img

        # # Debug
        # # ------
        # # Highlight Contour tiles - Red
        # img = alex_net_utils.highlight_tiles(img, frag.shape[0:2], c_frag_starts,  edge_color=(255, 0, 0))
        #
        # # # Highlight Background Fragment tiles - Green
        # # img = alex_net_utils.highlight_tiles(img, frag.shape[0:2], bg_frag_starts, edge_color=(0, 255, 0))
        #
        # # Highlight Removed tiles - Blue
        # img = alex_net_utils.highlight_tiles(img, frag.shape[0:2], removed_tiles, edge_color=(0, 0, 255))
        #
        # # Highlight Relocated tiles - Teal
        # img = alex_net_utils.highlight_tiles(img, frag.shape[0:2], relocated_tiles, edge_color=(0, 255, 255))
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
        # img = alex_net_utils.highlight_tiles(img, f_tile_size, f_tile_starts, edge_color=(255, 255, 0))
        #
        # plt.figure()
        # plt.imshow(img)

    return images


def get_gabor_params_from_target_filter(tgt_filt, match=None):
    """
    Get best fit 2D gabor parameters from feature extracting kernel
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
            print("Matching param {0}={1}".format(key, params[key]))

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

    mean_border_value = np.mean(border_points, axis=(0, 1, 2))
    mean_border_value = [np.uint8(ch) for ch in mean_border_value]

    return mean_border_value


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_key_dict, batch_size=32, img_size=(227, 227, 3), shuffle=True, labels_per_image=1,
                 preprocessing_cb=alex_net_utils.preprocessing_divide_255):
        """
        A Python generator (actually a keras sequencer object) that can be used to
        dynamically load images when the batch is run. Saves a lot on memory.

        Compared to a generator a sequencer object iterates over all images once during
        an epoch

        Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

        :param data_key_dict: dictionary of (image_location: label(enhancement gain)) of all
                              images in the data set.
        :param batch_size:
        :param img_size:
        :param shuffle: [default=True]
        self.labels_per_image = labels_per_image. If training multiple kernels simultaneously, set to 96.
                                Specifies the dimensions of the expected labels per image.

        """
        self.shuffle = shuffle
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_key_dict = data_key_dict
        self.labels_per_image = labels_per_image

        self.list_ids = self.data_key_dict.keys()
        self.on_epoch_end()

        self.image_preprocessing = preprocessing_cb

    def on_epoch_end(self):
        """
        Routines to run at the end of each epoch

        :return:
        """
        self.idx_arr = np.arange(len(self.data_key_dict))

        if self.shuffle:
            np.random.shuffle(self.idx_arr)

    def __data_generation(self, list_ids_temp):
        """

        :param list_ids_temp:
        :return:
        """
        x_arr = np.zeros((self.batch_size, self.img_size[2], self.img_size[1], self.img_size[0]))
        y_arr = np.zeros((self.batch_size, self.labels_per_image))

        # print("Loading a new batch")

        for idx, list_id in enumerate(list_ids_temp):

            # # print("{} Loading Image {}".format(idx, list_id))

            temp = load_img(list_id)
            in_img = keras.preprocessing.image.img_to_array(temp, dtype='float64', data_format='channels_last')

            in_img = self.image_preprocessing(in_img)  # channel first format

            in_img = np.transpose(in_img, axes=[2, 0, 1])  # channel to channel first format

            x_arr[idx, ] = in_img
            y_arr[idx, ] = self.data_key_dict[list_id]

        return x_arr, y_arr

    def __len__(self):
        """ The number of batches per epoch"""
        return int(np.floor(len(self.data_key_dict) / self.batch_size))

    def __getitem__(self, index):
        """
        Get one batch of data 
        :param index: 
        :return: 
        """
        idx_arr = self.idx_arr[index * self.batch_size: (index + 1) * self.batch_size]

        # find the list of ids
        list_ids_temp = [self.list_ids[k] for k in idx_arr]

        x_arr, y_arr = self.__data_generation(list_ids_temp)

        return x_arr, y_arr


if __name__ == '__main__':
    plt.ion()
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    #  Target Feature and its orientation
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 10
    tgt_filter = alex_net.get_target_feature_extracting_kernel(tgt_filter_idx)

    # # Display the target filter
    # plt.figure()
    # plt.imshow(normalize_fragment(tgt_filter))
    # plt.title('Target Filter')

    # -----------------------------------------------------------------------------------
    #  Contour Fragment
    # -----------------------------------------------------------------------------------
    fragment_gabor_params = get_gabor_params_from_target_filter(
        tgt_filter,
        # match=[ 'x0', 'y0', 'theta_deg', 'amp', 'sigma', 'lambda1', 'psi', 'gamma']
        # match=['x0', 'y0', 'theta_deg', 'amp', 'psi', 'gamma']
        match=['theta_deg']
    )
    fragment_gabor_params['theta_deg'] = np.int(fragment_gabor_params['theta_deg'])
    fragment_gabor_params = [fragment_gabor_params]

    fragment = gabor_fits.get_gabor_fragment(
        fragment_gabor_params, tgt_filter.shape[0:2])

    # # Display the contour fragment
    # plt.figure()
    # plt.imshow(fragment)
    # plt.title("Contour Fragment")

    # # Plot rotations of the fragment
    # gabor_fits.plot_fragment_rotations(fragment, fragment_gabor_params, delta_rot=15)

    # -----------------------------------------------------------------------------------
    #  Initializations
    # -----------------------------------------------------------------------------------
    image_size = np.array([227, 227, 3])

    # bg_value = np.mean(fragment, axis=(0, 1))
    # bg_value = [np.uint8(chan) for chan in bg_value]
    bg_value = get_mean_pixel_value_at_boundary(fragment)

    test_image = np.ones(image_size, dtype=np.uint8) * bg_value

    contour_len = 9
    beta_rotation = 15
    alpha_rotation = 0

    # In the Ref, the visible portion of the fragment moves around inside large tiles.
    # Here, full tile refers to the large tile & fragment tile refers to the visible stimulus
    fragment_size = np.array(fragment.shape[0:2])
    full_tile_size = np.array([18, 18])

    # -----------------------------------------------------------------------------------
    #  Add the Contour Path
    # -----------------------------------------------------------------------------------
    test_image, path_fragment_starts = add_contour_path_constant_separation(
        img=test_image,
        frag=fragment,
        frag_params=fragment_gabor_params,
        c_len=contour_len,
        beta=beta_rotation,
        alpha=alpha_rotation,
        d=full_tile_size[0],
        rand_inter_frag_direction_change=False,
        base_contour='sigmoid'
    )

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
        fragment_gabor_params,
        relocate_allowed=True
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

    # Highlight Removed tiles
    test_image = alex_net_utils.highlight_tiles(
        test_image, fragment.shape[0:2], bg_removed_tiles, edge_color=[0, 0, 255])

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
