# -------------------------------------------------------------------------------------------------
#  Various utility function used by many Files
#
# Author: Salman Khan
# Date  : 03/09/17
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imrotate

import keras.backend as K


def get_2d_gaussian_kernel(shape, sigma=1.0):
    """
    Returns a 2d (unnormalized) Gaussian kernel of the specified shape.

    :param shape: x,y dimensions of the gaussian
    :param sigma: standard deviation of generated Gaussian
    :return:
    """
    # ax = np.arange(-shape[0] // 2 + 1, shape[0] // 2 + 1)
    # ay = np.arange(-shape[1] // 2 + 1, shape[1] // 2 + 1)
    ax = np.linspace(-1, 1, shape[0])
    ay = np.linspace(-1, 1, shape[1])

    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    kernel = kernel.reshape(shape)

    return kernel


def get_activation_cb(model, layer_idx):
    """
    Return a callback that returns the output activation of the specified layer
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


def get_l1_and_l2_activations(img, l1_act_cb, l2_act_cb):
    """

    :param img:
    :param l1_act_cb:
    :param l2_act_cb:
    :return:
    """

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    l1_act = np.array(l1_act_cb([img, 0]))
    l2_act = np.array(l2_act_cb([img, 0]))

    l1_act = np.squeeze(l1_act, axis=0)
    l2_act = np.squeeze(l2_act, axis=0)

    return l1_act, l2_act


def plot_l1_and_l2_activations(img, l1_act_cb, l2_act_cb, tgt_filt_idx):
    """
    Plot 2 figures:
    [1] the test image,
    [2] l1_activations, l2_activations and the difference between the activations

    :param img:
    :param l1_act_cb:
    :param l2_act_cb:
    :param tgt_filt_idx:

    :return: Handles of the two images created
    """

    f1 = plt.figure()
    plt.imshow(img)

    l1_act, l2_act = get_l1_and_l2_activations(img, l1_act_cb, l2_act_cb)

    l1_act = l1_act[0, tgt_filt_idx, :, :]
    l2_act = l2_act[0, tgt_filt_idx, :, :]

    min_l2_act = l1_act.min()
    max_l2_act = l2_act.max()

    f2 = plt.figure()
    f2.add_subplot(1, 3, 1)
    plt.imshow(l1_act, cmap='seismic', vmin=min_l2_act, vmax=max_l2_act)
    plt.title('L1 Conv Layer Activation @ idx %d' % tgt_filt_idx)
    plt.colorbar(orientation='horizontal')
    plt.grid()

    f2.add_subplot(1, 3, 2)
    plt.imshow(l2_act, cmap='seismic', vmin=min_l2_act, vmax=max_l2_act)
    plt.title('L2 Contour Integration Layer Activation @ idx %d' % tgt_filt_idx)
    plt.colorbar(orientation='horizontal')
    plt.grid()

    f2.add_subplot(1, 4, 4)
    plt.imshow(l2_act - l1_act, cmap='seismic')
    plt.colorbar(orientation='horizontal')
    plt.title("Difference")
    plt.grid()

    return f1, f2


def plot_l1_and_l2_kernel_and_contour_fragment(model, tgt_filt_idx, frag=None):
    """

    :param model:
    :param tgt_filt_idx:
    :param frag:

    :return: None
    """
    l1_weights = K.eval(model.layers[1].weights[0])
    tgt_l1_filt = l1_weights[:, :, :, tgt_filt_idx]

    # Note this assume the final 'applied' kernels are stored in a kernel variable
    l2_weights = K.eval(model.layers[2].kernel)
    tgt_l2_filt = l2_weights[tgt_filt_idx, :, :]

    f = plt.figure()

    ax = plt.subplot2grid((2, 3), (0, 0), colspan=1)
    display_filt_l1 = (tgt_l1_filt - tgt_l1_filt.min()) * 1 / (tgt_l1_filt.max() - tgt_l1_filt.min())
    ax.imshow(display_filt_l1)  # normalized to [0, 1]
    ax.set_title("L1 filter")

    ax = plt.subplot2grid((2, 3), (0, 1), colspan=1)
    display_filt_l2 = (tgt_l2_filt - tgt_l2_filt.min()) * 1 / (tgt_l2_filt.max() - tgt_l2_filt.min())
    ax.imshow(display_filt_l2)  # normalized to [0, 1]
    ax.set_title("L2 filter")

    if frag is not None:
        ax = plt.subplot2grid((2, 3), (0, 2), colspan=1)
        ax.imshow(frag / 255.0)  # normalized to [0, 1]
        ax.set_title("Contour Fragment")

    # Plot individual channels of L1 filter separately
    ax = plt.subplot2grid((2, 3), (1, 0))
    ax.imshow(display_filt_l1[:, :, 0], cmap='seismic')
    ax.set_title("L1 filter, channel 0")

    ax = plt.subplot2grid((2, 3), (1, 1))
    ax.imshow(display_filt_l1[:, :, 1], cmap='seismic')
    ax.set_title("L1 filter, channel 1")

    ax = plt.subplot2grid((2, 3), (1, 2))
    ax.imshow(display_filt_l1[:, :, 2], cmap='seismic')
    ax.set_title("L1 filter, channel 2")

    f.suptitle("L1 and L2 Filters at index %d" % tgt_filt_idx)


def randomly_rotate_tile(tile, delta_rotation=45.0):
    """
    randomly rotate tile by 360/delta_rotation permutations

    :param delta_rotation: Angle in degrees, of which the rotated tile is a factor of
    :param tile: 2d contour fragment

    :return: rotated tile. Note this is an RGB format and values range b/w [0, 255]
    """
    num_possible_rotations = 360 // delta_rotation
    return imrotate(tile, angle=(np.random.randint(0, np.int(num_possible_rotations)) * delta_rotation))


def vertical_contour_generator(frag_len, bw_tile_spacing, cont_len, cont_start_loc, row_offset=0):
    """
    Generate the start co-ordinates of fragment squares that form a vertical contour of the
    specified length at the specified location

    :param frag_len:
    :param bw_tile_spacing: Between fragment square spacing in pixels
    :param cont_len: length of fragment in units of fragment squares
    :param cont_start_loc: start starting location where the contour should be places
    :param offset: Parameter not used for vertical contour generation. It is added because
           this function is used as a callback which requires this parameter.

    :return: tuple of (start_x, start_y)
    """
    mod_frag_len = frag_len + bw_tile_spacing

    start_x = range(
        cont_start_loc - (cont_len / 2) * mod_frag_len,
        cont_start_loc + (cont_len / 2 + 1) * mod_frag_len,
        mod_frag_len
    )
    start_y = np.ones_like(start_x) * cont_start_loc

    return start_x, start_y


def horizontal_contour_generator(frag_len, bw_tile_spacing, cont_len, cont_start_loc, row_offset=0):
    """
    Generate the start co-ordinates of fragment squares that form a horizontal contour of
    the specified length at the specified location

    :param frag_len:
    :param bw_tile_spacing:
    :param cont_len:
    :param cont_start_loc:
    :param offset: Parameter not used for vertical contour generation. It is added because
           this function is used as a callback which requires this parameter.

    :return: tuple of (start_x, start_y)
    """
    mod_frag_len = frag_len + bw_tile_spacing

    start_y = range(
        cont_start_loc - (cont_len / 2) * mod_frag_len,
        cont_start_loc + (cont_len / 2 + 1) * mod_frag_len,
        mod_frag_len
    )

    start_x = np.ones_like(start_y) * cont_start_loc

    return start_x, start_y


def diagonal_contour_generator(frag_len, row_offset, bw_tile_spacing, cont_len, cont_start_loc):
    """
    Generate the start co-ordinates of fragment squares that form a diagonal contour of
    the specified length & row_offset at the specified location

    :param frag_len:
    :param row_offset: row_offset to complete contours, found from the orientation of the fragment.
        See get_l1_filter_orientation_and_offset.
    :param bw_tile_spacing:
    :param cont_len:
    :param cont_start_loc: Start visual RF location of center neuron.

    :return: start_x_arr, start_y_arr locations of fragments that form the contour
    """
    frag_spacing = frag_len + bw_tile_spacing

    # 1st dimension stays the same
    start_x = range(
        cont_start_loc - (cont_len / 2) * frag_spacing,
        cont_start_loc + ((cont_len / 2) + 1) * frag_spacing,
        frag_spacing
    )

    # If there is nonzero spacing between tiles, the offset needs to be updated
    if bw_tile_spacing:
        row_offset = np.int(frag_spacing / np.float(frag_len) * row_offset)

    # 2nd dimension shifts with distance from the center row
    if row_offset is not 0:
        start_y = range(
            cont_start_loc - (cont_len / 2) * row_offset,
            cont_start_loc + ((cont_len / 2) + 1) * row_offset,
            row_offset
        )
    else:
        start_y = np.copy(start_x)

    return start_x, start_y


def get_background_tiles_locations(frag_len, img_len, row_offset, space_bw_tiles, tgt_n_visual_rf_start):
    """
    Starting locations for non-overlapping fragment tiles to cover the whole image.
    if row_offset is non-zero, tiles in each row are shifted by specified amount as you move further away
    from the center row.

    :param space_bw_tiles:
    :param tgt_n_visual_rf_start:
    :param row_offset:
    :param frag_len:
    :param img_len:

    :return: start_x, start_y
    """
    frag_spacing = frag_len + space_bw_tiles
    n_tiles = img_len // frag_spacing

    # To handle non-zero row shift, we have to add additional tiles, so that the whole image is populated
    add_tiles = 0
    if row_offset:
        max_shift = (n_tiles // 2 + 1) * row_offset
        add_tiles = abs(max_shift) // frag_spacing + 1
    # print("Number of tiles in image %d, number of additional tiles %d" % (n_tiles, add_tiles))

    n_tiles += add_tiles
    if n_tiles & 1 == 1:  # make even
        n_tiles += 1

    zero_offset_starts = range(
        tgt_n_visual_rf_start - (n_tiles / 2) * frag_spacing,
        tgt_n_visual_rf_start + (n_tiles / 2 + 1) * frag_spacing,
        frag_spacing,
    )

    # Fist dimension stays the same
    start_x = np.repeat(zero_offset_starts, len(zero_offset_starts))

    # In the second dimension each row is shifted by offset from the center row

    # # If there is nonzero spacing between tiles, the offset needs to be updated
    if space_bw_tiles:
        row_offset = np.int(frag_spacing / np.float(frag_len) * row_offset)

    start_y = []
    for row_idx in range(-n_tiles / 2, (n_tiles / 2) + 1):

        # print("processing row at idx %d, offset=%d" % (row_idx, row_idx * row_offset))

        ys = np.array(zero_offset_starts) + (row_idx * row_offset)
        start_y.append(ys)

    start_y = np.array(start_y)
    start_y = np.reshape(start_y, (start_y.shape[0] * start_y.shape[1]))

    loc_arr = np.array([start_x, start_y])
    loc_arr = loc_arr.T

    return loc_arr


def tile_image(img, frag, insert_loc_arr, rotate=True, delta_rotation=45, gaussian_smoothing=True, sigma=4.0):
    """
    Place tile 'fragments' at the specified starting positions (x, y) in the image.

    :param frag: contour fragment to be inserted
    :param insert_loc_arr: array of (x,y) starting positions of where tiles will be inserted
    :param img: image where tiles will be placed
    :param rotate: If true each tile is randomly rotated before insertion.
    :param delta_rotation: min rotation value
    :param gaussian_smoothing: If True, each fragment is multiplied with a Gaussian smoothing
            mask to prevent tile edges becoming part of stimuli [they will lie in the center of the RF of
            many neurons. [Default=True]
    :param sigma: Standard deviation of gaussian smoothing mask. Only used if gaussian smoothing is True

    :return: tiled image
    """
    img_len = img.shape[0]
    tile_len = frag.shape[0]

    g_kernel = get_2d_gaussian_kernel((tile_len, tile_len), sigma=sigma)
    g_kernel = np.reshape(g_kernel, (g_kernel.shape[0], g_kernel.shape[1], 1))
    g_kernel = np.repeat(g_kernel, 3, axis=2)

    if insert_loc_arr.ndim == 1:
        x_arr = [insert_loc_arr[0]]
        y_arr = [insert_loc_arr[1]]
    else:
        x_arr = insert_loc_arr[:, 0]
        y_arr = insert_loc_arr[:, 1]

    for idx in range(len(x_arr)):

        # print("Processing Fragment @ (%d,%d)" % (x_arr[idx], y_arr[idx]))

        if (-tile_len < x_arr[idx] < img_len) and (-tile_len < y_arr[idx] < img_len):

            start_x_loc = max(x_arr[idx], 0)
            stop_x_loc = min(x_arr[idx] + tile_len, img_len - 1)

            start_y_loc = max(y_arr[idx], 0)
            stop_y_loc = min(y_arr[idx] + tile_len, img_len - 1)

            # print("Placing Fragment at location  l1=(%d, %d), y = (%d, %d),"
            #       % (start_x_loc, stop_x_loc, start_y_loc, stop_y_loc))

            # Adjust incomplete beginning tiles
            if x_arr[idx] < 0:
                tile_x_start = tile_len - (stop_x_loc - start_x_loc)
            else:
                tile_x_start = 0

            if y_arr[idx] < 0:
                tile_y_start = tile_len - (stop_y_loc - start_y_loc)
            else:
                tile_y_start = 0
            #
            # print("Tile indices x = (%d,%d), y = (%d, %d)" % (
            #       tile_x_start, tile_x_start + stop_x_loc - start_x_loc,
            #       tile_y_start, tile_y_start + stop_y_loc - start_y_loc))

            if rotate:
                tile = randomly_rotate_tile(frag, delta_rotation)
            else:
                tile = frag

            # multiply the file with the gaussian smoothing filter
            # The edges between the tiles will lie within the stimuli of some neurons.
            # to prevent these prom being interpreted as stimuli, gradually decrease them.
            if gaussian_smoothing:
                tile = tile * g_kernel

            # only plot fragments who's start locations is within the image dimensions
            img[start_x_loc: stop_x_loc, start_y_loc: stop_y_loc, :] = \
                tile[tile_x_start: tile_x_start + stop_x_loc - start_x_loc,
                     tile_y_start: tile_y_start + stop_y_loc - start_y_loc, :]

    return img


def highlight_tiles(in_img, tile_shape, insert_loc_arr, edge_color=(255, 0, 0)):
    """
    Highlight specified tiles in the image


    :param in_img:
    :param tile_shape:
    :param insert_loc_arr:
    :param edge_color:

    :return: output image with the tiles highlighted
    """
    out_img = np.copy(in_img)

    img_len = in_img.shape[0]
    tile_len = tile_shape[0]

    if insert_loc_arr.ndim == 1:
        x_arr = [insert_loc_arr[0]]
        y_arr = [insert_loc_arr[1]]
    else:
        x_arr = insert_loc_arr[:, 0]
        y_arr = insert_loc_arr[:, 1]

    for idx in range(len(x_arr)):

        if (-tile_len < x_arr[idx] < img_len) and (-tile_len < y_arr[idx] < img_len):

            start_x_loc = max(x_arr[idx], 0)
            stop_x_loc = min(x_arr[idx] + tile_len, img_len-1)

            start_y_loc = max(y_arr[idx], 0)
            stop_y_loc = min(y_arr[idx] + tile_len, img_len-1)

            # print("Highlight tile @ tl=({0}, {1}), br=({2},{3})".format(
            #     start_x_loc, start_y_loc, stop_x_loc, stop_y_loc))

            out_img[start_x_loc: stop_x_loc, start_y_loc, :] = edge_color
            out_img[start_x_loc: stop_x_loc, stop_y_loc, :] = edge_color

            out_img[start_x_loc, start_y_loc: stop_y_loc, :] = edge_color
            out_img[stop_x_loc, start_y_loc: stop_y_loc, :] = edge_color

    return out_img


def find_most_active_l1_kernel_index(frag, l1_act_cb, plot=True):
    """
    Find the index of L1 conv layer kernel that is most responsive to the given fragment

    Input fragment is placed at the center of an input image. Generated image is passed through the network
    (using the specified l1_output callback). The index of the kernel that is most responsive is returned.

    :param frag: Input fragment
    :param l1_act_cb:
    :param plot: A plot of all activation to the given fragment. [Default=True]

    :return: index of most responsive kernel
    """
    start_x = 27 * 4  # Visual starting point of RF of neuron in the center of L1 activation map.
    start_y = 27 * 4

    test_image = np.zeros((227, 227, 3))

    test_image = tile_image(
        test_image,
        frag,
        (start_x, start_y),
        rotate=False,
        gaussian_smoothing=False
    )

    test_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format
    test_image = np.reshape(test_image, [1, test_image.shape[0], test_image.shape[1], test_image.shape[2]])
    # batch size is expected as the first dimension

    l1_act = l1_act_cb([test_image, 0])
    l1_act = np.squeeze(np.array(l1_act), axis=0)

    tgt_l1_act = l1_act[0, :, 27, 27]

    max_active_filt = tgt_l1_act.argmax()
    max_active_value = tgt_l1_act.max()

    # return the test image back to its original format
    test_image = test_image[0, :, :, :]
    test_image = np.transpose(test_image, (1, 2, 0))

    title = "Max active neuron at Index %d and value %0.2f" % (max_active_filt, max_active_value)
    print(title)

    if plot:
        f = plt.figure()
        f.add_subplot(1, 2, 1)

        plt.imshow(test_image / 255.0)
        plt.title('Input')
        f.add_subplot(1, 2, 2)
        plt.plot(tgt_l1_act)
        plt.xlabel('Kernel Index')
        plt.ylabel('Activation')
        plt.title(title)

    return max_active_filt


def plot_l2_visual_field(location, l2_kernel_mask, img, margin=4):
    """
    Plot the part of the visual scene seen by contour integration neuron @ specified location and all of its
    unmasked neighbors.

    :param margin: In the constructed tile plot, the distance the RFs of nearing neurons
    :param location: location of the target neurons. Should be a tuple of (x, y) l2 locations
    :param l2_kernel_mask: the mask that identifies the neighbor RFs to plot. 2D matrix.
    :param img: input image

    :return: None
    """
    l1_kernel_length = 11
    l1_conv_stride = 4

    n_rows = l2_kernel_mask.shape[0]
    n_cols = l2_kernel_mask.shape[1]
    # print("n_rows  n_col %d %d" % (n_rows, n_cols))

    # Initialize the tiled image
    width = (l1_kernel_length * n_rows) + ((n_rows - 1) * margin)
    height = (l1_kernel_length * n_cols) + ((n_cols - 1) * margin)
    tiled_image = np.zeros((width, height, img.shape[-1]))
    # print("Shape of tiled image", tiled_image.shape)

    for r_idx in range(n_rows):
        for c_idx in range(n_cols):

            cur_l2_neuron_x_loc = r_idx + location[0] - n_rows // 2
            cur_l2_neuron_y_loc = c_idx + location[1] - n_cols // 2

            cur_l2_neuron_vrf_start_x = cur_l2_neuron_x_loc * l1_conv_stride
            cur_l2_neuron_vrf_start_y = cur_l2_neuron_y_loc * l1_conv_stride

            tiled_image[
                (l1_kernel_length + margin) * r_idx: (l1_kernel_length + margin) * r_idx + l1_kernel_length,
                (l1_kernel_length + margin) * c_idx: (l1_kernel_length + margin) * c_idx + l1_kernel_length,
                :
            ] = img[
                cur_l2_neuron_vrf_start_x: cur_l2_neuron_vrf_start_x + l1_kernel_length,
                cur_l2_neuron_vrf_start_y: cur_l2_neuron_vrf_start_y + l1_kernel_length,
                :
            ] * l2_kernel_mask[r_idx, c_idx]

    plt.figure()
    plt.imshow(tiled_image)
    plt.title("Visual Field of neuron @ (%d,%d)" % (location[0], location[1]))
