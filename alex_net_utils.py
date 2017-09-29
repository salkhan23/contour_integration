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
    ax = np.arange(-shape[0] // 2 + 1, shape[0] // 2 + 1)
    ay = np.arange(-shape[1] // 2 + 1, shape[1] // 2 + 1)

    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

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

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    l1_act = np.array(l1_act_cb([img, 0]))
    l2_act = np.array(l2_act_cb([img, 0]))

    l1_act = np.squeeze(l1_act, axis=0)
    l2_act = np.squeeze(l2_act, axis=0)

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


def vertical_contour_generator(frag_len, bw_tile_spacing, cont_len, cont_start_loc):
    """
    Generate the start co-ordinates of fragment squares that form a vertical contour of the
    specified length at the specified location

    :param frag_len:
    :param bw_tile_spacing: Between fragment square spacing in pixels
    :param cont_len: length of fragment in units of fragment squares
    :param cont_start_loc: start starting location where the contour should be places

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


def horizontal_contour_generator(frag_len, bw_tile_spacing, cont_len, cont_start_loc):
    """
    Generate the start co-ordinates of fragment squares that form a horizontal contour of
    the specified length at the specified location

    :param frag_len:
    :param bw_tile_spacing:
    :param cont_len:
    :param cont_start_loc:

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


def tile_image(img, frag, insert_locs, rotate=True, gaussian_smoothing=True, sigma=4.0):
    """
    Place tile 'fragments' at the specified starting positions (x, y) in the image.

    :param frag: contour fragment to be inserted
    :param insert_locs: array of (x,y) starting positions of where tiles will be inserted
    :param img: image where tiles will be placed
    :param rotate: If true each tile is randomly rotated before insertion.
            Currently 8 possible orientations
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

    x_arr = insert_locs[0]
    y_arr = insert_locs[1]

    if isinstance(x_arr, int):
        x_arr = np.array([x_arr])
    if isinstance(y_arr, int):
        y_arr = np.array([y_arr])

    for idx in range(len(x_arr)):

        # print("Processing Fragment @ (%d,%d)" % (x_arr[idx], y_arr[idx]))

        if (-tile_len < x_arr[idx] < img_len) and (-tile_len < y_arr[idx] < img_len):

            start_x_loc = max(x_arr[idx], 0)
            stop_x_loc = min(x_arr[idx] + tile_len, img_len)

            start_y_loc = max(y_arr[idx], 0)
            stop_y_loc = min(y_arr[idx] + tile_len, img_len)

            # print("Placing Fragment at location  x=(%d, %d), y = (%d, %d),"
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
                tile = randomly_rotate_tile(frag, 45)
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
