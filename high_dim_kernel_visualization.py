# -------------------------------------------------------------------------------------------------
# Visualizing inputs the maximally excite a kernel using gradient accent.
#
# Technique can be used for higher layer filters for which the input channels are not 1 or 3.
#
# Compared to the original tutorial, the input images are resized to be the same size of the
# filters. This results in filters that are approximately the same as the kernel
# (for those that can be visualized)
#
# Ref: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
#      https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
#
# Author: Salman Khan
# Date  : 05/07/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import matplotlib.pyplot as plt

import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K

import utils
reload(utils)

# # Dimensions of the generated pictures for each filter
# IMG_WIDTH = 3
# IMG_HEIGHT = 3
# NUM_ITERATIONS = 20  # Iterations of the gradient ascent
#
# # The name of the layer we want to visualize
# LAYER_NAME = 'block1_conv1'


def normalize(x):
    """
    Utility function to normalize a tensor by its L2 norm

    :param x:
    :return:
    """
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def display_hd_filter_opt_stimuli(
        model, layer_idx, gen_img_row=7, gen_img_col=7, num_iterations=20, margin=1):
    """

    :param margin:
    :param num_iterations:
    :param gen_img_col:
    :param gen_img_row:
    :param layer_idx:
    :param model:
    :return: -1 if no weights are found, otherwise  0.
    """
    w_mat = model.layers[layer_idx].weights
    if not w_mat:
        print("No weights in layer at index %d" % layer_idx)
        return -1

    w_mat = w_mat[0]  # Exclude the biases

    # TODO: Generalize to Theano format
    r, c, in_ch, out_ch = K.int_shape(w_mat)

    input_img = model.input
    keep_filters = []

    for filter_idx in range(0, out_ch):
        print("Processing filter %d" % filter_idx)
        start_time = time.time()

        # Build a loss function that maximizes the activation of the nth filter of the layer
        layer_output = model.layers[layer_idx].output

        if K.image_data_format() == 'channel_first':
            loss = K.mean(layer_output[:, filter_idx, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_idx])

        # Compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]  # Gradient of the loss function wrt input image
        grads = normalize(grads)  # avoid very large or small gradients and ensures a smooth ascent

        # This function returns the loss and the grads given the input picture
        iterate = K.function([input_img, K.learning_phase()], [loss, grads])

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channel_first':
            # generates random floats b\w (0,1)
            input_img_data = np.random.random((1, 3, gen_img_row, gen_img_col))
        else:
            input_img_data = np.random.random((1, gen_img_row, gen_img_col, 3))
        input_img_data = (input_img_data - 0.5) * num_iterations + 128

        # Run gradient ascent for num_iterations steps
        loss_value = 0
        for i in range(num_iterations):
            loss_value, grads_value = iterate([input_img_data, 0])
            input_img_data += grads_value

            print("Current Loss value:", loss_value)
            if loss_value <= 0.0:
                break  # some filters get stuck to 0, we can skip them

        # Decode the resulting input image
        img = utils.deprocess_image(input_img_data[0])  # removes the batch index
        keep_filters.append((img, loss_value))

        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_idx, end_time - start_time))

    # Display the generated images
    n = np.int(np.round(np.sqrt(out_ch)))  # Single dimension of tiled image

    width = (n * gen_img_row) + ((n - 1) * margin)
    height = (n * gen_img_col) + ((n - 1) * margin)

    tiled_filters = np.zeros((width, height, 3))

    # Fill in in composite image with the filters
    for r_idx in range(n):
        for c_idx in range(n):

            filt_idx = (r_idx * n) + c_idx
            if filt_idx >= out_ch:
                break

            print("Processing filter %d" % filt_idx)
            img, loss = keep_filters[filt_idx]

            tiled_filters[
                (gen_img_row + margin) * r_idx: (gen_img_row + margin) * r_idx + gen_img_row,
                (gen_img_col + margin) * c_idx: (gen_img_col + margin) * c_idx + gen_img_col,
                :
            ] = utils.deprocess_image(img)

    # Plot the Composite Figure
    plt.ion()
    plt.figure()
    plt.imshow(tiled_filters)
    plt.colorbar()
    plt.title("Kernel visualization by gradient ascent.\n Layer %s (Idx %d). %d filters of size %dx%dx%d"
              % (model.layers[layer_idx].name, layer_idx, out_ch, r, c, in_ch))

    return 0


if __name__ == "__main__":

    plt.ion()

    vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False)
    vgg16_model.summary()

    layer_index = 1
    weights = vgg16_model.layers[layer_index].weights[0]

    # # First Layer
    # # --------------------------------------------------------------------
    # utils.display_filters(weights)
    # plt.title("Direct Filter Visualization, Layer %i, Filter shape %s"
    #           % (layer_index, weights.shape[:-1]))

    display_hd_filter_opt_stimuli(
        vgg16_model, layer_index, gen_img_row=3, gen_img_col=3, num_iterations=50)

    # # Second Layer
    # # --------------------------------------------------------------------
    layer_index = 2
    display_hd_filter_opt_stimuli(
        vgg16_model, layer_index, gen_img_row=9, gen_img_col=9, num_iterations=50)

    # # Third  Layer
    # # --------------------------------------------------------------------
    #
    # # Forth  Layer
    # # --------------------------------------------------------------------
    layer_index = 4
    display_hd_filter_opt_stimuli(
        vgg16_model, layer_index, gen_img_row=27, gen_img_col=27, num_iterations=50)
    #
    # # Fifth  Layer
    # # --------------------------------------------------------------------
    layer_index = 5
    display_hd_filter_opt_stimuli(
        vgg16_model, layer_index, gen_img_row=81, gen_img_col=81, num_iterations=50)