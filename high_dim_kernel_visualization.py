# -------------------------------------------------------------------------------------------------
# Visualizing inputs the maximally excite a kernel using gradient accent.
#
# Technique can be used for higher layer filters for which the input channels are not 1 or 3.
#
# Compared to the original tutorial, the input images are resized to be the same size of the filters.
# This results in filters that are approximately the same as the kernel
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

# Dimensions of the generated pictures for each filter
IMG_WIDTH = 3
IMG_HEIGHT = 3
NUM_ITERATIONS = 20  # Iterations of the gradient ascent

# The name of the layer we want to visualize
LAYER_NAME = 'block1_conv1'


def normalize(x):
    """
    Utility function to normalize a tensor by its L2 norm

    :param x:
    :return:
    """
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


if __name__ == "__main__":

    plt.ion()

    model = vgg16.VGG16(weights='imagenet', include_top=False)
    model.summary()

    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    weights = layer_dict[LAYER_NAME].weights[0]  # Only the weights not the bias
    r, c, in_ch, out_ch = K.int_shape(weights)

    # 1. Display the filters of the first layer
    # --------------------------------------------------------------------
    utils.display_filters(weights)
    plt.title("Actual Kernels of layer %s" %LAYER_NAME)

    # 2. Display activations that maximally activate a neuron
    # This is the main tutorial
    # --------------------------------------------------------------------
    input_img = model.input

    keep_filters = []
    for filter_idx in range(0, out_ch):
        print("Processing filter %d" % filter_idx)
        start_time = time.time()

        # Build a loss function that maximizes the activation of the nth filter of the layer being considered
        layer_output = layer_dict[LAYER_NAME].output

        if K.image_data_format() == 'channel_first':
            loss = K.mean(layer_output[:, filter_idx, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_idx])

        # Compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]  # Get the gradient of the loss function wrt input image
        grads = normalize(grads)  # avoids very large or very small gradients and ensures a smooth ascent

        # This function returns the loss and the grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # Step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channel_first':
            input_img_data = np.random.random((1, in_ch, IMG_WIDTH, IMG_HEIGHT))  # generates random floats b\w (0,1)
        else:
            input_img_data = np.random.random((1, IMG_WIDTH, IMG_HEIGHT, in_ch))  # generates random floats b\w (0,1)
        input_img_data = (input_img_data - 0.5) * NUM_ITERATIONS + 128  # Why x20? ## number of steps?

        # Run gradient ascent for 20 steps
        for i in range(NUM_ITERATIONS):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value

            print("Current Loss value:", loss_value)
            if loss_value <= 0.0:
                break  # some filters get stuck to 0, we can skip them

        # Decode the resulting input image
        # if loss_value > 0:
        img = utils.deprocess_image(input_img_data[0])  # removes the batch index
        keep_filters.append((img, loss_value))

        end_time = time.time()
        print('Filter %d processes in %ds' % (filter_idx, end_time - start_time))

    # Stitch the best 64 filters on a 8x8 grid
    n = 8
    #
    # # filters with the highest loss are assumed to be better looking
    # keep_filters.sort(key=lambda x: x[1], reverse=True)
    # keep_filters = keep_filters[:n * n]
    #
    # build a black picture with enough space for our 8x8 filters of size 128x128 with a 5pnt margin between
    margin = 1
    width = n * IMG_WIDTH + (n - 1) * margin
    height = n * IMG_HEIGHT + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill in the picture with the saved filters
    for i in range(n):
        for j in range(n):
            img, loss = keep_filters[i * n + j]
            stitched_filters[(IMG_WIDTH + margin) * i: (IMG_WIDTH + margin)*i + IMG_WIDTH,
                             (IMG_HEIGHT + margin) * j:(IMG_HEIGHT+margin)*j + IMG_HEIGHT, :] = img

    # save the results
    plt.figure()
    plt.imshow(stitched_filters)
    plt.title("Max activations of individual filters at Layer %s" % LAYER_NAME)
    plt.colorbar()
