# -------------------------------------------------------------------------------------------------
#  This is the Alex Net model trained on Imagenet.
#
#  Code Ref: https://github.com/heuritech/convnets-keras. Updated to use Keras V2 APIs
#
# Author: Salman Khan
# Date  : 21/07/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Activation, MaxPooling2D, Conv2D, Concatenate
from keras.layers import Dense, Flatten, ZeroPadding2D, Dropout
from keras.engine.topology import Layer
from keras.preprocessing.image import img_to_array, load_img
import keras.backend as K

import base_alex_net as alex_net
import learned_lateral_weights
import utils
reload(utils)
reload(learned_lateral_weights)
reload(alex_net)

np.random.seed(7)  # Set the random seed for reproducibility


def get_enhancement_model_contour_kernels():
    n = 3
    kernel = np.zeros((96, n, n))
    kernel[10, :, :] = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]) / 2.0
    kernel[5, :, :] = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]) / 2.0
    kernel[54, :, :] = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]) / 2.0
    kernel[67, :, :] = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]) / 2.0

    return kernel, n


def get_suppression_model_contour_kernels():
    n = 3
    kernel = np.zeros((96, n, n))
    kernel[10, :, :] = np.array([[0, 0, 0], [-1, 0, -1], [0, 0, 0]]) / 2.0
    kernel[5, :, :] = np.array([[0, -1, 0], [0, 0, 0], [0, -1, 0]]) / 2.0
    kernel[54, :, :] = np.array([[0, 0, -1], [0, 0, 0], [-1, 0, 0]]) / 2.0
    kernel[67, :, :] = np.array([[0, 0, -1], [0, 0, 0], [-1, 0, 0]]) / 2.0

    return kernel, n


def get_enhance_n_suppress_contour_kernels():
    n = 3
    kernel = np.zeros((96, n, n))
    kernel[10, :, :] = np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]]) / 4.0
    kernel[5, :, :] = np.array([[0, -1, 0], [1, 0, 1], [0, -1, 0]]) / 4.0
    kernel[54, :, :] = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4.0
    kernel[67, :, :] = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4.0

    return kernel, n


def get_enhance_n_suppress_5x5_contour_kernels():
    n = 5
    kernel = np.zeros((96, n, n))
    kernel[10, :, :] = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [-1, -1, 0, -1, -1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]) / 8.0

    kernel[5, :, :] = np.array([
        [0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0],
    ]) / 8.0

    kernel[54, :, :] = np.array([
        [1, 0, 0, 0, -1],
        [0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0],
        [0, -1, 0, 1, 0],
        [-1, 0, 0, 0, 1],
    ]) / 8.0

    kernel[67, :, :] = np.array([
        [1, 0, 0, 0, -1],
        [0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0],
        [0, -1, 0, 1, 0],
        [-1, 0, 0, 0, 1],
    ]) / 8.0

    return kernel, n


def get_enhance_n_suppress_non_overlap_contour_kernels():
    n = 7
    kernel = np.zeros((96, n, n))
    kernel[10, :, :] = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]) / 4.0

    kernel[5, :, :] = np.array([
        [0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0],
    ]) / 4.0

    kernel[54, :, :] = np.array([
        [1, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 1],
    ]) / 4.0

    kernel[67, :, :] = np.array([
        [1, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 1],
    ]) / 4.0

    return kernel, n


class ContourIntegrationLayer(Layer):
    def __init__(self, model_type, **kwargs):
        """

        :param n:
        :param kwargs:
        """
        model_type = model_type.lower()
        valid_model_types = [
            'enhance',
            'suppress',
            'enhance_n_suppress',
            'enhance_n_suppress_5',
            'enhance_n_suppress_non_overlap']

        if model_type not in valid_model_types:
            raise Exception("Need to specify a valid model type")

        if model_type == 'enhance':
            self.kernel, self.n = get_enhancement_model_contour_kernels()
        elif model_type == 'suppress':
            self.kernel, self.n = get_suppression_model_contour_kernels()
        elif model_type == 'enhance_n_suppress':
            self.kernel, self.n = get_enhance_n_suppress_contour_kernels()
        elif model_type == 'enhance_n_suppress_5':
            self.kernel, self.n = get_enhance_n_suppress_5x5_contour_kernels()
        else:
            self.kernel, self.n = get_enhance_n_suppress_non_overlap_contour_kernels()

        self.kernel = K.variable(self.kernel)
        super(ContourIntegrationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ContourIntegrationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        if K.image_data_format() == 'channels_last':
            _, r, c, ch = K.int_shape(inputs)
            # print("Call Fcn: Input shape ", K.int_shape(inputs))
        else:
            _, ch, r, c = K.int_shape(inputs)
            # print("Call Fcn: Input shape ", K.int_shape(inputs))

        # 1. Inputs Formatting
        # ----------------------------------
        # Pad the rows and columns to allow full matrix multiplication
        # Note that this function is aware of which dimension the columns and rows are
        padded_inputs = K.spatial_2d_padding(
            inputs,
            ((self.n / 2, self.n / 2), (self.n / 2, self.n / 2))
        )
        # print("Call Fcn: padded_inputs shape ", K.int_shape(padded_inputs))

        # Channel first, batch second. This is done to take the unknown batch size into the matrix multiply
        # where it can be handled more easily
        if K.image_data_format() == 'channels_last':
            inputs_chan_first = K.permute_dimensions(padded_inputs, [3, 0, 1, 2])
        else:
            inputs_chan_first = K.permute_dimensions(padded_inputs, [1, 0, 2, 3])
        # print("Call Fcn: inputs_chan_first shape: ", inputs_chan_first.shape)

        # 2. Kernel Formatting
        # --------------------
        if K.image_data_format() == 'channels_last':
            kernel_chan_first = K.permute_dimensions(self.kernel, (2, 0, 1))
        else:
            kernel_chan_first = self.kernel
        # print("Call Fcn: kernel_chan_first shape", kernel_chan_first.shape)

        # Flatten rows and columns into a single dimension
        k_ch, k_r, k_c = K.int_shape(kernel_chan_first)
        apply_kernel = K.reshape(kernel_chan_first, (k_ch, k_r * k_c, 1))
        # print("Call Fcn: kernel for matrix multiply: ", apply_kernel.shape)

        # 3. Get outputs at each spatial location
        # ----------------------------------------
        xs = []
        for i in range(r):
            for j in range(c):
                input_slice = inputs_chan_first[:, :, i:i + self.n, j:j + self.n]
                input_slice_apply = K.reshape(input_slice, (ch, -1, self.n ** 2))

                output_slice = K.batch_dot(input_slice_apply, apply_kernel)
                # Reshape the output slice to put batch first
                output_slice = K.permute_dimensions(output_slice, [1, 0, 2])
                xs.append(output_slice)

        # print("Call Fcn: len of xs", len(xs))
        # print("Call Fcn: shape of each element of xs", xs[0].shape)

        # Reshape the output to correct format
        outputs = K.concatenate(xs, axis=2)
        outputs = K.reshape(outputs, (-1, ch, r, c))  # Break into row and column

        if K.image_data_format() == 'channels_last':
            outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])  # Back to batch last

        # 4. Add the lateral and the feed-forward activations
        # ------------------------------------------------------
        outputs += inputs
        return outputs


def build_model(weights_path, model_type):
    """
    Build a modified AlexNet with a Contour Emphasizing layer
    Note: Layer names have to stay the same, to enable loading pre-trained weights

    :param model_type:
    :param weights_path:
    :return:
    """

    inputs = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(inputs)

    contour_int_layer = ContourIntegrationLayer(name='contour_integration', model_type=model_type)(conv_1)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(contour_int_layer)
    conv_2 = alex_net.crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)

    conv_2_1 = Conv2D(128, (5, 5), activation='relu', name='conv_2_1') \
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_2))
    conv_2_2 = Conv2D(128, (5, 5), activation='relu', name='conv_2_2') \
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_2))
    conv_2 = Concatenate(axis=1, name='conv_2')([conv_2_1, conv_2_2])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = alex_net.crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4_1 = Conv2D(192, (3, 3), activation='relu', name='conv_4_1') \
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_4))
    conv_4_2 = Conv2D(192, (3, 3), activation='relu', name='conv_4_2') \
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_4))
    conv_4 = Concatenate(axis=1, name='conv_4')([conv_4_1, conv_4_2])

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5_1 = Conv2D(128, (3, 3), activation='relu', name='conv_5_1') \
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_5))
    conv_5_2 = Conv2D(128, (3, 3), activation='relu', name='conv_5_2') \
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_5))
    conv_5 = Concatenate(axis=1, name='conv_5')([conv_5_1, conv_5_2])

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)

    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)

    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(inputs=inputs, outputs=prediction)

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model


def get_layer_activation(model, layer_idx, data_sample):
    """
    Return the activation volume of the specified layer.

    :param model:
    :param layer_idx:
    :param data_sample:

    :return: the whole activation volume
    """

    # Define a function to get the activation volume
    get_layer_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_idx].output]
    )

    # Get the activations in a usable format
    act_volume = np.asarray(get_layer_output(
        [data_sample, 0],  # second input specifies the learning phase 0=output, 1=training
    ))

    # Reshape the activations, the casting above adds another dimension
    act_volume = act_volume.reshape(
        act_volume.shape[1],
        act_volume.shape[2],
        act_volume.shape[3],
        act_volume.shape[4]
    )

    return act_volume


def add_fragment_at_location(image, fragment, loc_x, loc_y):
    """

    :param image:
    :param fragment:
    :param loc_x:
    :param loc_y:
    :return:
    """
    stride = 4  # The Stride using the in convolutional layer
    filt_dim_r = fragment.shape[0]
    filt_dim_c = fragment.shape[1]

    print("Fragment Placed at x: %d-%d, y: %d-%d"
          % (loc_x * stride, loc_x * stride + filt_dim_r, loc_y * stride, loc_y * stride + filt_dim_c))

    image[
        loc_x * stride: loc_x * stride + filt_dim_r,
        loc_y * stride: loc_y * stride + filt_dim_c,
        :,
    ] += fragment

    return image


def replace_fragment_at_location(image, fragment, loc_x, loc_y):
    """
    Similar to add_fragment_at_location, but instead of adding the fragment, replaces it

    :param image:
    :param fragment:
    :param loc_x:
    :param loc_y:
    :return:
    """
    stride = 4  # The Stride using the in convolutional layer
    filt_dim_r = fragment.shape[0]
    filt_dim_c = fragment.shape[1]

    print("Fragment Placed at x: %d-%d, y: %d-%d"
          % (loc_x * stride, loc_x * stride + filt_dim_r, loc_y * stride, loc_y * stride + filt_dim_c))

    image[
        loc_x * stride: loc_x * stride + filt_dim_r,
        loc_y * stride: loc_y * stride + filt_dim_c,
        :,
    ] = fragment

    return image


def generate_test_contour_image_from_fragment(fragment, overlap=4, img_dim=227):
    """
     Generates contours by spatially tiling fragments

    :param img_dim:  dimension of the generated image (square image generated)
    :param overlap:  Number of pixels that overlap between tiled fragments (stride of the convolutional layer)
    :param fragment:
    :return:
    """
    test_image = np.zeros((img_dim, img_dim, 3))

    # Single Point
    add_fragment_at_location(test_image, fragment, 10, 10)

    # VERTICAL: Overlapping fragments
    # # contour_2 = np.zeros(((conv_1_stride * (5 - 1) + 11), 11, 3))
    # # contour_2[:, (0, 3, 4, 5, 9, 10), :] = 1
    # # add_fragment_at_location(test_image, contour_2, 5, 20)
    add_fragment_at_location(test_image, fragment, 5, 20)
    add_fragment_at_location(test_image, fragment, 6, 20)
    add_fragment_at_location(test_image, fragment, 7, 20)
    add_fragment_at_location(test_image, fragment, 8, 20)
    add_fragment_at_location(test_image, fragment, 9, 20)
    test_image = np.clip(test_image, 0, 1)

    # VERTICAL: Non-overlapping fragments
    add_fragment_at_location(test_image, fragment, 16, 20)
    add_fragment_at_location(test_image, fragment, 20, 20)
    add_fragment_at_location(test_image, fragment, 24, 20)
    add_fragment_at_location(test_image, fragment, 28, 20)

    # HORIZONTAL: Overlapping fragments
    add_fragment_at_location(test_image, fragment, 5, 40)
    add_fragment_at_location(test_image, fragment, 5, 41)
    add_fragment_at_location(test_image, fragment, 5, 42)
    add_fragment_at_location(test_image, fragment, 5, 43)
    test_image = np.clip(test_image, 0, 1)

    # HORIZONTAL: Non-overlapping fragments
    add_fragment_at_location(test_image, fragment, 15, 40)
    add_fragment_at_location(test_image, fragment, 15, 44)
    add_fragment_at_location(test_image, fragment, 15, 48)
    add_fragment_at_location(test_image, fragment, 15, 52)

    # HORIZONTAL: Visually Non-overlapping fragments. Spatially adjacent with no gaps
    start_x = 25 * overlap  # Starting at location 25, 40
    start_y = 40 * overlap
    for ii in range(4):
        test_image[start_x: start_x + 11, start_y + ii * 11: start_y + (ii + 1) * 11, :] = fragment

    # DIAGONAL (backward slash): overlapping
    add_fragment_at_location(test_image, fragment, 30, 5)
    add_fragment_at_location(test_image, fragment, 31, 6)
    add_fragment_at_location(test_image, fragment, 32, 7)
    add_fragment_at_location(test_image, fragment, 33, 8)
    add_fragment_at_location(test_image, fragment, 34, 9)
    test_image = np.clip(test_image, 0, 1)

    # DIAGONAL (backward slash): Visually Non-overlapping fragments. Spatially adjacent with no gaps
    start_x = 40 * overlap
    start_y = 5 * overlap
    for ii in range(5):
        test_image[
            start_x + ii * 11: start_x + (ii + 1) * 11,
            start_y + ii * 11: start_y + (ii + 1) * 11,
            :
        ] = fragment

    # DIAGONAL (forward slash): Visually Non-overlapping fragments. Spatially adjacent with no gaps
    start_x = 51 * overlap
    start_y = 30 * overlap
    for ii in range(5):
        test_image[
            start_x - (ii * 11): start_x - (ii * 11) + 11,
            start_y + (ii * 11): start_y + (ii + 1) * 11,
            :
        ] = fragment

    return test_image


def plot_tgt_filters_activations(model, image, f_idx, image_normalization=False):
    """

    :param image_normalization:
    :param model: model to use
    :param image: test image
    :param f_idx: index of filter
    :return:
    """
    x = image
    x = np.transpose(x, (2, 0, 1))
    x = np.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])

    l1_activations = get_layer_activation(model, 1, x)
    l2_activations = get_layer_activation(model, 2, x)

    tgt_l1_activation = l1_activations[0, f_idx, :, :]
    tgt_l2_activation = l2_activations[0, f_idx, :, :]
    if image_normalization:
        tgt_l1_activation = utils.deprocess_image(tgt_l1_activation)
        tgt_l2_activation = utils.deprocess_image(tgt_l2_activation)

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    max_activation = tgt_l2_activation.max()
    min_activation = tgt_l2_activation.min()

    plt.imshow(tgt_l1_activation, cmap='seismic', vmin=min_activation, vmax=max_activation)
    plt.title('Raw Feature map of conv layer (l1) at index %d' % f_idx)
    plt.colorbar(orientation='horizontal')
    plt.grid()

    f.add_subplot(1, 3, 2)
    plt.imshow(tgt_l2_activation, cmap='seismic', vmin=min_activation, vmax=max_activation)
    plt.title('Raw Feature map of contour integration layer (l2) at index %d' % f_idx)
    plt.colorbar(orientation='horizontal')
    plt.grid()

    f.add_subplot(1, 3, 3)
    plt.imshow(tgt_l2_activation - tgt_l1_activation, cmap='seismic')
    plt.colorbar(orientation='horizontal')
    plt.title("Difference")
    plt.grid()


def main(model, fragment, f_idx):
    """
    This is the main routine for this file. First, the given contour fragment is tiled into
    a test image to form variously oriented contours. This is then passed to the model to
    visualize the activations of the first convolutional layer and contour integration layer

    :param model:
    :param fragment:
    :param f_idx:
    :return:
    """

    tgt_conv1_filter = K.eval(model.layers[1].weights[0])
    tgt_conv1_filter = tgt_conv1_filter[:, :, :, f_idx]

    tgt_cont_int_filter = K.eval(model.layers[2].kernel)
    tgt_cont_int_filter = tgt_cont_int_filter[f_idx, :, :]

    contour_test_image = generate_test_contour_image_from_fragment(fragment)

    f = plt.figure()
    ax = plt.subplot2grid((3, 8), (0, 0), colspan=2)
    ax.imshow(tgt_conv1_filter)
    ax.set_title('Conv 1 Filter @ idx %d' % f_idx)

    ax1 = plt.subplot2grid((3, 8), (0, 2), colspan=2)
    ax1.imshow(fragment)
    ax1.set_title('Contour Fragment')

    ax2 = plt.subplot2grid((3, 8), (0, 4), colspan=4, rowspan=3)
    ax2.imshow(contour_test_image)
    ax2.set_title('Test image')

    ax3 = plt.subplot2grid((3, 8), (1, 0), colspan=2)
    ax3.imshow(tgt_cont_int_filter, cmap='seismic')
    ax3.set_title('Cont. Int. Filter @ idx %d' % f_idx)

    # Show individual slices of the conv filter
    tgt_conv_filter_min = tgt_conv1_filter.min()
    tgt_conv_filter_max = tgt_conv1_filter.max()
    ax4 = plt.subplot2grid((3, 8), (2, 0))
    ax4.imshow(tgt_conv1_filter[:, :, 0], cmap='seismic', vmin=tgt_conv_filter_min, vmax=tgt_conv_filter_max)
    ax5 = plt.subplot2grid((3, 8), (2, 1))
    ax5.imshow(tgt_conv1_filter[:, :, 1], cmap='seismic', vmin=tgt_conv_filter_min, vmax=tgt_conv_filter_max)
    ax6 = plt.subplot2grid((3, 8), (2, 2))
    ax_for_cb = \
        ax6.imshow(tgt_conv1_filter[:, :, 2], cmap='seismic', vmin=tgt_conv_filter_min, vmax=tgt_conv_filter_max)
    f.colorbar(ax_for_cb, )

    plot_tgt_filters_activations(model, contour_test_image, f_idx)


if __name__ == "__main__":
    plt.ion()

    # 1. Build the model
    # --------------------------------------------------------------------
    K.set_image_dim_ordering('th')  # Model was originally defined with Theano backend.
    print("Building Contour Integration Model...")

    # m_type = 'enhance'
    # m_type = 'suppress'
    m_type = 'enhance_n_suppress'
    # m_type = 'enhance_n_suppress_5'
    # m_type = 'enhance_n_suppress_non_overlap'
    alex_net_cont_int_model = build_model("trained_models/AlexNet/alexnet_weights.h5", model_type=m_type)
    # alex_net_cont_int_model.summary()

    # # 2. Display filters in the first convolutional and contour integration layers
    # # --------------------------------------------------------------------
    # weights_ch_last = alex_net_cont_int_model.layers[1].weights[0]
    # utils.display_filters(weights_ch_last)

    # weights_ch_last = alex_net_cont_int_model.layers[2].kernel
    # utils.display_filters(weights_ch_last)

    # # 3. Display the activations of a test image
    # # ---------------------------------------------------------------------
    # img = load_img("trained_models/AlexNet/SampleImages/cat.7.jpg", target_size=(227, 227))
    # img = load_img("trained_models/AlexNet/SampleImages/zahra.jpg", target_size=(227, 227))
    # plt.figure()
    # plt.imshow(img)
    # plt.title('Original Image')
    #
    # x = img_to_array(img)
    # x = np.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])
    #
    # # y_hat = alex_net_cont_int_model.predict(x, batch_size=1, verbose=1)
    # # print("Prediction %s" % np.argmax(y_hat))
    #
    # utils.display_layer_activations(alex_net_cont_int_model, 1, x)
    # utils.display_layer_activations(alex_net_cont_int_model, 2, x)

    # 4. Contour Enhancement Visualizations
    # ---------------------------------------------------------------------
    # Vertical Filter
    tgt_filt_idx = 10
    frag_1 = np.zeros((11, 11, 3))
    frag_1[:, (0, 3, 4, 5, 9, 10), :] = 1
    main(alex_net_cont_int_model, frag_1, tgt_filt_idx)

    # Horizontal Filter
    tgt_filt_idx = 5
    frag_2 = np.zeros((11, 11, 3))
    frag_2[0:6, :, :] = 1
    main(alex_net_cont_int_model, frag_2, tgt_filt_idx)

    # # Diagonal Filter (back slash)
    # tgt_filt_idx = 54
    # # frag_3 = K.eval(alex_net_cont_int_model.layers[1].weights[0])
    # # frag_3 = frag_3[:, :, :, tgt_filt_idx]
    # frag_3 = np.zeros((11, 11))
    # frag_3[0, :] = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    # for i in range(1, 11):
    #     if (i % 2) == 0:
    #         frag_3[i, :] = np.roll(frag_3[i - 1, :], 1)
    #     else:
    #         frag_3[i, :] = frag_3[i - 1, :]
    # frag_3 = np.reshape(frag_3, (11, 11, 1))
    # frag_3 = np.repeat(frag_3, 3, axis=2)
    # main(alex_net_cont_int_model, frag_3, tgt_filt_idx)

    # 5. Output of contour enhancement on real image
    # ----------------------------------------------------------------------
    # test_real_img = load_img("trained_models/AlexNet/SampleImages/zahra.jpg", target_size=(227, 227))
    # # test_real_img = load_img("trained_models/AlexNet/SampleImages/cat.7.jpg", target_size=(227, 227))
    #
    # tgt_filt_idx = 5
    # plot_tgt_filters_activations(alex_net_cont_int_model, test_real_img, tgt_filt_idx, image_normalization=True)
    #
    # tgt_filt_idx = 10
    # plot_tgt_filters_activations(alex_net_cont_int_model, test_real_img, tgt_filt_idx, image_normalization=True)
