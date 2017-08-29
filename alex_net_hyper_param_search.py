# -------------------------------------------------------------------------------------------------
# The basic idea of this script is to match the gain from contour enhancement as see in
#
# Li, Piech and Gilbert - 2006 - Contour Saliency in Primary Visual Cortex
#
# Author: Salman Khan
# Date  : 27/08/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Conv2D
from keras.engine.topology import Layer
import keras.backend as K
import keras

import base_alex_net as alex_net
import learned_lateral_weights
import utils
import alex_net_add_cont_int as linear_cont_int_model
import alex_net_cont_int_complex_bg as complex_bg
reload(utils)
reload(learned_lateral_weights)
reload(alex_net)
reload(linear_cont_int_model)
reload(complex_bg)

np.random.seed(7)  # Set the random seed for reproducibility


class ContourIntegrationLayer(Layer):

    def __init__(self, model_type, **kwargs):
        """

        This is similar to the multiplicative nonlinear contour enhancment layer, but with two added
        learnable parameters.

        :param n:
        :param kwargs:
        """
        model_type = model_type.lower()
        valid_model_types = [
            'enhance',
            'suppress',
            'enhance_n_suppress',
            'enhance_n_suppress_5',
            'enhance_n_suppress_non_overlap',
            'non_overlap_full'
        ]

        if model_type not in valid_model_types:
            raise Exception("Need to specify a valid model type")

        if model_type == 'enhance':
            self.kernel, self.n = linear_cont_int_model.get_enhancement_model_contour_kernels()
        elif model_type == 'suppress':
            self.kernel, self.n = linear_cont_int_model.get_suppression_model_contour_kernels()
        elif model_type == 'enhance_n_suppress':
            self.kernel, self.n = linear_cont_int_model.get_enhance_n_suppress_contour_kernels()
        elif model_type == 'enhance_n_suppress_5':
            self.kernel, self.n = linear_cont_int_model.get_enhance_n_suppress_5x5_contour_kernels()
        elif model_type == 'enhance_n_suppress_non_overlap':
            self.kernel, self.n = linear_cont_int_model.get_enhance_n_suppress_non_overlap_contour_kernels()
        else:
            self.kernel, self.n = linear_cont_int_model.get_non_overlap_full_contour_kernels()

        self.kernel = K.variable(self.kernel)
        super(ContourIntegrationLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bias = self.add_weight(
            shape=(1, 1),
            initializer='ones',
            name='bias',
            trainable=True
        )

        self.alpha = self.add_weight(
            shape=(1, 1),
            initializer='ones',
            name='alpha',
            trainable=True
        )

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
                input_slice = inputs_chan_first[:, :, i:i+self.n, j:j+self.n]
                input_slice_apply = K.reshape(input_slice, (ch, -1, self.n**2))

                output_slice = K.batch_dot(input_slice_apply, apply_kernel) * self.alpha + self.bias
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
            outputs = outputs

        # 4. Add the lateral and the feed-forward activations
        # ------------------------------------------------------
        outputs = outputs * inputs
        # outputs = K.clip(outputs, -5, 5)
        return outputs + inputs


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

    model = Model(inputs=inputs, outputs=contour_int_layer)

    model.layers[1].trainable = False

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    model.compile(
        loss=keras.losses.categorical_crossentropy,  # Note this is not a function call.
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":

    plt.ion()

    # 1. Build the model
    # ---------------------------------------------------------------------
    K.clear_session()
    K.set_image_dim_ordering('th')  # Model was originally defined with Theano backend.
    print("Building Contour Integration Model...")

    m_type = 'non_overlap_full'
    alex_net_cont_int_model = build_model("trained_models/AlexNet/alexnet_weights.h5", model_type=m_type)
    alex_net_cont_int_model.summary()

    # 2. Create the test image
    # ---------------------------------------------------------------------
    fragment = np.zeros((11, 11, 3))  # Dimensions of the L1 convolutional layer of alexnet
    fragment[:, (0, 3, 4, 5, 9, 10), :] = 1

    start_x = np.array([97, 108, 119])  # stride of Conv L1
    start_y = np.array([108, 108, 108])

    test_image = np.zeros((227, 227, 3))

    test_image = complex_bg.tile_image(
        test_image,
        fragment,
        (start_x, start_y),
        rotate=False,
        gaussian_smoothing=False
    )

    input_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format
    input_image = np.reshape(input_image, [1, input_image.shape[0], input_image.shape[1], input_image.shape[2]])

    plt.figure()
    plt.imshow(test_image)
    plt.title("input image")

    linear_cont_int_model.plot_tgt_filters_activations(alex_net_cont_int_model, test_image, 10)
    plt.suptitle("Before  Adjustment")

    # 3. Meat!
    # ---------------------------------
    l1_output_cb = alex_net_cont_int_model.layers[1].output
    l2_output_cb = alex_net_cont_int_model.layers[2].output
    input_cb = alex_net_cont_int_model.input

    # New loss function that compares the current gain with the expected gain
    loss = 3 - K.square(l2_output_cb[:, 10, 27, 27] / (l1_output_cb[:, 10, 27, 27] + 1e-5))

    alpha = alex_net_cont_int_model.layers[2].alpha
    bias = alex_net_cont_int_model.layers[2].bias

    # Gradients of alpha and bias wrt to the loss function
    grads = K.gradients(loss, [alpha, bias])
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # Normalize the gradients, prevents too large and too small gradients

    iterate = K.function([input_cb], [loss, grads])

    l1_activations_cb = complex_bg.get_activation_cb(alex_net_cont_int_model, 1)
    l2_activations_cb = complex_bg.get_activation_cb(alex_net_cont_int_model, 2)

    step = 0.25
    old_loss = 10000000
    for r_idx in range(20):
        loss_value, grad_value = iterate([input_image])
        print("%d: loss value %s [old %s], grad value %s" % (r_idx, loss_value, old_loss, grad_value))

        alpha, bias = alex_net_cont_int_model.layers[2].get_weights()
        # if old_loss < loss_value:
        #     step /= 2.0
        # else:
        new_alpha = alpha + grad_value[0] * step
        new_bias = bias + grad_value[1] * step

        old_loss = loss_value

        print("New alpha=%0.4f, New bias =%0.4f" % (new_alpha, new_bias))
        alex_net_cont_int_model.layers[2].set_weights([new_alpha, new_bias])
        # print the new activations
        l1_act = np.array(l1_activations_cb([input_image, 0]))
        l1_act = np.squeeze(np.array(l1_act), axis=0)
        l2_act = np.array(l2_activations_cb([input_image, 0]))
        l2_act = np.squeeze(np.array(l2_act), axis=0)

        print("Contour Enhancement Gain %0.4f" % (l2_act[0, 10, 27, 27]/l1_act[0, 10, 27, 27]))

    linear_cont_int_model.plot_tgt_filters_activations(alex_net_cont_int_model, test_image, 10)
    plt.suptitle("AFTER")
