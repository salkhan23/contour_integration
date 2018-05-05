# -------------------------------------------------------------------------------------------------
# New model of contour integration that can be trained in the more standard way.
# It consists of two sub models with shared weights.
#
# (1) The first sub model is the Alexnet classification model with a contour integration layer
#     inserted after the first feature extracting layer
#
# (2) The second model is used for separately training the contour integration layer. It consists
#     if a feature extracting layer, contour integration layer and a layer to extract the
#     enhancement gain of the the the contour enhancing layer.
#
# Author: Salman Khan
# Date  : 10/02/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.engine.topology import Layer
import keras.activations as activations
from keras.regularizers import l1

from contour_integration_models.alex_net import masked_models as old_cont_int_models
import image_generator_linear
import alex_net_utils
import alex_net_hyper_param_search_multiplicative as mult_param_opt
import li_2006_routines
import gabor_fits

reload(old_cont_int_models)
reload(image_generator_linear)
reload(alex_net_utils)
reload(mult_param_opt)
reload(li_2006_routines)
reload(gabor_fits)


class EnhancementGainCalculatingLayer(Layer):
    def __init__(self, tgt_filt_idx, **kwargs):
        """
        Calculates the Enhancement gain of the neuron at the center of the activations for the
        specified kernel index

        :param tgt_kernel_idx:
        :param tgt_neuron_loc:
        :param kwargs:
        """
        self.tgt_filt_idx = tgt_filt_idx
        super(EnhancementGainCalculatingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        No parameter for this layer

        :param input_shape:
        :return:
        """
        super(EnhancementGainCalculatingLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], 1

    def call(self, inputs, **kwargs):
        """
        inputs is a tuple of feature layer and contour integration layer activations

        :param inputs:
        :param kwargs:
        :return:
        """
        feature_act = inputs[0]
        contour_act = inputs[1]

        _, ch, r, c = K.int_shape(inputs[0])

        center_neuron_loc = (r >> 1, c >> 1)
        # print('Call Fcn: center neuron location', center_neuron_loc)
        # print("Call Fcn: shape of input[0] and shape of inputs[1]", inputs[0].shape, inputs[1].shape)

        gain = contour_act[:, self.tgt_filt_idx, center_neuron_loc[0], center_neuron_loc[1]] / \
            (feature_act[:, self.tgt_filt_idx, center_neuron_loc[0], center_neuron_loc[1]] + 1e-8)

        return K.expand_dims(gain, axis=-1)


def build_contour_integration_training_model(tgt_filt_idx, rf_size=25, stride_length=(4, 4)):
    """

    :param tgt_filt_idx:
    :param stride_length:
    :param rf_size:
    :return:
    """
    input_layer = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=stride_length, activation='relu', name='conv_1')(input_layer)

    cont_int_layer = MultiplicativeContourIntegrationLayer(
        name='cont_int', rf_size=rf_size)(conv_1)

    # cont_int_layer = \
    #     old_cont_int_models.MultiplicativeContourIntegrationLayer(n=rf_size, activation='relu')(conv_1)

    cont_gain_calc_layer = EnhancementGainCalculatingLayer(tgt_filt_idx)([conv_1, cont_int_layer])

    model = Model(input_layer, outputs=cont_gain_calc_layer)

    model.load_weights("trained_models/AlexNet/alexnet_weights.h5", by_name=True)
    model.layers[1].trainable = False

    return model


class MultiplicativeContourIntegrationLayer(Layer):

    def __init__(self, rf_size=25, activation=None, **kwargs):
        """
        Contour Integration layer - Different from the old multiplicative contour integration layer
        no mask is assumed.

        :param tgt_filt_idx:
        :param tgt_neuron_loc:
        :param kwargs:
        """

        if 0 == (rf_size & 1):
            raise Exception("Specified RF size should be odd")

        self.n = rf_size
        self.activation = activations.get(activation)
        super(MultiplicativeContourIntegrationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        _, ch, r, c = input_shape
        # print("Build Fcn: Channel First Input shape ", input_shape)

        self.kernel = self.add_weight(
            shape=(ch, self.n, self.n,),
            initializer='glorot_normal',
            name='raw_kernel',
            trainable=True,
            regularizer=l1(0.05),
        )

        self.bias = self.add_weight(
            shape=(ch, 1, 1),
            initializer='zeros',
            name='bias',
            trainable=True,
            # regularizer=l1(0.01)
        )

        super(MultiplicativeContourIntegrationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        _, ch, r, c = K.int_shape(inputs)
        # print("Call Fcn: Channel First Input shape ", K.int_shape(inputs))

        # 1. Inputs Formatting
        # --------------------
        # Pad the rows and columns to allow full matrix multiplication
        # Note that this function is aware of which dimension the columns and rows are
        padded_inputs = K.spatial_2d_padding(
            inputs,
            ((self.n / 2, self.n / 2), (self.n / 2, self.n / 2))
        )
        # print("Call Fcn: padded_inputs shape ", K.int_shape(padded_inputs))

        # Channel first, batch second. This is done to take the unknown batch size into the matrix
        # multiply where it can be handled more easily
        inputs_chan_first = K.permute_dimensions(padded_inputs, [1, 0, 2, 3])
        # print("Call Fcn: inputs_chan_first shape: ", inputs_chan_first.shape)

        # 2. Kernel Formatting
        # --------------------
        # Flatten rows and columns into a single dimension
        k_ch, k_r, k_c = K.int_shape(self.kernel)
        apply_kernel = K.reshape(self.kernel, (k_ch, k_r * k_c, 1))
        # print("Call Fcn: kernel for matrix multiply: ", apply_kernel.shape)

        # 3. Get outputs at each spatial location
        # ---------------------------------------
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

        # 4. Add the lateral and the feed-forward activations
        # ------------------------------------------------------
        outputs = outputs * inputs + self.bias
        outputs = self.activation(outputs)

        return outputs + inputs


def plot_optimized_weights(tgt_model, tgt_filt_idx, start_w, start_b):
    """
    Plot starting and trained weights at the specified index

    :param tgt_model:
    :param tgt_filt_idx:
    :param start_w:
    :param start_b:
    :return:
    """
    opt_w, opt_b = tgt_model.layers[2].get_weights()
    max_v_opt = max(opt_w.max(), abs(opt_w.min()))
    max_v_start = max(start_w.max(), abs(start_w.min()))

    f = plt.figure()
    f.add_subplot(1, 2, 1)

    plt.imshow(start_w[tgt_filt_idx, :, :], vmin=-max_v_start, vmax=max_v_start)
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.tick_params(labelsize=20)
    plt.title("Start weights & bias=%0.4f" % start_b[tgt_filt_idx])

    f.add_subplot(1, 2, 2)
    plt.imshow(opt_w[tgt_filt_idx, :, :], vmin=-max_v_opt, vmax=max_v_opt)
    cb = plt.colorbar(orientation='horizontal')
    cb.ax.tick_params(labelsize=20)
    plt.title("Best weights & bias=%0.4f" % opt_b[tgt_filt_idx])


if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    tgt_filter_idx = 10

    # --------------------------------------------------------------------------------
    # Build the contour integration model
    # --------------------------------------------------------------------------------
    print("Building Model ...")
    contour_integration_model = build_contour_integration_training_model(
        rf_size=25,
        tgt_filt_idx=tgt_filter_idx
    )

    print contour_integration_model.summary()
