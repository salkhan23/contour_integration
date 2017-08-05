# -------------------------------------------------------------------------------------------------
#  This models a nonlinear interactions between the feedforward and lateral connections model.
# it is based on the model in alex_net_cont_integration.py
#
# Author: Salman Khan
# Date  : 03/08/17
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
import alex_net_add_cont_int as linear_cont_int_model
reload(utils)
reload(learned_lateral_weights)
reload(alex_net)
reload(linear_cont_int_model)

np.random.seed(7)  # Set the random seed for reproducibility


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
            self.kernel, self.n = linear_cont_int_model.get_enhancement_model_contour_kernels()
        elif model_type == 'suppress':
            self.kernel, self.n = linear_cont_int_model.get_suppression_model_contour_kernels()
        elif model_type == 'enhance_n_suppress':
            self.kernel, self.n = linear_cont_int_model.get_enhance_n_suppress_contour_kernels()
        elif model_type == 'enhance_n_suppress_5':
            self.kernel, self.n = linear_cont_int_model.get_enhance_n_suppress_5x5_contour_kernels()
        else:
            self.kernel, self.n = linear_cont_int_model.get_enhance_n_suppress_non_overlap_contour_kernels()

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
                input_slice = inputs_chan_first[:, :, i:i+self.n, j:j+self.n]
                input_slice_apply = K.reshape(input_slice, (ch, -1, self.n**2))

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
        outputs = outputs * inputs
        #outputs = K.clip(outputs, -5, 5)
        return outputs + inputs


# TODO: This is the exact same code as the one in alex_net_contour_integration,
# TODO: just the contour integration layer is different. Combine the two
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

    conv_2_1 = Conv2D(128, (5, 5), activation='relu', name='conv_2_1')\
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_2))
    conv_2_2 = Conv2D(128, (5, 5), activation='relu', name='conv_2_2')\
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_2))
    conv_2 = Concatenate(axis=1, name='conv_2')([conv_2_1, conv_2_2])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = alex_net.crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4_1 = Conv2D(192, (3, 3), activation='relu', name='conv_4_1')\
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_4))
    conv_4_2 = Conv2D(192, (3, 3), activation='relu', name='conv_4_2')\
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_4))
    conv_4 = Concatenate(axis=1, name='conv_4')([conv_4_1, conv_4_2])

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5_1 = Conv2D(128, (3, 3), activation='relu', name='conv_5_1')\
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_5))
    conv_5_2 = Conv2D(128, (3, 3), activation='relu', name='conv_5_2')\
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


if __name__ == "__main__":

    plt.ion()

    # 1. Build the model
    # --------------------------------------------------------------------
    K.set_image_dim_ordering('th')  # Model was originally defined with Theano backend.
    print("Building Contour Integration Model...")

    # m_type = 'enhance'
    # m_type = 'suppress'
    # m_type = 'enhance_n_suppress'
    # m_type = 'enhance_n_suppress_5'
    m_type = 'enhance_n_suppress_non_overlap'
    alex_net_cont_int_model = build_model("trained_models/AlexNet/alexnet_weights.h5", model_type=m_type)
    # alex_net_cont_int_model.summary()

    # TODO: Again same code as  alex_net_add_cont_int.py. Combine!
    # 4. Contour Enhancement Visualizations
    # ---------------------------------------------------------------------
    # Vertical Filter
    tgt_filt_idx = 10
    frag_1 = np.zeros((11, 11, 3))
    frag_1[:, (0, 3, 4, 5, 9, 10), :] = 1
    linear_cont_int_model.main(alex_net_cont_int_model, frag_1, tgt_filt_idx)

    # # Horizontal Filter
    tgt_filt_idx = 5
    frag_2 = np.zeros((11, 11, 3))
    frag_2[0:6, :, :] = 1
    linear_cont_int_model.main(alex_net_cont_int_model, frag_2, tgt_filt_idx)
    # #
    # # # 5. Output of contour enhancement on real image
    # # # ----------------------------------------------------------------------
    test_real_img = load_img("trained_models/AlexNet/SampleImages/zahra.jpg", target_size=(227, 227))
    # test_real_img = load_img("trained_models/AlexNet/SampleImages/cat.7.jpg", target_size=(227, 227))
    #
    tgt_filt_idx = 5
    linear_cont_int_model.plot_tgt_filters_activations(
        alex_net_cont_int_model, test_real_img, tgt_filt_idx, image_normalization=True)
    #
    # tgt_filt_idx = 10
    # linear_cont_int_model.plot_tgt_filters_activations(
    #     alex_net_cont_int_model, test_real_img, tgt_filt_idx, image_normalization=True)
