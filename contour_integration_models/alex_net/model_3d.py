# -------------------------------------------------------------------------------------------------
#  New Model of Contour Integration.
#
#  Compared to  previous models, the contour integration kernel used is 3D and connects across
#  feature maps
#
# Author: Salman Khan
# Date  : 27/04/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.layers import Input, Conv2D
from keras.engine.topology import Layer
import keras.activations as activations
from keras.regularizers import l1
import keras
from keras.models import Model


class ContourGainCalculatorLayer(Layer):
    def __init__(self, tgt_filt_idx, **kwargs):
        """
        A layer that calculates the enhancement gain of the neuron focused on the center of the image
        and at channel index = tgt_filt_idx.

        TODO: Make the tgt_filt_idx configurable. So that it can be specified in the call function.
        TODO: This will allow all contour integration kernels to be trained in the same layer.
        TODO: Alternatively try lambda layers

        :param tgt_filt_idx:
        :param kwargs:
        """
        self.tgt_filt_idx = tgt_filt_idx
        super(ContourGainCalculatorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ContourGainCalculatorLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], 1

    def call(self, inputs, **kwargs):
        """
        Calculate the enhancement gain of the neuron at the center of the image.

        :param inputs: a list/tuple of (feature extract activation, contour enhanced activation)
        :param kwargs:

        :return:
        """
        feature_act = inputs[0]
        contour_act = inputs[1]

        _, ch, r, c = K.int_shape(inputs[0])
        center_neuron_loc = (r >> 1, c >> 1)

        gain = contour_act[:, self.tgt_filt_idx, center_neuron_loc[0], center_neuron_loc[1]] / \
            (feature_act[:, self.tgt_filt_idx, center_neuron_loc[0], center_neuron_loc[1]] + 1e-8)

        return K.expand_dims(gain, axis=-1)


class ContourIntegrationLayer3D(Layer):

    def __init__(self, rf_size=25, activation=None, **kwargs):
        """
        Contour Integration layer. Different from previous contour integration layers,
        the contour integration kernel is 3D and allows connections between feature maps

        :param rf_size:
        :param activation:
        :param kwargs:
        """

        if 0 == (rf_size & 1):
            raise Exception("Specified RF size should be odd")

        self.n = rf_size
        self.activation = activations.get(activation)
        super(ContourIntegrationLayer3D, self).__init__(**kwargs)

    def build(self, input_shape):
        _, ch, r, c = input_shape
        # print("Build Fcn: Channel First Input shape ", input_shape)

        # Todo: Check which dimension is input and which one is output
        self.kernel = self.add_weight(
            shape=(self.n, self.n, ch, ch),
            initializer='glorot_normal',
            name='kernel',
            trainable=True,
            regularizer=l1(0.005)
        )

        self.bias = self.add_weight(
            shape=(ch,),
            initializer='zeros',
            name='bias',
            trainable=True,
            # regularizer=l1(0.05)
        )

        super(ContourIntegrationLayer3D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs, **kwargs):
        """
        Selectively enhance the gain of neurons in the feature extracting activation volume that
        are part of a smooth contour.

        :param inputs:
        :param kwargs:
        :return:
        """
        _, ch, r, c = K.int_shape(inputs)
        # print("Call Fcn: Channel First Input shape ", K.int_shape(inputs))

        outputs = K.conv2d(inputs, self.kernel, strides=(1, 1), padding='same')

        #outputs = outputs * inputs
        outputs = K.bias_add(outputs, self.bias)

        outputs = self.activation(outputs) #+ inputs

        return outputs


def build_contour_integration_model(tgt_filt_idx, rf_size=25):
    """
    Build a (short) model of 3D contour integration that can be used to train the model.

    THis is build on after the first feature extracting layer of object classifying network,
    and only trains the contour integration layer. THe complete model can still be used for
    object classification

    :param rf_size:
    :param tgt_filt_idx:
    :return:
    """
    input_layer = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(input_layer)

    contour_integrate_layer = ContourIntegrationLayer3D(rf_size=rf_size, activation='relu')(conv_1)

    contour_gain_layer = ContourGainCalculatorLayer(tgt_filt_idx)([
        conv_1, contour_integrate_layer])

    model = Model(input_layer, outputs=contour_gain_layer)

    model.layers[1].trainable = False  # Set the feature extracting layer as untrainable.

    model.load_weights("trained_models/AlexNet/alexnet_weights.h5", by_name=True)
    model.compile(optimizer='Adam', loss='mse')

    return model


if __name__ == '__main__':

    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    tgt_kernel_idx = 5

    np.random.seed(7)  # Set the random seed for reproducibility

    # -----------------------------------------------------------------------------------
    # Build the model
    # -----------------------------------------------------------------------------------
    print("Building the contour integration model...")
    cont_int_model = build_contour_integration_model(tgt_kernel_idx)
    # print cont_int_model.summary()

    # -----------------------------------------------------------------------------------
    # Validate the model is working properly
    # -----------------------------------------------------------------------------------
    image_name = "./data/sample_images/cat.7.jpg"

    # Option 1: Keras way
    # --------------------
    image = keras.preprocessing.image.load_img(
        image_name,
        target_size=[227, 227, 3]
    )

    # Takes care of putting channel first.
    input_image = keras.preprocessing.image.img_to_array(image)

    # # Option 2: pyplot and numpy only
    # # -------------------------------
    # # Note: This method only works for images that do not need to be resized.
    # image = plt.imread(image_name)
    # input_image = np.transpose(image, axes=(2, 0, 1))

    # plt.figure()
    # plt.imshow(image)

    y_hat = cont_int_model.predict(np.expand_dims(input_image, axis=0), batch_size=1)
    print("Model Prediction Enhancement Gain of {}".format(y_hat))
