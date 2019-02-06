# -------------------------------------------------------------------------------------------------
# Contour Integration Model Independent of target filter variable. All Contour integration
# filters to be trained together.
# --------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np

import keras


class ContourGainCalculatingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ContourGainCalculatingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        No Parameters
        """
        super(ContourGainCalculatingLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        b, ch, r, c = input_shape[1]
        return b, ch

    def call(self, inputs, **kwargs):
        """
        Calculate the enhancement gain of the neuron at the center of the image.

        :param inputs: a list/tuple of (feature extract activation, contour enhanced activation)
        :param kwargs:

        :return:
        """
        feature_act = inputs[0]
        contour_act = inputs[1]

        _, ch, r, c = keras.backend.int_shape(inputs[0])
        center_neuron_loc = (r >> 1, c >> 1)

        gain = contour_act[:, :, center_neuron_loc[0], center_neuron_loc[1]] / \
            (feature_act[:, :, center_neuron_loc[0], center_neuron_loc[1]] + 1e-4)

        return gain


class ContourIntegrationLayer3D(keras.layers.Layer):
    def __init__(self, inner_leaky_relu_alpha, outer_leaky_relu_alpha, rf_size=25,
                 activation=None, l1_reg_loss_weight=0.001, **kwargs):
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
        self.activation = keras.activations.get(activation)
        self.inner_leaky_relu_alpha = inner_leaky_relu_alpha
        self.outer_leaky_relu_alpha = outer_leaky_relu_alpha
        # TODO: Use l1_weight regularization
        self.l1_reg_loss_weight = l1_reg_loss_weight
        super(ContourIntegrationLayer3D, self).__init__(**kwargs)

    def build(self, input_shape):
        _, ch, r, c = input_shape
        # print("Build Fcn: Channel First Input shape ", input_shape)

        self.kernel = self.add_weight(
            shape=(self.n, self.n, ch, ch),
            name='kernel',
            initializer='glorot_normal',
            trainable=True,
            regularizer=keras.regularizers.l1(self.l1_reg_loss_weight)
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
        _, ch, r, c = keras.backend.int_shape(inputs)
        # print("Call Fcn: Channel First Input shape ", K.int_shape(inputs))

        outputs = keras.backend.conv2d(inputs, self.kernel, strides=(1, 1), padding='same')

        outputs = keras.backend.bias_add(outputs, self.bias)

        # Modulatory response: if not input, no output.
        outputs = outputs * inputs

        outputs = keras.layers.LeakyReLU(alpha=self.inner_leaky_relu_alpha)(outputs)
        outputs += inputs
        outputs = keras.layers.LeakyReLU(alpha=self.outer_leaky_relu_alpha)(outputs)

        # outputs = keras_backend.relu(outputs, alpha=self.inner_leaky_relu_alpha) + inputs
        # outputs = keras_backend.relu(outputs, alpha=self.outer_leaky_relu_alpha)

        return outputs


def training_model(rf_size=25, inner_leaky_relu_alpha=0.7, outer_leaky_relu_alpha=0.7, l1_reg_loss_weight=0.01):
    """

    :param rf_size:
    :param inner_leaky_relu_alpha:
    :param outer_leaky_relu_alpha:
    :param l1_reg_loss_weight:
    :return:
    """
    input_layer = keras.layers.Input(shape=(3, 227, 227))

    # AlexNet First Layer - Fixed
    conv_1 = keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(input_layer)

    contour_integrate_layer = ContourIntegrationLayer3D(
        rf_size=rf_size,
        inner_leaky_relu_alpha=inner_leaky_relu_alpha,
        outer_leaky_relu_alpha=outer_leaky_relu_alpha,
        l1_reg_loss_weight=l1_reg_loss_weight,
        name='contour_integration_layer')(conv_1)

    contour_gain_layer = ContourGainCalculatingLayer(name='contour_gain_layer')([conv_1, contour_integrate_layer])

    m = keras.Model(input_layer, outputs=contour_gain_layer)

    m.layers[1].trainable = False  # Set the feature extracting layer as untrainable.
    m.load_weights("trained_models/AlexNet/alexnet_weights.h5", by_name=True)
    m.compile(optimizer='Adam', loss='mse')

    return m


if __name__ == '__main__':

    plt.ion()
    np.random.seed(7)
    keras.backend.set_image_dim_ordering('th')  # For Theano / channel first

    # -----------------------------------------------------------------------------------
    # Build Contour Integration Model
    # -----------------------------------------------------------------------------------
    print("Building the Contour Integration Model ...")
    model = training_model(
        rf_size=35,
        inner_leaky_relu_alpha=0.9,
        outer_leaky_relu_alpha=1.,
        l1_reg_loss_weight=0.005,
    )

    # -----------------------------------------------------------------------------------
    # Validate the model is working properly
    # -----------------------------------------------------------------------------------
    image_name = "./data/sample_images/cat.7.jpg"

    image = keras.preprocessing.image.load_img(image_name, target_size=[227, 227, 3])

    # Channel First Transformation
    input_image = keras.preprocessing.image.img_to_array(image)

    y_hat = model.predict(np.expand_dims(input_image, axis=0), batch_size=1)
    print("Model Prediction Enhancement Gain of {}".format(y_hat))
