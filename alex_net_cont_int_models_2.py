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
import keras

import alex_net_cont_int_models as old_cont_int_models
import contour_image_generator
import alex_net_utils
import alex_net_hyper_param_search_multiplicative as mult_param_opt
import alex_net_cont_int_complex_bg as complex_bg

reload(old_cont_int_models)
reload(contour_image_generator)
reload(alex_net_utils)
reload(mult_param_opt)
reload(complex_bg)


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

    cont_int_layer = \
        old_cont_int_models.MultiplicativeContourIntegrationLayer(n=rf_size, activation='relu')(conv_1)

    cont_gain_calc_layer = EnhancementGainCalculatingLayer(tgt_filt_idx)([conv_1, cont_int_layer])

    model = Model(input_layer, outputs=cont_gain_calc_layer)

    model.load_weights("trained_models/AlexNet/alexnet_weights.h5", by_name=True)
    model.layers[1].trainable = False

    return model


if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # --------------------------------
    tgt_filter_idx = 10

    # Build the contour integration model
    # -----------------------------------
    print("Building Model ...")
    contour_integration_model = build_contour_integration_training_model(
        rf_size=25,
        tgt_filt_idx=tgt_filter_idx
    )
    print contour_integration_model.summary()

    # Define callback functions to get activations of L1 convolutional layer &
    # L2 contour integration layer
    l1_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    l2_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)

    # Store the start weights & bias for comparison later
    start_weights, start_bias = contour_integration_model.layers[2].get_weights()

    # Build the contour image generator
    # ----------------------------------
    print("Building Train Image Generator ...")
    feature_extract_kernels = K.eval(contour_integration_model.layers[1].weights[0])
    feature_extract_kernel = feature_extract_kernels[:, :, :, tgt_filter_idx]

    fragment = np.zeros((11, 11, 3))
    fragment[:, (0, 3, 4, 5, 9, 10), :] = 255.0

    train_image_generator = contour_image_generator.ContourImageGenerator(
        tgt_filt=feature_extract_kernel,
        tgt_filt_idx=tgt_filter_idx,
        contour_tile_loc_cb=alex_net_utils.vertical_contour_generator,
        row_offset=0,
        frag=fragment
    )

    # # Test the feature extract Kernel
    s = train_image_generator.generate(images_type='both')
    # X, y = s.next()
    # train_image_generator.show_image_batch(X, y)

    # Train the model
    # ---------------
    print("Starting Training ...")

    custom_optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    contour_integration_model.compile(optimizer='Adam', loss='mse')

    history = contour_integration_model.fit_generator(
        generator=s,
        steps_per_epoch=1,
        epochs=1000,
        verbose=2,
        # max_q_size=1,
        # workers=1,
    )

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    # Plot the Learnt weights
    # -----------------------
    mult_param_opt.plot_optimized_weights(
        contour_integration_model,
        tgt_filter_idx,
        start_weights,
        start_bias)

    # Plot Gain vs Contour Length after Optimization
    complex_bg.main_contour_length_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.vertical_contour_generator,
        tgt_filter_idx,
        smoothing=True,
        row_offset=0,
        n_runs=100,
    )

    # Plot Gain vs Contour Spacing after Optimization
    complex_bg.main_contour_spacing_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.vertical_contour_generator,
        tgt_filter_idx,
        smoothing=True,
        row_offset=0,
        n_runs=100)
