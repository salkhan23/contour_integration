from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle

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


def get_kernel_mask():
    """

    :return:
    """
    n = 25
    kernel = np.zeros((96, n, n))

    # Vertical Kernel
    kernel[10, (0, 3, 6, 9, 15, 18, 21, 24), (12, 12, 12, 12, 12, 12, 12, 12)] = 1
    # kernel[10, (12, 12, 12, 12, 12, 12, 12, 12), (0, 3, 6, 9, 15, 18, 21, 24)] = -1
    kernel[10, :, :] = kernel[10, :, :]

    # Horizontal Kernel
    # kernel[5, (0, 3, 6, 9, 15, 18, 21, 24), (12, 12, 12, 12, 12, 12, 12, 12)] = -1
    kernel[5, (12, 12, 12, 12, 12, 12, 12, 12), (0, 3, 6, 9, 15, 18, 21, 24)] = 1
    kernel[5, :, :] = kernel[5, :, :]

    # Diagonal Kernel (Leaning backwards)
    kernel[54, (0, 3, 6, 9, 15, 18, 21, 24), (0, 3, 6, 9, 15, 18, 21, 24)] = 1
    # kernel[54, (0, 3, 6, 9, 15, 18, 21, 24), (24, 21, 18, 15, 12, 6, 3, 0)] = -1
    kernel[54, :, :] = kernel[54, :, :]

    # Diagonal Kernel(Leaning backwards)
    kernel[67, :, :] = np.copy(kernel[54, :, :])

    return kernel, n


class ContourIntegrationLayer(Layer):

    def __init__(self, model_type, **kwargs):
        """

        :param kwargs:
        """
        self.kernel_mask, self.n = get_kernel_mask()

        super(ContourIntegrationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if K.image_data_format() == 'channels_last':
            _, r, c, ch = input_shape
            # print("Build Fcn: Channel Last Input shape ", input_shape)
        else:
            _, ch, r, c = input_shape
            # print("Build Fcn: Channel First Input shape ", input_shape)

        self.raw_kernel = self.add_weight(
            shape=(ch, self.n, self.n,),
            initializer='ones',
            name='raw_kernel',
            trainable=True
        )

        self.bias = self.add_weight(
            shape=(ch, 1, 1),
            initializer='zeros',
            name='bias',
            trainable=True
        )

        super(ContourIntegrationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs, **kwargs):

        if K.image_data_format() == 'channels_last':
            _, r, c, ch = K.int_shape(inputs)
            # print("Call Fcn: Channel Last Input shape ", K.int_shape(inputs))
        else:
            _, ch, r, c = K.int_shape(inputs)
            # print("Call Fcn: Channel First Input shape ", K.int_shape(inputs))

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
        #print("Call Fcn: inputs_chan_first shape: ", inputs_chan_first.shape)

        # 2. Kernel Formatting
        # --------------------
        # mask the kernel to keep only neighbors with overlapping RFs
        masked_kernel = self.raw_kernel * self.kernel_mask

        if K.image_data_format() == 'channels_last':

            kernel_chan_first = K.permute_dimensions(masked_kernel, (2, 0, 1))
        else:
            kernel_chan_first = masked_kernel
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
            outputs = outputs

        # 4. Add the lateral and the feed-forward activations
        # ------------------------------------------------------
        outputs = outputs * inputs + self.bias

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

    # Explore the model make sure everything is working as expected
    # --------------------------------------------------------------------
    # Define callback functions to get activations of L1 convolutional layer &
    # L2 contour integration layer
    l1_activations_cb = complex_bg.get_activation_cb(alex_net_cont_int_model, 1)
    l2_activations_cb = complex_bg.get_activation_cb(alex_net_cont_int_model, 2)

    # --------------------------------------------------------------------
    tgt_filter_index = 10
    tgt_neuron_location = (27, 27)

    # Simpler 'target filter' like contour fragment
    fragment = np.zeros((11, 11, 3))  # Dimensions of the L1 convolutional layer of alexnet
    fragment[:, (0, 3, 4, 5, 9, 10), :] = 255
    use_smoothing = True

    # test_image = np.zeros((227, 227, 3))
    # start_x, start_y = start_x, start_y = complex_bg.vertical_contour_generator(
    #     fragment.shape[0],
    #     bw_tile_spacing=0,
    #     cont_len=9,
    #     cont_start_loc=tgt_neuron_location[0] * 4
    # )
    #
    # test_image = complex_bg.tile_image(
    #     test_image,
    #     fragment,
    #     (start_x, start_y),
    #     rotate=False,
    #     gaussian_smoothing=True
    # )
    #
    # # Image preprocessing
    # test_image = test_image / 255.0  # Bring test_image back to the [0, 1] range.
    # #
    # # plt.figure()
    # # plt.imshow(test_image)
    # # plt.title('Input Image')
    #
    # test_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format
    # test_image = np.reshape(test_image, [1, test_image.shape[0], test_image.shape[1], test_image.shape[2]])
    #
    # l1_act = l1_activations_cb([test_image, 0])
    # l1_act = np.squeeze(np.array(l1_act), axis=0)
    # l2_act = l2_activations_cb([test_image, 0])
    # l2_act = np.squeeze(np.array(l2_act), axis=0)
    #
    # tgt_l1_act = l1_act[0, tgt_filter_index, :, :]
    # tgt_l2_act = l2_act[0, tgt_filter_index, :, :]
    #
    # test_image = test_image[0, :, :, :]
    # test_image = np.transpose(test_image, (1, 2, 0))
    # fig1, fig2 = complex_bg.plot_activations(test_image, tgt_l1_act, tgt_l2_act, tgt_filter_index)


    # 2. Extract the neural data we would like to match
    # ---------------------------------------------------------------------
    with open('.//neuro_data//Li2006.pickle', 'rb') as handle:
        data = pickle.load(handle)

    expected_gains = data['contour_len_avg_gain']

    # 2. Create a bank of images - a batch of multiple contour lengths
    # ---------------------------------------------------------------------
    tgt_filter_idx = 10
    tgt_neuron_loc = 27

    # Fragment
    fragment = np.zeros((11, 11, 3))
    fragment[:, (0, 3, 4, 5, 9, 10), :] = 255.0

    contour_len_arr = np.arange(1, 11, 2)

    images_list = []

    for c_len in contour_len_arr:
        # print("Creating image with contour of length %d" % c_len)
        test_image = np.zeros((227, 227, 3))

        start_x, start_y = complex_bg.vertical_contour_generator(
            fragment.shape[0],
            bw_tile_spacing=0,
            cont_len=c_len,
            cont_start_loc=tgt_neuron_loc * 4
        )

        test_image = complex_bg.tile_image(
            test_image,
            fragment,
            (start_x, start_y),
            rotate=False,
            gaussian_smoothing=False
        )

        # Image preprocessing
        test_image = test_image / 255.0  # Bring test_image back to the [0, 1] range.
        test_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format
        images_list.append(test_image)

    images_arr = np.stack(images_list, axis=0)

    # # For sanity check the generated images:
    # for image_idx in range(images_arr.shape[0]):
    #     print("display image at index %d" % image_idx)
    #     display_img = np.transpose(images_arr[image_idx, :, :, :], (1, 2, 0))
    #     plt.figure()
    #     plt.imshow(display_img)

    # 3. Build a cost function that minimizes the distance between the expected gain and actual gain
    # ----------------------------------------------------------------------------------------------
    l1_output_cb = alex_net_cont_int_model.layers[1].output
    l2_output_cb = alex_net_cont_int_model.layers[2].output
    input_cb = alex_net_cont_int_model.input

    l1_activations_cb = complex_bg.get_activation_cb(alex_net_cont_int_model, 1)
    l2_activations_cb = complex_bg.get_activation_cb(alex_net_cont_int_model, 2)

    # Mean Square Error Loss
    current_gain = l2_output_cb[:, tgt_filter_idx, tgt_neuron_loc, tgt_neuron_loc] / \
        (l1_output_cb[:, tgt_filter_idx, tgt_neuron_loc, tgt_neuron_loc] + 1e-5)
    loss = (expected_gains - current_gain) ** 2 / expected_gains.shape[0]

    weights_cb = alex_net_cont_int_model.layers[2].raw_kernel
    bias_cb = alex_net_cont_int_model.layers[2].bias

    # Gradients of weights and bias wrt to the loss function
    grads = K.gradients(loss, [weights_cb, bias_cb])
    grads = [gradient / (K.sqrt(K.mean(K.square(gradient))) + 1e-5) for gradient in grads]
    #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  # Normalize gradients, prevents too large or small gradients

    iterate = K.function([input_cb], [loss, grads[0], grads[1]])

    step = 0.0025
    old_loss = 10000000

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel('Loss')

    for r_idx in range(200):
        loss_value, grad_w, grad_b = iterate([images_arr])
        grad_value = [grad_w, grad_b]
        print("%d: loss %s" % (r_idx, loss_value))

        weights, bias = alex_net_cont_int_model.layers[2].get_weights()

        if old_loss < loss_value.mean():
            step /= 2.0
            print("Lowering step value to %0.4f" % step)
        else:
            new_weights = weights - grad_value[0] * step
            new_bias = bias - grad_value[1] * step

            # Print the new activations - only debug
            l1_act = np.array(l1_activations_cb([images_arr, 0]))
            l1_act = np.squeeze(np.array(l1_act), axis=0)
            l2_act = np.array(l2_activations_cb([images_arr, 0]))
            l2_act = np.squeeze(np.array(l2_act), axis=0)
            print("Contour Enhancement Gain %s" % (l2_act[:, 10, 27, 27] / l1_act[:, 10, 27, 27]))

            # f = plt.figure()
            # ax = f.add_subplot(2, 1, 1)
            # ax.imshow(grad_w[10, :, :])
            # ax2 = f.add_subplot(2, 1, 2)
            # ax2.imshow(new_weights[10, :, :])
            # plt.show()
            # raw_input("Next")

            alex_net_cont_int_model.layers[2].set_weights([new_weights, new_bias])

        old_loss = loss_value.mean()

        ax.plot(r_idx, loss_value.mean(), marker='.', color='blue')

    plt.figure()
    weights, bias = alex_net_cont_int_model.layers[2].get_weights()

    plt.imshow(weights[10, :, :])
    plt.title("Best weights, with bias=%0.4f" % bias[10])

    plt.figure()
    plt.plot(expected_gains, label='From Ref')
    plt.plot(l2_act[:, 10, 27, 27] / l1_act[:, 10, 27, 27], label='model')
    plt.legend()
