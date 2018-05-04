# -------------------------------------------------------------------------------------------------
#  Create a contour image generator
#
# Author: Salman Khan
# Date  : 01/02/18
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras.backend as K

import alex_net_utils
from contour_integration_models import alex_net_cont_int_models as cont_int_models
import alex_net_hyper_param_search_multiplicative as mult_param_opt
import alex_net_cont_int_complex_bg as complex_bg

reload(alex_net_utils)
reload(cont_int_models)
reload(mult_param_opt)
reload(complex_bg)


class ContourImageGenerator(object):

    def __init__(self, tgt_filt, tgt_filt_idx, contour_tile_loc_cb, row_offset, frag=None):
        """
        Creates a python generator that continuously generates train/test images
        for the specified tgt filter. The generator is as expected by Keras
        fit generator function.

        :param tgt_filt:
        :param tgt_filt_idx:
        :param contour_tile_loc_cb:
        :param frag:
        """
        self.tgt_filt_idx = tgt_filt_idx
        self.tgt_filt = tgt_filt

        if frag is None:
            self.frag = tgt_filt
        else:
            self.frag = frag
        self.frag_len = self.frag.shape[0]

        self.image_len = 227  # Alexnet expects input of size (227, 227, 3)
        self.tgt_neuron_rf_start = 27 * 4  # Start with the neuron at the center of the activation

        with open('.//neuro_data//Li2006.pickle', 'rb') as handle:
            data = pickle.load(handle)

        # Rather than calculate the contour fragment and background tiles locations everytime
        # Calculate them once and store them to be index later

        # Contour Length experiment variables
        self.c_len_expected_gains = data['contour_len_avg_gain']
        self.c_len_arr = data['contour_len_avg_len']

        self.c_len_cont_tile_loc = []
        for c_len in self.c_len_arr:
            tile_locations = contour_tile_loc_cb(
                frag_len=self.frag_len,
                bw_tile_spacing=0,
                cont_len=c_len,
                cont_start_loc=self.tgt_neuron_rf_start,
                row_offset=row_offset
            )

            self.c_len_cont_tile_loc.append(tile_locations)

        # Contour spacing experiment variables
        self.c_spacing_expected_gains = data['contour_separation_avg_gain']
        self.c_spacing_arr = np.floor(data['contour_separation_avg_rcd'] * self.frag_len) - self.frag_len

        self.c_spacing_cont_tile_loc = []
        self.bg_tile_loc = []
        for c_spacing in self.c_spacing_arr:

            c_spacing = int(c_spacing)

            tile_locations = contour_tile_loc_cb(
                frag_len=self.frag_len,
                bw_tile_spacing=c_spacing,
                cont_len=7,
                cont_start_loc=self.tgt_neuron_rf_start,
                row_offset=row_offset
            )
            self.c_spacing_cont_tile_loc.append(tile_locations)

            bg_tile_locations = alex_net_utils.get_background_tiles_locations(
                self.frag_len,
                self.image_len,
                row_offset,
                c_spacing,
                self.tgt_neuron_rf_start
            )
            self.bg_tile_loc.append(bg_tile_locations)

    @staticmethod
    def show_image(img, axis=None):
        """
        Generated images are in the format expected as inputs by the model. The model was
        first designed with a theano backend, and expects channel first. This function reverses
        the transformation, so that the image can be displayed in the format expected by matplotlib

        :param img:
        :param axis:
        :return:
        """
        display_img = np.transpose(img, (1, 2, 0))
        if axis is None:
            _, axis = plt.subplots()

        axis.imshow(display_img)

    def show_image_batch(self, img_stack, img_labels):
        """
        Similar to function show_image but for the whole batch of generated images.
        Additionally expects img_labels

        :param img_stack:
        :param img_labels:
        :return:
        """
        for i_idx in range(img_stack.shape[0]):
            display_img = img_stack[i_idx, :, :, :]
            self.show_image(display_img)
            plt.title("Expected Gain %0.4f" % img_labels[i_idx])

    def generate(self, images_type='both', batch_size=10):
        """
        This is the main function of this class.

        :param batch_size:
        :param images_type:
        :return:
        """

        images_type = images_type.lower()
        allowed_types = ['both', 'length', 'spacing']

        if images_type not in allowed_types:
            raise Exception("Invalid Image Type")

        if images_type == 'both':
            n_c_len_images = batch_size >> 1
            n_c_spacing_images = batch_size - n_c_len_images
        elif images_type == 'length':
            n_c_len_images = batch_size
            n_c_spacing_images = 0
        else:
            n_c_len_images = 0
            n_c_spacing_images = batch_size

        print("Generated batches will have %d contour length and %d contour spacing images"
              % (n_c_len_images, n_c_spacing_images))

        while True:

            image_arr = []
            label_arr = []

            if images_type is not 'spacing':
                for i in range(n_c_len_images):

                    # select_idx = i
                    select_idx = np.random.randint(0, len(self.c_len_arr))

                    test_image = np.zeros((227, 227, 3))

                    test_image = alex_net_utils.tile_image(
                        test_image,
                        self.frag,
                        self.bg_tile_loc[0],
                        rotate=True,
                        gaussian_smoothing=True
                    )

                    test_image = alex_net_utils.tile_image(
                        test_image,
                        self.frag,
                        self.c_len_cont_tile_loc[select_idx],
                        rotate=False,
                        gaussian_smoothing=True,
                    )

                    # Image preprocessing
                    test_image = test_image / 255.0  # Bring test_image back to the [0, 1] range.
                    test_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format

                    image_arr.append(test_image)
                    label_arr.append(self.c_len_expected_gains[select_idx])

            if images_type is not 'length':
                for i in range(n_c_spacing_images):

                    # select_idx = i
                    select_idx = np.random.randint(0, len(self.c_spacing_arr))

                    test_image = np.zeros((227, 227, 3))

                    test_image = alex_net_utils.tile_image(
                        test_image,
                        self.frag,
                        self.bg_tile_loc[select_idx],
                        rotate=True,
                        gaussian_smoothing=True
                    )

                    test_image = alex_net_utils.tile_image(
                        test_image,
                        self.frag,
                        self.c_spacing_cont_tile_loc[select_idx],
                        rotate=False,
                        gaussian_smoothing=True,
                    )

                    # Image preprocessing
                    test_image = test_image / 255.0  # Bring test_image back to the [0, 1] range.
                    test_image = np.transpose(test_image, (2, 0, 1))  # Theano back-end expects channel first format

                    image_arr.append(test_image)
                    label_arr.append(self.c_spacing_expected_gains[select_idx])

            image_arr = np.stack(image_arr, axis=0)
            label_arr = np.reshape(np.array(label_arr), (len(label_arr), 1))

            yield image_arr, label_arr


def optimize_contour_enhancement_weights(
        model, tgt_filt_idx, frag, contour_generator_cb,
        n_runs=1000, learning_rate=0.00025, offset=0, optimize_type='both', axis=None):
    """

    This function is similar to the function with the same name in
    alex_net_hyper_param_search_multiplicative.py but uses the new contour image generator

    :param model:
    :param tgt_filt_idx:
    :param frag:
    :param contour_generator_cb:
    :param n_runs:
    :param learning_rate:
    :param offset:
    :param optimize_type:
    :param axis:
    :return:
    """

    # Validate input parameters
    valid_optimize_type = ['length', 'spacing', 'both']
    if optimize_type.lower() not in valid_optimize_type:
        raise Exception("Invalid optimization type specified. Valid = [length, spacing, or both(Default)")
    optimize_type = optimize_type.lower()

    # Some Initialization
    tgt_n_loc = 27  # neuron looking @ center of RF

    # 1. Setup the Training image generator
    # -------------------------------------
    image_generator = ContourImageGenerator(
        tgt_filt=feature_extract_kernel,
        tgt_filt_idx=tgt_filt_idx,
        contour_tile_loc_cb=contour_generator_cb,
        row_offset=offset,
        frag=frag
    )

    generator = image_generator.generate(images_type='both')

    images, expected_gains = generator.next()

    # 2. Setup the optimization problem
    # ---------------------------------
    l1_output_cb = model.layers[1].output
    l2_output_cb = model.layers[2].output
    input_cb = model.input

    # Callbacks for the weights (learnable parameters)
    w_cb = model.layers[2].raw_kernel
    b_cb = model.layers[2].bias

    current_gains = l2_output_cb[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] / \
        (l1_output_cb[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] + 1e-8)
    current_gains = K.expand_dims(current_gains, axis=-1)

    y_true = K.placeholder(shape=(None, 1))

    loss = K.mean((y_true - current_gains) ** 2)

    # Gradients of weights and bias wrt to the loss function
    grads = K.gradients(loss, [w_cb, b_cb])
    grads = [gradient / (K.sqrt(K.mean(K.square(gradient))) + 1e-8) for gradient in grads]

    iterate = K.function([input_cb, y_true], [loss, grads[0], grads[1], l1_output_cb, l2_output_cb])

    # # Train the model using Adam
    # # -------------------------------
    # print("Starting Training")
    # opt = Adam()
    #
    # params = [w_cb, b_cb]
    #
    # updates = opt.get_updates(params, [], loss)
    #
    # train = K.function([input_cb, y_true], [loss], updates=updates)
    #
    # for epoch in range(n_runs):
    #     images, expected_gains = generator.next()
    #     loss = train([images, expected_gains])
    #     print("Epoch: {}, Loss: {}".format(epoch, np.mean(loss)))


    # 3. Training Loop
    # ----------------
    old_loss = 10000000
    losses = []
    # ADAM Optimization starting parameters
    m_w = 0
    v_w = 0

    m_b = 0
    v_b = 0

    for run_idx in range(n_runs):

        images, expected_gains = generator.next()

        loss_value, grad_w, grad_b, l1_out, l2_out = iterate([images, expected_gains])
        print("%d: loss %s" % (run_idx, loss_value.mean()))

        w, b = model.layers[2].get_weights()

        # ADAM weights update
        if loss_value.mean() > old_loss:
            # step /= 2.0
            # print("Lowering step value to %f" % step)
            pass
        else:
            m_w = 0.9 * m_w + (1 - 0.9) * grad_w
            v_w = 0.999 * v_w + (1 - 0.999) * grad_w ** 2

            new_w = w - learning_rate * m_w / (np.sqrt(v_w) + 1e-8)

            m_b = 0.9 * m_b + (1 - 0.9) * grad_b
            v_b = 0.999 * v_b + (1 - 0.999) * grad_b ** 2

            new_b = b - learning_rate * m_b / (np.sqrt(v_b) + 1e-8)

            # Print Contour Enhancement Gains
            model_gains = (l2_out[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] /
                           (l1_out[:, tgt_filt_idx, tgt_n_loc, tgt_n_loc] + 1e-8))

            print("Model Gain: ", np.around(model_gains, 3))
            print("Expected  : ", np.around(expected_gains.T, 3))

            model.layers[2].set_weights([new_w, new_b])

        old_loss = loss_value.mean()
        losses.append(loss_value.mean())

    # At the end of simulation plot loss vs iteration
    if axis is None:
        f, axis = plt.subplots()
    axis.plot(range(n_runs), losses, label='learning rate = %0.8f' % learning_rate)
    font_size = 20
    axis.set_xlabel("Iteration", fontsize=font_size)
    axis.set_ylabel("Loss", fontsize=font_size)
    axis.tick_params(axis='x', labelsize=font_size)
    axis.tick_params(axis='y', labelsize=font_size)


if __name__ == "__main__":
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # --------------------------
    tgt_filter_idx = 10

    # Build Contour Integration Model
    # -------------------------------
    print("Building Contour Integration Model...")

    # Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        n=25,
        activation='relu'
    )
    # contour_integration_model.summary()

    # Define callback functions to get activations of L1 convolutional layer &
    # L2 contour integration layer
    l1_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    l2_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)

    # Store the start weights & bias for comparison later
    start_weights, start_bias = contour_integration_model.layers[2].get_weights()

    # Build the Contour Image Generator
    # ---------------------------------
    print("Building Train Image Generator ...")
    feature_extract_kernels = K.eval(contour_integration_model.layers[1].weights[0])
    feature_extract_kernel = feature_extract_kernels[:, :, :, tgt_filter_idx]

    fragment = np.zeros((11, 11, 3))
    fragment[:, (0, 3, 4, 5, 9, 10), :] = 255.0

    # train_image_generator = ContourImageGenerator(
    #     tgt_filt=feature_extract_kernel,
    #     tgt_filt_idx=tgt_filter_idx,
    #     contour_tile_loc_cb=alex_net_utils.vertical_contour_generator,
    #     row_offset=0,
    #     frag=fragment
    # )

    # # Test the feature extract Kernel
    # s = train_image_generator.generate(images_type='both')
    # X, y = s.next()
    # train_image_generator.show_image_batch(X, y)

    # Train the model
    # ---------------
    optimize_contour_enhancement_weights(
        contour_integration_model,
        tgt_filter_idx,
        fragment,
        alex_net_utils.vertical_contour_generator,
        n_runs=500,
        offset=0,
        optimize_type='both',
        learning_rate=0.00025
    )

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
