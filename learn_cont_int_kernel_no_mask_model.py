# -------------------------------------------------------------------------------------------------
#  Learn Lateral weights based on Li2006 results for models in no_mask_models.py
#
#  Compared with the previous training method, this is more standard and uses python generators
#  and the regular training functions of Keras (fit generator)
#
# Author: Salman Khan
# Date  : 04/05/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K

import contour_integration_models.alex_net.no_mask_models as cont_int_models
import alex_net_utils
import image_generator_linear as contour_image_generator
import li_2006_routines
import gabor_fits

reload(cont_int_models)
reload(alex_net_utils)
reload(contour_image_generator)
reload(li_2006_routines)
reload(gabor_fits)


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

    # # -----------------------------------------------------------------------------------
    # # Vertical Contour Enhancement
    # # -----------------------------------------------------------------------------------
    # tgt_filter_idx = 10
    #
    # print("{0} Vertical Contour Enhancement (Feature kernel @ idx {1}) {0} ".format(
    #     '#' * 25, tgt_filter_idx))
    #
    # # Build the contour integration model
    # # -----------------------------------
    # print("Building Model ...")
    # contour_integration_model = cont_int_models.build_contour_integration_training_model(
    #     rf_size=25,
    #     tgt_filt_idx=tgt_filter_idx,
    # )
    #
    # # Define callback functions to get activations of L1 convolutional layer &
    # # L2 contour integration layer
    # l1_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    # l2_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)
    #
    # # Store the start weights & bias for comparison later
    # start_weights, start_bias = contour_integration_model.layers[2].get_weights()
    #
    # # Build the contour image generator
    # # ----------------------------------
    # print("Building Train Image Generator ...")
    # feature_extract_kernels = K.eval(contour_integration_model.layers[1].weights[0])
    # feature_extract_kernel = feature_extract_kernels[:, :, :, tgt_filter_idx]
    #
    # fragment = np.zeros((11, 11, 3))
    # fragment[:, (0, 3, 4, 5, 9, 10), :] = 255.0
    #
    # train_image_generator = contour_image_generator.ContourImageGenerator(
    #     tgt_filt=feature_extract_kernel,
    #     tgt_filt_idx=tgt_filter_idx,
    #     contour_tile_loc_cb=alex_net_utils.vertical_contour_generator,
    #     row_offset=0,
    #     frag=fragment
    # )
    #
    # s = train_image_generator.generate(images_type='both', batch_size=2)
    #
    # # Train the model
    # # ---------------
    # print("Starting Training ...")
    #
    # contour_integration_model.compile(optimizer='Adam', loss='mse')
    #
    # history = contour_integration_model.fit_generator(
    #     generator=s,
    #     steps_per_epoch=1,
    #     epochs=1000,
    #     verbose=2,
    #     # max_q_size=1,
    #     # workers=1,
    # )
    #
    # plt.figure()
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    #
    # # Plot the learnt weights
    # # -----------------------
    # plot_optimized_weights(
    #     contour_integration_model,
    #     tgt_filter_idx,
    #     start_weights,
    #     start_bias)
    #
    # # Compare Results with Neurophysiological Data
    # # --------------------------------------------
    # tgt_neuron_location = contour_integration_model.layers[2].output_shape[2:]
    # tgt_neuron_location = [loc >> 1 for loc in tgt_neuron_location]
    # print("Comparing Model & Neurophysiological results for neuron at location: ", tgt_neuron_location)
    #
    # # Plot Gain vs Contour Length after Optimization
    # li_2006_routines.main_contour_length_routine(
    #     fragment,
    #     l1_activations_cb,
    #     l2_activations_cb,
    #     alex_net_utils.vertical_contour_generator,
    #     tgt_filter_idx,
    #     smoothing=True,
    #     row_offset=0,
    #     n_runs=100,
    #     tgt_neuron_loc=tgt_neuron_location
    # )
    #
    # # Plot Gain vs Contour Spacing after Optimization
    # li_2006_routines.main_contour_spacing_routine(
    #     fragment,
    #     l1_activations_cb,
    #     l2_activations_cb,
    #     alex_net_utils.vertical_contour_generator,
    #     tgt_filter_idx,
    #     smoothing=True,
    #     row_offset=0,
    #     n_runs=100,
    #     tgt_neuron_loc=tgt_neuron_location
    # )
    #
    # alex_net_utils.plot_l1_and_l2_kernel_and_contour_fragment(
    #     contour_integration_model,
    #     tgt_filter_idx,
    #     fragment
    # )

    # -----------------------------------------------------------------------------------
    # Horizontal Contour Enhancement
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 5

    print("{0} Horizontal Contour Enhancement (Feature kernel @ idx {1}) {0} ".format(
        '#' * 25, tgt_filter_idx))

    # Build the contour integration model
    # -----------------------------------
    print("Building Model ...")
    contour_integration_model = cont_int_models.build_contour_integration_training_model(
        rf_size=25,
        tgt_filt_idx=tgt_filter_idx,
    )

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
    fragment[0:6, :, :] = 255.0

    train_image_generator = contour_image_generator.ContourImageGenerator(
        tgt_filt=feature_extract_kernel,
        tgt_filt_idx=tgt_filter_idx,
        contour_tile_loc_cb=alex_net_utils.horizontal_contour_generator,
        row_offset=0,
        frag=fragment
    )

    s = train_image_generator.generate(images_type='both', batch_size=2)

    # Train the model
    # ---------------
    print("Starting Training ...")

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

    # Plot the learnt weights
    # -----------------------
    plot_optimized_weights(
        contour_integration_model,
        tgt_filter_idx,
        start_weights,
        start_bias)

    # Compare Results with Neurophysiological Data
    # --------------------------------------------
    tgt_neuron_location = contour_integration_model.layers[2].output_shape[2:]
    tgt_neuron_location = [loc >> 1 for loc in tgt_neuron_location]
    print("Comparing Model & Neurophysiological results for neuron at location {}".format(
        tgt_neuron_location))

    # Plot Gain vs Contour Length after Optimization
    li_2006_routines.main_contour_length_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.horizontal_contour_generator,
        tgt_filter_idx,
        smoothing=True,
        row_offset=0,
        n_runs=100,
        tgt_neuron_loc=tgt_neuron_location
    )

    # Plot Gain vs Contour Spacing after Optimization
    li_2006_routines.main_contour_spacing_routine(
        fragment,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.horizontal_contour_generator,
        tgt_filter_idx,
        smoothing=True,
        row_offset=0,
        n_runs=100,
        tgt_neuron_loc=tgt_neuron_location
    )

    alex_net_utils.plot_l1_and_l2_kernel_and_contour_fragment(
        contour_integration_model,
        tgt_filter_idx,
        fragment
    )

    # # ---------------------------------------------------------------------------------
    # #  Diagonal Contour Enhancement
    # # ---------------------------------------------------------------------------------
    # tgt_filter_idx = 54
    #
    # print("{0} Diagonal Contour Enhancement (Feature kernel @ idx {1}) {0} ".format(
    #     '#' * 25,
    #     tgt_filter_idx
    # ))
    #
    # fragment = np.zeros((11, 11, 3))
    # fragment[0:6, :, :] = 255.0
    #
    # # Build the contour integration model
    # # -----------------------------------
    # print("Building Model ...")
    # contour_integration_model = cont_int_models.build_contour_integration_training_model(
    #     rf_size=25,
    #     tgt_filt_idx=tgt_filter_idx,
    #     stride_length=(4, 4)
    # )
    #
    # # Define callback functions to get activations of L1 convolutional layer &
    # # L2 contour integration layer
    # l1_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    # l2_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)
    #
    # # Store the start weights & bias for comparison later
    # start_weights, start_bias = contour_integration_model.layers[2].get_weights()
    #
    # # Build the contour image generator
    # # ----------------------------------
    # print("Building Train Image Generator ...")
    # feature_extract_kernels = K.eval(contour_integration_model.layers[1].weights[0])
    # feature_extract_kernel = feature_extract_kernels[:, :, :, tgt_filter_idx]
    #
    # orientation, tile_row_offset = gabor_fits.get_l1_filter_orientation_and_offset(
    #     feature_extract_kernel,
    #     tgt_filter_idx,
    #     show_plots=False
    # )
    # # tile_row_offset = -tile_row_offset
    # print("Orientation of Filter = %0.4f and Row offset %d" % (orientation, tile_row_offset))
    #
    # fragment = np.copy(feature_extract_kernel)
    # fragment = fragment.sum(axis=2)  # collapse all channels
    # fragment[fragment > 0] = 1
    # fragment[fragment <= 0] = 0
    # fragment *= 255
    # fragment = np.repeat(fragment[:, :, np.newaxis], 3, axis=2)
    #
    # train_image_generator = contour_image_generator.ContourImageGenerator(
    #     tgt_filt=feature_extract_kernel,
    #     tgt_filt_idx=tgt_filter_idx,
    #     contour_tile_loc_cb=alex_net_utils.diagonal_contour_generator,
    #     row_offset=tile_row_offset,
    #     frag=fragment
    # )
    #
    # s = train_image_generator.generate(images_type='both', batch_size=2)
    #
    # # Train the model
    # # ---------------
    # print("Starting Training ...")
    #
    # contour_integration_model.compile(optimizer='Adam', loss='mse')
    #
    # history = contour_integration_model.fit_generator(
    #     generator=s,
    #     steps_per_epoch=1,
    #     epochs=1000,
    #     verbose=2,
    #     # max_q_size=1,
    #     # workers=1,
    # )
    #
    # plt.figure()
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    #
    # # Plot the learnt weights
    # # -----------------------
    # plot_optimized_weights(
    #     contour_integration_model,
    #     tgt_filter_idx,
    #     start_weights,
    #     start_bias)
    #
    # # Compare Results with Neurophysiological Data
    # # --------------------------------------------
    # tgt_neuron_location = contour_integration_model.layers[2].output_shape[2:]
    # tgt_neuron_location = [loc >> 1 for loc in tgt_neuron_location]
    # print("Comparing Model & Neurophysiological results for neuron at location {}".format(
    #     tgt_neuron_location))
    #
    # # Plot Gain vs Contour Length after Optimization
    # li_2006_routines.main_contour_length_routine(
    #     fragment,
    #     l1_activations_cb,
    #     l2_activations_cb,
    #     alex_net_utils.diagonal_contour_generator,
    #     tgt_filter_idx,
    #     smoothing=True,
    #     row_offset=tile_row_offset,
    #     n_runs=100,
    #     tgt_neuron_loc=tgt_neuron_location
    # )
    #
    # # Plot Gain vs Contour Spacing after Optimization
    # li_2006_routines.main_contour_spacing_routine(
    #     fragment,
    #     l1_activations_cb,
    #     l2_activations_cb,
    #     alex_net_utils.diagonal_contour_generator,
    #     tgt_filter_idx,
    #     smoothing=True,
    #     row_offset=tile_row_offset,
    #     n_runs=100,
    #     tgt_neuron_loc=tgt_neuron_location
    # )
    #
    # alex_net_utils.plot_l1_and_l2_kernel_and_contour_fragment(
    #     contour_integration_model,
    #     tgt_filter_idx,
    #     fragment
    # )
