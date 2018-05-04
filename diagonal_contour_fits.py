from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K

from contour_integration_models.alex_net import masked_models as cont_int_models
import gabor_fits
import alex_net_utils
import alex_net_hyper_param_search_multiplicative as multiplicative_model
import li_2006_routines

reload(gabor_fits)
reload(cont_int_models)
reload(alex_net_utils)
reload(multiplicative_model)
reload(li_2006_routines)

np.random.seed(10)  # Set the random seed for reproducibility


if __name__ == "__main__":
    plt.ion()
    K.clear_session()

    # 1. Build the model
    # ---------------------------------------------------------------------
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # # Masked Multiplicative Model
    # contour_integration_model = cont_int_models.build_contour_integration_model(
    #     "masked_multiplicative",
    #     "trained_models/AlexNet/alexnet_weights.h5",
    #     weights_type='enhance',
    #     n=25,
    #     activation='relu'
    # )

    # Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        n=25,
        activation='relu'
    )
    # contour_integration_model.summary()

    feat_extract_kernels = K.eval(contour_integration_model.layers[1].weights[0])

    # callbacks to get activations of first feature extracting & contour integration layers
    feat_extract_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 1)
    cont_int_activations_cb = alex_net_utils.get_activation_cb(contour_integration_model, 2)

    # Store the start weights & bias for comparison later
    start_weights, start_bias = contour_integration_model.layers[2].get_weights()

    # Feature extracting kernel to target
    # ------------------------------------------------------------------------
    tgt_feat_extract_kernel_idx = 54

    tgt_filter_idx = feat_extract_kernels[:, :, :, tgt_feat_extract_kernel_idx]
    tgt_filter_len = tgt_filter_idx.shape[0]

    tgt_neuron_loc = (27, 27)
    target_neuron_rf_start = tgt_neuron_loc[0] * 4  # Visual RF start of center L2 neuron
    fragment_spacing = 0

    # Define the fragment
    fragment = np.copy(tgt_filter_idx)
    fragment = fragment.sum(axis=2)
    fragment[fragment > 0] = 1
    fragment[fragment <= 0] = 0
    fragment = fragment * 255
    fragment = np.repeat(fragment[:, :, np.newaxis], 3, axis=2)

    theta, offset = gabor_fits.get_l1_filter_orientation_and_offset(
        tgt_filter_idx, tgt_feat_extract_kernel_idx, show_plots=False)

    # -------------------------------------------------------------------------
    # 2. Plot parts of the visual fields seen by contour integration neurons and
    # its unmasked neighbors
    # -------------------------------------------------------------------------
    # The test image
    test_image = np.ones((227, 227, 3)) * 128

    start_x_arr, start_y_arr = alex_net_utils.diagonal_contour_generator(
        tgt_filter_len,
        offset,
        fragment_spacing,
        9,
        target_neuron_rf_start,
    )

    test_image = alex_net_utils.tile_image(
        test_image,
        fragment,
        (start_x_arr, start_y_arr),
        rotate=False,
        gaussian_smoothing=False
    )

    # Plot the image
    test_image = test_image / 255.0
    plt.figure()
    plt.imshow(test_image)
    plt.title("Input image")

    l2_masks = K.eval(contour_integration_model.layers[2].mask)
    tgt_l2_masks = l2_masks[tgt_feat_extract_kernel_idx, :, :]

    alex_net_utils.plot_l2_visual_field(tgt_neuron_loc, tgt_l2_masks, test_image)

    # -------------------------------------------------------------------------
    # 3. Find the best fit contour integration kernel
    # -------------------------------------------------------------------------
    # # learning_rate_array = [0.0025, 0.001, 0.00025, 0.0001, 0.00005]
    learning_rate_array = [0.0025]

    fig, ax = plt.subplots()

    for learning_rate in learning_rate_array:

        print("############ Processing Learning Rate %0.8f ###########" % learning_rate)

        contour_integration_model.layers[2].set_weights((start_weights, start_bias))

        images = multiplicative_model.optimize_contour_enhancement_layer_weights(
            contour_integration_model,
            tgt_feat_extract_kernel_idx,
            fragment,
            alex_net_utils.diagonal_contour_generator,
            n_runs=1000,
            offset=offset,
            optimize_type='both',
            learning_rate=learning_rate,
            axis=ax
        )
        ax.legend()

        multiplicative_model.plot_optimized_weights(
            contour_integration_model,
            tgt_feat_extract_kernel_idx,
            start_weights,
            start_bias
        )
        fig = plt.gcf()
        fig.suptitle("learning rate = %f" % learning_rate)

        # Plot Gain vs Contour Length after Optimization
        li_2006_routines.main_contour_length_routine(
            fragment,
            feat_extract_activations_cb,
            cont_int_activations_cb,
            alex_net_utils.diagonal_contour_generator,
            tgt_feat_extract_kernel_idx,
            smoothing=True,
            row_offset=offset,
            n_runs=100,
        )

        # Plot Gain vs Contour Spacing after Optimization
        li_2006_routines.main_contour_spacing_routine(
            fragment,
            feat_extract_activations_cb,
            cont_int_activations_cb,
            alex_net_utils.diagonal_contour_generator,
            tgt_feat_extract_kernel_idx,
            smoothing=True,
            row_offset=offset,
            n_runs=100
        )
