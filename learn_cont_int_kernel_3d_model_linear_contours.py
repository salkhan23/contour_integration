# -------------------------------------------------------------------------------------------------
#  Train the 3d model on linear contours to make sure it is working correctly.
#
# Author: Salman Khan
# Date  : 08/05/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.preprocessing.image import load_img

import image_generator_linear
import alex_net_utils
import base_models.alex_net as alex_net
import contour_integration_models.alex_net.model_3d as contour_integration_model_3d
import contour_integration_models.alex_net.no_mask_models as contour_integration_model_2d
import li_2006_routines

reload(image_generator_linear)
reload(alex_net_utils)
reload(alex_net)
reload(contour_integration_model_2d)
reload(contour_integration_model_3d)
reload(li_2006_routines)


def plot_contour_integration_weights_in_channels(weights, out_chan_idx, margin=1, axis=None):
    """
    Plot all input channels weights connected to a particular output channel
    All channels are plotted individually in a tiled image.

    :param weights:
    :param out_chan_idx:
    :param margin:
    :param axis:

    :return:
    """
    r, c, in_ch, out_ch = weights.shape

    n = np.int(np.round(np.sqrt(in_ch)))  # Single dimension of tiled image

    width = (n * r) + ((n - 1) * margin)
    height = (n * c) + ((n - 1) * margin)

    tiled_img = np.zeros((width, height))

    # Fill in in composite image with the filters
    for r_idx in range(n):
        for c_idx in range(n):

            in_chan_idx = (r_idx * n) + c_idx

            if in_chan_idx >= in_ch:
                break

            # print("Processing filter %d" % in_chan_idx)

            tiled_img[
                (r + margin) * r_idx: (r + margin) * r_idx + r,
                (c + margin) * c_idx: (c + margin) * c_idx + c,
            ] = weights[:, :, in_chan_idx, out_chan_idx]

    if axis is None:
        f, axis = plt.subplots()
    else:
        f = plt.gcf()

    cax = axis.imshow(tiled_img, cmap='seismic', vmax=np.max(abs(tiled_img)), vmin=-np.max(abs(tiled_img)))
    f.colorbar(cax, orientation='horizontal', ax=axis)

    # Put borders between tiles
    for r_idx in range(n):
        margin_lines = np.arange((r_idx * (r + margin)) + r, (r_idx * (r + margin)) + (r + margin))
        for line in margin_lines:
            axis.axvline(line, color='k')
            axis.axhline(line, color='k')


def plot_contour_integration_weights_out_channels(weights, in_chan_idx, margin=1, axis=None):
    """
    Plot all output channels weights connected to a particular input channel
    All channels are plotted individually in a tiled image.

    :param weights:
    :param in_chan_idx:
    :param margin:
    :param axis:

    :return:
    """
    r, c, in_ch, out_ch = weights.shape

    n = np.int(np.round(np.sqrt(in_ch)))  # Single dimension of tiled image

    width = (n * r) + ((n - 1) * margin)
    height = (n * c) + ((n - 1) * margin)

    tiled_img = np.zeros((width, height))

    # Fill in in composite image with the filters
    for r_idx in range(n):
        for c_idx in range(n):

            out_chan_idx = (r_idx * n) + c_idx

            if out_chan_idx >= in_ch:
                break

            print("Processing filter %d" % out_chan_idx)

            tiled_img[
                (r + margin) * r_idx: (r + margin) * r_idx + r,
                (c + margin) * c_idx: (c + margin) * c_idx + c,
            ] = weights[:, :, in_chan_idx, out_chan_idx]

    if axis is None:
        f, axis = plt.subplots()
    else:
        f = plt.gcf()

    cax = axis.imshow(tiled_img, cmap='seismic', vmax=np.max(abs(tiled_img)), vmin=-np.max(abs(tiled_img)))
    f.colorbar(cax, orientation='horizontal', ax=axis)


if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    image_size = (227, 227, 3)

    tgt_neuron_rf_start = 108
    gaussian_smoothing = False

    # # ===================================================================================
    # # Vertical Contours
    # # ===================================================================================
    # tgt_filter_idx = 10
    #
    # # Contour Integration Model
    # # -------------------------
    # cont_int_model = contour_integration_model_3d.build_contour_integration_model(
    #     tgt_filter_idx, rf_size=25)
    #
    # # cont_int_model = contour_integration_model_2d.build_contour_integration_training_model(
    # #     rf_size=25, tgt_filt_idx=tgt_filter_idx)
    # # cont_int_model.compile(optimizer='Adam', loss='mse')
    #
    # feat_extract_kernels = K.eval(cont_int_model.layers[1].weights[0])
    # tgt_filter = feat_extract_kernels[:, :, :, tgt_filter_idx]
    #
    # # Store the start weights & bias for comparison later
    # start_weights, start_bias = cont_int_model.layers[2].get_weights()
    #
    # # Fragment
    # # --------
    # frag = np.zeros((11, 11, 3))  # Dimensions of the L1 convolutional layer of alexnet
    # frag[:, (0, 3, 4, 5, 9, 10), :] = 255
    #
    # # # Show the contour fragment
    # # plt.figure()
    # # plt.imshow(frag)
    # # plt.title("Fragment")
    #
    # # Create a Linear contour image generator
    # # ---------------------------------------
    # gen_class = image_generator_linear.ContourImageGenerator(
    #     tgt_filt=tgt_filter,
    #     tgt_filt_idx=tgt_filter_idx,
    #     contour_tile_loc_cb=alex_net_utils.vertical_contour_generator,
    #     row_offset=0,
    #     frag=frag
    # )
    #
    # train_image_generator = gen_class.generate(images_type='both', batch_size=10)
    #
    # # X, y = train_image_generator.next()
    # # gen_class.show_image_batch(X, y)
    #
    # # Train the model
    # # ---------------
    # history = cont_int_model.fit_generator(
    #     generator=train_image_generator,
    #     steps_per_epoch=1,
    #     epochs=100,
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
    # # Plot the learned weights
    # # -------------------------
    # fig, ax_arr = plt.subplots(1, 2)
    # plot_contour_integration_weights_in_channels(start_weights, tgt_filter_idx, axis=ax_arr[0])
    #
    # learnt_weights, _ = cont_int_model.layers[2].get_weights()
    # plot_contour_integration_weights_in_channels(learnt_weights, tgt_filter_idx, axis=ax_arr[1])
    # fig.suptitle('Input channel of filter @ {}'.format(tgt_filter_idx))
    #
    # fig, ax_arr = plt.subplots(1, 2)
    # plot_contour_integration_weights_out_channels(start_weights, tgt_filter_idx, axis=ax_arr[0])
    #
    # learnt_weights, _ = cont_int_model.layers[2].get_weights()
    # plot_contour_integration_weights_out_channels(learnt_weights, tgt_filter_idx, axis=ax_arr[1])
    # fig.suptitle('Output channel of filter @ {}'.format(tgt_filter_idx))

    # ===================================================================================
    # Horizontal Contours
    # ===================================================================================
    tgt_filter_idx = 5

    # -----------------------------------------------------------------------------------
    # Contour Integration Model
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(
        tgt_filter_idx, rf_size=25)

    # cont_int_model = contour_integration_model_2d.build_contour_integration_training_model(
    #     rf_size=25, tgt_filt_idx=tgt_filter_idx)
    # cont_int_model.compile(optimizer='Adam', loss='mse')

    feat_extract_kernels = K.eval(cont_int_model.layers[1].weights[0])
    tgt_filter = feat_extract_kernels[:, :, :, tgt_filter_idx]

    l1_activations_cb = alex_net_utils.get_activation_cb(cont_int_model, 1)
    l2_activations_cb = alex_net_utils.get_activation_cb(cont_int_model, 2)

    # Store the start weights & bias for comparison later
    start_weights, _ = cont_int_model.layers[2].get_weights()

    # -----------------------------------------------------------------------------------
    # Fragment
    # -----------------------------------------------------------------------------------
    frag = np.zeros((11, 11, 3))
    frag[0:6, :, :] = 255.0

    # # Show the contour fragment
    # plt.figure()
    # plt.imshow(frag)
    # plt.title("Fragment")

    # -----------------------------------------------------------------------------------
    # Create a Linear contour image generator
    # -----------------------------------------------------------------------------------
    gen_class = image_generator_linear.ContourImageGenerator(
        tgt_filt=tgt_filter,
        tgt_filt_idx=tgt_filter_idx,
        contour_tile_loc_cb=alex_net_utils.horizontal_contour_generator,
        row_offset=0,
        frag=frag
    )

    train_image_generator = gen_class.generate(images_type='both', batch_size=10)

    # X, y = train_image_generator.next()
    # gen_class.show_image_batch(X, y)

    # -----------------------------------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------------------------------
    history = cont_int_model.fit_generator(
        generator=train_image_generator,
        steps_per_epoch=1,
        epochs=100,
        verbose=2,
        # max_q_size=1,
        # workers=1,
    )

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    # ----------------------------------------------------------------------------------
    #  Plot Results
    # ----------------------------------------------------------------------------------
    # 1. PLot all feature maps that feed into output mask at index tgt_filter_idx
    fig, ax_arr = plt.subplots(1, 2)
    plot_contour_integration_weights_in_channels(
        start_weights, tgt_filter_idx, axis=ax_arr[0])

    learnt_weights, _ = cont_int_model.layers[2].get_weights()
    plot_contour_integration_weights_in_channels(
        learnt_weights, tgt_filter_idx, axis=ax_arr[1])
    fig.suptitle('Input channel of filter @ {}'.format(tgt_filter_idx))

    # 2. Plot all output feature maps that receive from input channel at index tgt_filter_idx
    fig, ax_arr = plt.subplots(1, 2)
    plot_contour_integration_weights_out_channels(
        start_weights, tgt_filter_idx, axis=ax_arr[0])

    learnt_weights, _ = cont_int_model.layers[2].get_weights()
    plot_contour_integration_weights_out_channels(
        learnt_weights, tgt_filter_idx, axis=ax_arr[1])
    fig.suptitle('Output channel of filter @ {}'.format(tgt_filter_idx))

    # 3. Compare Results with Neurophysiological Data
    # --------------------------------------------
    tgt_neuron_location = cont_int_model.layers[2].output_shape[2:]
    tgt_neuron_location = [loc >> 1 for loc in tgt_neuron_location]
    print("Comparing Model & Neurophysiological results for neuron at location {}".format(
        tgt_neuron_location))

    # 3a. Plot Gain vs Contour Length after Optimization
    li_2006_routines.main_contour_length_routine(
        frag,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.horizontal_contour_generator,
        tgt_filter_idx,
        smoothing=True,
        row_offset=0,
        n_runs=100,
        tgt_neuron_loc=tgt_neuron_location
    )

    # 3.b Plot Gain vs Contour Spacing after Optimization
    li_2006_routines.main_contour_spacing_routine(
        frag,
        l1_activations_cb,
        l2_activations_cb,
        alex_net_utils.horizontal_contour_generator,
        tgt_filter_idx,
        smoothing=True,
        row_offset=0,
        n_runs=100,
        tgt_neuron_loc=tgt_neuron_location
    )

    #  4. Performance on sample  Li Stimuli
    # ----------------------------------------------------------------------------------
    # 1. Plot the activations of the model for an image with a straight contour
    d = load_img('./data/curved_contours/filter_5/c_len_9/beta_0/orient_92_clen_9_beta_0__0.png')
    d1 = np.array(d)
    alex_net_utils.plot_l1_and_l2_activations(
        d1 / 255.0, l1_activations_cb, l2_activations_cb, tgt_filter_idx)
    plt.title('straight')

    # 2. Plot the activations of the model for an image with a straight contour
    d = load_img('./data/curved_contours/filter_5/c_len_9/beta_15/orient_92_clen_9_beta_15__0.png')
    d1 = np.array(d)
    alex_net_utils.plot_l1_and_l2_activations(
        d1 / 255.0, l1_activations_cb, l2_activations_cb, tgt_filter_idx)
    plt.title('15 degrees')

