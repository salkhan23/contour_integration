# -------------------------------------------------------------------------------------------------
#  Visualize the Complete enhancement done by the model trained with multiple contour
#
# Author: Salman Khan
# Date  : 13/06/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

import keras

import contour_integration_models.alex_net.model_3d_all_kernels as model_3d_all_kernels
import contour_integration_models.alex_net.model_3d as model_3d_individually_trained
import image_generator_curve
import gabor_fits
import alex_net_utils

reload(model_3d_all_kernels)
reload(model_3d_individually_trained)
reload(image_generator_curve)
reload(gabor_fits)
reload(alex_net_utils)


def plot_max_contour_enhancement(img, input_cb, cont_int_cb):
    """
    Plot the maximum contour enhancement (Across all feature maps) @ each position
    :param img:
    :param input_cb:
    :param cont_int_cb:
    :return:
    """
    input_act, cont_int_act = alex_net_utils.get_l1_and_l2_activations(
        img,
        input_cb,
        cont_int_cb
    )

    # Max difference
    diff = cont_int_act - input_act
    max_diff_across_chan = np.max(diff, axis=1)
    max_diff_across_chan = max_diff_across_chan[0, ]  # get rid of batch dim
    max_max_diff = np.max(max_diff_across_chan)

    # Max Enhancement Gain
    gain = cont_int_act / (input_act + 10e-4)
    gain[input_act == 0] = 1
    max_gain = np.max(gain, axis=1)
    max_gain = max_gain[0, ]  # get rid of batch dim

    # Plots
    f = plt.figure(figsize=(20, 5))

    f.add_subplot(1, 4, 1)
    plt.imshow(img.astype('uint8'), cmap='Greys')
    plt.colorbar(orientation='horizontal')

    f.add_subplot(1, 4, 2)
    input_act_max = np.max(input_act, axis=1)
    input_act_max = input_act_max[0, ]
    max_max_input_act = np.max(input_act_max)
    plt.imshow(input_act_max, cmap='seismic', vmax=max_max_input_act, vmin=-max_max_input_act)
    plt.colorbar(orientation='horizontal')
    plt.title("Feature Extract activation")

    f.add_subplot(1, 4, 3)
    plt.imshow(max_diff_across_chan, cmap='seismic', vmax=max_max_diff, vmin=-max_max_diff)
    plt.colorbar(orientation='horizontal')
    plt.title("Maximum Difference @ each (x,y)")

    f.add_subplot(1, 4, 4)
    plt.imshow(max_gain, cmap='seismic', vmax=-0.5, vmin=2.5)
    plt.colorbar(orientation='horizontal')
    plt.title("Max Gain @ each (x,y)")


def plot_contour_enhancement_individual_kernels(img, input_cb, cont_int_cb, filt_idx_arr):
    """

    :param img:
    :param input_cb: Note this is input to the contour integration layer
    :param cont_int_cb:
    :param filt_idx_arr:
    :return:
    """
    for f_idx in filt_idx_arr:
        alex_net_utils.plot_l1_and_l2_activations(
            img,
            input_cb,
            cont_int_cb,
            f_idx,
            show_input_img=False,
        )
        plt.suptitle("Kernel {}".format(f_idx))


def main(model, g_params, preprocessing_cb, learnt_kernels, results_dir=None):
    """

    :param results_dir:
    :param preprocessing_cb:
    :param learnt_kernels:
    :param g_params:
    :param model:
    :return:
    """

    cont_int_layer_idx = alex_net_utils.get_layer_idx_by_name(model, 'contour_integration_layer')
    cont_int_input_layer_idx = cont_int_layer_idx - 1

    model.summary()

    # Callbacks
    cont_int_input_act_cb = alex_net_utils.get_activation_cb(model, cont_int_input_layer_idx)
    cont_int_act_cb = alex_net_utils.get_activation_cb(model, cont_int_layer_idx)

    learnt_weights, _ = model.layers[cont_int_layer_idx].get_weights()

    frag = gabor_fits.get_gabor_fragment(g_params, (11, 11))

    if type(g_params) is not list:
        g_params = [g_params]

    # ----------------------------------------------------------------------------------
    #  1. Curved contour @ center position [maximum rotation offset]
    # ----------------------------------------------------------------------------------
    c_len = 9
    beta = 15

    img = image_generator_curve.generate_contour_images(
        n_images=1,
        frag=frag,
        frag_params=g_params,
        c_len=c_len,
        beta=beta,
        alpha=0,
        f_tile_size=np.array((18, 18)),
        rand_inter_frag_direction_change=False
    )
    # returns a numpy array of shape (b, r, c, ch) of dtype = uint8.
    # Note format is channel last here

    img = np.squeeze(img, axis=0)
    img = img.astype(dtype='float64')
    test_image = preprocessing_cb(img)

    # functions below expect channel last format
    plot_max_contour_enhancement(test_image, cont_int_input_act_cb, cont_int_act_cb)
    if results_dir is not None:
        fig = plt.gcf()
        fig.savefig(os.path.join(results_dir, 'curved_contour_at_center_loc.eps'), format='eps')

    # plot_contour_enhancement_individual_kernels(
    #     test_image, cont_int_input_act_cb, cont_int_act_cb, learnt_kernels)

    # ----------------------------------------------------------------------------------
    #  2. Curved contour @ different location
    # ----------------------------------------------------------------------------------
    c_len = 12
    beta = 15

    img = image_generator_curve.generate_contour_images(
        n_images=1,
        frag=frag,
        frag_params=g_params,
        c_len=c_len,
        beta=beta,
        alpha=0,
        f_tile_size=np.array((18, 18)),
        center_frag_start=np.array([180, 120]),
        rand_inter_frag_direction_change=True
    )
    # returns a numpy array of shape (b, r, c, ch) of dtype = uint8.
    # Note format is channel last here

    img = np.squeeze(img, axis=0)
    img = img.astype(dtype='float64')
    test_image = preprocessing_cb(img)

    # functions below expect channel last format
    plot_max_contour_enhancement(test_image, cont_int_input_act_cb, cont_int_act_cb)
    if results_dir is not None:
        fig = plt.gcf()
        fig.savefig(os.path.join(results_dir, 'curved_contour_at_different_loc.eps'), format='eps')

    # plot_contour_enhancement_individual_kernels(
    #     test_image, cont_int_input_act_cb, cont_int_act_cb, learnt_kernels)

    # ----------------------------------------------------------------------------------
    # 3. Circle @ different position
    # ----------------------------------------------------------------------------------
    c_len = 21
    beta = 15

    img = image_generator_curve.generate_contour_images(
        n_images=1,
        frag=frag,
        frag_params=g_params,
        c_len=c_len,
        beta=beta,
        alpha=0,
        f_tile_size=np.array((18, 18)),
        center_frag_start=np.array([140, 140]),
        rand_inter_frag_direction_change=False,
        base_contour='circle'
    )
    # returns a numpy array of shape (b, r, c, ch) of dtype = uint8.
    # Note format is channel last here

    img = np.squeeze(img, axis=0)
    img = img.astype(dtype='float64')
    test_image = preprocessing_cb(img)

    # functions below expect channel last format
    plot_max_contour_enhancement(test_image, cont_int_input_act_cb, cont_int_act_cb)
    if results_dir is not None:
        fig = plt.gcf()
        fig.savefig(os.path.join(results_dir, 'circle.eps'), format='eps')

    # plot_contour_enhancement_individual_kernels(
    #     test_image, cont_int_input_act_cb, cont_int_act_cb, learnt_kernels)

    # ----------------------------------------------------------------------------------
    # 4. Irregular shape shape
    # ----------------------------------------------------------------------------------
    image_file = './data/sample_images/irregular_shape.jpg'

    temp = keras.preprocessing.image.load_img(image_file, target_size=(227, 227, 3))
    in_img = keras.preprocessing.image.img_to_array(temp, dtype='float64', data_format='channels_last')

    img = img.astype(dtype='float64')
    test_image = preprocessing_cb(in_img)

    # functions below expect channel last format
    plot_max_contour_enhancement(test_image, cont_int_input_act_cb, cont_int_act_cb)
    if results_dir is not None:
        fig = plt.gcf()
        fig.savefig(os.path.join(results_dir, 'irregular_shape.eps'), format='eps')

    # plot_contour_enhancement_individual_kernels(
    #     test_image, cont_int_input_act_cb, cont_int_act_cb, learnt_kernels)

    # ----------------------------------------------------------------------------------
    # 4. Natural Images
    # ----------------------------------------------------------------------------------
    list_of_images = [
        './data/sample_images/cat.23.jpg',
        './data/sample_images/cat.7.jpg',
        './data/sample_images/dog.12447.jpg',
        './data/sample_images/zahra.jpg',
        # './data/sample_images/square.png',
    ]

    for image_file in list_of_images:

        temp = keras.preprocessing.image.load_img(image_file, target_size=(227, 227, 3))
        in_img = keras.preprocessing.image.img_to_array(temp, dtype='float64', data_format='channels_last')

        img = img.astype(dtype='float64')
        test_image = preprocessing_cb(in_img)

        # functions below expect channel last format
        plot_max_contour_enhancement(test_image, cont_int_input_act_cb, cont_int_act_cb)

        if results_dir is not None:
            fig = plt.gcf()

            fig_name = image_file.replace('/', '_')
            fig_name = fig_name.replace('.', '_')

            fig.savefig(os.path.join(results_dir, '{}.eps'.format(fig_name)), format='eps')

        # plot_contour_enhancement_individual_kernels(
        #     test_image, cont_int_input_act_cb, cont_int_act_cb, learnt_kernels)


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    # immutable
    plt.ion()
    keras.backend.clear_session()
    keras.backend.set_image_dim_ordering('th')

    np.random.seed(20)

    # -----------------------------------------------------------------------------------
    # Base Gabor
    # -----------------------------------------------------------------------------------
    # Gabor Params for Filter @ index 5
    gabor_params = {
        'x0': 0,
        'y0': 0,
        'theta_deg': -90.0,
        'amp': 1,
        'sigma': 2.75,
        'lambda1': 15.0,
        'psi': 1.5,
        'gamma': 0.8
    }

    # # Gabor Params for Filter @ index 10
    # gabor_params = {
    #     'x0': 0,
    #     'y0': 0,
    #     'theta_deg': 0.0,
    #     'amp': 1,
    #     'sigma': 2.75,
    #     'lambda1': 5.5,
    #     'psi': 1.25,
    #     'gamma': 0.8
    # }

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("Loading Model ...")

    # # OLD INDIVIDUALLY trained model
    # # ------------------------------
    # contour_int_weights = "./results/beta_rotations_upt30_alpha_upto30/trained_weights.hf"
    #
    # cont_int_model = model_3d_individually_trained.build_contour_integration_model(
    #     tgt_filt_idx=5,
    #     rf_size=35,
    #     inner_leaky_relu_alpha=0.9,
    #     outer_leaky_relu_alpha=1.,
    #     l1_reg_loss_weight=0.0005,
    # )
    # cont_int_model.load_weights(contour_int_weights, by_name=True)

    # SIMULTANEOUSLY Trained model
    # ----------------------------
    # contour_int_weights = \
    #     "./results/optimal_gabors_with_rotations_5_10_orientation/contour_integration_layer_weights.hf"
    contour_int_weights = \
        "./results/imagenet_preprocessing/contour_integration_layer_weights.hf"

    cont_int_model = model_3d_all_kernels.training_model(
        rf_size=35,
        inner_leaky_relu_alpha=0.9,
        outer_leaky_relu_alpha=1.,
        l1_reg_loss_weight=0.0001 / 96,
    )
    cont_int_model.load_weights(contour_int_weights, by_name=True)

    trained_kernels = [5, 10, 19, 20, 21, 22, 48, 49, 51, 59, 60, 62, 64, 65, 66, 68, 69, 72, 73, 74, 76, 77, 79]
    # trained_kernels = [5, 10]

    # alex_net_utils.clear_unlearnt_contour_integration_kernels(cont_int_model, trained_kernels)

    # -----------------------------------------------------------------------------------
    # main routine
    # -----------------------------------------------------------------------------------

    # #  Plot All learnt kernels
    # # -------------------------
    # for kernels_idx in trained_kernels:
    #     alex_net_utils.plot_start_n_learnt_contour_integration_kernels(
    #         cont_int_model,
    #         kernels_idx
    #     )
    #     raw_input('Next?')

    main(
        model=cont_int_model,
        preprocessing_cb=alex_net_utils.preprocessing_imagenet,
        g_params=gabor_params,
        learnt_kernels=trained_kernels
    )

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    raw_input("Press any key to exit.")
