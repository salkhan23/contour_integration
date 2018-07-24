# -------------------------------------------------------------------------------------------------
#  Visualize the Complete enhancement done by the model trained with multiple contour
#
# Author: Salman Khan
# Date  : 13/06/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as keras_backend

import contour_integration_models.alex_net.model_3d as contour_integration_model_3d
import learn_cont_int_kernel_3d_model
import learn_cont_int_kernel_3d_model_linear_contours as linear_contour_training
import field_1993_routines
import image_generator_curve
import gabor_fits
import alex_net_utils

reload(contour_integration_model_3d)
reload(learn_cont_int_kernel_3d_model)
reload(linear_contour_training)
reload(field_1993_routines)
reload(image_generator_curve)
reload(gabor_fits)
reload(alex_net_utils)


def plot_max_contour_enhancement(img, feat_extract_cb, cont_int_cb):
    """
    Plot the maximum contour enhancement (Across all feature maps) @ each position
    :param img:
    :param feat_extract_cb:
    :param cont_int_cb:
    :return:
    """
    l1_act, l2_act = alex_net_utils.get_l1_and_l2_activations(
        img,
        feat_extract_cb,
        cont_int_cb
    )

    diff = l2_act - l1_act
    max_diff = np.max(diff, axis=1)

    plt.figure()
    plt.imshow(max_diff[0, ], cmap='seismic')
    plt.colorbar(orientation='horizontal')
    plt.title("Maximum contour enhancement @ each (x,y) ")


def plot_contour_enhancement_individual_kernels(img, feat_extract_cb, cont_int_cb, filt_idx_arr):
    """

    :param img:
    :param feat_extract_cb:
    :param cont_int_cb:
    :param filt_idx_arr:
    :return:
    """
    for f_idx in filt_idx_arr:
        alex_net_utils.plot_l1_and_l2_activations(
            img,
            feat_extract_cb,
            cont_int_cb,
            f_idx,
            show_img=False,
        )
        plt.suptitle("Kernel {}".format(f_idx))


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')

    learnt_weights_file = \
        "./trained_models/ContourIntegrationModel3d/orientation_matched/contour_integration_weights.hf"

    # learnt_weights_file = \
    #     "./trained_models/ContourIntegrationModel3d/filt_matched_frag/contour_integration_weights.hf"

    # -----------------------------------------------------------------------------------
    # Build Contour Integration Model
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(5)
    # cont_int_model.summary()

    # load pretrained weights
    cont_int_model.load_weights(learnt_weights_file, by_name=True)

    training_summary = learn_cont_int_kernel_3d_model.get_weights_training_summary_file(learnt_weights_file)
    trained_kernels = learn_cont_int_kernel_3d_model.get_prev_learnt_kernels(training_summary)
    print("Previously learnt kernels \n {}".format(trained_kernels))

    # Callbacks
    feat_extract_act_cb = alex_net_utils.get_activation_cb(cont_int_model, 1)
    cont_int_act_cb = alex_net_utils.get_activation_cb(cont_int_model, 2)

    # # -----------------------------------------------------------------------------------
    # #  Plot All learnt kernels
    # # -----------------------------------------------------------------------------------
    # learnt_weights, _ = cont_int_model.layers[2].get_weights()
    #
    # for kernels_idx in trained_kernels:
    #     learn_cont_int_kernel_3d_model.plot_start_n_learnt_contour_integration_kernels(
    #         cont_int_model,
    #         kernels_idx
    #     )

    # -----------------------------------------------------------------------------------
    #  Base Gabor Fragment
    # -----------------------------------------------------------------------------------
    frag_orient = 0
    gabor_params = {
        'x0': 0,
        'y0': 0,
        'theta_deg': frag_orient,
        'amp': 1,
        'sigma': 2.8,
        'lambda1': 12,
        'psi': 0,
        'gamma': 1
    }
    fragment = gabor_fits.get_gabor_fragment(gabor_params, (11, 11))

    # plt.figure()
    # plt.imshow(fragment)
    # plt.title("Contour Fragment")

    # ----------------------------------------------------------------------------------
    # Curved Contour @ center position
    # ----------------------------------------------------------------------------------
    c_len = 9
    beta = 0

    img_arr = image_generator_curve.generate_contour_images(
        n_images=1,
        frag=fragment,
        frag_params=gabor_params,
        c_len=c_len,
        beta=beta,
        f_tile_size=np.array((17, 17)),
        rand_inter_frag_direction_change=False
    )

    test_image = img_arr[0, ] / 255.0
    plt.imshow(test_image)
    plt.title("Curved Contour @ Center")

    plot_max_contour_enhancement(test_image, feat_extract_act_cb, cont_int_act_cb)
    plot_contour_enhancement_individual_kernels(
        test_image, feat_extract_act_cb, cont_int_act_cb, trained_kernels)

    # ----------------------------------------------------------------------------------
    # Curved Contour @ different position
    # ----------------------------------------------------------------------------------
    c_len = 12
    beta = 0

    img_arr = image_generator_curve.generate_contour_images(
        n_images=1,
        frag=fragment,
        frag_params=gabor_params,
        c_len=c_len,
        beta=beta,
        f_tile_size=np.array((17, 17)),
        center_frag_start=np.array([180, 120]),
        rand_inter_frag_direction_change=True
    )

    test_image = img_arr[0, ] / 255.0
    plt.imshow(test_image)
    plt.title("Curved Contour at Random Location")

    plot_max_contour_enhancement(test_image, feat_extract_act_cb, cont_int_act_cb)
    # plot_contour_enhancement_individual_kernels(
    #     test_image, feat_extract_act_cb, cont_int_act_cb, trained_kernels)

    # ----------------------------------------------------------------------------------
    # Circle @ different position
    # ----------------------------------------------------------------------------------
    c_len = 21
    beta = 15

    img_arr = image_generator_curve.generate_contour_images(
        n_images=1,
        frag=fragment,
        frag_params=gabor_params,
        c_len=c_len,
        beta=beta,
        f_tile_size=np.array((17, 17)),
        center_frag_start=np.array([10, 75]),
        rand_inter_frag_direction_change=False,
        base_contour='circle'
    )

    test_image = img_arr[0, ] / 255.0
    plt.imshow(test_image)
    plt.title("Curved Contour at Random Location")

    plot_max_contour_enhancement(test_image, feat_extract_act_cb, cont_int_act_cb)
    # plot_contour_enhancement_individual_kernels(
    #     test_image, feat_extract_act_cb, cont_int_act_cb, trained_kernels)

    # ----------------------------------------------------------------------------------
    # TODO: Add linear contours Generated the old way
    # ----------------------------------------------------------------------------------
    raw_input("Press any key to exit.")
