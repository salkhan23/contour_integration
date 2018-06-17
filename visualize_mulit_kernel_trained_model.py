# -------------------------------------------------------------------------------------------------
#  Visualize the Complete enhancement done by the model trained with multiple contour
#
# Author: Salman Khan
# Date  : 13/06/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

import keras.backend as K
import keras
from keras.preprocessing import image as image_preprocessing

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

DATA_DIR = './data/curved_contours/orientation_matched'
MODEL_STORE_DIR = './trained_models/ContourIntegrationModel3d/orientation_matched'

LEARNT_WEIGHTS = os.path.join(MODEL_STORE_DIR, "contour_integration_weights.hf")
LEARNT_SUMMARY = os.path.join(MODEL_STORE_DIR, "summary.txt")

if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    # Build Contour Integration Model
    # -----------------------------------------------------------------------------------
    tgt_kernel_idx = 5;
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(tgt_kernel_idx)
    cont_int_model.summary()

    # load pretrained weights
    cont_int_model.load_weights(LEARNT_WEIGHTS, by_name=True)

    # Print the list of kernels the model has been trained with
    with open(LEARNT_SUMMARY, 'r') as fid:
        read_in = fid.read()
        # print("previously trained Kernels @ indexes")
        # print(read_in)

    prev_learnt_kernels = set(read_in.split())
    prev_learnt_kernels = [int(idx) for idx in prev_learnt_kernels]
    print prev_learnt_kernels

    # -----------------------------------------------------------------------------------
    # Verify kernels have actually been learnt
    # -----------------------------------------------------------------------------------
    learnt_weights, _ = cont_int_model.layers[2].get_weights()

    # # Verify the Kernels are learnt, by displaying them
    # for learnt_kernel_idx in prev_learnt_kernels:
    #
    #     linear_contour_training.plot_contour_integration_weights_in_channels(
    #         learnt_weights, learnt_kernel_idx)
    #
    #     plt.title("Weights for Contour Integration kernel at index {}".format(learnt_kernel_idx))

    # -----------------------------------------------------------------------------------
    # Sample Image output
    # -----------------------------------------------------------------------------------
    # c_len = 9
    # beta = 15
    # img_idx = 15
    # filt_idx = 22
    # frag_orient = '75'
    #
    # sample_file = os.path.join(
    #     DATA_DIR, 'test', 'filter_{}'.format(filt_idx), 'c_len_{}'.format(c_len), 'beta_{}'.format(beta),
    #     "c_len_{0}_beta_{1}_rot_{2}__{3}.png".format(c_len, beta, frag_orient, img_idx)
    # )
    #
    # for filt_idx in prev_learnt_kernels:
    #     field_1993_routines.plot_activations(
    #         cont_int_model,
    #         sample_file,
    #         filt_idx)
    #     plt.suptitle("Kernel {}".format(filt_idx))

    # Base parameters of contour Fragment
    gabor_params = {
        'x0': 0,
        'y0': 0,
        'theta_deg': 0,
        'amp': 1,
        'sigma': 2.5,
        'lambda1': 8,
        'psi': 0,
        'gamma': 1
    }

    fragment = gabor_fits.get_gabor_fragment(gabor_params, np.array((11, 11)))

    img_arr = image_generator_curve.generate_contour_images(
        n_images=1,
        frag=fragment,
        frag_params=gabor_params,
        c_len=12,
        beta=30,
        f_tile_size=np.array((17, 17)),
        # inter_frag_rand_rot_direction=False
    )

    image = img_arr[0, ]

    # Callbacks to get activations of feature extract and contour integration layers
    feat_extract_act_cb = alex_net_utils.get_activation_cb(cont_int_model, 1)
    cont_int_act_cb = alex_net_utils.get_activation_cb(cont_int_model, 2)

    # PLot individual Filters
    # --------------------------------------------------
    # for filt_idx in prev_learnt_kernels:
    #     alex_net_utils.plot_l1_and_l2_activations(
    #         image / 255.0, feat_extract_act_cb, cont_int_act_cb, filt_idx)
    #     plt.suptitle("Kernel {}".format(filt_idx))

    # # Plot the maximum at each location
    # # ---------------------------------
    # l1_act, l2_act = alex_net_utils.get_l1_and_l2_activations(
    #     image / 255.0,
    #     feat_extract_act_cb,
    #     cont_int_act_cb)
    #
    # diff = l2_act - l1_act
    #
    # z = np.max(diff, axis=1)
    #
    # plt.figure()
    # plt.imshow(image/255.0)
    #
    # plt.figure()
    # plt.imshow(z[0, ], cmap='seismic')
    # plt.colorbar(orientation='horizontal')
    # plt.title("max across combined")

    # ---------------------------------------
    square_image = os.path.join('./data/sample_images/', 'square.png')
    square_image = os.path.join('./data/sample_images/', 'irregular_shape.jpg')

    img = image_preprocessing.load_img(square_image, target_size=(227, 227))

    in_x = image_preprocessing.img_to_array(img)

    plt.figure()
    plt.imshow(img)

    in_x = np.transpose(in_x, axes=(1, 2, 0))

    l1_act, l2_act = alex_net_utils.get_l1_and_l2_activations(
        in_x / 255.0,
        feat_extract_act_cb,
        cont_int_act_cb)

    diff = l2_act - l1_act

    diff_1 = diff[:, (22,48,66,73,78), :, :]

    z = np.max(diff, axis=1)

    plt.figure()
    plt.imshow(image/255.0)

    plt.figure()
    plt.imshow(z[0, ], cmap='seismic')
    plt.colorbar(orientation='horizontal')
    plt.title("max across combined")

    l2_max = l2_act.max()

    f, ax_arr = plt.subplots(1, 2)
    ax_arr[0].imshow(np.max(l1_act, axis=1)[0, :], vmin=-l2_max, vmax=l2_max)
    ax_arr[0].set_title('Feature extract (max across channels)')

    ax_arr[1].imshow(np.max(l2_act, axis=1)[0, :], vmin=-l2_max, vmax=l2_max)
    ax_arr[1].set_title('Contour Integration')













