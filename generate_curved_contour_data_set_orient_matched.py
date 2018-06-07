# -------------------------------------------------------------------------------------------------
# Scripts generates sets of training images for curved contours.
# Contour Fragments are matched for orientation only
#
#  The procedure for generating sample images is as follows.
#     1. A base contour fragment (gabor parameters) is described.
#     2. The most responsive feature extracting kernel is found.
#     3. The fragment is rotated by +15 degrees and the process is repeated until the whole
#        [0, 180] degree range is covered.
#
#
# Author: Salman Khan
# Date  : 21/04/18
# -------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pickle

import keras.backend as K

import contour_integration_models.alex_net.model_3d as contour_integration_model_3d
import gabor_fits
import alex_net_utils
import image_generator_curve


reload(contour_integration_model_3d)
reload(gabor_fits)
reload(alex_net_utils)
reload(image_generator_curve)


DATA_DIRECTORY = "./data/curved_contours/orientation_matched"


def generate_data_set(
        base_dir, tgt_filt_idx, n_img_per_set, frag, frag_params, f_tile_size,
        img_size=(227, 227, 3)):
    """

    :param base_dir:
    :param tgt_filt_idx:
    :param n_img_per_set:
    :param frag:
    :param frag_params:
    :param f_tile_size:
    :param img_size:
    :return:
    """
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    c_len_arr = np.array([1, 3, 5, 7, 9])
    beta_rot_arr = np.array([0, 15, 30, 45, 60])
    # TODO: Add different spacing

    # Neurophysiological data
    with open('.//data//neuro_data//Li2006.pickle', 'rb') as handle:
        li_2006_data = pickle.load(handle)

    absolute_gain_linear = {
        1: li_2006_data['contour_len_avg_gain'][0],
        3: li_2006_data['contour_len_avg_gain'][1],
        5: li_2006_data['contour_len_avg_gain'][2],
        7: li_2006_data['contour_len_avg_gain'][3],
        9: li_2006_data['contour_len_avg_gain'][4],
    }

    # TODO: Put data in Pickle Format
    relative_gain_curvature = {
        0: 1.00,
        15: 0.98,
        30: 0.87,
        45: 0.85,
        60: 0.61
    }

    # Create the destination directory
    # --------------------------------------
    filt_dir = os.path.join(base_dir, "filter_{0}".format(tgt_filt_idx))

    # -----------------------------------------------------------------------------------
    #  Generate the Data
    # -----------------------------------------------------------------------------------
    data_key_dict = {}

    for c_len in c_len_arr:

        c_len_dir = 'c_len_{0}'.format(c_len)

        for b_idx, beta in enumerate(beta_rot_arr):

            beta_n_clen_dir = os.path.join(c_len_dir, 'beta_{0}'.format(beta))

            abs_destination_dir = os.path.join(filt_dir, beta_n_clen_dir)
            if not os.path.exists(abs_destination_dir):
                os.makedirs(abs_destination_dir)

            img_arr = image_generator_curve.generate_contour_images(
                n_images=n_img_per_set,
                frag=frag,
                frag_params=frag_params,
                c_len=c_len,
                beta=beta,
                f_tile_size=f_tile_size,
                img_size=img_size
            )

            # Save the images to file & create a dictionary key of (Image, Expected gain)
            # that can be used by a python generator / keras sequence object

            #  Relative gain curvature is actually detectability.
            #  at 100% detectability, gain is full amount. @ 50 percent detectability, no gain (gain=1)
            abs_gain = 1 + 2 * (relative_gain_curvature[beta] - 0.5) * (absolute_gain_linear[c_len] - 1)

            beta_dict = {}
            for img_idx in range(img_arr.shape[0]):
                filename = "c_len_{0}_beta_{1}_rot_{2}__{3}.png".format(
                    c_len, beta, frag_params['theta_deg'], img_idx)

                plt.imsave(
                    os.path.join(abs_destination_dir, filename),
                    img_arr[img_idx, ],
                    format='PNG'
                )

                beta_dict[os.path.join(abs_destination_dir, filename)] = abs_gain

            # Add this dictionary to the dictionary of dictionaries
            data_key_dict['c_len_{0}_beta_{1}_rot_{2}'.format(
                c_len, beta, frag_params['theta_deg'])] = beta_dict

    # Store the dictionary of Dictionaries
    # Each entry in this dictionary is dictionary of image index and its absolute gain value
    # for a particular c_len, beta rotation value
    master_key_file_loc = os.path.join(filt_dir, 'data_key.pickle')

    if os.path.exists(master_key_file_loc):
        with open(master_key_file_loc, 'rb') as handle:
            prev_data_key_dict = pickle.load(handle)

        data_key_dict.update(prev_data_key_dict)

    with open(master_key_file_loc, 'wb') as handle:
        pickle.dump(data_key_dict, handle)


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # Gabor angles are wrt y axis (0 = vertical). We want fragments wrt to x axis. Hence -90
    theta_arr = -90 + np.arange(0, 180, 15)

    # Base parameters of contour Fragment
    gabor_params = {
        'x0': 0,
        'y0': 0,
        'theta_deg': -90,
        'amp': 1,
        'sigma': 2.5,
        'lambda1': 8,
        'psi': 0,
        'gamma': 1
    }

    # Create base directory if it doesnt exist
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
    else:
        ans = raw_input("Previous Data Set Exists. Overwrite? (Y/N)")

        if 'y' not in ans.lower():
            raise SystemExit()
        else:
            shutil.rmtree(DATA_DIRECTORY)

    # -----------------------------------------------------------------------------------
    # Load Contour Integration Model
    # -----------------------------------------------------------------------------------
    tgt_kernel_idx = 5  # Not important
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(
        tgt_kernel_idx)
    # cont_int_model.summary()

    feat_extract_kernels = K.eval(cont_int_model.layers[1].weights[0])

    feat_extract_act_cb = alex_net_utils.get_activation_cb(cont_int_model, 1)

    # # -----------------------------------------------------------------------------------
    # #  Fit most active neuron for each considered angle
    # # -----------------------------------------------------------------------------------
    # for theta in theta_arr:
    #     gabor_params['theta_deg'] = theta
    #     fragment = gabor_fits.get_gabor_fragment(gabor_params, feat_extract_kernels.shape[0:2])
    #
    #     kernel_idx, act_value = alex_net_utils.find_most_active_l1_kernel_index(
    #         fragment,
    #         feat_extract_act_cb,
    #         plot=False
    #     )
    #
    #     print("Fragment Orientation {0}: Max active kernel idx = {1}, value={2}".format(
    #         theta, kernel_idx, act_value))

    # -----------------------------------------------------------------------------------
    #  Generate Data for each Orientation
    # -----------------------------------------------------------------------------------
    n_train_images = 500
    n_test_images = 100

    image_size = np.array((227, 227, 3))

    full_tile_size = np.array((17, 17))
    frag_tile_size = np.array((11, 11))

    for theta in theta_arr:

        print("Processing contour fragment with base orientation {}".format(theta))

        gabor_params['theta_deg'] = theta
        fragment = gabor_fits.get_gabor_fragment(gabor_params, feat_extract_kernels.shape[0:2])

        # First max activated kernel
        kernel_idx, act_value = alex_net_utils.find_most_active_l1_kernel_index(
            fragment,
            feat_extract_act_cb,
            plot=False
        )
        print("Max responsive kernel index {0}, value={1}".format(kernel_idx, act_value))

        print("Generating Train Data Set")
        generate_data_set(
            base_dir=os.path.join(DATA_DIRECTORY, 'train'),
            tgt_filt_idx=kernel_idx,
            n_img_per_set=n_train_images,
            frag=fragment,
            frag_params=gabor_params,
            f_tile_size=full_tile_size,
            img_size=image_size
        )

        print("Generating Test Data Set")
        generate_data_set(
            base_dir=os.path.join(DATA_DIRECTORY, 'test'),
            tgt_filt_idx=kernel_idx,
            n_img_per_set=n_test_images,
            frag=fragment,
            frag_params=gabor_params,
            f_tile_size=full_tile_size,
            img_size=(227, 227, 3)
        )

        # # check the created image key is correct
        # master_key_file = os.path.join(
        #     DATA_DIRECTORY, 'train', "filter_{0}".format(kernel_idx), 'data_key.pickle')
        #
        # with open(master_key_file, 'rb') as fid:
        #     data_key = pickle.load(fid)
        #
        # print (data_key.keys())
        # raw_input("Continue?")
