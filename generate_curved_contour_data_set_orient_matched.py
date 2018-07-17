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
import datetime
import itertools

import keras.backend as keras_backend

import contour_integration_models.alex_net.model_3d as contour_integration_model_3d
import gabor_fits
import alex_net_utils
import image_generator_curve


reload(contour_integration_model_3d)
reload(gabor_fits)
reload(alex_net_utils)
reload(image_generator_curve)


DATA_DIRECTORY = "./data/curved_contours/orientation_matched2"


def generate_data_set(
        base_dir, tgt_filt_idx, n_img_per_set, frag, frag_params, f_tile_size,
        img_size=(227, 227, 3), overwrite_existing_data=False):
    """
    Given a contour fragment and its gabor params, generate test and train images in
    the base directory

    :param overwrite_existing_data:
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

    if type(frag_params) is not list:
        frag_params = [frag_params]

    # Create the destination directory
    # --------------------------------------
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    filt_dir = os.path.join(base_dir, "filter_{0}".format(tgt_filt_idx))

    if os.path.isdir(filt_dir):
        if overwrite_existing_data:
            print("Overwriting Existing Data for kernel at index {}".format(tgt_filt_idx))
            shutil.rmtree(filt_dir)
        else:
            ans = raw_input("Overwrite Existing Data for kernel at index {}?".format(tgt_filt_idx))

            if 'y' in ans.lower():
                shutil.rmtree(filt_dir)
            else:
                return

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
                    c_len, beta, frag_params[0]['theta_deg'], img_idx)

                plt.imsave(
                    os.path.join(abs_destination_dir, filename),
                    img_arr[img_idx, ],
                    format='PNG'
                )

                beta_dict[os.path.join(abs_destination_dir, filename)] = abs_gain

            # Add this dictionary to the dictionary of dictionaries
            data_key_dict['c_len_{0}_beta_{1}_rot_{2}'.format(
                c_len, beta, frag_params[0]['theta_deg'])] = beta_dict

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


def search_black_n_white_search_space(
        model_feat_extract_cb, lambda1_arr, psi_arr, sigma_arr, theta_arr, th=3.0):
    """

    :param model_feat_extract_cb:
    :param lambda1_arr:
    :param psi_arr:
    :param sigma_arr:
    :param theta_arr:
    :param th:

    :return: Dictionary of contour integration kernels along with the best fit parameters.
    Dictionary keys are kernel indexes, the value is another dictionary that contains 2 keys:
    gabor_params and max value
    """
    best_fit_params_dict = {}

    for theta, sigma, lambda1, psi in itertools.product(theta_arr, sigma_arr, lambda1_arr, psi_arr):

        # print("theta {0}, sigma {1} lambda1 {2}, psi {3}".format(
        #     theta, sigma, lambda1, psi))

        g_params = {
            'x0': 0,
            'y0': 0,
            'theta_deg': theta,
            'amp': 1,
            'sigma': sigma,
            'lambda1': lambda1,
            'psi': psi,
            'gamma': 1
        }

        frag = gabor_fits.get_gabor_fragment(g_params, (11, 11))

        k_idx, act_value = alex_net_utils.find_most_active_l1_kernel_index(
            frag,
            model_feat_extract_cb,
            plot=False
        )

        if act_value > th:
            if k_idx not in best_fit_params_dict:
                print("Adding Kernel {} to dictionary".format(k_idx))

                best_fit_params_dict[k_idx] = {
                    "gabor_params": g_params,
                    "max_act": act_value,
                }
            else:
                if act_value > best_fit_params_dict[k_idx]["max_act"]:
                    best_fit_params_dict[k_idx] = {
                        "gabor_params": g_params,
                        "max_act": act_value,
                    }
    return best_fit_params_dict


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')
    start_time = datetime.datetime.now()

    # -----------------------------------------------------------------------------------
    # Contour Integration Model
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(5)  # Index not important

    feat_extract_act_cb = alex_net_utils.get_activation_cb(cont_int_model, 1)

    # -----------------------------------------------------------------------------------
    #  Search over gabor parameters and find sets that maximally activate a feature
    #  extracting neuron
    # -----------------------------------------------------------------------------------
    print("Searching Gabor Parameter Ranges")

    # Full Range
    # -----------
    lambda1_array = np.arange(15, 2, -0.5)
    psi_array = np.concatenate((np.arange(0, 8, 0.25), np.arange(-0.5, -7, -0.25)))
    sigma_array = [2.5, 2.60, 2.70, 2.80]  # Any larger does not fit within the 11x11 fragment size.
    theta_array = -90 + np.arange(0, 180, 15)  # Gabor angles are wrt y axis (0 = vertical). To get wrt to x-axis -90

    # # Short Range [test functionality]
    # # ---------------------------------
    # lambda1_array = np.arange(15, 2, -1)
    # psi_array = [0]
    # theta_array = -90 + np.arange(0, 180, 30)
    # sigma_array = [2.5, 2.7]

    gabor_params_dict = search_black_n_white_search_space(
        feat_extract_act_cb,
        lambda1_arr=lambda1_array,
        psi_arr=psi_array,
        sigma_arr=sigma_array,
        theta_arr=theta_array
    )

    print('*' * 20)
    print("Number of trainable kernels {0}".format(len(gabor_params_dict)))
    for kernel_idx in gabor_params_dict.keys():
        print("Kernel {0}, max_activation {1}".format(kernel_idx, gabor_params_dict[kernel_idx]["max_act"]))

    print("Parameter Search took {}".format(datetime.datetime.now() - start_time))

    # # ------------------------------------------------------------------------------
    # # Plot all Gabors found to maximally activate neurons
    # # ------------------------------------------------------------------------------
    # for k_idx in gabor_params_dict.keys():
    #
    #     fragment = gabor_fits.get_gabor_fragment(
    #         gabor_params_dict[k_idx]["gabor_params"],
    #         (11, 11)
    #     )
    #
    #     gabor_fits.plot_fragment_rotations(fragment, gabor_params_dict[k_idx]["gabor_params"])
    #     plt.suptitle("Max active Kernel @ index {0}. Act Value {1}".format(
    #         k_idx,
    #         gabor_params_dict[k_idx]["max_act"]
    #     ))
    #
    #     print(gabor_params_dict[k_idx])
    #     raw_input()

    # ------------------------------------------------------------------------------
    # Generate the Data
    # ------------------------------------------------------------------------------
    n_train_images = 500
    n_test_images = 100

    full_tile_size = np.array((17, 17))
    frag_tile_size = np.array((11, 11))

    for kernel_idx in gabor_params_dict.keys():
        print("Generated Data for Contour Integration kernel @ index {0} [Feature Extract Activation {1}]...".format(
            kernel_idx, gabor_params_dict[kernel_idx]["max_act"]))

        kernel_data_gen_start_time = datetime.datetime.now()

        params = gabor_params_dict[kernel_idx]["gabor_params"]

        fragment = gabor_fits.get_gabor_fragment(params, frag_tile_size)

        print("Generating Train Data Set")
        generate_data_set(
            base_dir=os.path.join(DATA_DIRECTORY, 'train'),
            tgt_filt_idx=kernel_idx,
            n_img_per_set=n_train_images,
            frag=fragment,
            frag_params=params,
            f_tile_size=full_tile_size,
            img_size=(227, 227, 3)
        )

        print("Generating Test Data Set")
        generate_data_set(
            base_dir=os.path.join(DATA_DIRECTORY, 'test'),
            tgt_filt_idx=kernel_idx,
            n_img_per_set=n_test_images,
            frag=fragment,
            frag_params=params,
            f_tile_size=full_tile_size,
            img_size=(227, 227, 3)
        )

        print("Data Generation for kernel index {0} took {1}".format(
            kernel_idx,
            datetime.datetime.now() - kernel_data_gen_start_time
        ))

    # -----------------------------------------------------------------------------------
    #  End
    # -----------------------------------------------------------------------------------
    print("Total Time {}".format(datetime.datetime.now() - start_time))
