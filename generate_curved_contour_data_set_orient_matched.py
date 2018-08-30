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


DATA_DIRECTORY = "./data/curved_contours/param_search_black_and_white"


def get_neurophysiological_data_raw():
    """
    Retrieve neurophysiological data from pickle files.

    Four dictionaries are returned
    [1] absolute linear gain, indexed by c_len [Li 2006 - Experiment 1]
    [2] absolute linear gain, index by relative colinear distance [Li 2006 Experiment 2]
    [3] beta rotation detectability indexed by beta [Fields 1993 - Experiment 1]
    [4] alpha rotation detectability indexed by alpha then beta [Fields 19993 - Experiment 3]

    :return:
    """
    with open('.//data//neuro_data//Li2006.pickle', 'rb') as handle:
        li_2006_data = pickle.load(handle)

    abs_linear_gain_c_len = {
        1: li_2006_data['contour_len_avg_gain'][0],
        3: li_2006_data['contour_len_avg_gain'][1],
        5: li_2006_data['contour_len_avg_gain'][2],
        7: li_2006_data['contour_len_avg_gain'][3],
        9: li_2006_data['contour_len_avg_gain'][4],
    }

    abs_linear_gain_f_spacing = {
        1: li_2006_data['contour_separation_avg_gain'][0],
        1.2: li_2006_data['contour_separation_avg_gain'][1],
        1.4: li_2006_data['contour_separation_avg_gain'][2],
        1.6: li_2006_data['contour_separation_avg_gain'][3],
        1.9: li_2006_data['contour_separation_avg_gain'][4],
    }

    with open('.//data//neuro_data//fields_1993_exp_1_beta.pickle', 'rb') as handle:
        fields_1993_exp_1_beta = pickle.load(handle)
    # Use averaged data
    rel_beta_rot_detectability = fields_1993_exp_1_beta['ah_djf_avg_1s_proportion_correct']

    with open('.//data//neuro_data//fields_1993_exp_3_alpha.pickle', 'rb') as handle:
        fields_1993_exp_3_alpha = pickle.load(handle)
    # Use averaged data
    rel_alpha_rot_detectability = {
        0: fields_1993_exp_3_alpha['ah_djf_avg_alpha_0_proportion_correct'],
        15: fields_1993_exp_3_alpha['ah_djf_avg_alpha_15_proportion_correct'],
        30: fields_1993_exp_3_alpha['ah_djf_avg_alpha_30_proportion_correct']
    }

    return abs_linear_gain_c_len, abs_linear_gain_f_spacing, rel_beta_rot_detectability, rel_alpha_rot_detectability


def get_neurophysiological_data(results_type):
    """
    Returns a nested dictionary of absolute results that can be easily accessed.

    The way to reference the results is results[c_len or f_spacing][alpha][beta]

    :return:
    """
    valid_results_types = ['c_len', 'f_spacing']
    results_type = results_type.lower()
    if results_type not in valid_results_types:
        raise Exception("Invalid results type requested: {}. Allowed ={}".format(results_type, valid_results_types))

    abs_linear_gain_c_len, abs_linear_gain_f_spacing, rel_beta_detectability, rel_alpha_detectability = \
        get_neurophysiological_data_raw()

    alpha_rot_arr = [0, 15, 30]
    beta_rot_arr = [0, 15, 30, 45, 60]

    results_dict = {}

    if results_type == 'c_len':
        c_len_arr = [1, 3, 5, 7, 9]

        for c_len in c_len_arr:
            alpha_dict = {}

            for alpha in alpha_rot_arr:
                # Get Detectability Results
                if alpha == 0:
                    detectability_dict = {beta: rel_beta_detectability[beta] for beta in beta_rot_arr}
                else:
                    detectability_dict = {beta: rel_alpha_detectability[alpha][beta] for beta in beta_rot_arr}

                # Change to absolute gain values
                # Relative gain curvature is actually detectability.
                # at 100 % detectability, gain is full amount. @ 50 percent detectability, no gain (gain=1)
                alpha_beta_dict = \
                    {beta: 1 + 2 * (detectability_dict[beta] - 0.5) * (abs_linear_gain_c_len[c_len] - 1)
                     for beta in beta_rot_arr}

                alpha_dict[alpha] = alpha_beta_dict

            results_dict[c_len] = alpha_dict
    else:  # results_type == 'f_spacing'
        rcd_arr = [1, 1.2, 1.4, 1.6, 1.9]

        for rcd in rcd_arr:
            alpha_dict = {}

            for alpha in alpha_rot_arr:
                # Get Detectability Results
                if alpha == 0:
                    detectability_dict = {beta: rel_beta_detectability[beta] for beta in beta_rot_arr}
                else:
                    detectability_dict = {beta: rel_alpha_detectability[alpha][beta] for beta in beta_rot_arr}

                # Change to absolute gain values
                # Relative gain curvature is actually detectability.
                # at 100 % detectability, gain is full amount. @ 50 percent detectability, no gain (gain=1)
                alpha_beta_dict = \
                    {beta: 1 + 2 * (detectability_dict[beta] - 0.5) * (abs_linear_gain_f_spacing[rcd] - 1)
                     for beta in beta_rot_arr}

                alpha_dict[alpha] = alpha_beta_dict

            results_dict[rcd] = alpha_dict

    return results_dict


def generate_data_set(
        base_dir, tgt_filt_idx, n_img_per_set, frag, frag_params, f_tile_size,
        img_size=np.array((227, 227, 3)), overwrite_existing_data=False):
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
    if type(frag_params) is not list:
        frag_params = [frag_params]

    beta_rot_arr = np.array([0, 15, 30, 45, 60])  # main contour rotation
    alpha_rot_arr = np.array([0, 15, 30])   # fragment rotation wrt to contour direction

    data_key_dict = {}

    # -----------------------------------------------------------------------------------
    # Create the destination directory
    # -----------------------------------------------------------------------------------
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
    #  Li-2006 Experiment 1 (Contour Length vs Gain) + Fields-1993 Experiments
    # -----------------------------------------------------------------------------------
    c_len_arr = np.array([1, 3, 5, 7, 9])
    abs_gains_arr = get_neurophysiological_data('c_len')

    for c_len in c_len_arr:
        rcd_dir = 'c_len_{0}'.format(c_len)

        for b_idx, beta in enumerate(beta_rot_arr):
            beta_n_f_spacing_dir = os.path.join(rcd_dir, 'beta_{0}'.format(beta))

            for a_idx, alpha in enumerate(alpha_rot_arr):
                alpha_n_beta_n_c_len_dir = os.path.join(beta_n_f_spacing_dir, 'alpha_{0}'.format(alpha))

                abs_destination_dir = os.path.join(filt_dir, alpha_n_beta_n_c_len_dir)
                if not os.path.exists(abs_destination_dir):
                    os.makedirs(abs_destination_dir)

                abs_gain = abs_gains_arr[c_len][alpha][beta]

                print("Generating {0} images for [contour length {1}, beta {2}, alpha {3}]. Expected Gain {4}".format(
                    n_img_per_set, c_len, beta, alpha, abs_gain))

                img_arr = image_generator_curve.generate_contour_images(
                    n_images=n_img_per_set,
                    frag=frag,
                    frag_params=frag_params,
                    c_len=c_len,
                    beta=beta,
                    alpha=alpha,
                    f_tile_size=f_tile_size,
                    img_size=img_size
                )

                # Save the images to file & create a dictionary key of (Image, Expected gain)
                # that can be used by a python generator / keras sequence object
                curr_set_dict = {}
                set_base_filename = "c_len_{0}_beta_{1}_alpha_{2}_forient_{3}".format(
                    c_len, beta, alpha, frag_params[0]['theta_deg'])

                for img_idx in range(img_arr.shape[0]):
                    filename = os.path.join(
                        abs_destination_dir,
                        set_base_filename + '__{0}.png'.format(img_idx)
                    )

                    plt.imsave(filename, img_arr[img_idx, ], format='PNG')
                    curr_set_dict[filename] = abs_gain

                # Add this dictionary to the dictionary of dictionaries
                data_key_dict[set_base_filename] = curr_set_dict

    # -----------------------------------------------------------------------------------
    #  Li-2006 Experiment 2 (Fragment Spacing vs Gain) + Fields-1993 Experiments
    # -----------------------------------------------------------------------------------
    c_len = 7
    rcd_arr = np.array([1, 1.2, 1.4, 1.6, 1.9])
    abs_gains_arr = get_neurophysiological_data('f_spacing')

    for rcd in rcd_arr:

        # Relative colinear distance is used in a different way than in the original paper.
        # In the ref, RCD is defined as the ratio of distance between fragment centers to distance of some fixed
        # tile size. The stimuli generated here are based on Fields-1993 stimuli and require spacing between fragments
        # Since we are mostly interested in modeling the effects vs the actual results, we used rcd to change the
        # distance between fragments from some arbitrary reference.
        # rcd is use to see how much the fragment size increases, this is then used to increase the full tile size
        # (the actual fragment size stays the same)

        frag_size_inc = np.int(rcd * frag.shape[0]) - frag.shape[0]
        updated_f_tile_size = f_tile_size + frag_size_inc

        f_spacing_dir = 'f_spacingx10_{0}'.format(int(rcd * 10))  # prevent the . from appearing in the file name

        for b_idx, beta in enumerate(beta_rot_arr):
            beta_n_f_spacing_dir = os.path.join(f_spacing_dir, 'beta_{0}'.format(beta))

            for a_idx, alpha in enumerate(alpha_rot_arr):
                alpha_n_beta_n_c_len_dir = os.path.join(beta_n_f_spacing_dir, 'alpha_{0}'.format(alpha))

                abs_destination_dir = os.path.join(filt_dir, alpha_n_beta_n_c_len_dir)
                if not os.path.exists(abs_destination_dir):
                    os.makedirs(abs_destination_dir)

                abs_gain = abs_gains_arr[rcd][alpha][beta]

                print("Generating {0} images for [f_spacing {1}, full_tile {2}, beta {3}, alpha {4}]. "
                      "Expected Gain {5}".format(n_img_per_set, rcd, updated_f_tile_size, beta, alpha, abs_gain))

                img_arr = image_generator_curve.generate_contour_images(
                    n_images=n_img_per_set,
                    frag=frag,
                    frag_params=frag_params,
                    c_len=c_len,
                    beta=beta,
                    alpha=alpha,
                    f_tile_size=updated_f_tile_size,
                    img_size=img_size
                )

                # Save the images to file & create a dictionary key of (Image, Expected gain)
                # that can be used by a python generator / keras sequence object
                curr_set_dict = {}
                set_base_filename = "f_spacingx10_{0}_beta_{1}_alpha_{2}_forient_{3}".format(
                    int(rcd * 10), beta, alpha, frag_params[0]['theta_deg'])

                for img_idx in range(img_arr.shape[0]):
                    filename = os.path.join(
                        abs_destination_dir,
                        set_base_filename + '__{0}.png'.format(img_idx)
                    )

                    plt.imsave(filename, img_arr[img_idx, ], format='PNG')
                    curr_set_dict[filename] = abs_gain

                # Add this dictionary to the dictionary of dictionaries
                data_key_dict[set_base_filename] = curr_set_dict

    # print("Generated Data Dictionaries")
    # for key in sorted(data_key_dict.keys()):
    #     print(key)

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


def _search_black_n_white_search_space(
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


def _search_colored_parameter_space(
        model, lambda1_arr, psi_arr, sigma_arr, theta_arr, threshold=3.0):
    """
    Iterate through the provided lambda1_arr, psi_arr, sigma_arr & theta_arr, and store
    gabor parameters that maximally activate feature extracting kernels

    Each color channel is searched independently and the combined parameters are stored.

    TODO: single iter cycle and search individual channels for each kernel. (speed up this function)

    :param model:
    :param lambda1_arr:
    :param psi_arr:
    :param sigma_arr:
    :param theta_arr:
    :param threshold:

    :return: Dictionary of contour integration kernels along with the best fit parameters.
    Dictionary keys are kernel indexes, the value is another dictionary that contains 2 keys:
    gabor_params and max value. Gabor Param is a list of dictionaries one for each channel

    """
    w, b = model.layers[1].get_weights()

    best_fit_params_dict = {}

    for theta_c0, sigma_c0, lambda1_c0, psi_c0 in itertools.product(
            theta_arr, sigma_arr, lambda1_arr, psi_arr):

        print("C0: theta {0}, sigma {1}, lambda {2}, psi {3}".format(
            theta_c0, lambda1_c0, sigma_c0, psi_c0))

        c0_loop_start_time = datetime.datetime.now()

        for theta_c1, sigma_c1, lambda1_c1, psi_c1 in itertools.product(
                theta_arr, sigma_arr, lambda1_arr, psi_arr):

            # print("C1: theta {0}, sigma {1}, lambda {2}, psi {3}".format(
            #     theta_c1, lambda1_c1, sigma_c1, psi_c1))

            # c1_loop_start_time = datetime.datetime.now()

            for theta_c2, sigma_c2, lambda1_c2, psi_c2 in itertools.product(
                    theta_arr, sigma_arr, lambda1_arr, psi_arr):

                # print("C2: theta {0}, sigma {1}, lambda {2}, psi {3}".format(
                #     theta_c2, lambda1_c2, sigma_c2, psi_c2))

                # c2_loop_start_time = datetime.datetime.now()

                # ------------------------------------------------------------------
                # Main Loop
                # ------------------------------------------------------------------
                g_params_dict_list = [
                    {
                        'x0': 0,
                        'y0': 0,
                        'theta_deg': theta_c0,
                        'amp': 1,
                        'sigma': sigma_c0,
                        'lambda1': lambda1_c0,
                        'psi': psi_c0,
                        'gamma': 1
                    },
                    {
                        'x0': 0,
                        'y0': 0,
                        'theta_deg': theta_c1,
                        'amp': 1,
                        'sigma': sigma_c1,
                        'lambda1': lambda1_c1,
                        'psi': psi_c1,
                        'gamma': 1
                    },
                    {
                        'x0': 0,
                        'y0': 0,
                        'theta_deg': theta_c2,
                        'amp': 1,
                        'sigma': sigma_c2,
                        'lambda1': lambda1_c2,
                        'psi': psi_c2,
                        'gamma': 1
                    }
                ]

                frag = gabor_fits.get_gabor_fragment(
                    g_params_dict_list,
                    (11, 11)
                )

                act = np.tensordot(frag, w, axes=((0, 1, 2), (0, 1, 2)))

                k_idx = np.argmax(act)
                act_value = act[k_idx]

                if act_value > threshold:
                    if k_idx not in best_fit_params_dict:
                        best_fit_params_dict[k_idx] = {
                            "gabor_params": g_params_dict_list,
                            "max_act": act_value
                        }
                        print("Found best fit parameters for contour integration kernel @ {}".format(k_idx))
                    else:
                        if act_value > best_fit_params_dict[k_idx]["max_act"]:
                            best_fit_params_dict[k_idx] = {
                                "gabor_params": g_params_dict_list,
                                "max_act": act_value
                            }

        print("C0 search Cycle took {}".format(datetime.datetime.now() - c0_loop_start_time))

    return best_fit_params_dict


def search_parameter_ranges_for_gabor_fits(model_feat_extract_cb, model):
    """
    Search over gabor parameters and find sets that maximally activate a feature
    extracting neuron

    :param model_feat_extract_cb:
    :param model:

    :return: Dictionary of best fit params for each kernel found. Dictionary is indexed by kernel index.
             Each 'value' is another dictionary with two keys [max_act] and 'gabor_params'
             The way to reference g_param_dict[kernel_idx]['gabor_params']

    """
    print("Searching Gabor Parameter Ranges...")
    param_search_start_time = datetime.datetime.now()

    # Full Range
    # -----------
    lambda1_array = np.arange(15, 2, -0.5)
    psi_array = np.concatenate((np.arange(0, 8, 0.25), np.arange(-0.5, -7, -0.25)))
    sigma_array = [2.5, 2.60, 2.70, 2.75]  # Any larger does not fit within the 11x11 fragment size.
    # Gabor angles are wrt y axis (0 = vertical). To get wrt to x-axis -90
    theta_array = -90 + np.arange(0, 180, 15)

    # # Short Range [test functionality]
    # # ---------------------------------
    # lambda1_array = np.arange(15, 2, -1)
    # psi_array = [0]
    # theta_array = -90 + np.arange(0, 180, 30)
    # sigma_array = [2.5, 2.7]

    g_params_dict = _search_black_n_white_search_space(
        model_feat_extract_cb,
        lambda1_arr=lambda1_array,
        psi_arr=psi_array,
        sigma_arr=sigma_array,
        theta_arr=theta_array
    )

    # g_params_dict = search_colored_parameter_space(
    #     cont_int_model,
    #     lambda1_arr=lambda1_array,
    #     psi_arr=psi_array,
    #     sigma_arr=sigma_array,
    #     theta_arr=theta_array
    # )

    print("Parameter Search took {}".format(datetime.datetime.now() - param_search_start_time))
    return g_params_dict


def plot_fits_and_filters(g_params_dict, model, frag_size=(11, 11)):
    """

    :param g_params_dict:
    :param model:
    :param frag_size:

    :return:
    """
    w, b = model.layers[1].get_weights()

    for k_idx in g_params_dict.keys():

        tgt_w = w[:, :, :, k_idx]

        f, ax_arr = plt.subplots(1, 2)

        frag = gabor_fits.get_gabor_fragment(
            gabor_params_dict[k_idx]["gabor_params"],
            frag_size,
        )

        ax_arr[0].imshow(tgt_w)
        ax_arr[0].title('kernel')

        ax_arr[1].imshow(frag)
        ax_arr[1].title('fragment')

        f.suptitle("Max active Kernel @ index {0}. Act Value {1}".format(
            k_idx, gabor_params_dict[k_idx]["max_act"]))

        # Plot all rotations of the fragment
        gabor_fits.plot_fragment_rotations(frag, g_params_dict[k_idx]["gabor_params"])

        print(g_params_dict[k_idx])

        raw_input("Press any key to continue to next entry")

    print("All done")


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')
    start_time = datetime.datetime.now()

    n_train_images_per_set = 300
    n_test_images_per_set = 50

    full_tile_size = np.array((17, 17))
    frag_tile_size = np.array((11, 11))

    # -----------------------------------------------------------------------------------
    # Contour Integration Model
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(5)
    feat_extract_act_cb = alex_net_utils.get_activation_cb(cont_int_model, 1)

    # -----------------------------------------------------------------------------------
    # Find Best Fit Gabor Parameters
    # -----------------------------------------------------------------------------------
    # A. parameter_search_space method
    gabor_params_dict = search_parameter_ranges_for_gabor_fits(feat_extract_act_cb, cont_int_model)

    # print best fit params
    print("{0}\n Number of trainable kernels {1}.\n {0}, ".format('*' * 80, len(gabor_params_dict)))
    for kernel_idx in gabor_params_dict.keys():
        print("Kernel {0}, max_activation {1}".format(
            kernel_idx, gabor_params_dict[kernel_idx]["max_act"]))

    # Store best fit params
    if not os.path.exists(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)

    best_fit_params_store_file = os.path.join(DATA_DIRECTORY, 'best_fit_params.pickle')
    with open(best_fit_params_store_file, 'wb') as f_id:
        pickle.dump(gabor_params_dict, f_id)

    # ------------------------------------------------------------------------------
    # Plot all Gabors found to maximally activate neurons
    # ------------------------------------------------------------------------------
    plot_fits_and_filters(gabor_params_dict, cont_int_model)

    # ------------------------------------------------------------------------------
    # Generate the Data
    # ------------------------------------------------------------------------------
    for kernel_idx in gabor_params_dict.keys():
        print("Generated Data for kernel @ index {0} [Feature Extract Activation {1}]...".format(
            kernel_idx, gabor_params_dict[kernel_idx]["max_act"]))

        kernel_data_gen_start_time = datetime.datetime.now()

        params = gabor_params_dict[kernel_idx]["gabor_params"]

        fragment = gabor_fits.get_gabor_fragment(params, frag_tile_size)

        print("Generating Train Data Set")
        generate_data_set(
            base_dir=os.path.join(DATA_DIRECTORY, 'train'),
            tgt_filt_idx=kernel_idx,
            n_img_per_set=n_train_images_per_set,
            frag=fragment,
            frag_params=params,
            f_tile_size=full_tile_size,
        )

        print("Generating Test Data Set")
        generate_data_set(
            base_dir=os.path.join(DATA_DIRECTORY, 'test'),
            tgt_filt_idx=kernel_idx,
            n_img_per_set=n_test_images_per_set,
            frag=fragment,
            frag_params=params,
            f_tile_size=full_tile_size,
        )

        print("Data Generation for kernel index {0} took {1}".format(
            kernel_idx,
            datetime.datetime.now() - kernel_data_gen_start_time
        ))

    # -----------------------------------------------------------------------------------
    #  End
    # -----------------------------------------------------------------------------------
    print("Total Time {}".format(datetime.datetime.now() - start_time))
