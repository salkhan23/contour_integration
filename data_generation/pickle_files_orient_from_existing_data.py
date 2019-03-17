# ---------------------------------------------------------------------------------------
# Generate pickle file for orientation based training.
#
# Note: This does not generate data, it generates a new pickle file given the pickle
#       file for single contour training
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import keras

import base_models.alex_net as alex_net_module
import alex_net_utils
import gabor_fits

reload(alex_net_module)
reload(alex_net_utils)
reload(gabor_fits)


def gaussian(x, mu, sig):
    # return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return np.exp(-(x - mu)**2 / (2.0 * sig**2))


def get_wrapped_angle_diff(tgt_ang, orient_arr):

    ang_arr_diff = tgt_ang - orient_arr

    ang_arr_diff_wrapped = np.array([ang if ang < 180 else ang - 360 for ang in ang_arr_diff])
    ang_arr_diff_wrapped = np.array([ang if ang > -180 else ang + 360 for ang in ang_arr_diff_wrapped])

    return ang_arr_diff_wrapped


def generate_simultaneous_training_pickle_files(l1_act_cb, g_params_dict, data_dir, k_orient_arr):
    """

        :param k_orient_arr:
        :param l1_act_cb:
        :param g_params_dict:
        :param data_dir:
        :return:
        """
    # Should exist
    data_key_single_kernel_filename = 'data_key.pickle'

    # Will be generated
    data_key_match_orient_filename = 'data_key_orient_gaussian.pickle'
    angle_diff_cutoff = 5

    filt_dirs = os.listdir(data_dir)
    filt_dirs = sorted(filt_dirs, key=lambda x: int(x.split('_')[1]))

    for filt_dir_name in filt_dirs:

        filt_dir_path = os.path.join(data_dir, filt_dir_name)
        filt_dir_pickle_file = os.path.join(filt_dir_path, data_key_single_kernel_filename)  # Single Expected Gains
        filt_idx = np.int(filt_dir_name.split('_')[1])

        print("Processing Directory {}".format(filt_dir_path))

        # Get the Single gain pickle file
        if not os.path.exists(filt_dir_pickle_file):
            raise Exception("pickle file {} not found".format(data_key_single_kernel_filename))
        with open(filt_dir_pickle_file, 'rb') as h:
            single_gain_pickle_dict_of_dict = pickle.load(h)

        # Gabor Params of Filter
        # ------------------------
        # g_params = g_params_dict[filt_idx]['gabor_params']
        g_params = {'theta_deg': k_orient_arr[filt_idx]}

        other_filt_angle_diff = get_wrapped_angle_diff(g_params['theta_deg'], k_orient_arr)
        orient_gains_arr = gaussian(other_filt_angle_diff, 0, angle_diff_cutoff)
        # replace Nans with 1
        orient_gains_arr[np.isnan(orient_gains_arr)] = 1.0


        # Debug prints and plots
        # -----------------------
        # print("Filter orientation Angle {}".format(g_params['theta_deg']))
        # print("max gain {}".format(max(gains_arr)))
        # print("min abs angle diff {}".format(min(abs(other_filt_angle_diff))))
        #
        # x = np.arange(-180, 180)
        # x_diff_wrapped = get_wrapped_angle_diff(g_params['theta_deg'], x)
        #
        # plt.figure()
        # plt.plot(x, gaussian(x_diff_wrapped, 0, angle_diff_cutoff))
        # plt.stem(k_orient_arr, gains_arr)
        #
        # for idx, angle in enumerate(k_orient_arr):
        #     print("Filter idx {}, orient {}, angle_diff {} wrapped_angle_diff {}".format(
        #         idx, angle, angle - g_params['theta_deg'], x_diff_wrapped[idx]))
        #
        # raw_input("Continue?")

        list_of_folders = sorted(single_gain_pickle_dict_of_dict)
        orient_dict_of_dict = {}

        for folder in list_of_folders:

            # print("Processing folder {}".format(folder))
            folder_dict = single_gain_pickle_dict_of_dict[folder]

            orient_folder_dict = {}

            for k, v in folder_dict.iteritems():
                # print("Key = {}".format(k))

                # # max_active
                # new_v_max = v * mask_max_active
                # new_v_max = np.maximum(new_v_max, mask_non_zero)
                # max_active_folder_dict[k] = new_v_max
                # print("New label Max active : {}".format(new_v_max))

                # # Above Threshold
                # new_v_thres = (v ** 2) * mask_thres
                # # new_v_thres = np.maximum(new_v_thres, mask_non_zero)
                # thres_folder_dict[k] = new_v_thres
                # # print("New Label above Threshold: {}".format(new_v_thres))

                # Orientation
                new_v_orient = v * orient_gains_arr
                orient_folder_dict[k] = new_v_orient
                # print("New label matching orientation: {}".format(new_v_orient))

            # ------------------------------
            # update the dict of dict
            orient_dict_of_dict[folder] = orient_folder_dict

        # --------------------------------
        # Write the pickle Files
        # data_key_max_active_pkl_file = os.path.join(filt_dir_path, data_key_max_active_filename)
        # data_key_above_thres_pkl_file = os.path.join(filt_dir_path, data_key_above_thres_filename)
        data_key_match_orient_pkl_file = os.path.join(filt_dir_path, data_key_match_orient_filename)

        # print("max active pickle file: {}".format(data_key_max_active_pkl_file))
        # print("above threshold pickle file: {}".format(data_key_above_thres_pkl_file))
        # print("Similar orientation pickle files: {}".format(data_key_match_orient_pkl_file))

        # with open(data_key_max_active_pkl_file, 'wb') as h:
        #     pickle.dump(max_active_dict_of_dict, h)

        # with open(data_key_above_thres_pkl_file, 'wb') as h:
        #     pickle.dump(thres_dict_of_dict, h)

        with open(data_key_match_orient_pkl_file, 'wb') as h:
            pickle.dump(orient_dict_of_dict, h)


if __name__ == '__main__':

    base_data_dir = './data/curved_contours/frag_11x11_full_18x18_param_search'

    # Immutable
    plt.ion()
    keras.backend.set_image_dim_ordering('th')  # Model was originally defined with Theano backend.

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    alex_net_model = alex_net_module.alex_net("trained_models/AlexNet/alexnet_weights.h5")

    feat_extract_layer_idx = alex_net_utils.get_layer_idx_by_name(alex_net_model, 'conv_1')
    feature_extract_act_cb = alex_net_utils.get_activation_cb(alex_net_model, feat_extract_layer_idx)

    feature_extract_weights = alex_net_model.layers[feat_extract_layer_idx].get_weights()[0]

    # -----------------------------------------------------------------------------------
    # Gabor Params File
    # -----------------------------------------------------------------------------------
    gabor_params_file = './data_generation/gabor_fits_feature_extract_kernels.pickle'
    if os.path.exists(gabor_params_file):
        with open(gabor_params_file, 'rb') as handle:
            gabor_params_dic = pickle.load(handle)
    else:
        raise Exception("Gabors Parameters file not found")

    # -----------------------------------------------------------------------------------
    # Find Orientations of feature extracting kernels
    # -----------------------------------------------------------------------------------
    print("Finding Filter orientations ...")

    kernel_orient_arr = np.zeros(shape=feature_extract_weights.shape[-1])  # channel dimension
    for kernel_idx in np.arange(feature_extract_weights.shape[-1]):
        kernel = feature_extract_weights[:, :, :, kernel_idx]

        kernel_orient_arr[kernel_idx] = \
            gabor_fits.get_filter_orientation(kernel, o_type='max', display_params=False)

        print("kernel {} : {}".format(kernel_idx, kernel_orient_arr[kernel_idx]))
        # raw_input()

    # -----------------------------------------------------------------------------------
    # Data Generation
    # -----------------------------------------------------------------------------------
    generate_simultaneous_training_pickle_files(
        l1_act_cb=feature_extract_act_cb,
        g_params_dict=gabor_params_dic,
        data_dir=os.path.join(base_data_dir, 'train'),
        k_orient_arr=kernel_orient_arr
    )

    generate_simultaneous_training_pickle_files(
        l1_act_cb=feature_extract_act_cb,
        g_params_dict=gabor_params_dic,
        data_dir=os.path.join(base_data_dir, 'test'),
        k_orient_arr=kernel_orient_arr
    )
