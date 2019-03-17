# ---------------------------------------------------------------------------------------
# Given a directory with existing data (and pickle file data_key.pickle), generate
# pickle files for models that train all kernels simultaneously.
#
#  data_key.pickle =
#  single enhancement gain. Model picks the correct filter manually during training
#
#
#  This script generates three additional pickle files (similar to those generated
#  by data_generation/curved_contour_data.py). All pickle files contain gain values for
#  all 96 kernels.
#
#  [1] 'data_key_max_active.pickle': gain of max active kernel = neurophysiological gain
#  [2] 'data_key_above_threshold.pickle': gain of all neurons above a specified threshold
#       = neurophysiological gain
#  [3] 'data_key_matching_orientation': gain of all neurons with similar orientations
#       = neurophysiological gain
#
#  Additionally gains for all neurons with non zero activation is set to zero so that
#  contour integration layer learns not to enhance gains and serves as negative examples
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


def generate_simultaneous_training_pickle_files(l1_act_cb, g_params_dict, data_dir, k_orient_arr):
    """

    :param k_orient_arr:
    :param l1_act_cb:
    :param g_params_dict:
    :param data_dir:
    :return:
    """
    # should exist
    data_key_single_kernel_filename = 'data_key.pickle'

    # will be generated
    # data_key_max_active_filename = 'data_key_max_active.pickle'

    data_key_above_thres_filename = 'data_key_threshold_squared_gain.pickle'
    thres = 2.5

    # data_key_match_orient_filename = 'data_key_matching_orientation.pickle'
    # delta_orient = 5

    filt_dirs = os.listdir(data_dir)

    # sort the filters list numerically
    filt_dirs = sorted(filt_dirs, key=lambda x: int(x.split('_')[1]))

    for filt_dir_name in filt_dirs:

        filt_dir_path = os.path.join(data_dir, filt_dir_name)
        filt_dir_pickle_file = os.path.join(filt_dir_path, data_key_single_kernel_filename)  # Single Expected Gains
        filt_idx = np.int(filt_dir_name.split('_')[1])

        print("Processing Directory {}".format(filt_dir_path))

        # check that it contains a pickle file
        if not os.path.exists(filt_dir_pickle_file):
            raise Exception("pickle file {} not found".format(data_key_single_kernel_filename))

        # Gabor Params of Filter
        g_params = g_params_dict[filt_idx]['gabor_params']

        # Create a test image
        # -------------------
        img_size_single_dim = 227
        frag_size_single_dim = 11

        gabor_fragment = gabor_fits.get_gabor_fragment(
            g_params,
            (frag_size_single_dim, frag_size_single_dim)
        )

        # Generate a test image with the gabor fragment in the center
        # -----------------------------------------------------------
        test_image = np.zeros(shape=(img_size_single_dim, img_size_single_dim, 3), dtype='uint')

        # Tile the fragment in the center of the image
        img_center = img_size_single_dim // 2
        center_frag_start = img_center - (frag_size_single_dim // 2)

        test_image = alex_net_utils.tile_image(
            test_image,
            gabor_fragment,
            np.array([center_frag_start, center_frag_start]).T,
            rotate=False,
            gaussian_smoothing=False,
        )

        # preprocess image - ideally these should be put in a single place
        in_image = test_image.astype('float32')
        in_image = in_image / 255.0
        in_image = np.transpose(in_image, axes=(2, 0, 1))  # Channel First
        in_image = np.expand_dims(in_image, axis=0)

        first_layer_act = np.array(l1_act_cb([in_image, 0]))
        first_layer_act = np.squeeze(first_layer_act, axis=0)  # TODO: Why are there 5 dim here

        # Center Neuron activations of all filters
        # ----------------------------------------------
        center_neuron_act = first_layer_act[0, :, first_layer_act.shape[2] // 2, first_layer_act.shape[3] // 2]

        # # find all kernels with similar orientations
        # similar_orient_k_idxs = np.where(
        #     (k_orient_arr >= (g_params['theta_deg'] - delta_orient)) &
        #     (k_orient_arr <= (g_params['theta_deg'] + delta_orient))
        # )
        #
        # mask_similar_orient = np.zeros(96)
        # mask_similar_orient[similar_orient_k_idxs] = 1
        # mask_similar_orient = mask_similar_orient.astype(int)

        # find all kernels above threshold

        norm_center_neuron_act = 1 / (1 + np.exp(-3 * center_neuron_act + 3))
        mask_thres = norm_center_neuron_act

        # max_active_k = np.argmax(center_neuron_act)
        # mask_max_active = np.zeros(96)
        # mask_max_active[max_active_k] = 1

        mask_non_zero = center_neuron_act > 0
        mask_non_zero = mask_non_zero.astype(int)

        # create the pickle files
        # -----------------------
        # Create a new data_key pickle file for full training
        with open(filt_dir_pickle_file, 'rb') as h:
            single_gain_pickle_dict_of_dict = pickle.load(h)

        list_of_folders = sorted(single_gain_pickle_dict_of_dict)

        max_active_dict_of_dict = {}
        thres_dict_of_dict = {}
        orient_dict_of_dict = {}

        for folder in list_of_folders:
            # print("Processing folder {}".format(folder))
            folder_dict = single_gain_pickle_dict_of_dict[folder]

            max_active_folder_dict = {}
            thres_folder_dict = {}
            orient_folder_dict = {}

            for k, v in folder_dict.iteritems():
                # print("Key = {}".format(k))

                # # max_active
                # new_v_max = v * mask_max_active
                # new_v_max = np.maximum(new_v_max, mask_non_zero)
                # max_active_folder_dict[k] = new_v_max
                # print("New label Max active : {}".format(new_v_max))

                # Above Threshold
                new_v_thres = (v**2) * mask_thres
                # new_v_thres = np.maximum(new_v_thres, mask_non_zero)
                thres_folder_dict[k] = new_v_thres
                # print("New Label above Threshold: {}".format(new_v_thres))

                # # Orientation
                # new_v_orient = v * mask_similar_orient
                # new_v_orient = np.maximum(new_v_orient, mask_non_zero)
                # orient_folder_dict[k] = new_v_orient
                # print("New label matching orientation: {}".format(new_v_orient))

            # ------------------------------
            # update the dict of dict
            max_active_dict_of_dict[folder] = max_active_folder_dict
            thres_dict_of_dict[folder] = thres_folder_dict
            orient_dict_of_dict[folder] = orient_folder_dict

        # --------------------------------
        # Write the pickle Files
        # data_key_max_active_pkl_file = os.path.join(filt_dir_path, data_key_max_active_filename)
        data_key_above_thres_pkl_file = os.path.join(filt_dir_path, data_key_above_thres_filename)
        # data_key_match_orient_pkl_file = os.path.join(filt_dir_path, data_key_match_orient_filename)

        # print("max active pickle file: {}".format(data_key_max_active_pkl_file))
        # print("above threshold pickle file: {}".format(data_key_above_thres_pkl_file))
        # print("Similar orientation pickle files: {}".format(data_key_match_orient_pkl_file))

        # with open(data_key_max_active_pkl_file, 'wb') as h:
        #     pickle.dump(max_active_dict_of_dict, h)

        with open(data_key_above_thres_pkl_file, 'wb') as h:
            pickle.dump(thres_dict_of_dict, h)

        # with open(data_key_match_orient_pkl_file, 'wb') as h:
        #     pickle.dump(orient_dict_of_dict, h)


if __name__ == '__main__':

    base_data_dir = './data/curved_contours/frag_11x11_full_18x18_param_search'

    # Immutable
    plt.ion()
    keras.backend.set_image_dim_ordering('th')  # Model was originally defined with Theano backend.

    # --------------------------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------------------------
    alex_net_model = alex_net_module.alex_net("trained_models/AlexNet/alexnet_weights.h5")

    feat_extract_layer_idx = alex_net_utils.get_layer_idx_by_name(alex_net_model, 'conv_1')
    feature_extract_act_cb = alex_net_utils.get_activation_cb(alex_net_model, feat_extract_layer_idx)

    feature_extract_weights = alex_net_model.layers[feat_extract_layer_idx].get_weights()[0]

    # Gabor Params File
    # -----------------
    gabor_params_file = './data_generation/gabor_fits_feature_extract_kernels.pickle'
    if os.path.exists(gabor_params_file):
        with open(gabor_params_file, 'rb') as handle:
            gabor_params_dic = pickle.load(handle)
    else:
        raise Exception("Gabors Parameters file not found")

    # # -----------------------------------------------------------------------------------
    # # Find Orientations of feature extracting kernels
    # # -----------------------------------------------------------------------------------
    print("Finding Filter orientations ...")

    kernel_orient_arr = np.zeros(shape=feature_extract_weights.shape[-1])  # channel dimension
    # for kernel_idx in np.arange(feature_extract_weights.shape[-1]):
    #     kernel = feature_extract_weights[:, :, :, kernel_idx]
    #
    #     kernel_orient_arr[kernel_idx] = \
    #         gabor_fits.get_filter_orientation(kernel, o_type='max', display_params=False)
    #
    #     print("kernel {} : {}".format(kernel_idx, kernel_orient_arr[kernel_idx]))

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
