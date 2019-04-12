# ---------------------------------------------------------------------------------------
# Given a directory with existing data (and pickle file data_key.pickle), generate
# pickle files for models that train all kernels simultaneously.
#
#  data_key.pickle =
#  single enhancement gain. Model picks the correct filter manually during training
#
#  All pickle files contain gain values for all 96 kernels.
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


def sigmoid(x):
    return np.exp(x)/(np.exp(x) + 1)


def generate_simultaneous_training_pickle_files(
        l1_act_cb, g_params_dict, data_dir, sigmoid_center=0.5, sigmoid_scale=2):
    """

    :param sigmoid_scale: The slope of the sigmoid. Large values mean gain climbs to full value faster
    :param sigmoid_center: Specifies what percentage of the max gain of a kernel will be the center of
          its gain curve. At this value the gain will be 0.5
    :param l1_act_cb:
    :param g_params_dict:
    :param data_dir:

    :return:
    """

    # Should exist. This is the file used to generate the other pickle file
    # exist_data_key_filename = 'data_key.pickle'
    exist_data_key_filename = 'data_key_max_active.pickle'

    # Will be generated
    generate_data_key_filename = 'data_key_sigmoid_center_{}MaxAct_gain_{}_preprocessing_divide255.pickle'.format(
        sigmoid_center, sigmoid_scale)
    print("Generated pickle files name: {}".format(generate_data_key_filename))

    # Get max activation, stored in g_params_dict (divide 255 preprocessing)
    max_act = np.zeros(96)
    for k, v in g_params_dict.iteritems():
        max_act[k] = g_params_dict[k]['max_act']

    max_act = np.array(max_act)

    filt_dirs = os.listdir(data_dir)
    filt_dirs = sorted(filt_dirs, key=lambda x: int(x.split('_')[1]))  # sort the filters list numerically

    # Do not remove for debug
    prev_folder = None

    for filt_dir_name in filt_dirs:

        filt_dir_path = os.path.join(data_dir, filt_dir_name)
        filt_dir_pickle_file = os.path.join(filt_dir_path, exist_data_key_filename)  # Single Expected Gains
        filt_idx = np.int(filt_dir_name.split('_')[1])

        print("Processing Directory {}".format(filt_dir_path))

        # check that it contains a pickle file
        if not os.path.exists(filt_dir_pickle_file):
            raise Exception("pickle file {} not found".format(exist_data_key_filename))

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
        # TODO: include preprocessing function
        in_image = test_image.astype('float32')
        in_image = in_image / 255.0
        in_image = np.transpose(in_image, axes=(2, 0, 1))  # Channel First
        in_image = np.expand_dims(in_image, axis=0)

        first_layer_act = np.array(l1_act_cb([in_image, 0]))
        first_layer_act = np.squeeze(first_layer_act, axis=0)  # TODO: Why are there 5 dim here

        # Center Neuron activations of all filters
        # ------------------------------------------
        center_neuron_act = first_layer_act[0, :, first_layer_act.shape[2] // 2, first_layer_act.shape[3] // 2]

        # # Debug plots
        # plt.figure()
        # plt.plot(avg_acts, label='avg_act')
        # plt.plot(center_neuron_act, label='for image')
        # plt.legend()
        # plt.title("{}".format(filt_dir_name))
        # raw_input("Continue ?")

        # Find all kernels above threshold
        # --------------------------------

        # Apply sigmoid on kernels whose max act is known
        defined_max_act_k_idx_arr = g_params_dict.keys()

        sigmoid_gains = np.zeros(96)
        for k_idx in defined_max_act_k_idx_arr:

            # any neuron with is at half activation will get a gain of 0.5
            sigmoid_gains[k_idx] = sigmoid(sigmoid_scale * (center_neuron_act[k_idx] - max_act[k_idx] * sigmoid_center))

        # Set the gain of all non zero kernels with unknown max act to 1
        unknown_max_act_k_idx_arr = [x for x in np.arange(96) if x not in gabor_params_dic.keys()]

        # identify all kernels who's max act is unknown, but have a non zero center activation
        unknown_max_act_non_zero = np.zeros(96)
        for k_idx in unknown_max_act_k_idx_arr:
            if center_neuron_act[k_idx] > 0:
                unknown_max_act_non_zero[k_idx] = 1

        # create the pickle files
        # -----------------------
        # Create a new data_key pickle file for full training
        with open(filt_dir_pickle_file, 'rb') as h:
            single_gain_pickle_dict_of_dict = pickle.load(h)

        list_of_folders = sorted(single_gain_pickle_dict_of_dict)

        thres_dict_of_dict = {}

        for folder in list_of_folders:
            # print("Processing folder {}".format(folder))
            folder_dict = single_gain_pickle_dict_of_dict[folder]

            thres_folder_dict = {}

            for k, v in folder_dict.iteritems():

                v = max(v)
                # print("Key = {}".format(k))

                # Above Threshold
                new_v_thres = v * sigmoid_gains
                new_v_thres = np.maximum(new_v_thres, unknown_max_act_non_zero)

                thres_folder_dict[k] = new_v_thres
                # print("New Label above Threshold: {}".format(new_v_thres))

                # Debug
                # if v >= 2 and (prev_folder != filt_dir_name):
                #     plt.figure()
                #     prev_folder = filt_dir_name
                #
                #     plt.stem(max_act * sigmoid_center, label='{} of max act (Sigmoid Center)'.format(sigmoid_center))
                #     plt.plot(center_neuron_act, label='raw act to gabor frag', color='g')
                #     plt.plot(new_v_thres, label='gain', color='r')
                #
                #     plt.legend()
                #     plt.grid()
                #     plt.title("Gain={}. filt_dir_name {}".format(v, filt_dir_name))
                #
                #     print("sigmoid gain of filter {} (target) = {}".format(filt_idx, sigmoid_gains[filt_idx]))
                #
                #     import pdb
                #     pdb.set_trace()
                #
                #     plt.close()

            # ------------------------------
            # update the dict of dict
            thres_dict_of_dict[folder] = thres_folder_dict

        # --------------------------------
        # Write the pickle Files
        data_key_above_thres_pkl_file = os.path.join(filt_dir_path, generate_data_key_filename)

        with open(data_key_above_thres_pkl_file, 'wb') as h:
            pickle.dump(thres_dict_of_dict, h)


if __name__ == '__main__':
    # --------------------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------------------
    # base_data_dir = './data/curved_contours/frag_11x11_full_18x18_param_search'
    # gabor_params_file = './data_generation/gabor_fits_feature_extract_kernels.pickle'

    base_data_dir = './data/curved_contours/coloured_gabors_dataset'
    gabor_params_file = './data_generation/gabor_best_fit_coloured.pickle'

    # Immutable
    # ---------
    plt.ion()
    keras.backend.set_image_dim_ordering('th')  # Model was originally defined with Theano backend.

    if os.path.exists(gabor_params_file):
        with open(gabor_params_file, 'rb') as handle:
            gabor_params_dic = pickle.load(handle)
    else:
        raise Exception("Gabors Parameters file not found")

    # --------------------------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------------------------
    alex_net_model = alex_net_module.alex_net("trained_models/AlexNet/alexnet_weights.h5")

    feat_extract_layer_idx = alex_net_utils.get_layer_idx_by_name(alex_net_model, 'conv_1')
    feature_extract_act_cb = alex_net_utils.get_activation_cb(alex_net_model, feat_extract_layer_idx)

    feature_extract_weights = alex_net_model.layers[feat_extract_layer_idx].get_weights()[0]

    # -----------------------------------------------------------------------------------
    # Data Generation
    # -----------------------------------------------------------------------------------
    print("Starting Pickle Files Generation ...")

    generate_simultaneous_training_pickle_files(
        l1_act_cb=feature_extract_act_cb,
        g_params_dict=gabor_params_dic,
        data_dir=os.path.join(base_data_dir, 'train'),
    )

    generate_simultaneous_training_pickle_files(
        l1_act_cb=feature_extract_act_cb,
        g_params_dict=gabor_params_dic,
        data_dir=os.path.join(base_data_dir, 'test'),
    )
