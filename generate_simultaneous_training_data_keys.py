# ---------------------------------------------------------------------------------------
# Generate Training Data for Concurrent training contour Integration kernels.
#
#  This script uses single gain files pickle files and generates , gain values for all
#  kernels. First, all kernels that activate above a threshold for the base fragment are
#  identified. Next, expected gains for all of these kernels is set to
#  neurophysiological measured gain.
#
# BASE_DATA_DIR
#    ---> train
#            ---> filter_<5>
#                   ---> c_len_<1>
#                   ---> f_spacingx10_<10>
#                   ---> data_key.pickle
#    ---> test
#            ---> filter_<5>
#                   ---> c_len_<1>
#                   ---> f_spacingx10_<10>
#                   ---> data_key.pickle
#    ---> best_fit_params.pickle
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


BASE_DATA_DIR = './data/curved_contours/frag_11x11_full_18x18_param_search'
MIN_ACT_THRESHOLD = 3.0
FULL_TRAINING_DATA_KEY_FILE_NAME = 'all_kernels_data_key.pickle'


def generate_simultaneous_training_pickle_files(l1_act_cb, g_params_dict, data_dir):

    filt_dirs = os.listdir(data_dir)

    # sort the filters list numerically
    filt_dirs = sorted(filt_dirs, key=lambda x: int(x.split('_')[1]))

    for filt_dir_name in filt_dirs:

        filt_dir_path = os.path.join(data_dir, filt_dir_name)
        filt_dir_pickle_file = os.path.join(filt_dir_path, 'data_key.pickle')  # Single Expected Gains
        filt_idx = np.int(filt_dir_name.split('_')[1])

        print("Processing Filter {}".format(filt_dir_path))

        # check that it contains a pickle file
        if not os.path.exists(filt_dir_pickle_file):
            print("filter dir {}  does contains a data_key.pickle_file. Skipping ".format({filt_dir_path}))
            continue

        # Create a test image
        # -------------------
        img_size_single_dim = 227
        frag_size_single_dim = 11

        gabor_fragment = gabor_fits.get_gabor_fragment(
            g_params_dict[filt_idx]['gabor_params'],
            (frag_size_single_dim, frag_size_single_dim)
        )

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

        # # Debug
        # plt.figure()
        # plt.imshow(gabor_fragment)
        # plt.title("Gabor Fragment for  Filter {}".format(filt_idx))
        # plt.figure()
        # plt.imshow(test_image)
        # plt.title("Test Image for  Filter {}".format(filt_idx))

        # Center Neuron activations of all filters
        # ----------------------------------------------
        # preprocess image - ideally these should be put in a single place
        in_image = test_image.astype('float32')
        in_image = in_image / 255.0
        in_image = np.transpose(in_image, axes=(2, 0, 1))  # Channel First
        in_image = np.expand_dims(in_image, axis=0)

        first_layer_act = np.array(l1_act_cb([in_image, 0]))
        first_layer_act = np.squeeze(first_layer_act, axis=0)  # TODO: Why are there 5 dim here

        center_neuron_act = first_layer_act[0, :, first_layer_act.shape[2] // 2, first_layer_act.shape[3] // 2]

        # # Debug
        # plt.figure()
        # plt.plot(center_neuron_act)
        # plt.title("Center Neuron activations for Filter {}".format(filt_idx))

        # Create a mask for all neurons above threshold
        # ---------------------------------------------
        mask_above_thres = center_neuron_act > MIN_ACT_THRESHOLD
        mask_above_thres = mask_above_thres.astype(int)

        mask_non_zero = center_neuron_act > 0
        mask_non_zero = mask_non_zero.astype(int)

        # Create a new data_key pickle file for full training
        with open(filt_dir_pickle_file, 'rb') as h:
            single_gain_pickle_dict_of_dict = pickle.load(h)

        list_of_folders = sorted(single_gain_pickle_dict_of_dict)

        new_dict_of_dict = {}

        # Old dictionary = {image_name: expected gain}
        # New Dictionary = {image_name: [expected_gain, 1, 1, 1, expected_gain, 1, 1] (96 times)}
        for folder in list_of_folders:
            # print("Processing folder {}".format(folder))
            folder_dict = single_gain_pickle_dict_of_dict[folder]

            new_folder_dict = {}
            for k, v in folder_dict.iteritems():
                new_v = v * mask_above_thres
                # new_v[new_v == 0] = 1

                # set all non zero activations (below threshold) to 1
                new_v = np.maximum(new_v, mask_non_zero)

                new_folder_dict[k] = new_v
                new_dict_of_dict[folder] = new_folder_dict

        full_training_data_key_file = os.path.join(filt_dir_path, FULL_TRAINING_DATA_KEY_FILE_NAME)
        with open(full_training_data_key_file, 'wb') as h:
            pickle.dump(new_dict_of_dict, h)


if __name__ == '__main__':

    keras.backend.set_image_dim_ordering('th')  # Model was originally defined with Theano backend.
    plt.ion()

    # Base Model
    alex_net_model = alex_net_module.alex_net("trained_models/AlexNet/alexnet_weights.h5")
    feature_extract_act_cb = alex_net_utils.get_activation_cb(alex_net_model, 1)

    # Gabor Params
    gabor_params_file = os.path.join(BASE_DATA_DIR, 'best_fit_params.pickle')
    if os.path.exists(gabor_params_file):
        with open(gabor_params_file, 'rb') as handle:
            gabor_params_dic = pickle.load(handle)
    else:
        raise Exception("Gabors Parameters file not found")

    generate_simultaneous_training_pickle_files(
        l1_act_cb=feature_extract_act_cb,
        g_params_dict=gabor_params_dic,
        data_dir=os.path.join(BASE_DATA_DIR, 'train'),
    )

    generate_simultaneous_training_pickle_files(
        l1_act_cb=feature_extract_act_cb,
        g_params_dict=gabor_params_dic,
        data_dir=os.path.join(BASE_DATA_DIR, 'test'),
    )
