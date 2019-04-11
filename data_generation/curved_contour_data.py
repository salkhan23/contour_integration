# ---------------------------------------------------------------------------------------
# Generate Curved Contours Data Set
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import shutil

import keras

import base_models.alex_net as alex_net_module
import gabor_fits
import alex_net_utils
import image_generator_curve

reload(alex_net_module)
reload(alex_net_utils)
reload(gabor_fits)
reload(image_generator_curve)


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
    with open('.//data//neuro_data//Li2006.pickle', 'rb') as h:
        li_2006_data = pickle.load(h)

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

    with open('.//data//neuro_data//fields_1993_exp_1_beta.pickle', 'rb') as h:
        fields_1993_exp_1_beta = pickle.load(h)
    # Use averaged data
    rel_beta_rot_detectability = fields_1993_exp_1_beta['ah_djf_avg_1s_fitted_proportion_correct']

    with open('.//data//neuro_data//fields_1993_exp_3_alpha.pickle', 'rb') as h:
        fields_1993_exp_3_alpha = pickle.load(h)

    with open('.//data//neuro_data//fields_1993_exp_2_alpha_90_no_rotation.pickle', 'rb') as h:
        fields_1993_exp_2_alpha_no_rotation = pickle.load(h)

    # Use averaged data
    rel_alpha_rot_detectability = {
        0: fields_1993_exp_3_alpha['ah_djf_avg_alpha_0_proportion_correct'],
        15: fields_1993_exp_3_alpha['ah_djf_avg_alpha_15_proportion_correct'],
        30: fields_1993_exp_3_alpha['ah_djf_avg_alpha_30_proportion_correct'],
        90: fields_1993_exp_2_alpha_no_rotation['ah_djf_avg_alpha_90_no_rotation_proportion_correct']
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

    alpha_rot_arr = [0, 15, 30, 90]
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


def get_center_neuron_feature_extract_act(frag, l1_act_cb, img_size=(np.array((227, 227, 3)))):
    """

    :param frag:
    :param l1_act_cb:
    :param img_size:
    :return:
    """
    test_image = np.zeros(shape=img_size, dtype='uint')

    # Tile the fragment in the center of the image
    img_center = img_size[0] // 2
    center_frag_start = img_center - (frag.shape[0] // 2)

    test_image = alex_net_utils.tile_image(
        test_image,
        frag,
        np.array([center_frag_start, center_frag_start]).T,
        rotate=False,
        gaussian_smoothing=False,
    )

    in_image = test_image.astype('float32')
    in_image = in_image / 255.0
    in_image = np.transpose(in_image, axes=(2, 0, 1))  # Channel First
    in_image = np.expand_dims(in_image, axis=0)

    first_layer_act = np.array(l1_act_cb([in_image, 0]))
    first_layer_act = np.squeeze(first_layer_act, axis=0)  # TODO: Why are there 5 dim here

    center_neuron_act = first_layer_act[0, :, first_layer_act.shape[2] // 2, first_layer_act.shape[3] // 2]

    return center_neuron_act


def generate_data_set(
        base_dir, subfolder_name, n_img_per_set, frag, frag_params, f_tile_size, l1_act_cb, k_orient_arr,
        img_size=np.array((227, 227, 3)), delta_orient=5, thres=2.5):
    """

    :param k_orient_arr:
    :param base_dir:
    :param subfolder_name:
    :param n_img_per_set:
    :param frag:
    :param frag_params:
    :param f_tile_size:
    :param l1_act_cb:
    :param img_size:
    :param delta_orient:
    :param thres:
    :return:
    """
    if type(frag_params) is not list:
        frag_params = [frag_params]

    beta_rot_arr = np.array([0, 15, 30, 45, 60])  # main contour rotation
    alpha_rot_arr = np.array([0, 15, 30])  # fragment rotation wrt to contour direction

    data_key_max_active_filename = 'data_key_max_active.pickle'
    data_key_above_thres_filename = 'data_key_above_threshold.pickle'
    data_key_match_orient_filename = 'data_key_matching_orientation.pickle'

    data_key_dict_max_active = {}
    data_key_thres = {}
    data_key_orient = {}

    # -----------------------------------------------------------------------------------
    # For Labels
    # -----------------------------------------------------------------------------------
    # find all kernels with similar orientations
    similar_orient_k_idxs = np.where(
        (k_orient_arr >= (frag_params[0]['theta_deg'] - delta_orient)) &
        (k_orient_arr <= (frag_params[0]['theta_deg'] + delta_orient))
    )

    mask_similar_orient = np.zeros(96)
    mask_similar_orient[similar_orient_k_idxs] = 1
    mask_similar_orient = mask_similar_orient.astype(int)
    print("Fragment orientation: {}. kernels with similar orientations @ {}".format(
        frag_params[0]['theta_deg'], similar_orient_k_idxs))

    # find all kernels above threshold
    center_neuron_act = get_center_neuron_feature_extract_act(frag, l1_act_cb)

    mask_above_thres = center_neuron_act > thres
    mask_above_thres = mask_above_thres.astype(int)
    print("Fragment orientation: {}. kernels above threshold @ {}".format(
        frag_params[0]['theta_deg'], np.nonzero(mask_above_thres)))

    # Find max active kernel index
    max_active_k = np.argmax(center_neuron_act)
    mask_max_active = np.zeros(96)
    mask_max_active[max_active_k] = 1
    print("Fragment orientation: {}. Max Active kernel at @ {}".format(
        frag_params[0]['theta_deg'], max_active_k))

    # Find all kernels with activation above 0
    mask_non_zero = center_neuron_act > 0
    mask_non_zero = mask_non_zero.astype(int)
    print("Fragment orientation: {}. kernels non zero @ {}".format(
        frag_params[0]['theta_deg'], np.nonzero(mask_non_zero)))

    # ----------------------------------------------------------------------------
    # Create Filter directory
    # ----------------------------------------------------------------------------
    filt_dir = os.path.join(base_dir, subfolder_name)

    data_key_max_active_file = os.path.join(filt_dir, data_key_max_active_filename)
    data_key_above_thres_file = os.path.join(filt_dir, data_key_above_thres_filename)
    data_key_match_orient_file = os.path.join(filt_dir, data_key_match_orient_filename)

    if not os.path.exists(filt_dir):
        os.makedirs(filt_dir)

    # -----------------------------------------------------------------------------------
    #  Li-2006 Experiment 1 (Contour Length vs Gain) + Fields-1993 Experiments
    # -----------------------------------------------------------------------------------
    c_len_arr = np.array([1, 3, 5, 7, 9])
    abs_gains_arr = get_neurophysiological_data('c_len')

    for c_len in c_len_arr:
        clen_dir = 'c_len_{0}'.format(c_len)

        for b_idx, beta in enumerate(beta_rot_arr):
            beta_n_clen_dir = os.path.join(clen_dir, 'beta_{0}'.format(beta))

            for a_idx, alpha in enumerate(alpha_rot_arr):
                alpha_n_beta_n_c_len_dir = os.path.join(beta_n_clen_dir, 'alpha_{0}'.format(alpha))

                abs_destination_dir = os.path.join(filt_dir, alpha_n_beta_n_c_len_dir)
                if not os.path.exists(abs_destination_dir):
                    os.makedirs(abs_destination_dir)

                abs_gain = abs_gains_arr[c_len][alpha][beta]

                print("Generating {0} images for [c_len {1}, beta {2}, alpha {3}, forient {4}]. Gain {5}".format(
                    n_img_per_set, c_len, beta, alpha, frag_params[0]['theta_deg'], abs_gain))

                img_arr = image_generator_curve.generate_contour_images(
                    n_images=n_img_per_set,
                    frag=frag,
                    frag_params=frag_params,
                    c_len=c_len,
                    beta=beta,
                    alpha=alpha,
                    f_tile_size=f_tile_size,
                    img_size=img_size,
                    random_alpha_rot=True
                )

                set_base_filename = "clen_{0}_beta_{1}_alpha_{2}_forient_{3}".format(
                    c_len, beta, alpha, frag_params[0]['theta_deg'])

                # -----------------------------------------------------------------------
                # Generate Labels
                # -----------------------------------------------------------------------
                curr_set_max_active = {}
                curr_set_thres = {}
                curr_set_orient = {}

                for img_idx in range(img_arr.shape[0]):

                    # Save the image
                    filename = os.path.join(
                        abs_destination_dir,
                        set_base_filename + '__{0}.png'.format(img_idx)
                    )

                    plt.imsave(filename, img_arr[img_idx, ], format='PNG')

                    # labels
                    label_max_active = mask_max_active * abs_gain
                    label_max_active = np.maximum(label_max_active, mask_non_zero)
                    curr_set_max_active[filename] = label_max_active

                    label_thres = mask_above_thres * abs_gain
                    label_thres = np.maximum(label_thres, mask_non_zero)
                    curr_set_thres[filename] = label_thres

                    label_orient = mask_similar_orient * abs_gain
                    label_orient = np.maximum(label_orient, mask_non_zero)
                    curr_set_orient[filename] = label_orient

                # Add this dictionary to the dictionary of dictionaries
                data_key_dict_max_active[set_base_filename] = curr_set_max_active
                data_key_thres[set_base_filename] = curr_set_thres
                data_key_orient[set_base_filename] = curr_set_orient

    # -----------------------------------------------------------------------------------
    #  Li-2006 Experiment 2 (Fragment Spacing vs Gain) + Fields-1993 Experiments
    # -----------------------------------------------------------------------------------
    c_len = 7
    rcd_arr = np.array([1, 1.2, 1.4, 1.6, 1.9])  # Relative Colinear Distance
    abs_gains_arr = get_neurophysiological_data('f_spacing')

    for rcd in rcd_arr:

        # NOTE: relative colinear distance (RCD) used in a different way than in the original paper.
        # In the ref, RCD is defined as the ratio of distance between fragment centers to distance of some fixed
        # tile size. The stimuli generated here are based on Fields-1993 stimuli and require spacing between fragments
        # Since we are mostly interested in modeling the effects vs the actual results, we used rcd to change the
        # distance between fragments from some arbitrary reference.
        # rcd is use to see how much the fragment size increases, this is then used to increase the full tile size
        # (the actual fragment size stays the same)

        frag_size_inc = np.int(rcd * frag.shape[0]) - frag.shape[0]
        updated_f_tile_size = f_tile_size + frag_size_inc

        fspacing_dir = 'f_spacingx10_{0}'.format(int(rcd * 10))  # prevent the . from appearing in the file name

        for b_idx, beta in enumerate(beta_rot_arr):
            beta_n_fspacing_dir = os.path.join(fspacing_dir, 'beta_{0}'.format(beta))

            for a_idx, alpha in enumerate(alpha_rot_arr):
                alpha_n_beta_n_fspacing_dir = os.path.join(beta_n_fspacing_dir, 'alpha_{0}'.format(alpha))

                abs_destination_dir = os.path.join(filt_dir, alpha_n_beta_n_fspacing_dir)
                if not os.path.exists(abs_destination_dir):
                    os.makedirs(abs_destination_dir)

                abs_gain = abs_gains_arr[rcd][alpha][beta]

                print("Generating {0} images for [f_spacing {1}, full_tile {2}, beta {3}, alpha {4}, forient {5}]. "
                      "Expected Gain {6}".format(
                        n_img_per_set, rcd, updated_f_tile_size, beta, alpha, frag_params[0]['theta_deg'], abs_gain))

                img_arr = image_generator_curve.generate_contour_images(
                    n_images=n_img_per_set,
                    frag=frag,
                    frag_params=frag_params,
                    c_len=c_len,
                    beta=beta,
                    alpha=alpha,
                    f_tile_size=updated_f_tile_size,
                    img_size=img_size,
                    random_alpha_rot=True
                )

                set_base_filename = "f_spacingx10_{0}_beta_{1}_alpha_{2}_forient_{3}".format(
                    int(rcd * 10), beta, alpha, frag_params[0]['theta_deg'])

                # -----------------------------------------------------------------------
                # Generate Labels
                # -----------------------------------------------------------------------
                curr_set_max_active = {}
                curr_set_thres = {}
                curr_set_orient = {}

                for img_idx in range(img_arr.shape[0]):
                    # Save the image
                    filename = os.path.join(
                        abs_destination_dir,
                        set_base_filename + '__{0}.png'.format(img_idx)
                    )

                    plt.imsave(filename, img_arr[img_idx, ], format='PNG')

                    # labels
                    label_max_active = mask_max_active * abs_gain
                    label_max_active = np.maximum(label_max_active, mask_non_zero)
                    curr_set_max_active[filename] = label_max_active

                    label_thres = mask_above_thres * abs_gain
                    label_thres = np.maximum(label_thres, mask_non_zero)
                    curr_set_thres[filename] = label_thres

                    label_orient = mask_similar_orient * abs_gain
                    label_orient = np.maximum(label_orient, mask_non_zero)
                    curr_set_orient[filename] = label_orient

                # Add this dictionary to the dictionary of dictionaries
                data_key_dict_max_active[set_base_filename] = curr_set_max_active
                data_key_thres[set_base_filename] = curr_set_thres
                data_key_orient[set_base_filename] = curr_set_orient

    with open(data_key_max_active_file, 'wb') as h:
        pickle.dump(data_key_dict_max_active, h)

    with open(data_key_above_thres_file, 'wb') as h:
        pickle.dump(data_key_thres, h)

    with open(data_key_match_orient_file, 'wb') as h:
        pickle.dump(data_key_orient, h)


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    # A set is defined as one combination of clen/fspacing, beta, alpha.
    # There are (2*5) * 5 * 3 = 150 sets
    n_train_images_per_set = 200
    n_test_images_per_set = 20

    full_tile_size = np.array((18, 18))
    frag_tile_size = np.array((11, 11))

    # target_kernels = []
    target_kernels = [0, 2, 5, 10]

    # where the data should be stored
    data_store_dir = "./data/curved_contours/test"

    # Optimal Gabor Fits for all kernels
    gabor_params_file = './data_generation/gabor_best_fit_coloured.pickle'
    # gabor_params_file = "./data_generation/gabor_fits_feature_extract_kernels.pickle"

    # Immutable ----------------------------------
    plt.ion()
    keras.backend.set_image_dim_ordering('th')
    np.random.seed(random_seed)

    # Check that in correct directory to run the code
    cur_dir = os.getcwd()
    if cur_dir.split('/')[-1] != 'contour_integration':
        raise Exception("Script must be run from directory contour_integration."
                        "Command: data_generation.curved_contour_data.py")

    start_time = datetime.now()

    if os.path.exists(data_store_dir):
        ans = raw_input("Data Folder {} already Exists. overwrite? y/n".format(data_store_dir))
        if 'y' in ans.lower():
            shutil.rmtree(data_store_dir)
        else:
            raise SystemExit()
    os.makedirs(data_store_dir)

    if not os.path.exists(gabor_params_file):
        raise Exception("Gabor params files not found")

    # Load & Filter Gabor Params
    # --------------------------
    with open(gabor_params_file, 'rb') as handle:
        gabor_params_dict = pickle.load(handle)

    # Get params of target kernels only
    if target_kernels:
        all_kernel_idxs = gabor_params_dict.keys()
        for kernel_idx in all_kernel_idxs:
            if kernel_idx not in target_kernels:
                del (gabor_params_dict[kernel_idx])

    # Model
    # -----------------------
    alex_net_model = alex_net_module.alex_net("./trained_models/AlexNet/alexnet_weights.h5")
    feature_extract_act_cb = alex_net_utils.get_activation_cb(alex_net_model, 1)

    feature_extract_weights = alex_net_model.layers[1].get_weights()[0]

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

    # -----------------------------------------------------------------------------------
    # Generate the Data
    # -----------------------------------------------------------------------------------
    for kernel_idx in gabor_params_dict.keys():

        kernel_data_gen_start_time = datetime.now()
        g_params = gabor_params_dict[kernel_idx]['gabor_params']

        print("Generated data for kernel {0} ]...".format(kernel_idx))

        fragment = gabor_fits.get_gabor_fragment(g_params, frag_tile_size)

        sub_folder_name = 'filter_{}'.format(kernel_idx)

        print("Generating Train Data Set")
        generate_data_set(
            base_dir=os.path.join(data_store_dir, 'train'),
            subfolder_name=sub_folder_name,
            n_img_per_set=n_train_images_per_set,
            frag=fragment,
            frag_params=g_params,
            f_tile_size=full_tile_size,
            l1_act_cb=feature_extract_act_cb,
            k_orient_arr=kernel_orient_arr
        )

        print("Generating Test Data Set")
        generate_data_set(
            base_dir=os.path.join(data_store_dir, 'test'),
            subfolder_name=sub_folder_name,
            n_img_per_set=n_train_images_per_set,
            frag=fragment,
            frag_params=g_params,
            f_tile_size=full_tile_size,
            l1_act_cb=feature_extract_act_cb,
            k_orient_arr=kernel_orient_arr
        )

        print("Generating data for kernel {} took {}".format(
            kernel_idx, datetime.now() - kernel_data_gen_start_time))

    print("Total Time spent in Data Generation {}".format(datetime.now() - start_time))
