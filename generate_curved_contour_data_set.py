# -------------------------------------------------------------------------------------------------
# Scripts generates sets of training images for curved contours
#
# Author: Salman Khan
# Date  : 21/04/18
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import pickle

import keras.backend as K
from keras.preprocessing.image import load_img

import image_generator_curve
from base_models import alex_net
import gabor_fits

reload(image_generator_curve)
reload(alex_net)
reload(gabor_fits)

DATA_DIRECTORY = "./data/curved_contours"

if not os.path.exists(DATA_DIRECTORY):
    os.makedirs(DATA_DIRECTORY)


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

    # -----------------------------------------------------------------------------------
    # Create the destination directory
    # -----------------------------------------------------------------------------------
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Inform existing data will be overwritten if it exists
    filt_dir = os.path.join(base_dir, "filter_{0}".format(tgt_filt_idx))

    if os.path.isdir(filt_dir):
        ans = raw_input("Generated Images will overwrite existing images. Continue? (Y/N)")

        if 'y' not in ans.lower():
            raise SystemExit()
        else:
            shutil.rmtree(filt_dir)

    # -----------------------------------------------------------------------------------
    #  Generate the Data
    # -----------------------------------------------------------------------------------
    data_key_dict = {}

    for c_len in c_len_arr:

        c_len_dir = 'c_len_{0}'.format(c_len)

        for b_idx, beta in enumerate(beta_rot_arr):

            if c_len == 1 and beta != 0:
                continue

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
            abs_gain = absolute_gain_linear[c_len] * relative_gain_curvature[beta]

            beta_dict = {}
            for img_idx in range(img_arr.shape[0]):
                filename = "c_len_{0}_beta_{1}__{2}.png".format(c_len, beta, img_idx)

                plt.imsave(
                    os.path.join(abs_destination_dir, filename),
                    img_arr[img_idx, ],
                    format='PNG'
                )

                beta_dict[os.path.join(abs_destination_dir, filename)] = abs_gain

            # Add this dictionary to the dictionary of dictionaries
            data_key_dict['c_len_{}_beta_{}'.format(c_len, beta)] = beta_dict

    # Store the dictionary of Dictionaries
    # Each entry in this dictionary is dictionary of image index and its absolute gain value
    # for a particular c_len, beta rotation value
    master_key_file_loc = os.path.join(filt_dir, 'data_key.pickle')
    with open(master_key_file_loc, 'wb') as handle:
        pickle.dump(data_key_dict, handle)


if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 10

    n_train_images = 500
    n_test_images = 100

    image_size = np.array((227, 227, 3))

    full_tile_size = np.array((17, 17))
    frag_tile_size = np.array((11, 11))

    # contour_len_arr = np.array([1, 3, 5, 7, 9])
    # beta_rotation_arr = np.array([0, 15, 30, 45, 60])

    # -----------------------------------------------------------------------------------
    # Target Kernel
    # -----------------------------------------------------------------------------------
    tgt_filter = alex_net.get_target_feature_extracting_kernel(tgt_filter_idx)
    print("Feature extracting kernel @ index {} selected.".format(tgt_filter_idx))

    # -----------------------------------------------------------------------------------
    # Contour Fragment
    # -----------------------------------------------------------------------------------
    # Gabor fit parameters target filter. Params that are not fit are defaulted.
    fragment_gabor_params = image_generator_curve.get_gabor_params_from_target_filter(
        tgt_filter,
        # match=[ 'x0', 'y0', 'theta_deg', 'amp', 'sigma', 'lambda1', 'psi', 'gamma']
        match=[ 'x0', 'y0', 'theta_deg', 'amp', 'lambda1', 'psi', 'gamma']
        # match=['x0', 'y0', 'theta_deg', 'amp', 'psi', 'gamma']
        # match=['theta_deg']
    )
    fragment_gabor_params['theta_deg'] = np.int(fragment_gabor_params['theta_deg'])

    # Generate a gabor fragment
    fragment = gabor_fits.get_gabor_fragment(fragment_gabor_params, frag_tile_size[0:2])

    # # Display the contour fragment
    # plt.figure()
    # plt.imshow(fragment)
    # plt.title("Contour Fragment")
    #
    # # Plot rotations of the fragment
    # image_generator_curve.plot_fragment_rotations(
    #     fragment, fragment_gabor_params, delta_rot=15)

    # ------------------------------------------------------------------------------------
    # Generate Images
    # ------------------------------------------------------------------------------------
    print("Generating Train Data Set")
    generate_data_set(
        base_dir=os.path.join(DATA_DIRECTORY, 'train'),
        tgt_filt_idx=tgt_filter_idx,
        n_img_per_set=n_train_images,
        frag=fragment,
        frag_params=fragment_gabor_params,
        f_tile_size=full_tile_size,
        img_size=(227, 227, 3)
    )

    print("Generating Test Data Set")
    generate_data_set(
        base_dir=os.path.join(DATA_DIRECTORY, 'test'),
        tgt_filt_idx=tgt_filter_idx,
        n_img_per_set=n_test_images,
        frag=fragment,
        frag_params=fragment_gabor_params,
        f_tile_size=full_tile_size,
        img_size=(227, 227, 3)
    )

    # -----------------------------------------------------------------------------------
    #  Debug
    # -----------------------------------------------------------------------------------
    # Load a sample image to see that it is created and stored correctly
    contour_len = 9
    inter_fragment_rotation = 15
    image_idx = 0
    test_img_loc = os.path.join(
        DATA_DIRECTORY,
        'test',
        'filter_{}'.format(tgt_filter_idx),
        "c_len_{0}".format(contour_len),
        "beta_{0}".format(inter_fragment_rotation),
        "c_len_{0}_beta_{1}__{2}.png".format(contour_len, inter_fragment_rotation, image_idx)
    )

    img = load_img(test_img_loc)
    plt.figure()
    plt.imshow(img)
    plt.title("Sample Generated Image")
