# -------------------------------------------------------------------------------------------------
# Scripts generates sets of training images for curved contours
#
# Author: Salman Khan
# Date  : 21/04/18
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import keras.backend as K

import image_generator_curve
from base_models import alex_net
import gabor_fits

reload(image_generator_curve)
reload(alex_net)
reload(gabor_fits)

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data/curved_contours")

if not os.path.exists(DATA_DIRECTORY):
    os.makedirs(DATA_DIRECTORY)

if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 10

    n_images = 100

    image_size = np.array((227, 227, 3))

    full_tile_size = np.array((17, 17))
    frag_tile_size = np.array((11, 11))

    contour_len_arr = np.array([1, 3, 5, 7, 9])
    beta_rotation_arr = np.array([0, 15, 30, 45, 60])

    # -----------------------------------------------------------------------------------
    # Get neurophysiological Data
    # -----------------------------------------------------------------------------------
    # TODO: Add relative spacing data

    with open('.//data//neuro_data//Li2006.pickle', 'rb') as handle:
        Li2006Data = pickle.load(handle)

    # TODO: Get Data from Fields - 1993
    relative_gain_curvature = {
        0: 1.00,
        15: 0.98,
        30: 0.87,
        45: 0.85,
        60: 0.61
    }

    # -----------------------------------------------------------------------------------
    # Target Kernel
    # -----------------------------------------------------------------------------------
    tgt_filter = alex_net.get_target_feature_extracting_kernel(tgt_filter_idx)
    print("Feature extracting kernel @ index {} selected.".format(tgt_filter_idx))

    # -----------------------------------------------------------------------------------
    # Contour Fragment
    # -----------------------------------------------------------------------------------
    # Gabor fit parameters target filter. Params that are not fit are defaulted.
    fragment_gabor_params = image_generator_curve.get_gabor_from_target_filter(
        tgt_filter,
        # match=[ 'x0', 'y0', 'theta_deg', 'amp', 'sigma', 'lambda1', 'psi', 'gamma']
        match=['x0', 'y0', 'theta_deg', 'amp', 'psi', 'gamma']
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
    # curved_contour_image_generator.plot_fragment_rotations(
    #     fragment, fragment_gabor_params, delta_rot=15)

    # ------------------------------------------------------------------------------------
    # Generate Images
    # ------------------------------------------------------------------------------------
    # Inform existing data will be overwritten.
    base_dir = "filter_{0}".format(tgt_filter_idx)

    if os.path.isdir(os.path.join(DATA_DIRECTORY, base_dir)) and \
            os.listdir(os.path.join(DATA_DIRECTORY, base_dir)):

        ans = raw_input("Generated Images will overwrite existing images. Continue? (Y/N)")

        if 'y' not in ans.lower():
            raise SystemExit()

    data_key_dict = {}
    for c_len in contour_len_arr:

        c_len_dir = base_dir + '/c_len_{0}'.format(c_len)

        for b_idx, beta in enumerate(beta_rotation_arr):

            if c_len == 1 and beta != 0:
                continue

            beta_dir = c_len_dir + '/beta_{0}'.format(beta)
            abs_destination_dir = os.path.join(DATA_DIRECTORY, beta_dir)
            if not os.path.exists(abs_destination_dir):
                os.makedirs(abs_destination_dir)

            file_names = image_generator_curve.generate_contour_images(
                n_images,
                fragment,
                fragment_gabor_params,
                c_len,
                beta,
                full_tile_size,
                abs_destination_dir,
                img_size=image_size
            )

            c_len_idx = (c_len - 1) / 2
            abs_gain = Li2006Data['contour_len_avg_gain'][c_len_idx] * relative_gain_curvature[beta]
            # print("clen {0}, beta {1}, abs gain={2}".format(c_len, beta, abs_gain))

            beta_dict = {}
            for filename in file_names:
                beta_dict[filename] = abs_gain

            data_key_dict['c_len_{}_beta_{}'.format(c_len, beta)] = beta_dict

    # Store the data_key_dict (X,y) pairs
    pickle_file_loc = os.path.join(DATA_DIRECTORY, base_dir, 'data_key.pickle')
    with open(pickle_file_loc, 'wb') as handle:
        pickle.dump(data_key_dict, handle)
