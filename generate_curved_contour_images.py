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

import curved_contour_image_generator
import base_alex_net
import gabor_fits

reload(curved_contour_image_generator)
reload(base_alex_net)
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

    n_images = 20

    image_size = np.array((227, 227, 3))

    full_tile_size = np.array((17, 17))
    frag_tile_size = np.array((11, 11))

    contour_len_arr = np.array([9])
    beta_rotation_arr = np.array([15, 30, 45, 60])

    # -----------------------------------------------------------------------------------
    # Target Kernel
    # -----------------------------------------------------------------------------------
    tgt_filter = base_alex_net.get_target_feature_extracting_kernel(tgt_filter_idx)
    print("Feature extracting kernel @ index {} selected.".format(tgt_filter_idx))

    # -----------------------------------------------------------------------------------
    # Contour Fragment
    # -----------------------------------------------------------------------------------
    # Check to confirm data will be overwritten
    base_dir = "filter_{0}".format(tgt_filter_idx)

    if os.path.isdir(os.path.join(DATA_DIRECTORY, base_dir)) and \
       os.listdir(os.path.join(DATA_DIRECTORY, base_dir)):

        ans = raw_input("Generated Images will overwrite existing images. Continue? (Y/N)")

        if 'y' not in ans.lower():
            raise SystemExit()

    # Gabor fit parameters derived from the Target Filter
    fragment_gabor_params = curved_contour_image_generator.get_gabor_from_target_filter(
        tgt_filter,
        # match=[ 'x0', 'y0', 'theta_deg', 'amp', 'sigma', 'lambda1', 'psi', 'gamma']
        match=['x0', 'y0', 'theta_deg', 'amp', 'psi', 'gamma']
        # match=['theta_deg']
    )
    fragment_gabor_params['theta_deg'] = np.int(fragment_gabor_params['theta_deg'])

    # Generate a gabor fragment
    fragment = gabor_fits.get_gabor_fragment(
        fragment_gabor_params, frag_tile_size[0:2])

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

    # Temp for now just choose some arbitrary gain enhancement.
    # TODO: get actual enhancement gain, absolute for straight contour from Li-2006
    # TODO: relative for curved contours from Fields 1993
    enhancement_gain_arr = np.array([2.0, 1.6, 1.0, 1.0])

    train_dict = {}

    for c_len in contour_len_arr:

        destination_dir = 'filter_{0}/c_len_{1}'.format(tgt_filter_idx, c_len)

        for b_idx, beta in enumerate(beta_rotation_arr):

            abs_destination_dir = os.path.join(DATA_DIRECTORY, destination_dir, 'beta_{0}'.format(beta))
            if not os.path.exists(abs_destination_dir):
                os.makedirs(abs_destination_dir)

            file_names = curved_contour_image_generator.generate_contour_images(
                n_images,
                fragment,
                fragment_gabor_params,
                c_len,
                beta,
                full_tile_size,
                abs_destination_dir,
                img_size=image_size
            )

            for filename in file_names:
                train_dict[filename] = enhancement_gain_arr[b_idx]

    pickle_file_loc = 'filter_{}'.format(tgt_filter_idx)
    abs_destination_dir = os.path.join(DATA_DIRECTORY, pickle_file_loc, 'trainKey.pickle')

    with open(abs_destination_dir, 'wb') as handle:
        pickle.dump(train_dict, handle)
