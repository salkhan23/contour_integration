# -------------------------------------------------------------------------------------------------
# Generate Curved Contour Images
#
# Author: Salman Khan
# Date  : 21/04/18
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import imrotate

import keras.backend as K

import curved_contour_image_generator
import base_alex_net
import gabor_fits

reload(curved_contour_image_generator)

BASE_DIRECTORY = os.path.join(os.path.dirname(__file__), "data/curved_contours")

if not os.path.exists(BASE_DIRECTORY):
    os.makedirs(BASE_DIRECTORY)


if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 5

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

    # -----------------------------------------------------------------------------------
    # Contour Fragment
    # -----------------------------------------------------------------------------------
    fragment_gabor_params = curved_contour_image_generator.get_gabor_from_target_filter(
        tgt_filter,
        # match=[ 'x0', 'y0', 'theta_deg', 'amp', 'sigma', 'lambda1', 'psi', 'gamma']
        match=['x0', 'y0', 'theta_deg', 'amp', 'psi', 'gamma']
        # match=['theta_deg']
    )
    fragment_gabor_params['theta_deg'] = np.int(fragment_gabor_params['theta_deg'])

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
    for c_len in contour_len_arr:

        destination_dir = 'filter_{0}_orient_{1}/c_len_{2}'.format(
            tgt_filter_idx, fragment_gabor_params['theta_deg'], c_len)

        for beta in beta_rotation_arr:

            abs_destination_dir = os.path.join(BASE_DIRECTORY, destination_dir, 'beta_{0}'.format(beta))
            if not os.path.exists(abs_destination_dir):
                os.makedirs(abs_destination_dir)

            curved_contour_image_generator.generate_contour_images(
                n_images,
                fragment,
                fragment_gabor_params,
                c_len,
                beta,
                full_tile_size,
                abs_destination_dir,
                img_size=image_size
            )
