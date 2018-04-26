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
    contour_len_arr = np.array([9])
    beta_rotation_arr = np.array([15, 30, 45, 60])

    n_images = 20

    full_tile_size = np.array([17, 17])

    # ------------------------------------------------------------------------------------
    # Contour Fragment
    # ------------------------------------------------------------------------------------
    tgt_filter_idx = 10
    tgt_filter = base_alex_net.get_target_feature_extracting_kernel(tgt_filter_idx)

    tgt_filter_orientation = gabor_fits.get_filter_orientation(tgt_filter, o_type='average')
    tgt_filter_orientation = np.int(np.floor(tgt_filter_orientation))

    fragment_gabor_params = {
        'x0': 0,
        'y0': 0,
        'theta_deg': tgt_filter_orientation,
        'amp': 1,
        'sigma': 4,
        'lambda1': 8,
        'psi': 0,
        'gamma': 1
    }

    fragment = gabor_fits.get_gabor_fragment(fragment_gabor_params, tgt_filter.shape[0:2])

    # # Display the contour fragment
    # plt.figure()
    # plt.imshow(fragment)
    # plt.title("Contour Fragment")

    # ------------------------------------------------------------------------------------
    # Generate Images
    # ------------------------------------------------------------------------------------
    for c_len in contour_len_arr:

        destination_dir = 'filter_{0}_orient_{1}/c_len_{2}'.format(
            tgt_filter_idx, tgt_filter_orientation, c_len)

        for beta in beta_rotation_arr:

            abs_destination_dir = os.path.join(BASE_DIRECTORY, destination_dir, 'beta_{0}'.format(beta))
            if not os.path.exists(abs_destination_dir):
                os.makedirs(abs_destination_dir)

            curved_contour_image_generator.generate_contour_images(
                n_images, fragment, fragment_gabor_params, c_len, beta, full_tile_size, abs_destination_dir)
