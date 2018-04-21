# -------------------------------------------------------------------------------------------------
# Generate Curved Contour Images
#
# Author: Salman Khan
# Date  : 21/04/18
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import os

import keras.backend as K

import curved_contour_image_generator

reload(curved_contour_image_generator)

BASE_DIRECTORY = os.path.join(os.path.dirname(__file__), "data/curved_contours")

if not os.path.exists(BASE_DIRECTORY):
    os.makedirs(BASE_DIRECTORY)


if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # Initialization
    tgt_filter_idx = 10
    contour_len = 9
    beta_rotation = 15
    n_images = 20

    target_directory = 'filter_{0}/c_len_{1}/beta_{2}'.format(tgt_filter_idx, contour_len, beta_rotation)
    tgt_location = os.path.join(BASE_DIRECTORY, target_directory)
    if not os.path.exists(tgt_location):
        os.makedirs(tgt_location)

    curved_contour_image_generator.generate_contour_images(
        n_images, tgt_filter_idx, contour_len, beta_rotation, tgt_location)
