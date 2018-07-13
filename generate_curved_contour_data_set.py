# -------------------------------------------------------------------------------------------------
# Scripts generates sets of training images for curved contours
# Contour Fragments are generating by matching all gabor fragments from target filter
#
# Author: Salman Khan
# Date  : 21/04/18
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

import keras.backend as keras_backend

import image_generator_curve
from base_models import alex_net
import gabor_fits
import generate_curved_contour_data_set_orient_matched as generate_contour_images

reload(image_generator_curve)
reload(alex_net)
reload(gabor_fits)
reload(generate_contour_images)

DATA_DIRECTORY = "./data/curved_contours/filt_matched_frag"


if __name__ == '__main__':
    plt.ion()
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')
    start_time = datetime.datetime.now()

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 0

    n_train_images = 500
    n_test_images = 100

    image_size = np.array((227, 227, 3))

    full_tile_size = np.array((17, 17))
    frag_tile_size = np.array((11, 11))

    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

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
    best_fit_params_list = gabor_fits.find_best_fit_2d_gabor(tgt_filter)
    #
    # Create list of dictionaries of gabor params
    gabor_params_dict_list = []
    for chan_params in best_fit_params_list:

        params = {
            'x0': chan_params[0],
            'y0': chan_params[1],
            'theta_deg': np.int(chan_params[2]),
            'amp': chan_params[3],
            'sigma': 2.75,
            'lambda1': chan_params[5],
            'psi': chan_params[6],
            'gamma': chan_params[7]
        }
        gabor_params_dict_list.append(params)

    fragment = gabor_fits.get_gabor_fragment(gabor_params_dict_list, frag_tile_size)

    # Plot rotations of the fragment
    gabor_fits.plot_fragment_rotations(fragment, gabor_params_dict_list[0], delta_rot=15)

    # ------------------------------------------------------------------------------------
    # Generate Images
    # ------------------------------------------------------------------------------------
    print("Generating Train Data Set")
    generate_contour_images.generate_data_set(
        base_dir=os.path.join(DATA_DIRECTORY, 'train'),
        tgt_filt_idx=tgt_filter_idx,
        n_img_per_set=n_train_images,
        frag=fragment,
        frag_params=gabor_params_dict_list,
        f_tile_size=full_tile_size,
        img_size=(227, 227, 3)
    )

    print("Generating Test Data Set")
    generate_contour_images.generate_data_set(
        base_dir=os.path.join(DATA_DIRECTORY, 'test'),
        tgt_filt_idx=tgt_filter_idx,
        n_img_per_set=n_test_images,
        frag=fragment,
        frag_params=gabor_params_dict_list,
        f_tile_size=full_tile_size,
        img_size=(227, 227, 3)
    )

    print("Total Time {}".format(datetime.datetime.now() - start_time))

    # # -----------------------------------------------------------------------------------
    # #  Debug
    # # -----------------------------------------------------------------------------------
    # # Load a sample image to see that it is created and stored correctly
    # contour_len = 9
    # inter_fragment_rotation = 15
    # image_idx = 0
    # test_img_loc = os.path.join(
    #     DATA_DIRECTORY,
    #     'test',
    #     'filter_{}'.format(tgt_filter_idx),
    #     "c_len_{0}".format(contour_len),
    #     "beta_{0}".format(inter_fragment_rotation),
    #     "c_len_{0}_beta_{1}__{2}.png".format(contour_len, inter_fragment_rotation, image_idx)
    # )
    #
    # img = load_img(test_img_loc)
    # plt.figure()
    # plt.imshow(img)
    # plt.title("Sample Generated Image")
