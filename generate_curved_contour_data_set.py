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
import pickle

import keras.backend as keras_backend

import image_generator_curve

import gabor_fits
import generate_curved_contour_data_set_orient_matched as generate_contour_images
import contour_integration_models.alex_net.model_3d as contour_integration_model_3d
import alex_net_utils

reload(image_generator_curve)
reload(gabor_fits)
reload(generate_contour_images)
reload(contour_integration_model_3d)
reload(alex_net_utils)

DATA_DIRECTORY = "./data/curved_contours/filter_matched"

if __name__ == '__main__':
    plt.ion()
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')
    start_time = datetime.datetime.now()

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    cont_int_kernel_arr = [19, 10, 20, 21, 48, 49, 51, 59, 62, 64, 65, 68]

    n_train_images = 1000
    n_test_images = 100

    image_size = np.array((227, 227, 3))

    full_tile_size = np.array((17, 17))
    frag_tile_size = np.array((11, 11))

    # -----------------------------------------------------------------------------------
    # Contour Integration Model
    # -----------------------------------------------------------------------------------
    print("Building Contour Integration Model...")

    cont_int_model = contour_integration_model_3d.build_contour_integration_model(
        tgt_filt_idx=0,
        rf_size=25,
        inner_leaky_relu_alpha=0.7,
        outer_leaky_relu_alpha=0.94,
        l1_reg_loss_weight=0.01
    )
    # cont_int_model.summary()

    feat_extract_callback = alex_net_utils.get_activation_cb(cont_int_model, 1)
    feat_extract_kernels, _ = cont_int_model.layers[1].get_weights()

    # -----------------------------------------------------------------------------------
    # Contour Fragment
    # -----------------------------------------------------------------------------------
    for tgt_filter_idx in cont_int_kernel_arr:

        tgt_filter = feat_extract_kernels[:, :, :, tgt_filter_idx]
        print("Generating Data for Contour Integration kernel @ {0}".format(tgt_filter_idx))
        data_gen_start_time = datetime.datetime.now()

        best_fit_params_list = gabor_fits.find_best_fit_2d_gabor(tgt_filter)

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

        # Get most responsive kernel and activation value
        max_active_kernel, max_act_value = alex_net_utils.find_most_active_l1_kernel_index(
            fragment, feat_extract_callback)
        print("Most active Kernel @ {}, Activation Value {}".format(
            max_active_kernel, max_act_value))

        # raw_input("Continue?")

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
            img_size=image_size
        )

        print("Generating Test Data Set")
        generate_contour_images.generate_data_set(
            base_dir=os.path.join(DATA_DIRECTORY, 'test'),
            tgt_filt_idx=tgt_filter_idx,
            n_img_per_set=n_test_images,
            frag=fragment,
            frag_params=gabor_params_dict_list,
            f_tile_size=full_tile_size,
            img_size=image_size
        )

        print("Data generation for kernel @ {0} took {1}".format(
            tgt_filter_idx, datetime.datetime.now() - data_gen_start_time))

        # Store the best fit params
        best_fit_params_store_file = os.path.join(DATA_DIRECTORY, 'best_fit_params.pickle')
        if os.path.exists(best_fit_params_store_file):
            with open(best_fit_params_store_file, 'rb') as f_id:
                best_fit_params_dict = pickle.load(f_id)
        else:
            best_fit_params_dict = {}

        with open(best_fit_params_store_file, 'wb') as f_id:
            best_fit_params_dict[tgt_filter_idx] = gabor_params_dict_list
            pickle.dump(best_fit_params_dict, f_id)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("Total Run Time {}".format(datetime.datetime.now() - start_time))
