# ---------------------------------------------------------------------------------------
# Find Gabor Filters for all Feature extracting Filters
#
# Note: Run from base directory
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import keras
from datetime import datetime
import os
import pickle

import alex_net_utils
import base_models.alex_net as alex_net_module
import gabor_fits
import image_generator_curve

reload(alex_net_utils)
reload(alex_net_module)
reload(gabor_fits)
reload(image_generator_curve)


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    full_tile_size = np.array((18, 18))
    frag_tile_size = np.array((11, 11))

    gabor_params_pickle_file = './data_generation/gabor_best_fit_coloured.pickle'

    # Immutable
    plt.ion()
    np.random.seed(random_seed)
    keras.backend.set_image_dim_ordering('th')
    start_time = datetime.now()

    if os.path.exists(gabor_params_pickle_file):
        ans = raw_input("{}  already exits. Enter y to overwrite.".format(gabor_params_pickle_file))
        if 'y' in ans.lower():
            os.remove(gabor_params_pickle_file)
        else:
            raise SystemExit()

    # -----------------------------------------------------------------------------------
    # Contour Integration Model
    # -----------------------------------------------------------------------------------
    print("Loading Model {}".format('*' * 80))

    # Model only needed for feature extracting kernels
    alex_net_model = alex_net_module.alex_net("./trained_models/AlexNet/alexnet_weights.h5")
    feature_extract_act_cb = alex_net_utils.get_activation_cb(alex_net_model, 1)

    feat_extract_kernels, _ = alex_net_model.layers[1].get_weights()
    n_kernels = feat_extract_kernels.shape[-1]

    saved_gabor_param_dict = {}

    # -----------------------------------------------------------------------------------
    #  Best Fit Parameters
    # -----------------------------------------------------------------------------------
    for k_idx in np.arange(n_kernels):
        print("Finding Best Fit Parameters for kernel {}".format(k_idx))

        tgt_kernel = feat_extract_kernels[:, :, :, k_idx]

        # Find the Gabor Fits for each Channel independently
        gabor_params = gabor_fits.find_best_fit_2d_gabor(tgt_kernel, verbose=1)

        # Change the parameters into a list of Gabor Param dictionaries
        gabor_params_dict_list = []

        for chan_idx, chan_params in enumerate(gabor_params):
            if chan_params is not None:
                params = {
                    'x0': chan_params[0],
                    'y0': chan_params[1],
                    'theta_deg': chan_params[2],
                    'amp': chan_params[3],
                    'sigma': chan_params[4],
                    'lambda1': chan_params[5],
                    'psi': chan_params[6],
                    'gamma': chan_params[7]
                }

                gabor_params_dict_list.append(params)

        # Error Checking
        if len(gabor_params_dict_list) != 3:
            print("All Gabor Params for filter {} not found".format(k_idx))
            continue

        fragment = gabor_fits.get_gabor_fragment(gabor_params_dict_list, frag_tile_size)

        # Get most responsive kernel and activation value
        max_active_k, max_act_value = alex_net_utils.find_most_active_l1_kernel_index(
            fragment, feature_extract_act_cb, plot=False, tgt_filt=k_idx)

        if max_active_k != k_idx:
            print("Generated Fragment not optimal for kernel {}. Max Active kernel {}. ".format(k_idx, max_active_k))
            continue
        else:
            print("Fragment for filter {} found. Max. activation {}".format(k_idx, max_act_value))

        # Plot the kernel and the generated fragment
        fig, ax_array = plt.subplots(1, 2)
        disp_tgt_filter = (tgt_kernel - tgt_kernel.min()) / (tgt_kernel.max() - tgt_kernel.min())
        ax_array[0].imshow(disp_tgt_filter)
        ax_array[0].set_title("Kernel {}".format(k_idx))
        ax_array[1].imshow(fragment)
        ax_array[1].set_title("Generated Fragment")

        # Plot Channel-wise Gabor Fits
        gabor_fits.plot_kernel_and_best_fit_gabors(tgt_kernel, k_idx, gabor_params)

        # See if Gabor Fits are good for Contour Image Generation
        # --------------------------------------------------------
        gabor_fits.plot_fragment_rotations(fragment, gabor_params_dict_list)

        # Create a Test image
        c_len = 9
        beta = 15
        alpha = 0

        img_arr = image_generator_curve.generate_contour_images(
            n_images=1,
            frag=fragment,
            frag_params=gabor_params_dict_list,
            c_len=c_len,
            beta=beta,
            alpha=alpha,
            f_tile_size=full_tile_size,
            img_size=np.array((227, 227, 3)),
            random_alpha_rot=True
        )

        plt.figure()
        plt.imshow(img_arr[0, ])
        plt.title("Test Image, Filter {}, c_len={}, beta={}, alpha={}".format(k_idx, c_len, beta, alpha))

        ans = raw_input("Add Gabor parameter to list? y/n")
        if 'y' in ans.lower():
            print("Storing parameters for kernel {}".format(k_idx))
            saved_gabor_param_dict[k_idx] = {
                "gabor_params": gabor_params_dict_list,
                "max_act": max_act_value
            }

        plt.close('all')

    print("Fined processing all Feature Extracting kernels.")
    print("Parameters of {} kernels stored.".format(len(saved_gabor_param_dict)))

    # Store the pickle File
    with open(gabor_params_pickle_file, 'wb') as handle:
        pickle.dump(saved_gabor_param_dict, handle)
