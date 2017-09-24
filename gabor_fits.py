# -------------------------------------------------------------------------------------------------
#  Find the parameters of the best fit 2D Gabor Filter for a given l1 layer kernel. Bet fit Gabors
#  for each channel are found independently.
#
# Author: Salman Khan
# Date  : 17/09/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

import keras.backend as K

import alex_net_cont_int_models as cont_int_models
import alex_net_utils

reload(cont_int_models)
reload(alex_net_utils)

np.random.seed(7)  # Set the random seed for reproducibility


def gabor_2d((x, y), x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma):
    """
    2D Spatial Gabor Filter (Real Component only).

    Ref: [1] J. Movellan - 2002 - Tutorial on Gabor Filters
         [2] https://en.wikipedia.org/wiki/Gabor_filter

    Note: Compared to the definitions in the references above. The theta (orientation of the gaussian envelope)
    angle is reversed (compare x_prime and y_prime definitions with reference). In the references, theta rotates
    in the clockwise direction. Here we change it to rotate in the more conventional counter clockwise direction
    (theta=0, corresponds to the x axis.). Note also that this is the angle of the gaussian envelop with
    is orthogonal to the direction of the stripes.

    :param x0: x-coordinate of center of Gaussian
    :param y0: y-coordinate of center of Gaussian
    :param theta_deg: Orientation of the Gaussian or the orientation of the normal to the sinusoid (carrier)
        component of the Gabor. It is specified in degrees to allow curve_fit greater resolution when finding the
        optimal parameters
    :param amp: Amplitude of the Gabor Filter
    :param sigma: width of the Gaussian (envelope) component
    :param lambda1: Wavelength of the sinusoid (carrier) component
    :param psi: phase offset of the sinusoid component
    :param gamma: Scale ratio of the x vs y spatial extent of the Gaussian envelope.

    :return: 1D vector of 2D spatial gabor function over (x, y). Note it needs to be reshaped to get the 2D
        version. It is done this way because curve fit function, expect a single vector of inputs to optimize over
    """
    sigma = np.float(sigma)

    theta = theta_deg * np.pi / 180.0

    x_prime = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
    y_prime = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

    out = amp * np.exp(-(x_prime ** 2 + (gamma ** 2 * y_prime ** 2)) / (2 * sigma ** 2)) * \
        np.cos(2 * np.pi * x_prime / lambda1 + psi)

    # print(x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma)

    return out.ravel()


def find_best_fit_2d_gabor(kernel):
    """
    Find the bit fit parameters of a 2D gabor for each input channel of kernel.

    :param kernel: Alexnet l1 kernel

    :return: list of best fit parameters for each channel of kernel. Format: [x, y, chan]
    """
    n_channels = kernel.shape[-1]

    x_arr = np.arange(-0.5, 0.5, 1 / np.float(kernel.shape[0]))
    y_arr = np.copy(x_arr)

    xx, yy = np.meshgrid(x_arr, y_arr)

    opt_params_list = []

    for chan_idx in range(n_channels):

        opt_params_found = False

        theta = 0

        # gabor_2d((x, y), x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma):
        bounds = ([-1, -1,   0, -np.inf, 0.1,      0,         0, -2],
                  [ 1,  1, 180,  np.inf,   4, np.inf, 2 * np.pi,  6])

        while not opt_params_found:

            p0 = [0, 0, theta, 1, 1, 2, 0, 1]

            try:
                popt, pcov = optimize.curve_fit(
                    gabor_2d, (xx, yy), kernel[:, :, chan_idx].ravel(), p0=p0, bounds=bounds)

                # 1 SD of error in estimate
                one_sd_error = np.sqrt(np.diag(pcov))

                # Check that error in the estimate is reasonable
                if one_sd_error[2] <= 1.0:

                    opt_params_found = True
                    opt_params_list.append(popt)

                    print("Optimal Parameter for channel %d:" % chan_idx)
                    print( "(x0,y0)=(%0.2f, %0.2f), theta=%0.2f, A=%0.2f, sigma=%0.2f, lambda=%0.2f, "
                           "psi=%0.2f, gamma=%0.2f"
                           % (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]))

                    print("Err: (x0,y0)=(%0.2f, %0.2f), theta=%0.2f, A=%0.2f, sigma=%0.2f, "
                          "lambda=%0.2f, psi=%0.2f, gamma=%0.2f"
                          % (one_sd_error[0], one_sd_error[1], one_sd_error[2], one_sd_error[3], one_sd_error[4],
                             one_sd_error[5], one_sd_error[6], one_sd_error[7]))
                else:
                    theta += 10

            except RuntimeError:
                theta += 10

                if theta == 180:
                    print("Optimal parameters could not be found")
                    opt_params_found = True
                    opt_params_list.append(None)

            except ValueError:
                print("Optimal parameters could not be found")
                opt_params_found = True
                opt_params_list.append(None)

    return opt_params_list


def plot_kernel_and_best_fit_gabors(kernel, kernel_idx, fitted_gabors_params):
    """

    :param kernel:
    :param kernel_idx: Index of the kernel (only fir title)
    :param fitted_gabors_params: list of fitted parameters for each channel of kernel

    :return: None
    """
    n_channels = kernel.shape[-1]

    x_arr = np.arange(-0.5, 0.5, 1 / np.float(kernel.shape[0]))
    y_arr = np.copy(x_arr)
    xx, yy = np.meshgrid(x_arr, y_arr)

    # Higher resolution display
    x2_arr = np.arange(-0.5, 0.5, 1 / np.float((kernel.shape[0]) + 100))
    y2_arr = np.copy(x2_arr)
    xx2, yy2 = np.meshgrid(x2_arr, y2_arr)

    f = plt.figure()

    # Normalize the kernel to [0, 1] to display it properly
    display_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

    for chan_idx in range(n_channels):

        # Plot the kernel
        f.add_subplot(n_channels, 3, (chan_idx * 3) + 1)
        plt.imshow(display_kernel[:, :, chan_idx], cmap='seismic')
        plt.title(r"$Chan=%d $" % chan_idx)

        if np.any(fitted_gabors_params[chan_idx]):  # if it is not none

            x0, y0, theta, amp, sigma, lambda1, psi, gamma = fitted_gabors_params[chan_idx]

            # Fitted gabor - same resolution (with which fit was done)
            f.add_subplot(n_channels, 3, (chan_idx * 3) + 2)
            fitted_gabor = gabor_2d((xx, yy), x0, y0, theta, amp, sigma, lambda1, psi, gamma)
            fitted_gabor = fitted_gabor.reshape((x_arr.shape[0], y_arr.shape[0]))
            display_gabor = (fitted_gabor - fitted_gabor.min()) / (fitted_gabor.max() - fitted_gabor.min())
            plt.imshow(display_gabor, cmap='seismic')
            plt.title(r"$\theta_{opt}=%0.2f$" % theta)

            # # Fitted gabor - higher resolution
            f.add_subplot(n_channels, 3, (chan_idx * 3) + 3)
            fitted_gabor = gabor_2d((xx2, yy2), x0, y0, theta, amp, sigma, lambda1, psi, gamma)
            fitted_gabor = fitted_gabor.reshape((x2_arr.shape[0], y2_arr.shape[0]))
            display_gabor = (fitted_gabor - fitted_gabor.min()) / (fitted_gabor.max() - fitted_gabor.min())
            plt.imshow(display_gabor, cmap='seismic')

    f.suptitle("Target Filter Index %d" % kernel_idx)


if __name__ == "__main__":
    plt.ion()
    K.clear_session()

    # 1. Build the model
    # ---------------------------------------------------------------------
    K.set_image_dim_ordering('th')
    print("Building Contour Integration Model...")

    # Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        n=25,
        activation='relu'
    )
    # contour_integration_model.summary()

    l1_weights = K.eval(contour_integration_model.layers[1].weights[0])

    # 2. Select the target L1 filter to find the
    # ---------------------------------------------------------------------
    # # A. For a particular target filter
    # tgt_filter_idx = 54
    #
    # tgt_filter = l1_weights[:, :, :, tgt_filter_idx]
    # optimal_params = find_best_fit_2d_gabor(tgt_filter)
    # plot_kernel_and_best_fit_gabors(tgt_filter, tgt_filter_idx, optimal_params)

    # B. For a rage of target filters
    for tgt_filter_idx in np.arange(76, 96):
        tgt_filter = l1_weights[:, :, :, tgt_filter_idx]

        optimal_params = find_best_fit_2d_gabor(tgt_filter)
        plot_kernel_and_best_fit_gabors(tgt_filter, tgt_filter_idx, optimal_params)
