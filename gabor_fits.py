# -------------------------------------------------------------------------------------------------
#  Find the parameters of the best fit 2D Gabor Filter for a given l1 layer of alex_net kernel.
#  One of the parameters that is returned is the orientation of the Gabor.
#
#  Once this parameter is known it can be used to construct various contour test stimuli for the
#  target neuron
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


def gabor_2d((x, y), x0, y0, amp, sigma, theta, lambda1, psi, gamma):
    """
    2D spatial Gabor Filter (Real Component only)

    Ref: [1] J. Movellan - 2002 - Tutorial on Gabor Filters
         [2] https://en.wikipedia.org/wiki/Gabor_filter

    Note: Compared to the definitions in the reference above. Orientation of theta is reversed. This is because
    theta as defined in the references rotates clockwise. This modification makes the gabor rotate in the
    more conventional counter clockwise direction. theta=0, corresponds to the x axis.

    :param x0: x-coordinate of center of Gaussian
    :param y0: y-coordinate of center of Gaussian
    :param amp: Amplitude of the Gabor Filter
    :param sigma: width of the Gaussian component (Envelope) of the Gabor Function
    :param theta: Orientation of the Gaussian or the orientation of the normal to the sinusoid (carrier)
         component of the Gabor (Radians)
    :param lambda1: Wavelength of the sinusoid component
    :param psi: phase offset of the sinusoid component
    :param gamma: Scale ratio of the extend of spatial spread in the x -direction compared to the y
        direction.

    :return: 2D spatial gabor function over (x, y)
    """

    sigma = np.float(sigma)

    x_prime = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
    y_prime = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

    out = amp * np.exp(-(x_prime ** 2 + (gamma ** 2 * y_prime ** 2)) / (2 * sigma ** 2)) * \
        np.cos(2 * np.pi * x_prime / lambda1 + psi)

    return out.ravel()


def find_best_fit_2d_gabor(kernel):
    """
    :param kernel: [x, y, chan]

    :return: array of optimal parameters for the gabor_2d function for each input channel
    """

    half_kernel_len = kernel.shape[0] // 2
    n_channels = kernel.shape[-1]

    x_arr = np.arange(-half_kernel_len, half_kernel_len + 1, 1)
    y_arr = np.copy(x_arr)

    xx, yy = np.meshgrid(x_arr, y_arr)

    opt_params_list = []

    for chan_idx in range(n_channels):
        popt, pcov = optimize.curve_fit(gabor_2d, (xx, yy), kernel[:, :, chan_idx].ravel())

        x0, y0, amp, sigma, theta, lambda1, psi, gamma = popt

        print("best fit params for(x0,y0)=(%0.4f,%0.4f), A=%0.4f, sigma=%0.4f, theta=%0.2f,"
              "lambda=%0.4f, psi=%0.4f, gamma=%0.4f"
              % (x0, y0, amp, sigma, theta * 180.0 / np.pi, lambda1, psi, gamma))

        print("1 SD of fits %s" % np.sqrt(np.diag(pcov)))

        opt_params_list.append(popt)

    return np.array(opt_params_list)


def plot_kernel_and_best_fit_gabor(gabor_params, kernel_idx, kernel, resolution=1):
    """

    :param kernel_idx:
    :param gabor_params:
    :param kernel:
    :param resolution:
    :return:
    """
    half_kernel_len = kernel.shape[0] // 2
    n_channels = kernel.shape[-1]

    x_arr = np.arange(-half_kernel_len, half_kernel_len + 1, resolution)
    y_arr = np.copy(x_arr)

    xx, yy = np.meshgrid(x_arr, y_arr)

    # Normalize input kernel for display
    display_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

    f = plt.figure()
    f.suptitle("Kernel @ index %d" % kernel_idx)

    for chan_idx in range(n_channels):

        f.add_subplot(2, n_channels, chan_idx + 1)
        plt.imshow(display_kernel[:, :, chan_idx], cmap='seismic')
        plt.title("channel %d" % chan_idx)

        x0, y0, amp, sigma, theta, lambda1, psi, gamma = gabor_params[chan_idx]
        fitted_gabor = gabor_2d((xx, yy), x0, y0, amp, sigma, theta, lambda1, psi, gamma)
        fitted_gabor = fitted_gabor.reshape(x_arr.shape[0], x_arr.shape[0])

        display_gabor = (fitted_gabor - fitted_gabor.min()) / (fitted_gabor.max() - fitted_gabor.min())
        f.add_subplot(2, n_channels, n_channels + chan_idx + 1)
        plt.imshow(display_gabor, cmap='seismic')
        plt.title("theta= %0.2f" % (theta * 180.0 / np.pi))


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

    # 2. L1 kernel index to fit
    # ---------------------------------------------------------------------
    tgt_filter_idx = 54

    # 3. Find the best fit Gabor
    # ---------------------------------------------------------------------
    tgt_filter = l1_weights[:, :, :, tgt_filter_idx]

    gabor_fit_params = find_best_fit_2d_gabor(tgt_filter)
    plot_kernel_and_best_fit_gabor(gabor_fit_params, tgt_filter_idx, tgt_filter, resolution=0.01)
