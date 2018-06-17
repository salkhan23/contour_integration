# -------------------------------------------------------------------------------------------------
#  Find the parameters of the best fit 2D Gabor Filter for a given L1 layer kernel. Bet fit Gabors
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

from contour_integration_models.alex_net import masked_models as cont_int_models
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


def get_gabor_fragment(params, spatial_size):
    """
    Constructed a 2D Gabor fragment from the specified params of the specified size.
    A 3 channel fragment is always generated.

    params is  either a dictionary of a list of dictionaries of the type specified below.

    params = {
        'x0': x0,
        'y0': y0,
        'theta_deg': theta_deg,
        'amp': amp,
        'sigma': sigma,
        'lambda1': lambda1,
        'psi': psi,
        'gamma': gamma
    }

    If a single dictionary is specified all three channels have the same parameters, else if
    a list of size 3 is specified each channel has its own parameters.

    :param params: is either a dictionary of a list of dictionaries
    :param spatial_size:
    :return:
    """
    half_x = spatial_size[0] // 2
    half_y = spatial_size[1] // 2

    x = np.linspace(-half_x, half_x, spatial_size[0])
    y = np.linspace(-half_y, half_y, spatial_size[1])

    xx, yy = np.meshgrid(x, y)

    if type(params) is list and len(params) not in (1, 3):
        raise Exception("Only length 3 list of parameters can be specified")

    if type(params) is not list:

        frag = gabor_2d(
            (xx, yy),
            x0=params['x0'],
            y0=params['y0'],
            theta_deg=params['theta_deg'],
            amp=params['amp'],
            sigma=params['sigma'],
            lambda1=params['lambda1'],
            psi=params['psi'],
            gamma=params['gamma']
        )

        frag = frag.reshape((x.shape[0], y.shape[0]))
        frag = np.stack((frag, frag, frag), axis=2)
    else:

        frag = np.zeros((spatial_size[0], spatial_size[1], 3))

        for idx, chan_params in enumerate(params):

            frag_chan = gabor_2d(
                (xx, yy),
                x0=chan_params['x0'],
                y0=chan_params['y0'],
                theta_deg=chan_params['theta_deg'],
                amp=chan_params['amp'],
                sigma=chan_params['sigma'],
                lambda1=chan_params['lambda1'],
                psi=chan_params['psi'],
                gamma=chan_params['gamma']
            )

            frag_chan = frag_chan.reshape((x.shape[0], y.shape[0]))
            frag[:, :, idx] = frag_chan

    # Normalize to range 0 - 255
    frag = (frag - frag.min()) / (frag.max() - frag.min()) * 255

    frag = frag.astype(np.uint8)

    return frag


def find_best_fit_2d_gabor(kernel):
    """
    Find the best fit parameters of a 2D gabor for each input channel of kernel.

    :param kernel: Alexnet l1 kernel

    :return: list of best fit parameters for each channel of kernel. Format: [x, y, chan]
    """
    n_channels = kernel.shape[-1]

    half_x = kernel.shape[0] // 2
    half_y = kernel.shape[1] // 2

    x = np.linspace(-half_x, half_x, kernel.shape[0])
    y = np.linspace(-half_y, half_y, kernel.shape[1])

    xx, yy = np.meshgrid(x, y)

    opt_params_list = []

    for c_idx in range(n_channels):

        opt_params_found = False

        theta = 0

        # gabor_2d(     x0,      y0, theta_deg,     amp, sigma, lambda1,       psi, gamma):
        bounds = ([-half_x, -half_y,      -160, -np.inf,   0.1,       0,         0,     0],
                  [ half_x,  half_y,       180,  np.inf,     4,  np.inf, 2 * np.pi,     6])

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

                    # print( "[%d]: (x0,y0)=(%0.2f, %0.2f), theta=%0.2f, A=%0.2f, sigma=%0.2f, lambda=%0.2f, "
                    #        "psi=%0.2f, gamma=%0.2f"
                    #        % (chan_idx, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]))

                    # print("Err: (x0,y0)=(%0.2f, %0.2f), theta=%0.2f, A=%0.2f, sigma=%0.2f, "
                    #       "lambda=%0.2f, psi=%0.2f, gamma=%0.2f"
                    #       % (one_sd_error[0], one_sd_error[1], one_sd_error[2], one_sd_error[3], one_sd_error[4],
                    #          one_sd_error[5], one_sd_error[6], one_sd_error[7]))
                else:
                    theta += 10

            except RuntimeError:
                theta += 10

                if theta == 360:
                    # print("Optimal parameters could not be found")
                    opt_params_found = True
                    opt_params_list.append(
                        np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                    )

            except ValueError:
                # print("Optimal parameters could not be found")
                opt_params_found = True
                opt_params_list.append(
                    np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                )

    return opt_params_list


def plot_kernel_and_best_fit_gabors(kernel, kernel_idx, fitted_gabors_params):
    """

    :param kernel:
    :param kernel_idx: Index of the kernel (only fir title)
    :param fitted_gabors_params: list of fitted parameters for each channel of kernel

    :return: None
    """
    n_channels = kernel.shape[-1]

    half_x = kernel.shape[0] // 2
    half_y = kernel.shape[1] // 2

    x = np.linspace(-half_x, half_x, kernel.shape[0])
    y = np.linspace(-half_y, half_y, kernel.shape[1])
    xx, yy = np.meshgrid(x, y)

    # Higher resolution display
    x2 = np.linspace(-half_x, half_x, (kernel.shape[0] + 100))
    y2 = np.linspace(-half_y, half_y, (kernel.shape[1] + 100))
    xx2, yy2 = np.meshgrid(x2, y2)

    # -----------------------------------------------------------------------------------
    # Channel Fitting
    # -----------------------------------------------------------------------------------
    f = plt.figure()

    # Normalize the kernel to [0, 1] to display it properly
    display_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

    for c_idx in range(n_channels):

        # Plot the kernel
        f.add_subplot(n_channels, 3, (c_idx * 3) + 1)
        plt.imshow(display_kernel[:, :, c_idx], cmap='seismic')
        plt.title(r"$Chan=%d $" % c_idx)

        if np.any(fitted_gabors_params[c_idx]):  # if it is not none

            x0, y0, theta, amp, sigma, lambda1, psi, gamma = fitted_gabors_params[c_idx]

            # Fitted gabor - same resolution (with which fit was done)
            f.add_subplot(n_channels, 3, (c_idx * 3) + 2)
            fitted_gabor = gabor_2d((xx, yy), x0, y0, theta, amp, sigma, lambda1, psi, gamma)
            fitted_gabor = fitted_gabor.reshape((x.shape[0], y.shape[0]))
            display_gabor = (fitted_gabor - fitted_gabor.min()) / (fitted_gabor.max() - fitted_gabor.min())
            plt.imshow(display_gabor, cmap='seismic')

            # # Fitted gabor - higher resolution
            f.add_subplot(n_channels, 3, (c_idx * 3) + 3)
            fitted_gabor = gabor_2d((xx2, yy2), x0, y0, theta, amp, sigma, lambda1, psi, gamma)
            fitted_gabor = fitted_gabor.reshape((x2.shape[0], y2.shape[0]))
            display_gabor = (fitted_gabor - fitted_gabor.min()) / (fitted_gabor.max() - fitted_gabor.min())
            plt.imshow(display_gabor, cmap='seismic')

    f.suptitle("2D Gabor Fits for L1 Filter @ Index %d" % kernel_idx)

    # -----------------------------------------------------------------------------------
    # Full Filter Fitting
    # -----------------------------------------------------------------------------------
    # create a list of dictionaries
    gabor_dict_list = []
    for chan_params in fitted_gabors_params:
        chan_dict = {
            'x0': chan_params[0],
            'y0': chan_params[1],
            'theta_deg': chan_params[2],
            'amp': chan_params[3],
            'sigma': chan_params[4],
            'lambda1': chan_params[5],
            'psi': chan_params[6],
            'gamma': chan_params[7]
        }

        gabor_dict_list.append(chan_dict)

    frag = get_gabor_fragment(gabor_dict_list, kernel.shape[0: 2])

    f, ax_arr = plt.subplots(1, 2)

    ax_arr[0].imshow(display_kernel)
    ax_arr[0].set_title("Kernel")

    # display_frag = (frag - frag.min()) / (frag.max() - frag.min())
    ax_arr[1].imshow(frag)
    ax_arr[1].set_title("Regenerated Fragment")


def get_l1_filter_orientation_and_offset(tgt_filt, tgt_filt_idx, show_plots=True):
    """
    Given a Target AlexNet L1 Convolutional Filter, fit to a 2D spatial Gabor. Use this to as
    the orientation of the filter and calculate the row shift offset to use when tiling fragments.
    This offset represents the shift in pixels to use for each row  as you move away from the
    center row. Thereby allowing contours for the target filter to be generated.

    Raises an exception if no best fit parameters are found for any of the channels of the target
    filter.

    :param show_plots:
    :param tgt_filt_idx:
    :param tgt_filt:

    :return: optimal orientation, row offset.
    """
    tgt_filt_len = tgt_filt.shape[0]

    best_fit_params_list = find_best_fit_2d_gabor(tgt_filt)

    # Plot the best fit params
    if show_plots:
        plot_kernel_and_best_fit_gabors(tgt_filt, tgt_filt_idx, best_fit_params_list)

    # Remove all empty entries
    best_fit_params_list = [params for params in best_fit_params_list if params is not None]
    if not best_fit_params_list:
        # raise Exception("Optimal Params could not be found")
        return np.NaN, np.NaN

    # Find channel with highest energy (Amplitude) and use its preferred orientation
    # Best fit parameters: x0, y0, theta, amp, sigma, lambda1, psi, gamma
    best_fit_params_list = np.array(best_fit_params_list)
    amp_arr = best_fit_params_list[:, 3]
    amp_arr = np.abs(amp_arr)
    max_amp_idx = np.argmax(amp_arr)

    theta_opt = best_fit_params_list[max_amp_idx, 2]

    # TODO: Fix me - Explain why this is working
    # TODO: How to handle horizontal (90) angles
    # # Convert the orientation angle into a y-offset to be used when tiling fragments
    contour_angle = theta_opt + 90.0  # orientation is of the Gaussian envelope with is orthogonal to
    # # sinusoid carrier we are interested in.
    # contour_angle = np.mod(contour_angle, 180.0)

    # if contour_angle >= 89:
    #     contour_angle -= 180  # within the defined range of tan

    # contour_angle = contour_angle * np.pi / 180.0
    # offset = np.int(np.ceil(tgt_filter_len / np.tan(contour_angle)))
    row_offset = np.int(np.ceil(tgt_filt_len / np.tan(np.pi - contour_angle * np.pi / 180.0)))

    # print("L1 kernel %d, optimal orientation %0.2f(degrees), vertical offset of tiles %d"
    #       % (tgt_filt_idx, theta_opt, row_offset))

    return theta_opt, row_offset


def get_filter_orientation(tgt_filt, o_type='average', display_params=True):
    """
    Fit the target filter to a gabor filter and find the orientation of each channel.
    If type=average, the average orientation across the filter is returned, else if
    type=max, the orientation of channel with the maximum orientation is returned

    :param tgt_filt:
    :param o_type: ['average', 'max']
    :param display_params:

    :return: orientation of the type specified
    """

    gabor_fit_params = find_best_fit_2d_gabor(tgt_filt)
    gabor_fit_params = np.array(gabor_fit_params)

    if display_params:
        print("Gabor Filt Parameters:")
        for c_idx, p in enumerate(gabor_fit_params):
            print("Chan {0}: (x0,y0)=({1:0.2f},{2:0.2f}), theta_deg={3:0.1f}, A={4:0.2f}, sigma={5:0.2f}, "
                  "lambda={6:0.2f}, psi={7:0.2f}, gamma={8:0.2f}".format(
                    c_idx, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]))

    o_type = o_type.lower()
    if o_type == 'average':
        orientation_arr = gabor_fit_params[:, 2]
        orientation = np.mean(orientation_arr)
    elif o_type == 'max':
        amp_arr = gabor_fit_params[:, 3]
        orientation_idx = np.argmax(abs(amp_arr))
        orientation = gabor_fit_params[orientation_idx, 2]
    else:
        raise Exception("Unknown o_type!")

    return orientation


if __name__ == "__main__":

    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # 1. Build the model
    # ---------------------------------------------------------------------
    print("Building Contour Integration Model...")

    # Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        n=25,
        activation='relu'
    )
    # contour_integration_model.summary()

    feature_extract_kernels = K.eval(contour_integration_model.layers[1].weights[0])

    # ---------------------------------------------------------------------
    # Plot fragment and regenerated fragment from best fit parmeters
    # ---------------------------------------------------------------------
    tgt_filter_idx = 0

    tgt_filter = feature_extract_kernels[:, :, :, tgt_filter_idx]
    optimal_params = find_best_fit_2d_gabor(tgt_filter)
    plot_kernel_and_best_fit_gabors(tgt_filter, tgt_filter_idx, optimal_params)
    plt.suptitle("Feature extracting kernel at index {}".format(tgt_filter_idx))

    # -----------------------------------------------------------------------------------
    # Get Orientation preferences of each channel for each filter
    # -----------------------------------------------------------------------------------
    n_filters = feature_extract_kernels.shape[3]
    opt_orient_list = []

    np.set_printoptions(precision=1)

    for filter_idx in range(n_filters):

        tgt_filter = feature_extract_kernels[:, :, :, filter_idx]
        opt_params = find_best_fit_2d_gabor(tgt_filter)

        per_chan_orient = [param[2] for param in opt_params]
        opt_orient_list.append(np.array(per_chan_orient))

        print('kernel @ {0}: {1:<10}'.format(filter_idx, np.array(per_chan_orient)))

    opt_orient_list = np.array(opt_orient_list)

    # ---------------------------------------------------------------------------------
    # Plot the orientation preferences for each filter
    # ---------------------------------------------------------------------------------
    color_arr = ['r', 'g', 'b']
    plt.figure()

    for chan_idx in np.arange(3):
        plt.stem(
            np.arange(n_filters), opt_orient_list[:, chan_idx],
            color_arr[chan_idx], label='chan_{}'.format(chan_idx), markerfmt=color_arr[chan_idx]+'o',
        )

    plt.legend()
