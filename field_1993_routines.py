# -------------------------------------------------------------------------------------------------
#  Experiments from Fields - 1993 -
#
# Author: Salman Khan
# Date  : 25/05/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pickle

import image_generator_curve
import alex_net_utils

from keras.preprocessing.image import load_img

reload(image_generator_curve)
reload(alex_net_utils)


def contour_gain_vs_inter_fragment_rotation(
        model, data_key, c_len, frag_orient, n_runs=100, axis=None):
    """
    Compare model performance with Fields-1993 Experiment 1 - Contour enhancement gain as a
    function of inter-fragment rotation.

    The behavioural results in Fields are presented as a relative detectability metric.
    To get absolute enhancement gain, these results are combined with the results of Li-2006.

    For example, for a contour length of 7, Li-2006 gives absolute gain values for a linear
    contour. To get absolute enhancement gains for  contour of length 7 with inter-fragment
    rotations of 15 degrees, the detectability results from Fields -1993 are multiplied
    with the absolute gain of the linear contour of the same length.

    Detectability = 100%, Absolute enhancement gain = Full linear gain
    Detectability = 50%, absolute enhancement gain = 1

    TODO: Inter-fragment spacing is currently not accounted for. Li-2006 presents
    TODO: enhancement gain for relative spacing between fragments. The considered spacing
    TODO: are too small for the base spacing considered in Fields-1993.
    TODO: Need to properly account for these.

    :param model: Contour Integration Model.
        (Should be training model with last layer = enhancement gain calculating layer)
    :param data_key: data key (dictionary of dictionaries) that describes the data
    :param c_len: Contour Length
    :param frag_orient: base orientation of fragment [-90 = horizontal, 0 = vertical]
    :param n_runs: number of runs to average results over for each point
    :param axis: [Default=None]

    :return:
    """
    # --------------------------------------
    # Validation
    # --------------------------------------
    if c_len not in [1, 3, 5, 7, 9]:
        raise Exception("Invalid contour length {0} specified. Allowed = [1, 3, 5, 7, 9]")

    print("Model Contour Gain vs inter-fragment rotation "
          "for contour length {0}, frag orientation {1}".format(c_len, frag_orient))

    # --------------------------------------
    # Get Neurophysiological Data
    # --------------------------------------
    with open('.//data//neuro_data//Li2006.pickle', 'rb') as handle:
        li_2006_data = pickle.load(handle)

    absolute_gain_linear = {
        1: li_2006_data['contour_len_avg_gain'][0],
        3: li_2006_data['contour_len_avg_gain'][1],
        5: li_2006_data['contour_len_avg_gain'][2],
        7: li_2006_data['contour_len_avg_gain'][3],
        9: li_2006_data['contour_len_avg_gain'][4],

    }

    # TODO: Retrieve this from a pickle object.
    relative_gain_curvature = {
        0: 1.00,
        15: 0.98,
        30: 0.87,
        45: 0.85,
        60: 0.61
    }

    inter_frag_rotation_arr = np.array([0, 15, 30, 45, 60])

    # Plot Neurophysiological data
    # --------------------------------------
    if axis is None:
        f, axis = plt.subplot()

    # Relative gain curvature is actually detectability.
    # at 100% detectability, gain is full amount. @ 50 percent detectability, no gain (gain=1)
    absolute_gains = [
        1 + 2 * (relative_gain_curvature[beta] - 0.5) * (absolute_gain_linear[c_len] - 1)
        for beta in inter_frag_rotation_arr
    ]

    axis.plot(inter_frag_rotation_arr, absolute_gains,
              label='Fields-1993-c_len_{}'.format(c_len), marker='s', linestyle='--')

    # --------------------------------------
    # Model Results
    # --------------------------------------
    avg_gain_per_angle = []
    std_gain_per_angle = []

    for beta in inter_frag_rotation_arr:

        print("Processing c_len = {}, beta = {}".format(c_len, beta))

        # Image Retriever
        active_train_set = data_key["c_len_{0}_beta_{1}_rot_{2}".format(c_len, beta, frag_orient)]

        image_generator = image_generator_curve.DataGenerator(
            active_train_set,
            batch_size=1,
            shuffle=True,
        )

        gen_out = iter(image_generator)

        # Get the results
        y_hat_arr = []
        for r_idx in range(n_runs):

            x_in, y = gen_out.next()

            # TODO: look into using activations callbacks. Than this routine can be used by
            # TODO: the full contour integration model, which does not have a gain calculating layer.
            y_hat = model.predict(x_in, batch_size=1)
            y_hat_arr.append(y_hat)
            # print("Predicted gain {0}, Expected gain {1}".format(y_hat, y))

        avg_gain_per_angle.append(np.mean(y_hat_arr))
        std_gain_per_angle.append(np.std(y_hat_arr))

    axis.errorbar(
        inter_frag_rotation_arr, avg_gain_per_angle, std_gain_per_angle,
        marker='o', label='model-c_len_{}'.format(c_len), linestyle='-')

    axis.legend()
    axis.set_xlabel("Inter-fragment rotation (Deg)")
    axis.set_ylabel("Gain")
    axis.set_title("Enhancement gain vs inter-fragment rotation - Fields -1993 (Exp 1)")


def contour_gain_vs_length(model, data_key, beta, frag_orient, n_runs=100, axis=None):
    """
    Model contour enhancement gain as a function of contour length. This is similar
    to an experiment from Li-2006 except that additionally contour curvature is considered.

    This is a derived Neurophysiological Result. Expected Gain for a given length is found by
    multiplying the relative curvature gain from Fields 1993 with the absolute gain for a linear
    contour as specified by Li -2006. Not that inter-fragment spacing is different from
    Li 2006 Results.

    Detectability = 100%, Absolute enhancement gain = Full linear gain
    Detectability = 50%, absolute enhancement gain = 1

    TODO: Inter-fragment spacing is currently not accounted for. Li-2006 presents
    TODO: enhancement gain for relative spacing between fragments. The considered spacing
    TODO: are too small for the base spacing considered in Fields-1993.
    TODO: Need to properly account for these.

    Different from the previous version of this routine (Li2006Routines), here stimuli are
    generated based on Fields -1993 Method.

    :param model: Contour Integration Model
        (Should be training model with last layer = enhancement gain calculating layer)
    :param data_key: data key (dictionary of dictionaries) that describes the data
    :param beta: Consider contours with inter-fragment rotations of this amount
    :param frag_orient: Default orientation of contour fragment. [-90 = horizontal, 0 = vertical]
    :param n_runs: number of runs to average results over for each point
    :param axis: [Default None]

    :return:
    """
    # --------------------------------------
    # Validation
    # --------------------------------------
    if beta not in [0, 15, 30, 45, 60]:
        raise Exception("Invalid inter-fragment rotation {}. Allowed [0, 15, 30, 45, 60]".format(beta))

    print("Model Contour Gain vs contour length"
          "for inter-fragment rotation {0}, frag orientation {1}".format(beta, frag_orient))

    # --------------------------------------
    # Get Neurophysiological Data
    # --------------------------------------
    with open('.//data//neuro_data//Li2006.pickle', 'rb') as handle:
        li_2006_data = pickle.load(handle)

    # TODO: Retrieve this from a pickle object.
    relative_gain_curvature = {
        0: 1.00,
        15: 0.98,
        30: 0.87,
        45: 0.85,
        60: 0.61
    }

    c_len_arr = np.array([1, 3, 5, 7, 9])

    # Relative gain curvature is actually detectability.
    # at 100% detectability, gain is full amount. @ 50 percent detectability, no gain (gain=1)
    absolute_gains = 1 + (li_2006_data['contour_len_avg_gain'] - 1) \
        * 2 * (relative_gain_curvature[beta] - 0.5)

    # Plot Neurophysiological data
    # --------------------------------------
    axis.plot(c_len_arr, absolute_gains,
              label='Fields1993+Li2006-beta{}'.format(beta), marker='s', linestyle='--')

    # --------------------------------------
    # Model Results
    # --------------------------------------
    avg_gain_per_len = []
    std_gain_per_len = []

    for c_len in c_len_arr:

        print("Processing c_len = {}, beta = {}".format(c_len, beta))

        # if c_len == 1 and beta != 0:
        #     continue

        # Image Retriever
        active_train_set = data_key["c_len_{0}_beta_{1}_rot_{2}".format(c_len, beta, frag_orient)]

        image_generator = image_generator_curve.DataGenerator(
            active_train_set,
            batch_size=1,
            shuffle=True,
        )

        gen_out = iter(image_generator)

        # Get the results
        y_hat_arr = []
        for r_idx in range(n_runs):
            x_in, y = gen_out.next()

            # TODO: look into using activations callbacks. Than this routine can be used by
            # TODO: the full contour integration model, which does not have a gain calculating layer.
            y_hat = model.predict(x_in, batch_size=1)
            y_hat_arr.append(y_hat)
            # print("Predicted gain {0}, Expected gain {1}".format(y_hat, y))

        avg_gain_per_len.append(np.mean(y_hat_arr))
        std_gain_per_len.append(np.std(y_hat_arr))

    axis.errorbar(
        c_len_arr, avg_gain_per_len, std_gain_per_len,
        marker='o', label='Model-beta_{}'.format(beta), linestyle='-')

    axis.legend()
    axis.set_xlabel("Contour Length")
    axis.set_ylabel("Gain")
    axis.set_title("Enhancement gain vs Contour Length")


def plot_activations(model, img_file, tgt_filt_idx,):
    """

    PLot the Feature Extract and Contour Integration Activations for the specified image.

    :param model:
    :param img_file:
    :param tgt_filt_idx:
    :return:
    """
    # Callbacks to get activations of feature extract and contour integration layers
    feat_extract_act_cb = alex_net_utils.get_activation_cb(model, 1)
    cont_int_act_cb = alex_net_utils.get_activation_cb(model, 2)

    d = load_img(img_file)
    d1 = np.array(d)

    alex_net_utils.plot_l1_and_l2_activations(
        d1 / 255.0, feat_extract_act_cb, cont_int_act_cb, tgt_filt_idx)
