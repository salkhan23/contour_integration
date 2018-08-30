# -------------------------------------------------------------------------------------------------
#  Experiments from Fields - 1993 -
#
# Author: Salman Khan
# Date  : 25/05/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

import generate_curved_contour_data_set_orient_matched
import image_generator_curve
import alex_net_utils

reload(image_generator_curve)
reload(alex_net_utils)
reload(generate_curved_contour_data_set_orient_matched)


def contour_gain_vs_inter_fragment_rotation(
        model, data_key, c_len, frag_orient=None, alpha=0, n_runs=100, axis=None):
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

    :param alpha:
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
    valid_c_len = [1, 3, 5, 7, 9]
    if c_len not in valid_c_len:
        raise Exception("Invalid contour length {0} specified. Allowed = {1}".format(c_len, valid_c_len))

    valid_alpha = [0, 15, 30]
    if alpha not in valid_alpha:
        raise Exception("Invalid alpha {0} specified. Allowed {1}".format(alpha, valid_alpha))

    print("Model Contour Gain vs inter-fragment rotation "
          "for contour length {0}, frag orientation {1}, alpha = {2}".format(c_len, frag_orient, alpha))

    # --------------------------------------
    # Get Neurophysiological Data
    # --------------------------------------
    abs_gains_arr = generate_curved_contour_data_set_orient_matched.get_neurophysiological_data('c_len')
    inter_frag_rotation_arr = np.array([0, 15, 30, 45, 60])

    # Plot Neurophysiological data
    # --------------------------------------
    if axis is None:
        f, axis = plt.subplots()

    expected_gains = [abs_gains_arr[c_len][alpha][beta] for beta in inter_frag_rotation_arr]

    axis.plot(inter_frag_rotation_arr, expected_gains,
              label='Fields-1993-c_len_{}'.format(c_len), marker='s', linestyle='--')

    # --------------------------------------
    # Model Results
    # --------------------------------------
    avg_gain_per_angle = []
    std_gain_per_angle = []

    for beta in inter_frag_rotation_arr:

        print("Processing c_len = {}, beta = {}".format(c_len, beta))

        # filter the data keys
        use_set = data_key
        use_set = [x for x in use_set if 'c_len_{}'.format(c_len) in x]
        use_set = [x for x in use_set if 'beta_{}'.format(beta) in x]
        use_set = [x for x in use_set if 'alpha_{}'.format(alpha) in x]
        if frag_orient is not None:
            use_set = [x for x in use_set if 'forient_{}'.format(frag_orient) in x]

        active_test_set = {}
        for set_id in use_set:
            active_test_set.update(data_key[set_id])

        image_generator = image_generator_curve.DataGenerator(
            active_test_set,
            batch_size=1,
            shuffle=True,
        )

        gen_out = iter(image_generator)

        # Get the results
        y_hat_arr = []
        for r_idx in range(n_runs):

            x_in, y = gen_out.next()

            # TODO: look into using activations callbacks. Then this routine can be used by
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


def contour_gain_vs_length(model, data_key, beta, frag_orient=None, alpha=0, n_runs=100, axis=None):
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
    :param alpha:
    :param frag_orient: Default orientation of contour fragment. [-90 = horizontal, 0 = vertical]
    :param n_runs: number of runs to average results over for each point
    :param axis: [Default None]

    :return:
    """
    # --------------------------------------
    # Validation
    # --------------------------------------
    valid_beta = [0, 15, 30, 45, 60]
    if beta not in valid_beta:
        raise Exception("Invalid inter-fragment rotation {0}. Allowed {1}".format(beta, valid_beta))

    valid_alpha = [0, 15, 30]
    if alpha not in valid_alpha:
        raise Exception("Invalid alpha {0} specified. Allowed {1}".format(alpha, valid_alpha))

    print("Model Contour Gain vs contour length "
          "for inter-fragment rotation {0}, frag orientation {1}, alpha={2}".format(beta, frag_orient, alpha))

    # --------------------------------------
    # Get Neurophysiological Data
    abs_gains_arr = generate_curved_contour_data_set_orient_matched.get_neurophysiological_data('c_len')
    c_len_arr = np.array([1, 3, 5, 7, 9])

    expected_gains = [abs_gains_arr[c_len][alpha][beta] for c_len in c_len_arr]

    # Plot Neurophysiological data
    # --------------------------------------
    axis.plot(c_len_arr, expected_gains,
              label='Fields1993+Li2006-beta{}'.format(beta), marker='s', linestyle='--')

    # --------------------------------------
    # Model Results
    # --------------------------------------
    avg_gain_per_len = []
    std_gain_per_len = []

    for c_len in c_len_arr:

        print("Processing c_len = {}, beta = {}".format(c_len, beta))

        # filter the data keys
        use_set = data_key
        use_set = [x for x in use_set if 'c_len_{}'.format(c_len) in x]
        use_set = [x for x in use_set if 'beta_{}'.format(beta) in x]
        use_set = [x for x in use_set if 'alpha_{}'.format(alpha) in x]
        if frag_orient is not None:
            use_set = [x for x in use_set if 'forient_{}'.format(frag_orient) in x]

        active_test_set = {}
        for set_id in use_set:
            active_test_set.update(data_key[set_id])

        image_generator = image_generator_curve.DataGenerator(
            active_test_set,
            batch_size=1,
            shuffle=True,
        )

        gen_out = iter(image_generator)

        # Get the results
        y_hat_arr = []
        for r_idx in range(n_runs):
            x_in, y = gen_out.next()

            # TODO: look into using activations callbacks. Then this routine can be used by
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


def contour_gain_vs_spacing(model, data_key, beta, frag_orient=None, alpha=0, n_runs=100, axis=None):
    """

    todo: Fix the description!
    Model contour enhancement gain as a function of fragment spacing. This is similar
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
    :param alpha:
    :param frag_orient: Default orientation of contour fragment. [-90 = horizontal, 0 = vertical]
    :param n_runs: number of runs to average results over for each point
    :param axis: [Default None]

    :return:
    """
    # --------------------------------------
    # Validation
    # --------------------------------------
    valid_beta = [0, 15, 30, 45, 60]
    if beta not in valid_beta:
        raise Exception("Invalid inter-fragment rotation {0}. Allowed {1}".format(beta, valid_beta))

    valid_alpha = [0, 15, 30]
    if alpha not in valid_alpha:
        raise Exception("Invalid alpha {0} specified. Allowed {1}".format(alpha, valid_alpha))

    print("Model Contour Gain vs fragment spacing "
          "for inter-fragment rotation {0}, frag orientation {1}, alpha={2}".format(beta, frag_orient, alpha))

    # --------------------------------------
    # Get Neurophysiological Data
    abs_gains_arr = generate_curved_contour_data_set_orient_matched.get_neurophysiological_data('f_spacing')

    f_spacing_arr = np.array([1, 1.2, 1.4, 1.6, 1.9])

    expected_gains = [abs_gains_arr[f_spacing][alpha][beta] for f_spacing in f_spacing_arr]

    # Plot Neurophysiological data
    # --------------------------------------
    axis.plot(f_spacing_arr, expected_gains,
              label='Fields1993+Li2006-beta{}'.format(beta), marker='s', linestyle='--')

    # --------------------------------------
    # Model Results
    # --------------------------------------
    avg_gain_per_f_spacing = []
    std_gain_per_f_spacing = []

    for f_spacing in f_spacing_arr:

        print("Processing fragment spacing = {}, beta = {}".format(f_spacing, beta))

        # filter the data keys
        use_set = data_key
        use_set = [x for x in use_set if 'f_spacingx10_{}'.format(int(f_spacing * 10)) in x]
        use_set = [x for x in use_set if 'beta_{}'.format(beta) in x]
        use_set = [x for x in use_set if 'alpha_{}'.format(alpha) in x]
        if frag_orient is not None:
            use_set = [x for x in use_set if 'forient_{}'.format(frag_orient) in x]

        active_test_set = {}
        for set_id in use_set:
            active_test_set.update(data_key[set_id])

        image_generator = image_generator_curve.DataGenerator(
            active_test_set,
            batch_size=1,
            shuffle=True,
        )

        gen_out = iter(image_generator)

        # Get the results
        y_hat_arr = []
        for r_idx in range(n_runs):
            x_in, y = gen_out.next()

            # TODO: look into using activations callbacks. Then this routine can be used by
            # TODO: the full contour integration model, which does not have a gain calculating layer.
            y_hat = model.predict(x_in, batch_size=1)
            y_hat_arr.append(y_hat)
            # print("Predicted gain {0}, Expected gain {1}".format(y_hat, y))

        avg_gain_per_f_spacing.append(np.mean(y_hat_arr))
        std_gain_per_f_spacing.append(np.std(y_hat_arr))

    axis.errorbar(
        f_spacing_arr, avg_gain_per_f_spacing, std_gain_per_f_spacing,
        marker='o', label='Model-beta_{}'.format(beta), linestyle='-')

    axis.legend()
    axis.set_xlabel("Contour Length")
    axis.set_ylabel("Gain")
    axis.set_title("Enhancement gain vs Fragment Spacing")


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
