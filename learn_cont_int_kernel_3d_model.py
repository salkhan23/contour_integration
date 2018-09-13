# -------------------------------------------------------------------------------------------------
#  Learn contour enhancement kernels for the 3d contour integration model
#  Model is trained on stimuli from Fields - 1993.
#
# Author: Salman Khan
# Date  : 06/05/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from time import time
import gc
from datetime import datetime

import keras.backend as keras_backend
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import load_img
from keras import losses
from keras import optimizers

import alex_net_utils
import contour_integration_models.alex_net.model_3d as contour_integration_model_3d
import image_generator_curve
import learn_cont_int_kernel_3d_model_linear_contours as linear_contour_training
import field_1993_routines

reload(alex_net_utils)
reload(contour_integration_model_3d)
reload(image_generator_curve)
reload(linear_contour_training)
reload(field_1993_routines)


IMAGE_SIZE = (227, 227, 3)
TEMP_WEIGHT_STORE_FILE = 'best_weights.hf'


def load_pretrained_weights(model, prev_trained_weights_file):
    """

    :param model:
    :param prev_trained_weights_file:
    :return:
    """
    print("Loading Pretrained Weights")

    train_summary_file = get_weights_training_summary_file(prev_trained_weights_file)

    prev_learnt_kernels_set = get_prev_learnt_kernels(train_summary_file)
    print("Previously Trained Kernel Indices {}".format(prev_learnt_kernels_set))

    if os.path.exists(prev_trained_weights_file):
        model.load_weights(prev_trained_weights_file, by_name=True)

    return prev_learnt_kernels_set


def get_weights_training_summary_file(w_store_file):
    train_summary_file = os.path.splitext(w_store_file)[0] + '_training_summary.txt'

    # print("Previously trained summary file {}".format(train_summary_file))
    return train_summary_file


def get_prev_learnt_kernels(train_summary_file):

    prev_learnt_idx_set = set()

    if os.path.exists(train_summary_file):
        with open(train_summary_file, 'rb+') as fid:
            for line in fid:
                k_idx = line.split()[0]
                prev_learnt_idx_set.add(int(k_idx))

    # print("Previously learnt kernels {}".format(prev_learnt_idx_set))
    return prev_learnt_idx_set


def save_learnt_weights(model, tgt_filt_idx, w_store_file):

    # print("Saving Learnt Weights @ {}".format(w_store_file))

    basedir = os.path.dirname(w_store_file)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    model.save_weights(w_store_file)

    train_summary_file = get_weights_training_summary_file(w_store_file)

    prev_learnt_kernels_set = get_prev_learnt_kernels(train_summary_file)
    prev_learnt_kernels_set.update([tgt_filt_idx])

    with open(train_summary_file, 'w+') as fid:
        for entry in prev_learnt_kernels_set:
            fid.write(str(entry) + '\n')


def _get_train_n_test_dictionary_of_dictionaries(tgt_filt_idx, data_dir):
    """

    :param tgt_filt_idx:
    :param data_dir:
    :return:
    """
    train_data_key_loc = os.path.join(
        data_dir, "train/filter_{}".format(tgt_filt_idx), "data_key.pickle")

    test_data_key_loc = os.path.join(
        data_dir, "test/filter_{}".format(tgt_filt_idx), "data_key.pickle")

    with open(train_data_key_loc, 'rb') as handle:
        train_data_list_of_dict = pickle.load(handle)  # Returns a list of dictionaries

    with open(test_data_key_loc, 'rb') as handle:
        test_data_list_of_dict = pickle.load(handle)  # Returns a list of dictionaries

    return train_data_list_of_dict, test_data_list_of_dict


def get_train_n_test_data_keys(
        tgt_filt_idx, data_dir, c_len=None, beta=None, alpha=None, f_spacing=None, frag_orient=None):
    """
    A data key is a dictionary of file location:expected gain tuples.
    Two Dictionaries are returned: one for testing and one for testing model performance

    :param tgt_filt_idx:
    :param data_dir:
    :param c_len:
    :param beta:
    :param alpha:
    :param f_spacing: fragment spacing.
    :param frag_orient:
    :return:
    """
    valid_c_len = [1, 3, 5, 7, 9]
    valid_beta = [0, 15, 30, 45, 60]
    valid_alpha = [0, 15, 30]
    valid_f_spacing = [1, 1.2, 1.4, 1.6, 1.9]

    if c_len is None:
        c_len = valid_c_len
    elif isinstance(c_len, int):
        c_len = [c_len]

    if beta is None:
        beta = valid_beta
    elif isinstance(beta, int):
        beta = [beta]

    if alpha is None:
        alpha = valid_alpha
    elif isinstance(alpha, int):
        alpha = [alpha]

    if f_spacing is None:
        f_spacing = valid_f_spacing
    elif isinstance(f_spacing, int):
        f_spacing = [f_spacing]

    if any(x not in valid_c_len for x in c_len):
        raise Exception("Invalid c_len elements {}. All should be in {}".format(c_len, valid_c_len))

    if any(x not in valid_beta for x in beta):
        raise Exception("Invalid beta elements {}. All should be in {}".format(beta, valid_beta))

    if any(x not in valid_alpha for x in alpha):
        raise Exception("Invalid alpha elements {}. All should be in {}".format(alpha, valid_alpha))

    if any(x not in valid_f_spacing for x in f_spacing):
        raise Exception("Invalid f_spacing elements {}. All should be in {}".format(f_spacing, valid_f_spacing))

    train_data_dict_of_dict, test_data_dict_of_dict =\
        _get_train_n_test_dictionary_of_dictionaries(tgt_filt_idx, data_dir)

    full_set = train_data_dict_of_dict.keys()
    use_set = []

    # Get all base files (c_len and frag_spacing )
    for entry in c_len:
        use_set.extend([x for x in full_set if 'c_len_{}'.format(entry) in x])
    for entry in f_spacing:
        use_set.extend([x for x in full_set if 'f_spacingx10_{}'.format(int(entry * 10)) in x])

    full_set = use_set
    use_set = []
    for entry in beta:
        use_set.extend([x for x in full_set if 'beta_{}'.format(entry) in x])

    full_set = use_set
    use_set = []
    for entry in alpha:
        use_set.extend([x for x in full_set if 'alpha_{}'.format(entry) in x])

    if frag_orient is not None:
        full_set = use_set
        use_set = []

        if isinstance(frag_orient, int):
            frag_orient = [frag_orient]

        for entry in frag_orient:
            use_set.extend([x for x in full_set if 'forient_{}'.format(entry) in x])

    # for entry in sorted(use_set):
    #     print entry
    # print("Number of Internal dictionaries selected {}".format(len(use_set)))

    # Single dictionary containing (image file location, expected gain)
    active_train_set = {}
    active_test_set = {}

    for set_id in use_set:
        active_train_set.update(train_data_dict_of_dict[set_id])
        active_test_set.update(test_data_dict_of_dict[set_id])

    return active_train_set, active_test_set


def train_contour_integration_kernel(
        model, tgt_filt_idx, data_dir, b_size=32, n_epochs=200, training_cb=None,
        c_len=None, beta=None, alpha=None, f_spacing=None, axis=None):
    """

    :param model:
    :param tgt_filt_idx:
    :param data_dir:
    :param b_size:
    :param n_epochs:
    :param c_len:
    :param beta:
    :param alpha:
    :param training_cb:
    :param f_spacing:
    :param axis:
    :return: (min_train_loss, min_test_loss, size_of_train_data_dict, size_of_test_data_dict)
    """
    print("Learning contour integration kernel @ index {} ...".format(tgt_filt_idx))

    # Modify the contour integration training model to train the target kernel
    contour_integration_model_3d.update_contour_integration_kernel(model, tgt_filt_idx)
    model.compile(
        optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        loss=losses.mean_squared_error
    )

    # -----------------------------------------------------------------------------------
    # Build the Data Generators
    # -----------------------------------------------------------------------------------
    print("Building data generators...")

    train_data_dict, test_data_dict = get_train_n_test_data_keys(
        tgt_filt_idx, data_dir, c_len=c_len, beta=beta, alpha=alpha, f_spacing=f_spacing)

    if b_size > len(train_data_dict):
        print("WARN: Specified batch size is > than number of actual data")
        b_size = len(train_data_dict)

    train_image_generator = image_generator_curve.DataGenerator(
        train_data_dict,
        batch_size=b_size,
        shuffle=True,
    )

    # Load the entire validation set
    # Tensorboard callback cannot use a generator for validation Data
    # Just load all the test images
    test_image_generator = image_generator_curve.DataGenerator(
        test_data_dict,
        batch_size=len(test_data_dict.keys()),
        shuffle=True,
    )
    gen_out = iter(test_image_generator)
    test_images, test_labels = gen_out.next()

    # -----------------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------------
    if training_cb is None:
        training_cb = []

    n_train_imgs = len(train_data_dict)
    n_test_imgs = len(test_data_dict)
    steps_per_epoch = n_train_imgs / b_size

    print("Training ...")
    history = model.fit_generator(
        generator=train_image_generator,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        validation_data=(test_images, test_labels),
        validation_steps=1,
        # max_q_size=1,
        workers=8,
        callbacks=training_cb
    )

    min_loss_train = min(history.history['loss'])
    min_loss_test = min(history.history['val_loss'])

    print("Minimum Training Loss {0}, Validation loss {1}".format(min_loss_train, min_loss_test))

    # Plot Loss vs Time
    # --------------------------------------
    if axis is None:
        f, axis = plt.subplots()

    axis.plot(history.history['loss'], label='train_loss_{0}'.format(tgt_filt_idx))
    axis.plot(history.history['val_loss'], label='validation_loss_{0}'.format(tgt_filt_idx))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()

    # Clean up
    # --------------------------------------
    del train_image_generator, test_image_generator
    del test_images, test_labels
    del train_data_dict, test_data_dict
    del history
    gc.collect()

    return min_loss_train, min_loss_test, n_train_imgs, n_test_imgs


def plot_start_n_learnt_contour_integration_kernels(model, tgt_filt_idx, start_w=None):
    """

    :param model:
    :param tgt_filt_idx:
    :param start_w: complete set of weights at star of training [Optional]
    :return:
    """
    f, ax_arr = plt.subplots(1, 3)

    learnt_w, _ = model.layers[2].get_weights()
    linear_contour_training.plot_contour_integration_weights_in_channels(
        learnt_w,
        tgt_filt_idx,
        axis=ax_arr[0]
    )
    ax_arr[0].set_title("Learnt Contour Int")

    if start_w is not None:
        linear_contour_training.plot_contour_integration_weights_in_channels(
            start_w,
            tgt_filt_idx,
            axis=ax_arr[1]
        )
        ax_arr[1].set_title("Initial Contour Int")

    feat_extract_w, _ = model.layers[1].get_weights()
    tgt_feat_extract_w = feat_extract_w[:, :, :, tgt_filt_idx]

    normalized_tgt_feat_extract_w = (tgt_feat_extract_w - tgt_feat_extract_w.min()) / \
        (tgt_feat_extract_w.max() - tgt_feat_extract_w.min())

    ax_arr[2].imshow(normalized_tgt_feat_extract_w)
    ax_arr[2].set_title("Feature Extract")

    f.suptitle("Input channels feeding of contour integration kernel @ index {}".format(tgt_filt_idx))
    f.set_size_inches(18, 9)


def clear_unlearnt_contour_integration_kernels(model, trained_kernels):
    """

    :param model:
    :param trained_kernels:
    :return:
    """
    w, b = model.layers[2].get_weights()
    n_kernels = w.shape[3]

    print("All Contour Integration Kernels other than {} will be cleared".format(trained_kernels))

    for i in range(n_kernels):
        if i not in trained_kernels:
            w[:, :, :, i] = np.zeros_like(w[:, :, :, i])
            b[i] = 0

    model.layers[2].set_weights([w, b])


def test_sample_output(data_dir, tgt_filt_idx, feat_extract_cb, cont_int_cb, rslt_dir, c_len, beta, alpha, img_idx=0):
    """

    :param rslt_dir:
    :param cont_int_cb:
    :param feat_extract_cb:
    :param tgt_filt_idx:
    :param data_dir:
    :param c_len:
    :param beta:
    :param alpha:
    :param img_idx:
    :return:
    """
    test_image_dir = os.path.join(
        data_dir,
        'test/filter_{0}/c_len_{1}/beta_{2}/alpha_{3}'.format(
            tgt_filt_idx, c_len, beta, alpha)
    )

    image_file = os.listdir(test_image_dir)[img_idx]
    test_image = load_img(os.path.join(test_image_dir, image_file))
    test_image = np.array(test_image) / 255.0

    sample_img_fig, sample_img_act_fig = alex_net_utils.plot_l1_and_l2_activations(
        test_image,
        feat_extract_cb,
        cont_int_cb,
        tgt_filt_idx
    )

    sample_img_act_fig.suptitle("Contour Integration kernel @ index {0}. [c_len={1}, beta={2}, alpha={3}]".format(
        tgt_filt_idx, c_len, beta, alpha))

    sample_img_act_fig.set_size_inches(18, 9)

    image_id = 'kernel_{}_clen_{}_beta_{}_alpha_{}_img_idx_{}'.format(tgt_filt_idx, c_len, beta, alpha, img_idx)

    sample_img_act_fig.savefig(os.path.join(
        rslt_dir, 'sample_image_activation_kernel_' + image_id + '.png'))
    sample_img_fig.savefig(os.path.join(
        rslt_dir, 'sample_image_' + image_id + '.png'))


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    keras_backend.set_image_dim_ordering('th')
    keras_backend.clear_session()
    start_time = datetime.now()

    np.random.seed(7)

    batch_size = 32
    num_epochs = 100

    save_weights = True
    prev_train_weights = None

    target_kernel_idx_arr = [
        5, 10, 19, 20, 21, 22, 48, 49, 51, 59,
        60, 62, 64, 65, 66, 68, 69, 72, 73, 74,
        76, 77, 79, 80, 82,
    ]

    data_directory = "./data/curved_contours/frag_11x11_full_18x18_param_search"
    results_identifier = 'beta_rotations_upto30'

    # What data to train with (None means everything)
    contour_lengths = None
    fragment_spacing = None
    beta_rotations = [0, 15, 30]
    alpha_rotations = [0]

    # prev_train_weights = \
    #     './trained_models/ContourIntegrationModel3d/filter_matched/contour_integration_weights.hf'

    # Immutable  ------------------------------------------------------------------------
    base_results_dir = './results'
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)

    results_dir = os.path.join(base_results_dir, results_identifier)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    weights_store_file = os.path.join(results_dir, 'trained_weights.hf')

    if contour_lengths is None:
        contour_lengths = [1, 3, 5, 7, 9]
    if beta_rotations is None:
        beta_rotations = [0, 15, 30, 45, 60]
    if alpha_rotations is None:
        alpha_rotations = [0, 15, 30]
    if fragment_spacing is None:
        fragment_spacing = [1, 1.2, 1.4, 1.6, 1.9]

    print("*"*80)
    print("Data Source: {}".format(data_directory))
    print("Results will be stored @ {}".format(results_dir))
    print("*"*80)

    # Turn off figure display when running in batch mode
    if len(target_kernel_idx_arr) > 2:
        plt.ioff()

    # -----------------------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(
        tgt_filt_idx=0,
        rf_size=35,
        inner_leaky_relu_alpha=0.9,
        outer_leaky_relu_alpha=1.,
        l1_reg_loss_weight=0.0005,
    )

    prev_trained_kernel_idx_arr = []
    if prev_train_weights is not None:
        prev_trained_kernel_idx_arr = load_pretrained_weights(cont_int_model, prev_train_weights)

    start_weights, _ = cont_int_model.layers[2].get_weights()

    # Callbacks
    feat_extract_callback = alex_net_utils.get_activation_cb(cont_int_model, 1)
    cont_int_callback = alex_net_utils.get_activation_cb(cont_int_model, 2)

    # -------------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------------
    fig_losses, loss_vs_epoch_ax = plt.subplots()

    min_loss_arr = []
    n_training_images = 0
    n_test_images = 0

    for target_kernel_idx in target_kernel_idx_arr:

        if target_kernel_idx in prev_trained_kernel_idx_arr:
            # Skip kernels that are already trained
            print("Skipping Contour Integration kernel @ index {}. Already trained. ".format(target_kernel_idx))
            continue

        kernel_training_start_time = datetime.now()

        if os.path.exists(TEMP_WEIGHT_STORE_FILE):
            os.remove(TEMP_WEIGHT_STORE_FILE)

        # Callbacks for training
        # ----------------------
        tensorboard = TensorBoard(
            log_dir='logs/{}'.format(time()),
            # histogram_freq=1,
            # write_grads=True,
            # write_images=False,
            # batch_size=1,  # For histogram
        )

        checkpoint = ModelCheckpoint(
            TEMP_WEIGHT_STORE_FILE,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='min',
            save_weights_only=True,
        )
        callbacks = [tensorboard, checkpoint]
        # callbacks = [checkpoint]

        min_train_loss, min_test_loss, n_training_images, n_test_images = train_contour_integration_kernel(
            model=cont_int_model,
            tgt_filt_idx=target_kernel_idx,
            data_dir=data_directory,
            b_size=batch_size,
            n_epochs=num_epochs,
            training_cb=callbacks,
            axis=loss_vs_epoch_ax,
            c_len=contour_lengths,
            f_spacing=fragment_spacing,
            beta=beta_rotations,
            alpha=alpha_rotations
        )

        fig_losses.savefig(os.path.join(results_dir, 'losses.png'), dpi=fig_losses.dpi)

        min_loss_arr.append((min_train_loss, min_test_loss))

        # load best weights
        cont_int_model.load_weights(TEMP_WEIGHT_STORE_FILE)  # load best weights

        # Save the learnt weights
        if save_weights:
            save_learnt_weights(cont_int_model, target_kernel_idx, weights_store_file)

        # Cleanup
        del checkpoint, tensorboard, callbacks

        print("Training kernel {0} took {1}".format(
            target_kernel_idx, datetime.now() - kernel_training_start_time))

        # -------------------------------------------------------------------------------
        # Plot Learnt Kernels
        # -------------------------------------------------------------------------------
        plot_start_n_learnt_contour_integration_kernels(
            cont_int_model,
            target_kernel_idx,
            start_weights,
        )

        learnt_kernel_fig = plt.gcf()
        learnt_kernel_fig.savefig(os.path.join(
            results_dir, 'learnt_contour_integration_kernel_{}.png'.format(target_kernel_idx)))

        # -------------------------------------------------------------------------------
        # Todo: Should be moved to another File
        train_data_dict_of_dicts, test_data_dict_of_dicts = _get_train_n_test_dictionary_of_dictionaries(
            target_kernel_idx,
            data_directory,
        )
        # get list of considered orientations
        list_of_data_sets = train_data_dict_of_dicts.keys()

        if 'forient' in list_of_data_sets[0]:
            fragment_orientation_arr = [np.int(item.split("forient_")[1]) for item in list_of_data_sets]
            fragment_orientation_arr = set(fragment_orientation_arr)
        else:
            fragment_orientation_arr = [None]

        # -------------------------------------------------------------------------------
        #  Fields - 1993 - Experiment 1 - Curvature vs Gain
        # -------------------------------------------------------------------------------
        print("Checking gain vs curvature performance ...")
        for fragment_orientation in fragment_orientation_arr:
            fig_curvature_perf, ax = plt.subplots()

            field_1993_routines.contour_gain_vs_inter_fragment_rotation(
                cont_int_model,
                test_data_dict_of_dicts,
                c_len=9,
                frag_orient=fragment_orientation,
                n_runs=100,
                axis=ax
            )

            field_1993_routines.contour_gain_vs_inter_fragment_rotation(
                cont_int_model,
                test_data_dict_of_dicts,
                c_len=7,
                frag_orient=fragment_orientation,
                n_runs=100,
                axis=ax
            )

            fig_curvature_perf.suptitle(
                "Contour Curvature (Beta Rotations) Performance. Kernel @ index {0}, Frag orientation {1}".format(
                    target_kernel_idx, fragment_orientation))

            fig_curvature_perf.set_size_inches(11, 9)
            fig_curvature_perf.savefig(os.path.join(
                results_dir, 'beta_rotations_performance_kernel_{}.png'.format(target_kernel_idx)))

        # -------------------------------------------------------------------------------
        # Enhancement Gain vs Contour Length
        # -------------------------------------------------------------------------------
        print("Checking gain vs contour length performance ...")
        for fragment_orientation in fragment_orientation_arr:

            fig_c_len_perf, ax = plt.subplots()

            # Linear contours
            field_1993_routines.contour_gain_vs_length(
                cont_int_model,
                test_data_dict_of_dicts,
                beta=0,
                frag_orient=fragment_orientation,
                n_runs=100,
                axis=ax
            )

            # For inter-fragment rotation of 15 degrees
            field_1993_routines.contour_gain_vs_length(
                cont_int_model,
                test_data_dict_of_dicts,
                beta=15,
                frag_orient=fragment_orientation,
                n_runs=100,
                axis=ax
            )

            fig_c_len_perf.suptitle("Contour Length Performance. kernel @ index {0}, Frag orientation {1}".format(
                target_kernel_idx, fragment_orientation))

            fig_c_len_perf.set_size_inches(11, 9)
            fig_c_len_perf.savefig(os.path.join(
                results_dir, 'contour_len_performance_kernel_{}.png'.format(target_kernel_idx)))

        # -------------------------------------------------------------------------------
        # Enhancement Gain vs Fragment Spacing
        # -------------------------------------------------------------------------------
        print("Checking gain vs fragment spacing performance ...")
        for fragment_orientation in fragment_orientation_arr:

            fig_spacing_perf, ax = plt.subplots()

            # Linear contours
            field_1993_routines.contour_gain_vs_spacing(
                cont_int_model,
                test_data_dict_of_dicts,
                beta=0,
                frag_orient=fragment_orientation,
                n_runs=100,
                axis=ax
            )

            # For inter-fragment rotation of 15 degrees
            field_1993_routines.contour_gain_vs_spacing(
                cont_int_model,
                test_data_dict_of_dicts,
                beta=15,
                frag_orient=fragment_orientation,
                n_runs=100,
                axis=ax
            )

            fig_spacing_perf.suptitle("Fragment Spacing Performance. kernel @ index {0}, Frag orientation {1}".format(
                target_kernel_idx, fragment_orientation))

            fig_spacing_perf.set_size_inches(11, 9)
            fig_spacing_perf.savefig(os.path.join(
                results_dir, 'fragment_spacing_performance_kernel_{}.png'.format(target_kernel_idx)))

        # -------------------------------------------------------------------------------
        # Debug - Plot the performance on a test image
        # -------------------------------------------------------------------------------
        print("Checking performance on sample images ...")

        test_sample_output(
            data_dir=data_directory,
            tgt_filt_idx=target_kernel_idx,
            feat_extract_cb=feat_extract_callback,
            cont_int_cb=cont_int_callback,
            rslt_dir=results_dir,
            c_len=9,
            beta=15,
            alpha=0,
            img_idx=0
        )

        test_sample_output(
            data_dir=data_directory,
            tgt_filt_idx=target_kernel_idx,
            feat_extract_cb=feat_extract_callback,
            cont_int_cb=cont_int_callback,
            rslt_dir=results_dir,
            c_len=9,
            beta=30,
            alpha=0,
            img_idx=10
        )

    # -----------------------------------------------------------------------------------
    #  End
    # -----------------------------------------------------------------------------------
    # print Elapsed Time
    print("Total Elapsed Time {}".format(datetime.now() - start_time))

    os.remove(TEMP_WEIGHT_STORE_FILE)

    # Write the Summary of the run to a file
    with open(os.path.join(results_dir, 'summary.txt'), 'wb') as f_id:

        f_id.write("Model Hyper-Parameters : --------------------------------------\n")
        f_id.write("L1_loss: {}\n".format(cont_int_model.layers[2].l1_reg_loss_weight))
        f_id.write("Contour Integration rf size {}\n".format(cont_int_model.layers[2].n))
        f_id.write("Outer Relu alpha {}\n".format(cont_int_model.layers[2].outer_leaky_relu_alpha))
        f_id.write("Inner Relu alpha {}\n".format(cont_int_model.layers[2].inner_leaky_relu_alpha))
        f_id.write("\n")

        f_id.write("Training Parameters : --------------------------------------\n")
        f_id.write("Number of (train, test) images for each kernel: ({0}, {1}).\n".format(
            n_training_images, n_test_images))
        f_id.write("Number of Epochs: {}.\n".format(num_epochs))
        f_id.write("Batch Size: {} images.\n".format(batch_size))
        f_id.write("\n")

        f_id.write("Training Data Set Details: ---------------------------------\n")
        f_id.write("Data Directory: {}\n".format(data_directory))
        f_id.write("Contour Lengths: {}\n".format(contour_lengths))
        f_id.write("Fragment_spacing: {}\n".format(fragment_spacing))
        f_id.write("Beta: {}\n".format(beta_rotations))
        f_id.write("Alpha: {}\n".format(alpha_rotations))
        f_id.write("\n")

        f_id.write("Min Losses : -----------------------------------------------\n")

        for idx, target_kernel_idx in enumerate(target_kernel_idx_arr):
            loss_string = "Min Losses for kernel {0}: Train {1}, Test{2}".format(
                target_kernel_idx, min_loss_arr[idx][0], min_loss_arr[idx][1])
            print(loss_string)
            f_id.write(loss_string + '\n')

    # At end of Training, clear all contour integration kernels that are not trained
    train_sum_file = get_weights_training_summary_file(weights_store_file)
    prev_trained_idxes = np.array(list(get_prev_learnt_kernels(train_sum_file)))

    trained_kernel_idxes = np.concatenate((prev_trained_idxes, np.array(target_kernel_idx_arr)))
    trained_kernel_idxes = set(trained_kernel_idxes)

    clear_unlearnt_contour_integration_kernels(cont_int_model, trained_kernel_idxes)
