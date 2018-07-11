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
    if not os.path.exists(os.path.dirname(basedir)):
        os.makedirs(os.path.dirname(basedir))

    model.save_weights(w_store_file)

    train_summary_file = get_weights_training_summary_file(w_store_file)

    prev_learnt_kernels_set = get_prev_learnt_kernels(train_summary_file)
    prev_learnt_kernels_set.update([tgt_filt_idx])

    with open(train_summary_file, 'w+') as fid:
        for entry in prev_learnt_kernels_set:
            fid.write(str(entry) + '\n')


def _get_train_n_test_dictionary_of_dictionaries(tgt_filt_idx, data_dir):
    train_data_key_loc = os.path.join(
        data_dir, "train/filter_{}".format(tgt_filt_idx), "data_key.pickle")

    test_data_key_loc = os.path.join(
        data_dir, "test/filter_{}".format(tgt_filt_idx), "data_key.pickle")

    with open(train_data_key_loc, 'rb') as handle:
        train_data_list_of_dict = pickle.load(handle)  # Returns a list of dictionaries

    with open(test_data_key_loc, 'rb') as handle:
        test_data_list_of_dict = pickle.load(handle)  # Returns a list of dictionaries

    return train_data_list_of_dict, test_data_list_of_dict


def get_train_n_test_data_keys(tgt_filt_idx, data_dir, c_len=None, beta=None, frag_orient=None):
    """
    A data key is a dictionary of file location:expected gain tuples.
    Two Dictionaries are returned: one for testing and one for testing model performance

    :param tgt_filt_idx:
    :param data_dir:
    :param c_len:
    :param beta:
    :param frag_orient:
    :return:
    """
    train_data_dict_of_dict, test_data_dict_of_dict =\
        _get_train_n_test_dictionary_of_dictionaries(tgt_filt_idx, data_dir)

    train_set = train_data_dict_of_dict.keys()
    if c_len is not None:
        train_set = [x for x in train_set if 'c_len_{}'.format(c_len) in x]
    if beta is not None:
        train_set = [x for x in train_set if 'beta_{}'.format(beta) in x]
    if frag_orient is not None:
        train_set = [x for x in train_set if 'rot_{}'.format(frag_orient) in x]

    # Single dictionary containing (image file location, expected gain)
    active_train_set = {}
    active_test_set = {}

    for set_id in train_set:
        active_train_set.update(train_data_dict_of_dict[set_id])
        active_test_set.update(test_data_dict_of_dict[set_id])

    return active_train_set, active_test_set


def train_contour_integration_kernel(
        model, tgt_filt_idx, data_dir, b_size=32, n_epochs=200, training_cb=None, steps_per_epoch=10, axis=None):
    """

    :param model:
    :param tgt_filt_idx:
    :param data_dir:
    :param b_size:
    :param n_epochs:
    :param steps_per_epoch:
    :param training_cb:
    :param axis:
    :return:
    """
    print("Learning contour integration kernel @ index {} ...".format(tgt_filt_idx))

    # Modify the contour integration training model to train the target kernel
    contour_integration_model_3d.update_contour_integration_kernel(model, tgt_filt_idx)
    model.compile(optimizer='Adam', loss='mse')

    # -----------------------------------------------------------------------------------
    # Build the Data Generators
    # -----------------------------------------------------------------------------------
    print("Building data generators...")

    train_data_dict, test_data_dict = get_train_n_test_data_keys(tgt_filt_idx, data_dir)
    train_image_generator = image_generator_curve.DataGenerator(
        test_data_dict,
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

    print("Training ...")
    history = model.fit_generator(
        generator=train_image_generator,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        validation_data=(test_images, test_labels),
        validation_steps=1,
        # max_q_size=1,
        workers=8,
        callbacks=training_cb
    )

    print("Minimum Training Loss {0}, Validation loss {1}".format(
        min(history.history['loss']),
        min(history.history['val_loss'])
    ))

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

    feat_extract_w, _ = cont_int_model.layers[1].get_weights()
    tgt_feat_extract_w = feat_extract_w[:, :, :, tgt_filt_idx]

    normalized_tgt_feat_extract_w = (tgt_feat_extract_w - tgt_feat_extract_w.min()) / \
        (tgt_feat_extract_w.max() - tgt_feat_extract_w.min())

    ax_arr[2].imshow(normalized_tgt_feat_extract_w)
    ax_arr[2].set_title("Feature Extract")

    f.suptitle("Input channels feeding of contour integration kernel @ index {}".format(tgt_filt_idx))


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


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    keras_backend.set_image_dim_ordering('th')
    keras_backend.clear_session()
    start_time = datetime.now()

    batch_size = 32
    num_epochs = 200

    save_weights = True
    prev_train_weights = None

    # target_kernel_idx_arr = [5, 10]
    # data_directory = './data/curved_contours/filt_matched_frag'
    # weights_store_file =
    # './trained_models/ContourIntegrationModel3d/filt_matched_frag/contour_integration_weights_2.hf'
    # prev_train_weights =
    # './trained_models/ContourIntegrationModel3d/filt_matched_frag/contour_integration_weights.hf'

    # target_kernel_idx_arr = [5, 10, 19, 20, 21, 22]
    target_kernel_idx_arr = [48, 49, 51, 59, 62, 64, 65, 66, 68, 72, 73, 74, 76, 77, 79, 80, 82, 85]
    data_directory = './data/curved_contours/orientation_matched2'
    weights_store_file = \
        './trained_models/ContourIntegrationModel3d/orientation_matched/contour_integration_weights_2.hf'
    prev_train_weights = \
        './trained_models/ContourIntegrationModel3d/orientation_matched/contour_integration_weights.hf'

    # -----------------------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(
        tgt_filt_idx=0,
        rf_size=25,
        inner_leaky_relu_alpha=0.7,
        outer_leaky_relu_alpha=0.94,
        l1_reg_loss_weight=0.01
    )

    if prev_train_weights is not None:
        load_pretrained_weights(cont_int_model, prev_train_weights)

    start_weights, _ = cont_int_model.layers[2].get_weights()

    # -------------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------------
    fig, loss_vs_epoch_ax = plt.subplots()

    for target_kernel_idx in target_kernel_idx_arr:

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

        train_contour_integration_kernel(
            model=cont_int_model,
            tgt_filt_idx=target_kernel_idx,
            data_dir=data_directory,
            b_size=batch_size,
            n_epochs=num_epochs,
            training_cb=callbacks,
            steps_per_epoch=10,
            axis=loss_vs_epoch_ax
        )

        # load best weights
        cont_int_model.load_weights(TEMP_WEIGHT_STORE_FILE)  # load best weights

        # Save the learnt weights
        if save_weights:
            save_learnt_weights(cont_int_model, target_kernel_idx, weights_store_file)

        # Cleanup
        del checkpoint, tensorboard, callbacks

        # -------------------------------------------------------------------------------
        # Plot Learnt Kernels
        # -------------------------------------------------------------------------------
        plot_start_n_learnt_contour_integration_kernels(
            cont_int_model,
            target_kernel_idx,
            start_weights,
        )

        # # -------------------------------------------------------------------------------
        # # Todo: Should be moved to another File
        # train_data_dict_of_dicts, test_data_dict_of_dicts = _get_train_n_test_dictionary_of_dictionaries(
        #     target_kernel_idx,
        #     data_directory,
        # )
        # # get list of considered orientations
        # list_of_data_sets = train_data_dict_of_dicts.keys()
        #
        # if 'rot' in list_of_data_sets[0]:
        #     fragment_orientation_arr = [np.int(item.split("rot_")[1]) for item in list_of_data_sets]
        #     fragment_orientation_arr = set(fragment_orientation_arr)
        # else:
        #     fragment_orientation_arr = [None]
        #
        # # -------------------------------------------------------------------------------
        # #  Fields - 1993 - Experiment 1 - Curvature vs Gain
        # # -------------------------------------------------------------------------------
        # for fragment_orientation in fragment_orientation_arr:
        #     fig, ax = plt.subplots()
        #
        #     field_1993_routines.contour_gain_vs_inter_fragment_rotation(
        #         cont_int_model,
        #         test_data_dict_of_dicts,
        #         c_len=9,
        #         frag_orient=fragment_orientation,
        #         n_runs=100,
        #         axis=ax
        #     )
        #
        #     field_1993_routines.contour_gain_vs_inter_fragment_rotation(
        #         cont_int_model,
        #         test_data_dict_of_dicts,
        #         c_len=7,
        #         frag_orient=fragment_orientation,
        #         n_runs=100,
        #         axis=ax
        #     )
        #
        #     fig.suptitle("Contour Integration kernel @ index {0}, Fragment orientation {1}".format(
        #         target_kernel_idx, fragment_orientation))
        #
        # # -------------------------------------------------------------------------------
        # # Enhancement gain vs contour length
        # # -------------------------------------------------------------------------------
        # for fragment_orientation in fragment_orientation_arr:
        #
        #     fig, ax = plt.subplots()
        #
        #     # Linear contours
        #     field_1993_routines.contour_gain_vs_length(
        #         cont_int_model,
        #         test_data_dict_of_dicts,
        #         beta=0,
        #         frag_orient=fragment_orientation,
        #         n_runs=100,
        #         axis=ax
        #     )
        #
        #     # For inter-fragment rotation of 15 degrees
        #     field_1993_routines.contour_gain_vs_length(
        #         cont_int_model,
        #         test_data_dict_of_dicts,
        #         beta=15,
        #         frag_orient=fragment_orientation,
        #         n_runs=100,
        #         axis=ax
        #     )
        #
        #     fig.suptitle("Contour Integration kernel @ index {0}, Fragment orientation {1}".format(
        #         target_kernel_idx, fragment_orientation))

        print("Training kernel {0} took {1}".format(target_kernel_idx, datetime.now() - kernel_training_start_time))

    # ----------------------------------------------------------------------------------------------
    # At the end of training set all contour integration kernels that were not trained to zero
    # ----------------------------------------------------------------------------------------------
    train_sum_file = get_weights_training_summary_file(weights_store_file)

    prev_trained_idxes = np.array(list(get_prev_learnt_kernels(train_sum_file)))
    trained_kernel_idxes = np.concatenate((prev_trained_idxes, np.array(target_kernel_idx_arr)))
    trained_kernel_idxes = set(trained_kernel_idxes)

    clear_unlearnt_contour_integration_kernels(cont_int_model, trained_kernel_idxes)

    # Verify Kernels are cleared properly
    # ------------------------------------
    plot_start_n_learnt_contour_integration_kernels(
        cont_int_model,
        target_kernel_idx_arr[0],
    )

    plot_start_n_learnt_contour_integration_kernels(
        cont_int_model,
        0,
    )

    print("Total Elapsed Time {}".format(datetime.now() - start_time))
