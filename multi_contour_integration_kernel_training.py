# -------------------------------------------------------------------------------------------------
#  Learn contour enhancement kernels for the 3d contour integration model
#  Model is trained on stimuli from Fields - 1993.
#
# Author: Salman Khan
# Date  : 06/05/18
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pickle
import os
from time import time
import numpy as np

import keras.backend as keras_backend
from keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import gc

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


def get_train_n_test_data_dictionaries(tgt_filt_idx, data_dir):
    """
    Given the target filter to train, pull out its train/test data dictionaries.


    :param tgt_filt_idx:
    :param data_dir:
    :return:
    """
    train_data_key_loc = os.path.join(
        data_dir, "train/filter_{}".format(tgt_filt_idx), "data_key.pickle")

    test_data_key_loc = os.path.join(
        data_dir, "test/filter_{}".format(tgt_filt_idx), "data_key.pickle")

    with open(train_data_key_loc, 'rb') as handle:
        train_data_dict = pickle.load(handle)  # Returns a list of dictionaries

    with open(test_data_key_loc, 'rb') as handle:
        test_data_dict = pickle.load(handle)  # Returns a list of dictionaries

    return train_data_dict, test_data_dict


def get_train_summary_file(weights_store_file):
    basedir = os.path.dirname(weights_store_file)
    return os.path.join(basedir, 'train_summary.txt')


def get_trained_contour_integration_kernel_indices(training_summary_file):
    prev_learnt_idx_set = set()

    if os.path.exists(training_summary_file):
        with open(training_summary_file, 'rb+') as fid:
            for line in fid:
                k_idx = line.split()[0]
                prev_learnt_idx_set.add(int(k_idx))

    # print("Previously learnt kernels {}".format(prev_learnt_idx_set))
    return prev_learnt_idx_set


def save_learnt_weights(model, tgt_filt_idx, weights_store_file, training_summary_file):
    """

    :param model:
    :param tgt_filt_idx:
    :param weights_store_file:
    :param training_summary_file:
    :return:
    """
    print("Saving learnt weights")
    prev_learn_idx_set = get_trained_contour_integration_kernel_indices(training_summary_file)
    prev_learn_idx_set.update([tgt_filt_idx])

    if not os.path.exists(os.path.dirname(weights_store_file)):
        os.makedirs(os.path.dirname(weights_store_file))

    model.save_weights(weights_store_file)

    with open(training_summary_file, 'w+') as fid:
        for entry in prev_learn_idx_set:
            fid.write(str(entry) + '\n')


def load_learnt_weights(model, learnt_weights_file, training_summary_file):
    """

    :param model:
    :param learnt_weights_file:
    :param training_summary_file:
    :return:
    """
    print("Loading previously learnt contour Integration kernels")
    prev_learnt_idx_list = []

    if os.path.exists(learnt_weights_file):
        model.load_weights(learnt_weights_file, by_name=True)

        if os.path.exists(training_summary_file):
            # Format: kernel_idx
            prev_learnt_idx_list = get_trained_contour_integration_kernel_indices(training_summary_file)

        else:
            print("weights loaded but no training summary file")
    else:
        print("No previously trained weights file")

    return prev_learnt_idx_list


def learn_contour_integration_kernel(
        tgt_filt_idx, data_dir, weights_store_file=None, b_size=32, n_epochs=2, store_weights=True):
    """

    :param tgt_filt_idx:
    :param data_dir: 
    :param weights_store_file: 
    :param b_size: Batch size
    :param n_epochs:
    :param store_weights: [True], whether to update stored weights
    
    :return: trained_model
    
    """
    print("Learning Contour Integration Kernel @ {}".format(tgt_filt_idx))
    # -----------------------------------------------------------------------------------
    # Build the model
    # -----------------------------------------------------------------------------------
    print("Loading model...")
    model = contour_integration_model_3d.build_contour_integration_model(tgt_filt_idx)
    # model.summary()clear

    # Load the weights
    # ----------------
    # The weights summary file keeps track of which  contour integration kernels have been learnt
    # Code assumes a fixed name and location (same as weights file) for the summary file
    training_summary_file = get_train_summary_file(weights_store_file)

    load_learnt_weights(model, weights_store_file, training_summary_file)

    # Compile the Model
    model.compile(optimizer='Adam', loss='mse')

    # Save starting weights for comparision later
    start_w, _ = model.layers[2].get_weights()

    # -----------------------------------------------------------------------------------
    # Build the Data Generators
    # -----------------------------------------------------------------------------------
    print("Building data generators...")

    train_data_dict, test_data_dict = get_train_n_test_data_dictionaries(tgt_filt_idx, data_dir)
    train_set = train_data_dict  # Use all the data to train the model

    # Data generator
    # --------------------
    active_train_set = {}
    active_test_set = {}

    for set_id in train_set:
        active_train_set.update(train_data_dict[set_id])
        active_test_set.update(test_data_dict[set_id])

    train_image_generator = image_generator_curve.DataGenerator(
        active_train_set,
        batch_size=b_size,
        shuffle=True,
    )

    test_image_generator = image_generator_curve.DataGenerator(
        active_test_set,
        batch_size=len(active_test_set.keys()),
        shuffle=True,
    )
    # Because the tensorboard callback does not work with data generators, just load all the test images
    gen_out = iter(test_image_generator)
    test_images, test_labels = gen_out.next()

    # -----------------------------------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------------------------------
    print("Learning contour integration kernel @ index {}...".format(tgt_filt_idx))

    # n_images_to_process
    n_train_images = len(active_train_set.keys())
    n_train_images_to_process = n_train_images * n_epochs
    # Usage of epochs is strange with fit generator, for some reason weights are updated only after an epoch
    # but we want the weights to be updated much more frequently
    steps_per_epoch = 10
    images_per_step = steps_per_epoch * b_size
    train_epochs = n_train_images_to_process / images_per_step

    n_test_images = 1000
    # Model takes a long time to run if we validate with the full set of test images,
    # for now limit the test images to size 500
    if test_images.shape[0] > n_test_images:
        test_images = test_images[0:n_test_images, ]
        test_labels = test_labels[0:n_test_images, ]

    print("Number of train images {0}, epochs {1}, training epochs {2}, test images {3}. Images per step {4}".format(
        n_train_images,
        n_epochs,
        train_epochs,
        test_images.shape[0],
        images_per_step
    ))

    best_learnt_weights_file = 'best_weights.hf'

    # -----------------------------------------------------
    # Callbacks
    tensorboard = TensorBoard(
        log_dir='logs/{}'.format(time()),
        # histogram_freq=1,
        # write_grads=True,
        # write_images=False,
        # batch_size=1,  # For histogram
    )

    checkpoint = ModelCheckpoint(
        best_learnt_weights_file,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        save_weights_only=True,
    )

    # Training
    # --------
    history = model.fit_generator(
        generator=train_image_generator,
        epochs=train_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        validation_data=(test_images, test_labels),
        validation_steps=1,
        # max_q_size=1,
        workers=8,
        callbacks=[tensorboard, checkpoint]
    )

    # Save the Weights
    if store_weights:
        model.load_weights(best_learnt_weights_file, by_name=True)
        os.remove(best_learnt_weights_file)
        save_learnt_weights(model, tgt_filt_idx, weights_store_file, training_summary_file)

    # Plot Loss vs Time
    # --------------------------------------
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('model loss. Contour Integration kernel @ {}'.format(tgt_filt_idx))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()

    # Clean Up
    del train_image_generator, test_image_generator, test_images, test_labels

    return model, start_w


def plot_start_n_learnt_contour_integration_kernels(model, tgt_filt_idx, start_w=None, feat_extract_filt=None):
    """

    :param model:
    :param tgt_filt_idx:
    :param start_w: complete set of weights at star of training [Optional]
    :param feat_extract_filt: [Optional]
    :return:
    """
    f, ax_arr = plt.subplots(1, 3)

    # Learnt weights
    learnt_w, _ = model.layers[2].get_weights()
    linear_contour_training.plot_contour_integration_weights_in_channels(
        learnt_w, tgt_filt_idx, axis=ax_arr[0])
    ax_arr[0].set_title("Learnt Contour Int")

    if start_w is not None:
        linear_contour_training.plot_contour_integration_weights_in_channels(
            start_w, tgt_filt_idx, axis=ax_arr[1])
        ax_arr[1].set_title("Initial Contour Int")

    if feat_extract_filt is not None:
        ax_arr[2].imshow(feat_extract_filt)
        ax_arr[2].set_title("Feature Extract")

    f.suptitle("Input channels feeding of contour integration kernel @ index {}".format(tgt_filt_idx))


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    keras_backend.set_image_dim_ordering('th')

    # data_directory = './data/curved_contours/filt_matched_frag'
    # model_weights_store_file = \
    #     './trained_models/ContourIntegrationModel3d/filt_matched_frag/contour_integration_weights.hf'

    data_directory = './data/curved_contours/orientation_matched'
    model_weights_store_file = \
        './trained_models/ContourIntegrationModel3d/orientation_matched/contour_integration_weights.hf'

    # -----------------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------------
    target_kernel_idx_arr = [22, 48, 66, 73, 78]

    cont_int_model = None
    for target_kernel_idx in target_kernel_idx_arr:

        # Train the model
        cont_int_model, start_weights = learn_contour_integration_kernel(
            target_kernel_idx,
            data_directory,
            model_weights_store_file,
            n_epochs=20
        )

        # Plot the trained kernel
        feat_extract_kernels, _ = cont_int_model.layers[1].get_weights()
        tgt_feat_extract_kernel = feat_extract_kernels[:, :, :, target_kernel_idx]
        normalized_tgt_feat_extract_kernel = \
            (tgt_feat_extract_kernel - tgt_feat_extract_kernel.min()) / \
            (tgt_feat_extract_kernel.max() - tgt_feat_extract_kernel.min())

        plot_start_n_learnt_contour_integration_kernels(
            cont_int_model,
            target_kernel_idx,
            start_weights,
            normalized_tgt_feat_extract_kernel
        )

        # Cleanup after every iteration
        temp = tf.get_default_graph().get_operations()
        print("Number of tensorflow operations {}".format(len(temp)))
        del cont_int_model
        gc.collect()
        keras_backend.clear_session()
        # raw_input("Continue?")

    # ----------------------------------------------------------------------------------





    # -----------------------------------------------------------------------------------
    # # Plot all learnt kernels
    # if cont_int_model is not None:
    #     train_summary_file = get_train_summary_filename_from_stored_weights_file(model_weights_store_file)
    #     train_kernel_idxs = get_trained_weights_indices(train_summary_file)
    #     print ("Previously Trained kernels Indices @ {}".format(train_kernel_idxs))
    #
    #     for idx in train_kernel_idxs:
    #         plot_start_n_learnt_contour_integration_kernels(
    #             cont_int_model,
    #             idx)
    #         plt.suptitle("[Previously Trained] Contour Integration Kernel @ index {}".format(idx))

    # -----------------------------------------------------------------------------------
    #  Fields Experiment 1- Contour Gain vs contour Curvature
    # -----------------------------------------------------------------------------------
    keras_backend.clear_session()
    gc.collect()

    c_len_arr = [7, 9]

    # Get list of trained contour integration kernels
    train_summary = get_train_summary_file(model_weights_store_file)
    trained_kernel_idx_arr = list(get_trained_contour_integration_kernel_indices(train_summary))
    print("Contour Integration kernel @ indices {} are trained".format(trained_kernel_idx_arr))

    # Load the model
    # Any kernel is fine, all weights are loaded
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(trained_kernel_idx_arr[0])
    load_learnt_weights(cont_int_model, model_weights_store_file, train_summary)

    for target_filter in trained_kernel_idx_arr:
        _, test_data_dictionary = get_train_n_test_data_dictionaries(target_filter, data_directory)

        list_of_data_sets = test_data_dictionary.keys()

        if 'rot' in list_of_data_sets[0]:
            frag_orientation_arr = [np.int(item.split("rot_")[1]) for item in list_of_data_sets]
            frag_orientation_arr = set(frag_orientation_arr)
        else:
            frag_orientation_arr = [None]

        for frag_orientation in frag_orientation_arr:
            fig, ax = plt.subplots()

            field_1993_routines.contour_gain_vs_inter_fragment_rotation(
                cont_int_model,
                test_data_dictionary,
                c_len=9,
                frag_orient=frag_orientation,
                n_runs=100,
                axis=ax
            )

            field_1993_routines.contour_gain_vs_inter_fragment_rotation(
                cont_int_model,
                test_data_dictionary,
                c_len=7,
                frag_orient=frag_orientation,
                n_runs=100,
                axis=ax
            )

            fig.suptitle("Contour Rotation {}".format(frag_orientation))
