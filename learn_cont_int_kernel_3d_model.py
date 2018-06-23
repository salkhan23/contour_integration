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

import keras.backend as K
from keras.callbacks import TensorBoard

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


DATA_DIR = './data/curved_contours/filt_matched_frag'
MODEL_STORE_DIR = './trained_models/ContourIntegrationModel3d/filt_matched_frag'
# DATA_DIR = './data/curved_contours/orientation_matched'
# MODEL_STORE_DIR = './trained_models/ContourIntegrationModel3d/orientation_matched'

PREV_LEARNT_WEIGHTS = os.path.join(MODEL_STORE_DIR, "contour_integration_weights.hf")
# This file lists indices of contour integration kernels whose weights are stored
PREV_LEARNT_SUMMARY = os.path.join(MODEL_STORE_DIR, "summary.txt")

IMAGE_SIZE = (227, 227, 3)


def load_learnt_weights(model):

    print("Loading previously learnt weights")
    prev_learn_idx_set = set()

    if os.path.exists(PREV_LEARNT_WEIGHTS):
        model.load_weights(PREV_LEARNT_WEIGHTS, by_name=True)

        if os.path.exists(PREV_LEARNT_SUMMARY):

            with open(PREV_LEARNT_SUMMARY, 'r') as fid:
                read_in = fid.read()
                read_in = read_in.split()
                read_in = [int(k_idx) for k_idx in read_in]
                print("previously trained Kernels @ indexes {}".format(read_in))
                prev_learn_idx_set.update(read_in)

        else:
            print('summary File Does not exist')

    else:
        print("No previously trained weights file")

    return prev_learn_idx_set


def save_learnt_weights(model, tgt_filt_idx):

    print("Saving Learnt Weights")

    tgt_filt_prev_learnt = False
    prev_learn_idx_set = set()

    if os.path.exists(PREV_LEARNT_SUMMARY):
        with open(PREV_LEARNT_SUMMARY, 'r') as fid:
            read_in = fid.read()
            read_in = read_in.split()
            read_in = [int(k_idx) for k_idx in read_in]
            prev_learn_idx_set.update(read_in)

    if tgt_filt_idx in prev_learn_idx_set:
        tgt_filt_prev_learnt = True

    if not os.path.exists(MODEL_STORE_DIR):
        os.makedirs(MODEL_STORE_DIR)
    model.save_weights(PREV_LEARNT_WEIGHTS)

    # update the summary file
    if not tgt_filt_prev_learnt:
        with open(PREV_LEARNT_SUMMARY, 'a+') as fid:
            fid.write(str(tgt_filt_idx) + '\n')


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    tgt_kernel_idx = 10
    batch_size = 8

    update_common_shared_weights = True

    # ------------------------------------------------------------------------------------
    # Load data set keys
    # ------------------------------------------------------------------------------------
    train_data_key_loc = os.path.join(
        DATA_DIR, "train/filter_{}".format(tgt_kernel_idx), "data_key.pickle")

    test_data_key_loc = os.path.join(
        DATA_DIR, "test/filter_{}".format(tgt_kernel_idx), "data_key.pickle")

    with open(train_data_key_loc, 'rb') as handle:
        train_data_dict = pickle.load(handle)  # Returns a list of dictionaries

    with open(test_data_key_loc, 'rb') as handle:
        test_data_dict = pickle.load(handle)  # Returns a list of dictionaries

    # Set of images to train with
    # ---------------------------
    # train_set = [
    #     'c_len_1_beta_0',
    #     'c_len_1_beta_15',
    #     'c_len_1_beta_30',
    #     'c_len_1_beta_45',
    #     'c_len_1_beta_60',
    #
    #     'c_len_3_beta_0',
    #     'c_len_3_beta_15',
    #     'c_len_3_beta_30',
    #     'c_len_3_beta_45',
    #     'c_len_3_beta_60',
    #
    #     'c_len_5_beta_0',
    #     'c_len_5_beta_15',
    #     'c_len_5_beta_30',
    #     'c_len_5_beta_45',
    #     'c_len_5_beta_60',
    #
    #     'c_len_7_beta_0',
    #     'c_len_7_beta_15',
    #     'c_len_7_beta_30',
    #     'c_len_7_beta_45',
    #     'c_len_7_beta_60',
    #
    #     'c_len_9_beta_0',
    #     'c_len_9_beta_15',
    #     'c_len_9_beta_30',
    #     'c_len_9_beta_45',
    #     'c_len_9_beta_60',
    # ]
    train_set = train_data_dict

    # -----------------------------------------------------------------------------------
    # Contour Integration Model
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(tgt_kernel_idx)
    # cont_int_model.summary()

    # Target feature extracting kernel
    feat_extract_kernels = K.eval(cont_int_model.layers[1].weights[0])
    tgt_feat_extract_kernel = feat_extract_kernels[:, :, :, tgt_kernel_idx]

    # Load previously learnt weights
    # -------------------------------------------
    prev_trained_kernel_idx_set = load_learnt_weights(cont_int_model)

    # # Verify weights were loaded correctly
    # linear_contour_training.plot_contour_integration_weights_in_channels(
    #     start_weights, prev_trained_kernel_idx)

    # Store starting weights for comparision later
    start_weights, _ = cont_int_model.layers[2].get_weights()

    # Compile the model and setup Tensorboard callbacks
    # -------------------------------------------------
    cont_int_model.compile(optimizer='Adam', loss='mse')

    tensorboard = TensorBoard(
        log_dir='logs/{}'.format(time()),
        histogram_freq=1,
        write_grads=True,
        batch_size=1,
    )

    # ------------------------------------------------------------------------------------
    # Image Generators
    # ------------------------------------------------------------------------------------
    active_train_set = {}
    active_test_set = {}

    for set_id in train_set:
        if set_id in train_data_dict:
            active_train_set.update(train_data_dict[set_id])
            active_test_set.update(test_data_dict[set_id])
        else:
            ans = raw_input("{0} image set not in data key. Continue without (Y/anything else)".format(set_id))
            if 'y' in ans.lower():
                continue
            else:
                raise SystemExit()

    train_image_generator = image_generator_curve.DataGenerator(
        active_train_set,
        batch_size=batch_size,
        img_size=IMAGE_SIZE,
        shuffle=True,
    )

    test_image_generator = image_generator_curve.DataGenerator(
        active_test_set,
        batch_size=1000,
        img_size=IMAGE_SIZE,
        shuffle=True,
    )

    gen_out = iter(train_image_generator)
    test_images, test_labels = gen_out.next()

    # # Test the generator (sequence) object
    # gen_out = iter(train_image_generator)
    # X, y = gen_out.next()
    #
    # plt.figure()
    # plt.imshow(np.transpose(X[0, ], (1, 2, 0)))
    # plt.title("Expected gain {0}".format(y[0]))

    # -----------------------------------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------------------------------
    print("Learning Contour Integration kernels for Filter @ index {}".format(tgt_kernel_idx))

    history = cont_int_model.fit_generator(
        generator=train_image_generator,
        epochs=20,
        steps_per_epoch=5,
        verbose=2,
        validation_data=(test_images, test_labels),
        validation_steps=1,
        # max_q_size=1,
        workers=8,
        callbacks=[tensorboard]
    )

    # Save the model/weights
    # --------------------------------------
    # Update stored shared weights
    if update_common_shared_weights:
        save_learnt_weights(cont_int_model, tgt_kernel_idx)

    # Save the weights of the kernel trained individually
    individually_trained_model_dir = os.path.join(MODEL_STORE_DIR, "filter_{}".format(tgt_kernel_idx))
    if not os.path.exists(individually_trained_model_dir):
        os.makedirs(individually_trained_model_dir)

    stored_weights_file = os.path.join(
        MODEL_STORE_DIR, "filter_{}".format(tgt_kernel_idx), 'trained_model.hf')
    cont_int_model.save_weights(stored_weights_file)

    # -----------------------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------------------
    # 1.  Plot Loss vs Time
    # --------------------------------------
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()

    # 2.  Plot Learnt Kernels
    # --------------------------------------
    learnt_weights, _ = cont_int_model.layers[2].get_weights()

    # Target Kernel
    fig, ax_arr = plt.subplots(1, 2)
    linear_contour_training.plot_contour_integration_weights_in_channels(
        start_weights, tgt_kernel_idx, axis=ax_arr[0])
    linear_contour_training.plot_contour_integration_weights_in_channels(
        learnt_weights, tgt_kernel_idx, axis=ax_arr[1])
    fig.suptitle('Input channel feeding into output channel @ {}'.format(tgt_kernel_idx))

    # # All output channels receiving input from target input channel
    # fig, ax_arr = plt.subplots(1, 2)
    # linear_contour_training.plot_contour_integration_weights_out_channels(
    #     start_weights, tgt_kernel_idx, axis=ax_arr[0])
    # linear_contour_training.plot_contour_integration_weights_out_channels(
    #     learnt_weights, tgt_kernel_idx, axis=ax_arr[1])
    # fig.suptitle('Output channel feed by input channel @ {}'.format(tgt_kernel_idx))

    # Previously trained kernels
    for prev_trained_kernel_idx in prev_trained_kernel_idx_set:
        fig, ax_arr = plt.subplots(1, 2)
        linear_contour_training.plot_contour_integration_weights_in_channels(
            start_weights, prev_trained_kernel_idx, axis=ax_arr[0])
        linear_contour_training.plot_contour_integration_weights_in_channels(
            learnt_weights, prev_trained_kernel_idx, axis=ax_arr[1])
        fig.suptitle('Input channel feeding into output channel @ {} (Previously learnt)'.format(
            prev_trained_kernel_idx))

    # 3. Fields - 1993 - Experiment 1 - Curvature vs Gain
    # --------------------------------------------------------------------------
    list_of_data_sets = train_data_dict.keys()

    if 'rot' in list_of_data_sets[0]:
        frag_orientation_arr = [np.int(item.split("rot_")[1]) for item in list_of_data_sets]
        frag_orientation_arr = set(frag_orientation_arr)
    else:
        frag_orientation_arr = [None]

    for frag_orientation in frag_orientation_arr:
        fig, ax = plt.subplots()

        field_1993_routines.contour_gain_vs_inter_fragment_rotation(
            cont_int_model,
            test_data_dict,
            c_len=9,
            frag_orient=frag_orientation,
            n_runs=100,
            axis=ax
        )

        field_1993_routines.contour_gain_vs_inter_fragment_rotation(
            cont_int_model,
            test_data_dict,
            c_len=7,
            frag_orient=frag_orientation,
            n_runs=100,
            axis=ax
        )

        fig.suptitle("Contour Rotation {}".format(frag_orientation))

    # 4. Enhancement gain vs contour length
    # -------------------------------------------------------------------------
    list_of_data_sets = train_data_dict.keys()

    if 'rot' in list_of_data_sets[0]:
        frag_orientation_arr = [np.int(item.split("rot_")[1]) for item in list_of_data_sets]
        frag_orientation_arr = set(frag_orientation_arr)
    else:
        frag_orientation_arr = [None]

    for frag_orientation in frag_orientation_arr:

        fig, ax = plt.subplots()

        # Linear contours
        field_1993_routines.contour_gain_vs_length(
            cont_int_model,
            test_data_dict,
            beta=0,
            frag_orient=frag_orientation,
            n_runs=100,
            axis=ax
        )

        # For inter-fragment rotation of 15 degrees
        field_1993_routines.contour_gain_vs_length(
            cont_int_model,
            test_data_dict,
            beta=15,
            frag_orient=frag_orientation,
            n_runs=100,
            axis=ax
        )

        fig.suptitle("Contour Rotation {}".format(frag_orientation))
