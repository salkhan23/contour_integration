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
import gc

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


DATA_DIR = './data/curved_contours'
IMAGE_SIZE = (227, 227, 3)

PREV_LEARNT_WEIGHTS = os.path.join(
        "trained_models", "ContourIntegrationModel3d", "contour_integration_weights.hf")
PREV_LEARNT_SUMMARY = os.path.join(
        "trained_models", "ContourIntegrationModel3d", "summary.txt")


def load_learnt_weights(model):

    print("Loading previously learnt weights")

    if os.path.exists(PREV_LEARNT_WEIGHTS):
        model.load_weights(PREV_LEARNT_WEIGHTS, by_name=True)

        if os.path.exists(PREV_LEARNT_SUMMARY):

            with open(PREV_LEARNT_SUMMARY, 'r') as fid:
                read_in = fid.read()
                print("previously trained Kernels @ indexes")
                print(read_in)
        else:
            print('summary File Does not exist')

    else:
        print("No previously trained weights file")


def save_learnt_weights(model, tgt_filt_idx):

    tgt_filt_prev_learnt = False

    print("Saving Learnt Weights")

    if os.path.exists(PREV_LEARNT_SUMMARY):
        with open(PREV_LEARNT_SUMMARY, 'rb+') as fid:
            read_in = fid.read()

        for line in read_in:
            if str(tgt_filt_idx) in line:
                tgt_filt_prev_learnt = True
                print ("Updating stored weights at index {}".format(tgt_filt_idx))
                break

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
    gc.collect()
    K.set_image_dim_ordering('th')

    tgt_kernel_idx = 5
    batch_size = 8

    prev_trained_kernel_idx = 10

    update_common_shared_weights = True

    # Set of images to train with
    train_set = [
        'c_len_1_beta_0',
        'c_len_1_beta_15',
        'c_len_1_beta_30',
        'c_len_1_beta_45',
        'c_len_1_beta_60',

        'c_len_3_beta_0',
        'c_len_3_beta_15',
        'c_len_3_beta_30',
        'c_len_3_beta_45',
        'c_len_3_beta_60',

        'c_len_5_beta_0',
        'c_len_5_beta_15',
        'c_len_5_beta_30',
        'c_len_5_beta_45',
        'c_len_5_beta_60',

        'c_len_7_beta_0',
        'c_len_7_beta_15',
        'c_len_7_beta_30',
        'c_len_7_beta_45',
        'c_len_7_beta_60',

        'c_len_9_beta_0',
        'c_len_9_beta_15',
        'c_len_9_beta_30',
        'c_len_9_beta_45',
        'c_len_9_beta_60',
    ]

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
    load_learnt_weights(cont_int_model)

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
    train_data_key_loc = os.path.join(
        DATA_DIR, "train/filter_{}".format(tgt_kernel_idx), "data_key.pickle")

    test_data_key_loc = os.path.join(
        DATA_DIR, "test/filter_{}".format(tgt_kernel_idx), "data_key.pickle")

    with open(train_data_key_loc, 'rb') as handle:
        train_data_dict = pickle.load(handle)  # Returns a list of dictionaries

    with open(test_data_key_loc, 'rb') as handle:
        test_data_dict = pickle.load(handle)  # Returns a list of dictionaries

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
        epochs=40,
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
    stored_weights_file = os.path.join(
        DATA_DIR, "train", "filter_{}".format(tgt_kernel_idx), 'trained_model.hf')
    cont_int_model.save_weights(stored_weights_file)

    # Make a new model and load the saved weights
    # new_model = contour_integration_model_3d.build_contour_integration_model(tgt_kernel_idx)
    # new_model.load_weights(stored_weights_file)

    # -----------------------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------------------
    # 1. Loss vs Time
    # --------------------------------------
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()

    # 2. Learnt Kernels
    # --------------------------------------
    learnt_weights, _ = cont_int_model.layers[2].get_weights()

    # --------------
    # Target Kernel
    # --------------
    # All input channels feeding into target output filter
    fig, ax_arr = plt.subplots(1, 2)
    linear_contour_training.plot_contour_integration_weights_in_channels(
        start_weights, tgt_kernel_idx, axis=ax_arr[0])
    linear_contour_training.plot_contour_integration_weights_in_channels(
        learnt_weights, tgt_kernel_idx, axis=ax_arr[1])

    fig.suptitle('Input channel feeding into output channel @ {}'.format(tgt_kernel_idx))

    # All output channels receiving input from target input channel
    fig, ax_arr = plt.subplots(1, 2)
    linear_contour_training.plot_contour_integration_weights_out_channels(
        start_weights, tgt_kernel_idx, axis=ax_arr[0])

    linear_contour_training.plot_contour_integration_weights_out_channels(
        learnt_weights, tgt_kernel_idx, axis=ax_arr[1])

    fig.suptitle('Output channel feed by input channel @ {}'.format(tgt_kernel_idx))

    # -----------------
    # Non Target Kernel
    # -----------------
    fig, ax_arr = plt.subplots(1, 2)
    linear_contour_training.plot_contour_integration_weights_in_channels(
        start_weights, prev_trained_kernel_idx, axis=ax_arr[0])
    linear_contour_training.plot_contour_integration_weights_in_channels(
        learnt_weights, prev_trained_kernel_idx, axis=ax_arr[1])

    fig.suptitle('Input channel feeding into output channel @ {} (Previously learnt)'.format(
        prev_trained_kernel_idx))

    # 3. Fields - 1993 - Experiment 1 - Curvature vs Gain
    # --------------------------------------------------------------------------
    _, ax = plt.subplots()

    field_1993_routines.contour_gain_vs_inter_fragment_rotation(
        cont_int_model,
        test_data_dict,
        c_len=9,
        n_runs=100,
        axis=ax
    )

    field_1993_routines.contour_gain_vs_inter_fragment_rotation(
        cont_int_model,
        test_data_dict,
        c_len=7,
        n_runs=100,
        axis=ax
    )

    # 4. Enhancement gain vs contour length
    # -------------------------------------------------------------------------
    _, ax = plt.subplots()

    # Linear contours
    field_1993_routines.contour_gain_vs_length(
        cont_int_model,
        test_data_dict,
        0,
        n_runs=100,
        axis=ax
    )

    # For inter-fragment rotation of 15 degrees
    field_1993_routines.contour_gain_vs_length(
        cont_int_model,
        test_data_dict,
        15,
        n_runs=100,
        axis=ax
    )

    # 5. Enhancement Visualization over Sample Images
    # -----------------------------------------------
    # 1. Linear Contour
    image_idx = 0
    beta = 0
    c_len = 9

    image_loc = os.path.join(
        DATA_DIR,
        'train',
        'filter_{}'.format(tgt_kernel_idx),
        'c_len_{}'.format(c_len),
        'beta_{}'.format(beta),
        'c_len_{0}_beta_{1}__{2}.png'.format(c_len, beta, image_idx)
    )
    field_1993_routines.plot_activations(cont_int_model, image_loc, tgt_kernel_idx)
    plt.suptitle("Linear Contour")

    # 2. 15 Rotation Contour
    image_idx = 0
    beta = 15
    c_len = 9

    image_loc = os.path.join(
        DATA_DIR,
        'train',
        'filter_{}'.format(tgt_kernel_idx),
        'c_len_{}'.format(c_len),
        'beta_{}'.format(beta),
        'c_len_{0}_beta_{1}__{2}.png'.format(c_len, beta, image_idx)
    )
    field_1993_routines.plot_activations(cont_int_model, image_loc, tgt_kernel_idx)
    plt.suptitle("Contour with inter-fragment rotation of 15")
