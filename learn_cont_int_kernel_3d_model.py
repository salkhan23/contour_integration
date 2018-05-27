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

import keras.backend as K
from keras.preprocessing.image import load_img

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

if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    tgt_kernel_idx = 5
    batch_size = 10

    # Set of images to train with
    train_set = [
        'c_len_1_beta_0',

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

    # Callbacks to get activations of feature extract and contour integration layers
    feat_extract_act_cb = alex_net_utils.get_activation_cb(cont_int_model, 1)
    cont_int_act_cb = alex_net_utils.get_activation_cb(cont_int_model, 2)

    # Store starting weights for comparision later
    start_weights, _ = cont_int_model.layers[2].get_weights()

    # ------------------------------------------------------------------------------------
    # Image Generators
    # ------------------------------------------------------------------------------------
    train_data_key_loc = os.path.join(
        DATA_DIR, "train//filter_{}".format(tgt_kernel_idx), "data_key.pickle")

    test_data_key_loc = os.path.join(
        DATA_DIR, "test//filter_{}".format(tgt_kernel_idx), "data_key.pickle")

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
        batch_size=batch_size,
        img_size=IMAGE_SIZE,
        shuffle=True,
    )

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
    history = cont_int_model.fit_generator(
        generator=train_image_generator,
        epochs=100,
        steps_per_epoch=10,
        verbose=2,
        validation_data=test_image_generator,
        validation_steps=10,
        # max_q_size=1,
        # workers=1,
    )

    # Save the models/weights
    stored_weights_file = os.path.join(
        DATA_DIR, "train", "filter_{}".format(tgt_kernel_idx), 'trained_model.hf')

    cont_int_model.save_weights(stored_weights_file)

    # Save the models/weights
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

    fig.suptitle('Input channel feeding into output channel @ {}'.format(tgt_kernel_idx))

    # 3. Fields - 1993 - Experiment 1 - Curvature vs Gain
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots()

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

    # 4. Linear Enhancement gain vs Contour Length
    # -------------------------------------------------------------------------
    # Get Neurological Data
    with open('.//data//neuro_data//Li2006.pickle', 'rb') as handle:
        li_2006_data = pickle.load(handle)

    relative_gain_curvature = {
        0: 1.00,
        15: 0.98,
        30: 0.87,
        45: 0.85,
        60: 0.61
    }

    curve_rot_arr = np.array([0, 15, 30, 45, 60])

    expected_gain = li_2006_data['contour_len_avg_gain']
    contour_len_arr = li_2006_data['contour_len_avg_len']

    # Initialization
    n_runs = 100

    plt.figure()
    plt.plot(contour_len_arr, expected_gain, label='Li 2006 Results', marker='square')

    avg_gain_per_len = []
    sd_gain_per_len = []

    for c_len_idx, c_len in enumerate(contour_len_arr):
        print("Getting Results for c_len {}".format(c_len))

        # image generator
        train_dict = train_data_dict['c_len_{0}_beta_{1}'.format(c_len, 0)]
        train_image_generator = image_generator_curve.DataGenerator(
            train_dict,
            batch_size=1,
            img_size=IMAGE_SIZE,
            shuffle=True,
        )
        gen_out = iter(train_image_generator)

        # get the predictions
        y_hat_arr = []
        for r_idx in range(n_runs):
            X, y = gen_out.next()

            y_hat = cont_int_model.predict(X, batch_size=1, verbose=0)
            y_hat_arr.append(y_hat)

        avg_gain_per_len.append(np.mean(y_hat_arr))
        sd_gain_per_len.append(np.std(y_hat_arr))

    plt.errorbar(contour_len_arr, avg_gain_per_len, sd_gain_per_len, marker='o', label='model', color='g')
    plt.legend()
    plt.xlabel("Contour Length")
    plt.ylabel("Gain")
    plt.title("Enhancement Gain vs. Contour Length - Linear Contours")

    # 5. Enhancement Visualization over Sample Images
    # -----------------------------------------------
    # 1. Linear Contour
    image_idx = 0
    image_loc = './data/curved_contours/train/filter_{0}/c_len_9/beta_0/c_len_9_beta_0__{1}.png'.format(
        tgt_kernel_idx, image_idx)
    d = load_img(image_loc)
    d1 = np.array(d)

    alex_net_utils.plot_l1_and_l2_activations(
        d1 / 255.0, feat_extract_act_cb, cont_int_act_cb, tgt_kernel_idx)
    plt.suptitle("Linear Contour")

    # 2. 15 Rotation Array
    image_idx = 0
    image_loc = './data/curved_contours/train/filter_{0}/c_len_9/beta_15/c_len_9_beta_15__{1}.png'.format(
        tgt_kernel_idx, image_idx)
    d = load_img(image_loc)
    d1 = np.array(d)

    alex_net_utils.plot_l1_and_l2_activations(
        d1 / 255.0, feat_extract_act_cb, cont_int_act_cb, tgt_kernel_idx)
    plt.title("Contour with inter-fragment rotation of 15")
