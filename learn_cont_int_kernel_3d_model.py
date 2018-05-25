# -------------------------------------------------------------------------------------------------
#  Learn contour enhancement kernels for the 3d model contour integration model
#
# Author: Salman Khan
# Date  : 06/05/18
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pickle
from contour_integration_models.alex_net import model_3d as contour_integration_model
import os

import keras.backend as K
from keras.models import save_model

import image_generator_curve
import learn_cont_int_kernel_3d_model_linear_contours as linear_contour_training

reload(image_generator_curve)
reload(linear_contour_training)

DATA_DIR = 'data/curved_contours'

if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 5

    image_size = (227, 227, 3)

    batch_size = 10

    train_set = [
        'c_len_1_beta_0',

        'c_len_3_beta_0',
        # 'c_len_3_beta_15',
        # 'c_len_3_beta_30',
        # 'c_len_3_beta_45',
        # 'c_len_3_beta_60',

        'c_len_5_beta_0',
        # 'c_len_5_beta_15',
        # 'c_len_5_beta_30',
        # 'c_len_5_beta_45',
        # 'c_len_5_beta_60',

        'c_len_7_beta_0',
        # 'c_len_7_beta_15',
        # 'c_len_7_beta_30',
        # 'c_len_7_beta_45',
        # 'c_len_7_beta_60',
        #
        'c_len_9_beta_0',
        # 'c_len_9_beta_15',
        # 'c_len_9_beta_30',
        # 'c_len_9_beta_45',
        # 'c_len_9_beta_60',
    ]

    # -----------------------------------------------------------------------------------
    # Contour Integration Model
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model.build_contour_integration_model(5)

    # Store the start weights & bias for comparison later
    start_weights, _ = cont_int_model.layers[2].get_weights()

    # -----------------------------------------------------------------------------------
    # Online Image Generator for training
    # -----------------------------------------------------------------------------------
    train_data_key_loc = './data/curved_contours/filter_{}/data_key.pickle'.format(tgt_filter_idx)

    with open(train_data_key_loc, 'rb') as handle:
        train_data_dict = pickle.load(handle)

    active_train_set = {}
    for set_id in train_set:
        if set_id in train_data_dict:
            active_train_set.update(train_data_dict[set_id])
        else:
            ans = raw_input('{0} Image set not in Data key? Continue without ? (Y/N)'.format(set_id))
            if 'y' in ans.lower():
                continue
            else:
                raise SystemExit()

    train_image_generator = image_generator_curve.DataGenerator(
        active_train_set,
        batch_size=batch_size,
        img_size=image_size,
        shuffle=True,
        data_dir=DATA_DIR
    )

    # # Test the generator (sequence) object
    # gen_out = iter(train_image_generator)
    # X, y = gen_out.next()
    #
    # plt.figure()
    # plt.imshow(np.transpose(X[0, ], (1, 2, 0)))

    # -----------------------------------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------------------------------
    history = cont_int_model.fit_generator(
        generator=train_image_generator,
        epochs=100,
        steps_per_epoch=10,
        verbose=2,
        # max_q_size=1,
        # workers=1,
    )

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    model_file_name = os.path.join(DATA_DIR, "filter_{}".format(tgt_filter_idx), 'trained_model.hf')
    save_model(cont_int_model, model_file_name)

    # Plot the learned weights
    # -------------------------
    fig, ax_arr = plt.subplots(1, 2)
    linear_contour_training.plot_contour_integration_weights_in_channels(
        start_weights, tgt_filter_idx, axis=ax_arr[0])

    learnt_weights, _ = cont_int_model.layers[2].get_weights()
    linear_contour_training.plot_contour_integration_weights_in_channels(
        learnt_weights, tgt_filter_idx, axis=ax_arr[1])
    fig.suptitle('Input channel of filter @ {}'.format(tgt_filter_idx))

    fig, ax_arr = plt.subplots(1, 2)
    linear_contour_training.plot_contour_integration_weights_out_channels(
        start_weights, tgt_filter_idx, axis=ax_arr[0])

    learnt_weights, _ = cont_int_model.layers[2].get_weights()
    linear_contour_training.plot_contour_integration_weights_out_channels(
        learnt_weights, tgt_filter_idx, axis=ax_arr[1])
    fig.suptitle('Output channel of filter @ {}'.format(tgt_filter_idx))
