# ---------------------------------------------------------------------------------------
# Training Script for the new contour integration model that works on all contour
# integration kernels simultaneously
# ---------------------------------------------------------------------------------------
import pickle
import keras
import os
import shutil
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import image_generator_curve
from contour_integration_models.alex_net import model_3d_all_kernels
import learn_cont_int_kernel_3d_model
import alex_net_utils

reload(image_generator_curve)
reload(model_3d_all_kernels)
reload(learn_cont_int_kernel_3d_model)
reload(alex_net_utils)

DISPLAY_FIGURES = False


def create_data_generator(list_pickle_file_paths, b_size=1, shuffle=True):
    """

    :param shuffle:
    :param b_size:
    :param list_pickle_file_paths:
    :return:
    """
    data_dict = {}

    for pkl_file_path in list_pickle_file_paths:

        pkl_file = os.path.join(pkl_file_path, data_key_file_name)
        if not os.path.exists(pkl_file):
            raise Exception("{} does not exist".format(pkl_file))

        print("Loading Data from {}".format(pkl_file))

        with open(pkl_file, 'rb') as h:
            curr_dict_of_dicts = pickle.load(h)

        for k, v in curr_dict_of_dicts.iteritems():

            if 'alpha_0' in k:
                if 'beta_0' in k or 'beta_15' in k or 'beta_30:' in k:
                    # print("Adding {}".format(k))
                    data_dict.update(curr_dict_of_dicts[k])
                    # print("beta_0 in dict {}".format('beta_0' in k))
                    # print("beta_15 in dict {}".format('beta_15' in k))
                    # print("beta_30 in dict {}".format('beta_30' in k))

    n_data_pts = len(data_dict)
    print("Number of data points {}".format(n_data_pts))

    if n_data_pts < b_size:
        b_size = n_data_pts
        print("WARN: Num data points > then requested batch size. Changing batch size to {}".format(n_data_pts))

    data_generator = image_generator_curve.DataGenerator(
        data_dict,
        batch_size=b_size,
        shuffle=shuffle,
        labels_per_image=96
    )

    return data_generator, n_data_pts


def get_list_pf_pickle_files(data_dir):
    """

    :return:
    """
    print("Getting all pickles in {}".format(data_dir))

    filters_list = os.listdir(data_dir)
    list_of_pickle_files = []

    for filter_dir in filters_list:
        filter_dir_path = os.path.join(data_dir, filter_dir)
        list_of_pickle_files.append(filter_dir_path)

    return list_of_pickle_files


def plot_max_contour_enhancement(img, feat_extract_cb, cont_int_cb):
    """
    Plot the maximum contour enhancement (Across all feature maps) @ each position
    :param img:
    :param feat_extract_cb:
    :param cont_int_cb:
    :return:
    """
    l1_act, l2_act = alex_net_utils.get_l1_and_l2_activations(
        img,
        feat_extract_cb,
        cont_int_cb
    )

    diff = l2_act - l1_act
    max_diff = np.max(diff, axis=1)

    plt.figure()
    plt.imshow(max_diff[0, ], cmap='seismic')
    plt.colorbar(orientation='horizontal')
    plt.title("Maximum contour enhancement @ each (x,y) ")


if __name__ == '__main__':
    keras.backend.set_image_dim_ordering('th')  # Model was originally defined with Theano backend.

    batch_size = 128
    num_test_points = 10000
    num_epochs = 20

    # results_dir = './results/all_kernels_no_alpha_rotations'
    results_dir = './results/all_kernels_alpha_0_beta_upto30'

    # Immutable ---------------------------------------------------------
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    weights_store_name = 'contour_integration_layer_weights.hf'
    weights_store_file = os.path.join(results_dir, weights_store_name)

    summary_file_name = 'summary.txt'

    data_key_file_name = 'all_kernels_data_key.pickle'

    plt.ion()

    # -----------------------------------------------------------------------------------
    print("Creating Data Generators ...")

    # # Manually by explicitly stating
    # train_list_of_pickle_file_paths = [
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/train/filter_5',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/train/filter_10',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/train/filter_19',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/train/filter_20',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/train/filter_21',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/train/filter_22',
    # ]
    #
    # test_list_of_pickle_file_paths = [
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/test/filter_5',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/test/filter_10',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/test/filter_19',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/test/filter_20',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/test/filter_21',
    #     './data/curved_contours/frag_11x11_full_18x18_param_search/test/filter_22',
    # ]

    # # All filters in a base directory directory
    base_data_directory = './data/curved_contours/frag_11x11_full_18x18_param_search'
    train_list_of_pickle_file_paths = get_list_pf_pickle_files(os.path.join(base_data_directory, 'train'))
    test_list_of_pickle_file_paths = get_list_pf_pickle_files(os.path.join(base_data_directory, 'test'))

    # ------------------------------------------------------
    train_data_generator, num_training_points = \
        create_data_generator(train_list_of_pickle_file_paths, b_size=batch_size)

    # Get all test data points in one iteration.
    # Tensorboard does not like a generator for validation data
    test_data_generator, total_test_points = \
        create_data_generator(test_list_of_pickle_file_paths, b_size=num_test_points)

    gen_out = iter(test_data_generator)
    X, y = gen_out.next()

    filter_idxs = [np.int(x.split('/')[-1].split('_')[1]) for x in train_list_of_pickle_file_paths]

    # -----------------------------------------------------------------------------------
    print("Building the model ...")
    model = model_3d_all_kernels.training_model(
        rf_size=35,
        inner_leaky_relu_alpha=0.9,
        outer_leaky_relu_alpha=1.,
        l1_reg_loss_weight=0.0001/96,
    )

    optimizer = keras.optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.mean_squared_error
    )

    start_weights, _ = model.layers[2].get_weights()
    # Callbacks
    feat_extract_act_cb = alex_net_utils.get_activation_cb(model, 1)
    cont_int_act_cb = alex_net_utils.get_activation_cb(model, 2)

    # -----------------------------------------------------------------------------------
    print("Training the model ...")

    start_time = datetime.now()

    checkpoint = keras.callbacks.ModelCheckpoint(
        weights_store_file,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        save_weights_only=True,
    )

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(time()),
        # histogram_freq=1,
        # write_grads=True,
        # write_images=False,
        # batch_size=1,  # For histogram
    )

    callbacks = [tensorboard, checkpoint]

    steps_per_epoch = num_training_points / batch_size

    history = model.fit_generator(
        generator=train_data_generator,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        validation_data=(X, y),
        validation_steps=1,
        # max_q_size=1,
        workers=8,
        callbacks=callbacks
    )

    # Plot Losses
    fig, axis = plt.subplots()
    axis.plot(history.history['loss'], label='train_loss')
    axis.plot(history.history['val_loss'], label='validation_loss')
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    fig.savefig(os.path.join(results_dir, 'loss_vs_epoch.eps'), format='eps')

    training_time = datetime.now() - start_time
    print("Training took {}".format(training_time))

    # -------------------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------------------
    # 1. Display learnt kernels
    learnt_weights_visualize_dir = os.path.join(results_dir, 'filter_visualizations')
    if not os.path.exists(learnt_weights_visualize_dir):
        os.mkdir(learnt_weights_visualize_dir)

    for kernel_idx in filter_idxs:
        learn_cont_int_kernel_3d_model.plot_start_n_learnt_contour_integration_kernels(
            model,
            kernel_idx,
            start_weights,
        )

        learnt_kernel_fig = plt.gcf()

        learnt_kernel_fig.savefig(os.path.join(
            learnt_weights_visualize_dir, 'learnt_contour_integration_kernel_{}.eps'.format(kernel_idx)), format='eps')

    # For a Sample Image plot the expected gain vs actual gain
    image_idx = 7

    # 2. predict output on a single image
    test_image = X[image_idx, ]
    test_label = y[image_idx, ]

    input_image = np.expand_dims(test_image, axis=0)
    y_hat = model.predict(input_image)

    display_image = np.transpose(test_image, axes=(1, 2, 0))
    plt.figure()
    plt.imshow(display_image)

    plt.figure()
    plt.stem(test_label.T, 'r', label='Expected')
    plt.stem(y_hat.T, 'sb', label='Predicted')
    plt.legend()

    # 3. Plot Max Enhancement
    z = np.transpose(test_image, axes=(1, 2, 0))
    plot_max_contour_enhancement(z, feat_extract_act_cb, cont_int_act_cb)

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    # Write Summary File
    with open(os.path.join(results_dir, summary_file_name), 'wb') as f_id:

        f_id.write("Final training Loss: {} @ Epoch {}\n".format(
            np.min(history.history['loss']),  np.argmin(history.history['loss'])))
        f_id.write("Final Validation Loss: {} @ Epoch {}\n".format(
            np.min(history.history['val_loss']), np.argmin(history.history['val_loss'])))
        f_id.write("Training Duration: {}\n".format(training_time))
        f_id.write("\n")

        f_id.write("Model Hyper-Parameters : --------------------------------------\n")
        f_id.write("L1 Loss Weight {}\n".format(model.layers[2].l1_reg_loss_weight))
        f_id.write("Contour Integration rf size {}\n".format(model.layers[2].n))
        f_id.write("Outer Relu alpha {}\n".format(model.layers[2].outer_leaky_relu_alpha))
        f_id.write("Inner Relu alpha {}\n".format(model.layers[2].inner_leaky_relu_alpha))
        f_id.write("\n")

        f_id.write("Training Parameters : --------------------------------------\n")
        f_id.write("Trained Filters: [{} Total]: ".format(len(filter_idxs)))
        for filter_idx in filter_idxs:
            f_id.write("{},".format(filter_idx))
        f_id.write("\n")

        f_id.write("Data set: training {}, test {} out of {}\n".format(
            num_training_points, num_test_points, total_test_points
        ))

        f_id.write("Number of Epochs: {}.\n".format(num_epochs))
        f_id.write("Batch Size: {}.\n".format(batch_size))
        f_id.write("Learning Rate: {}.\n".format(keras.backend.eval(optimizer.lr)))
        f_id.write("Optimizer Type: {}\n".format(optimizer.__class__))
        f_id.write("\n")

        f_id.write("Data Directories : --------------------------------------\n")
        for idx, folder in enumerate(train_list_of_pickle_file_paths):
            f_id.write('\t{}: {}\n'.format(idx, folder))
