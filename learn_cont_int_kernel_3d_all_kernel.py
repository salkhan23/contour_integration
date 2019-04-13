# ---------------------------------------------------------------------------------------
# Training Script for the new contour integration model that works on all contour
# integration kernels simultaneously
# ---------------------------------------------------------------------------------------
import pickle
import keras
import os
import shutil
from time import time
import numpy as np
from datetime import datetime
# Import this to run without displaying figures
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import image_generator_curve
from contour_integration_models.alex_net import model_3d_all_kernels
import learn_cont_int_kernel_3d_model
import alex_net_utils
import visualize_multi_kernel_trained_model


reload(image_generator_curve)
reload(model_3d_all_kernels)
reload(learn_cont_int_kernel_3d_model)
reload(alex_net_utils)
reload(visualize_multi_kernel_trained_model)


def get_alpha_or_beta_value_from_key(key_name, look_for):
    """
    key_name = beta
    :param look_for:
    :param key_name:
    :return:
    """
    allowed_look_for = ['alpha', 'beta']

    if look_for not in allowed_look_for:
        raise Exception("Invalid lookfor value {}. Must be one of {}".format(look_for, allowed_look_for))

    look_for_value_string = key_name.split(look_for)[1]
    look_for_value = look_for_value_string.split('_')[1]

    return look_for_value


def create_data_generator(list_pickle_files, preprocessing_cb, b_size=1, shuffle=True):
    """

    :param preprocessing_cb:
    :param shuffle:
    :param b_size:
    :param list_pickle_files:
    :return:
    """
    data_dict = {}

    allowed_alpha = ['alpha_0', 'alpha_15', 'alpha_30']
    allowed_beta = ['beta_0', 'beta_15', 'beta_30']

    included_alpha_set = set()
    included_beta_set = set()

    for pkl_file in list_pickle_files:

        print("Loading Data from {}".format(pkl_file))

        with open(pkl_file, 'rb') as h:
            curr_dict_of_dicts = pickle.load(h)

        list_of_used_data_keys = []

        for k, v in curr_dict_of_dicts.iteritems():

            # # Debug
            # print("Checking: {}".format(k))
            # print("Meets Beta Requirement {}".format(any(x in k for x in allowed_beta)))
            # print("Meets Alpha Requirement {}".format(any(x in k for x in allowed_alpha)))
            # print("Should add {}".format(
            #     any(x in k for x in allowed_beta) and
            #     any(x in k for x in allowed_alpha))
            # )

            if any(x in k for x in allowed_beta) and any(x in k for x in allowed_alpha):
                # print("Adding {}".format(k))

                list_of_used_data_keys.append(k)

                included_alpha_set.add(get_alpha_or_beta_value_from_key(k, look_for='alpha'))
                included_beta_set.add(get_alpha_or_beta_value_from_key(k, look_for='beta'))

                data_dict.update(curr_dict_of_dicts[k])

            # # Debug: print the list of data_keys that will be used
            # print("List of data sets that will be used for file {}".format(pkl_file))
            # for i_item, item in enumerate(sorted(list_of_used_data_keys)):
            #     print("[{}]: {}".format(i_item, item))

    # Note this is the number of data points from all pickle files
    n_data_pts = len(data_dict)
    if n_data_pts == 0:
        raise Exception("Number of data points = 0")

    print("Total number of data points (all pickle files) {}".format(n_data_pts))
    print("Number of data points from single pickle file {}".format(n_data_pts / len(list_pickle_files)))

    if n_data_pts < b_size:
        b_size = n_data_pts
        print("WARN: Num data points > then requested batch size. Changing batch size to {}".format(n_data_pts))

    data_generator = image_generator_curve.DataGenerator(
        data_dict,
        batch_size=b_size,
        shuffle=shuffle,
        labels_per_image=96,
        preprocessing_cb=preprocessing_cb
    )

    return data_generator, n_data_pts, included_alpha_set, included_beta_set


def get_list_pf_pickle_files(data_dir, pickle_filename):
    """

    :return:
    """
    print("Getting all '{}' in {}".format(pickle_filename, data_dir))

    filters_list = os.listdir(data_dir)
    list_of_pickle_files = []

    for filter_dir in filters_list:

        filter_dir_path = os.path.join(data_dir, filter_dir)

        if pickle_filename in os.listdir(filter_dir_path):
            list_of_pickle_files.append(os.path.join(filter_dir_path, pickle_filename))
        else:
            print("WARN: pickle file {} not found in {}".format(pickle_filename, data_dir))

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


def main(l1_loss, loss_fcn, lr, preprocessing_fcn):
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    batch_size = 128
    num_epochs = 50

    # Data Directory
    # ===============
    # base_data_directory = './data/curved_contours/frag_11x11_full_18x18_param_search'
    # display_kernel_idxs = [
    #     5, 10, 19, 20, 21, 22, 48, 49, 51, 59,
    #     60, 62, 64, 65, 66, 68, 69, 72, 73, 74,
    #     76, 77, 79
    # ]
    # data_key_file_name = 'data_key_thres_above_half_max_act_preprocessing_divide255.pickle'

    base_data_directory = './data/curved_contours/coloured_gabors_dataset'
    display_kernel_idxs = [
        0,   2,  5, 10, 13, 19, 20, 21, 22, 23,
        25, 27, 30, 33, 34, 35, 39, 48, 49, 51,
        52, 54, 55, 56, 59, 62, 64, 65, 66, 68,
        69, 70, 72, 73, 74, 77, 78, 79, 80, 81,
        83, 89
    ]
    data_key_file_name = 'data_key_sigmoid_center_0.5MaxAct_gain_2_preprocessing_divide255.pickle'
    # data_key_file_name = 'data_key_above_mean_act_preprocessing_divide255.pickle'

    if len(display_kernel_idxs) >= 2:
        plt.ioff()

    # Results Directory
    # ==================
    base_results = './results/lr_search'
    results_identifier = 'lossFcn_{}_l1LossWeight_{}_lr_{}_beta_30_alpha_30'.format(
        str(loss_fcn).split(' ')[1],
        str(l1_loss),
        str(lr)
    )
    results_dir = os.path.join(base_results, results_identifier)

    if os.path.exists(results_dir):
        ans = raw_input("{} directory already exists. Overwrite? y/n".format(results_dir))
        if 'y' in ans.lower():
            shutil.rmtree(results_dir)
        else:
            raise SystemExit
    os.makedirs(results_dir)

    print("{}\nData Directory {}".format('*' * 80, base_data_directory))
    print("Results @      {}.\n {}".format(results_dir, '*' * 80))

    # Immutable
    weights_store_name = 'contour_integration_layer_weights.hf'
    summary_file_name = 'summary.txt'

    if len(display_kernel_idxs) > 2:
        plt.ioff()
    # -----------------------------------------------------------------------------------
    # Data Generators
    # -----------------------------------------------------------------------------------
    print("Creating Data Generators ...")

    # Get pickle data key files
    # =========================
    # # Manually
    # train_list_of_pickle_file_paths = [
    #     os.path.join(base_data_directory, 'train/filter_0'),
    #     os.path.join(base_data_directory, 'train/filter_5'),
    #     os.path.join(base_data_directory, 'train/filter_10'),
    #     os.path.join(base_data_directory, 'train/filter_20'),
    #     os.path.join(base_data_directory, 'train/filter_21'),
    #     os.path.join(base_data_directory, 'train/filter_22'),
    # ]
    # train_list_of_pickle_files = \
    #     [os.path.join(path, data_key_file_name) for path in train_list_of_pickle_file_paths]
    #
    # test_list_of_pickle_file_paths = [
    #     os.path.join(base_data_directory, 'test/filter_0'),
    #     os.path.join(base_data_directory, 'test/filter_5'),
    #     os.path.join(base_data_directory, 'test/filter_10'),
    #     os.path.join(base_data_directory, 'test/filter_20'),
    #     os.path.join(base_data_directory, 'test/filter_21'),
    #     os.path.join(base_data_directory, 'test/filter_22'),
    # ]
    # test_list_of_pickle_files = \
    #     [os.path.join(path, data_key_file_name) for path in test_list_of_pickle_file_paths]

    # Automatically get all pickle files.
    train_list_of_pickle_files = \
        get_list_pf_pickle_files(os.path.join(base_data_directory, 'train'), data_key_file_name)
    test_list_of_pickle_files = \
        get_list_pf_pickle_files(os.path.join(base_data_directory, 'test'), data_key_file_name)

    print("{} Training and {} pickle files".format(
        len(train_list_of_pickle_files), len(test_list_of_pickle_files)))

    train_data_generator, num_training_points, included_alpha, included_beta = create_data_generator(
        train_list_of_pickle_files,
        preprocessing_cb=preprocessing_fcn,
        b_size=batch_size
    )

    # Get all test data points in one iteration.
    # Tensorboard does not like a generator for validation data
    num_test_points = 1000
    test_data_generator, total_test_points, _, _ = create_data_generator(
        test_list_of_pickle_files,
        preprocessing_cb=preprocessing_fcn,
        b_size=num_test_points)

    gen_out = iter(test_data_generator)
    test_images, test_labels = gen_out.next()

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print("Building the model ...")
    model = model_3d_all_kernels.training_model(
        rf_size=35,
        inner_leaky_relu_alpha=0.9,
        outer_leaky_relu_alpha=1.,
        l1_reg_loss_weight=l1_loss,
    )
    model.summary()

    optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(optimizer=optimizer, loss=loss_fcn)

    cont_int_layer_idx = alex_net_utils.get_layer_idx_by_name(model, 'contour_integration_layer')
    # feat_extract_layer_idx = alex_net_utils.get_layer_idx_by_name(model, 'conv_1')

    start_weights, _ = model.layers[cont_int_layer_idx].get_weights()

    # # Callbacks
    # feat_extract_act_cb = alex_net_utils.get_activation_cb(model, feat_extract_layer_idx)
    # cont_int_act_cb = alex_net_utils.get_activation_cb(model, cont_int_layer_idx)

    # -----------------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------------
    print("Training the model ...")

    start_time = datetime.now()
    weights_store_file = os.path.join(results_dir, weights_store_name)

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
        validation_data=(test_images, test_labels),
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

    # -----------------------------------------------------------------------------------
    #  Write Summary File
    # -----------------------------------------------------------------------------------
    with open(os.path.join(results_dir, summary_file_name), 'wb') as f_id:

        f_id.write("Final training Loss: {} @ Epoch {}\n".format(
            np.min(history.history['loss']), np.argmin(history.history['loss'])))
        f_id.write("Final Validation Loss: {} @ Epoch {}\n".format(
            np.min(history.history['val_loss']), np.argmin(history.history['val_loss'])))
        f_id.write("Training Duration: {}\n".format(training_time))
        f_id.write("\n")

        f_id.write("Model Hyper-Parameters : --------------------------------------\n")
        f_id.write("L1 Loss Weight {}\n".format(model.layers[cont_int_layer_idx].l1_reg_loss_weight))
        f_id.write("Contour Integration rf size {}\n".format(model.layers[cont_int_layer_idx].n))
        f_id.write("Outer Relu alpha {}\n".format(model.layers[cont_int_layer_idx].outer_leaky_relu_alpha))
        f_id.write("Inner Relu alpha {}\n".format(model.layers[cont_int_layer_idx].inner_leaky_relu_alpha))
        f_id.write("\n")

        f_id.write("Training Details : --------------------------------------\n")
        f_id.write("Number of Epochs: {}.\n".format(num_epochs))
        f_id.write("Batch Size: {}.\n".format(batch_size))
        f_id.write("Start Learning rate: {}.\n".format(lr))
        f_id.write("Final Learning Rate: {}.\n".format(keras.backend.eval(optimizer.lr)))
        f_id.write("Optimizer Type: {}\n".format(optimizer.__class__))
        f_id.write("Loss Function: {}\n".format(str(loss_fcn).split(' ')[1]))
        f_id.write("\n")

        f_id.write("Data Details : ..............................................\n")
        f_id.write("Base Data Directory: {}\n".format(base_data_directory))
        f_id.write("Data key filename: '{}'\n".format(data_key_file_name))
        f_id.write("Data points: training {}, test {} out of {}\n".format(
            num_training_points, num_test_points, total_test_points
        ))
        f_id.write("Included alpha values: {}\n".format(sorted(included_alpha)))
        f_id.write("Included beta values: {}\n".format(sorted(included_beta)))

        f_id.write("Training pickle files: \n")
        for idx, pickle_file in enumerate(train_list_of_pickle_files):
            f_id.write('\t{}: {}\n'.format(idx, pickle_file))
        f_id.write("Test pickle files: \n")
        for idx, pickle_file in enumerate(test_list_of_pickle_files):
            f_id.write('\t{}: {}\n'.format(idx, pickle_file))

    # -------------------------------------------------------------------------------------
    # Save Learnt Contour Integration kernels
    # -------------------------------------------------------------------------------------
    print("Plotting Learnt Contour Integration kernels")
    learnt_weights_visualize_dir = os.path.join(results_dir, 'filter_visualizations')
    if not os.path.exists(learnt_weights_visualize_dir):
        os.mkdir(learnt_weights_visualize_dir)

    for kernel_idx in display_kernel_idxs:
        alex_net_utils.plot_start_n_learnt_contour_integration_kernels(
            model,
            kernel_idx,
            start_weights,
        )

        learnt_kernel_fig = plt.gcf()

        learnt_kernel_fig.savefig(os.path.join(
            learnt_weights_visualize_dir, 'learnt_contour_integration_kernel_{}.eps'.format(kernel_idx)), format='eps')
    plt.close('all')

    # -----------------------------------------------------------------------------------
    # Save Sample Results
    # -----------------------------------------------------------------------------------
    print("Getting Sample Image Results")

    sample_results_dir = os.path.join(results_dir, 'sample_results')
    if not os.path.exists(sample_results_dir):
        os.mkdir(sample_results_dir)

    gabor_params_list = [

        # Gabor Params for Filter @ index 5
        {
            'x0': 0,
            'y0': 0,
            'theta_deg': -90.0,
            'amp': 1,
            'sigma': 2.75,
            'lambda1': 15.0,
            'psi': 1.5,
            'gamma': 0.8
        },
        # Gabor Params for Filter @ index 10
        {
            'x0': 0,
            'y0': 0,
            'theta_deg': 0.0,
            'amp': 1,
            'sigma': 2.75,
            'lambda1': 5.5,
            'psi': 1.25,
            'gamma': 0.8
        }
    ]

    for gabor_params in gabor_params_list:
        visualize_multi_kernel_trained_model.main_contour_images(
            model=model,
            preprocessing_cb=preprocessing_fcn,
            g_params=gabor_params,
            learnt_kernels=display_kernel_idxs,
            results_dir=sample_results_dir
        )

    visualize_multi_kernel_trained_model.main_natural_images(
        model=model,
        preprocessing_cb=preprocessing_fcn,
        learnt_kernels=display_kernel_idxs,
        results_dir=sample_results_dir
    )


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    # Model was originally defined with Theano backend.
    keras.backend.set_image_dim_ordering('th')
    np.random.seed(random_seed)

    plt.ion()

    # -----------------------------------------------------------------------------------
    # Main Routine(s)
    # -----------------------------------------------------------------------------------
    print("Iteration 1 {}".format('=' * 80))
    main(
        l1_loss=0.00001,
        loss_fcn=keras.losses.mean_squared_error,
        lr=0.0001,
        preprocessing_fcn=alex_net_utils.preprocessing_divide_255
    )

    print("Iteration 2 {}".format('=' * 80))
    main(
        l1_loss=0.00001,
        loss_fcn=keras.losses.mean_squared_error,
        lr=0.00001,
        preprocessing_fcn=alex_net_utils.preprocessing_divide_255
    )

    print("Iteration 3 {}".format('=' * 80))
    main(
        l1_loss=0.00001,
        loss_fcn=keras.losses.mean_squared_error,
        lr=0.000001,
        preprocessing_fcn=alex_net_utils.preprocessing_divide_255
    )

    print("Iteration 4 {}".format('=' * 80))
    main(
        l1_loss=0.00001,
        loss_fcn=keras.losses.mean_squared_error,
        lr=0.0000001,
        preprocessing_fcn=alex_net_utils.preprocessing_divide_255
    )
