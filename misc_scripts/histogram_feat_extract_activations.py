# ---------------------------------------------------------------------------------------
# PLot Histogram of activations
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import keras

from base_models.alex_net import alex_net
import alex_net_utils

reload(alex_net_utils)


IMAGE_NET_TRAIN_DIR = "./data/imagenet-data/train"


def load_image(img_file):
    """

    :param img_file:
    :return:
    """
    img = keras.preprocessing.image.load_img(img_file, target_size=(227, 227, 3))
    img = keras.preprocessing.image.img_to_array(img,  dtype='float64', data_format='channels_first')

    return img


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    random_seed = 7
    np.random.seed(random_seed)
    keras.backend.set_image_dim_ordering('th')

    preprocessing_fcn = alex_net_utils.preprocessing_divide_255

    batch_size = 1024

    # -----------------------------------------------------------------------------------
    # Load the model
    # -----------------------------------------------------------------------------------
    print("Loading Model ...")
    alex_net_model = alex_net("trained_models/AlexNet/alexnet_weights.h5")
    alex_net_model.summary()

    alex_net_model.compile(
        loss=keras.losses.categorical_crossentropy,  # Note this is not a function call.
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    feat_extract_layer_idx = alex_net_utils.get_layer_idx_by_name(alex_net_model, 'conv_1')
    feat_extract_layer_cb = alex_net_utils.get_activation_cb(alex_net_model, feat_extract_layer_idx)

    # -----------------------------------------------------------------------------------
    # Images to collect Histograms over
    # -----------------------------------------------------------------------------------
    print("Building Imagenet Generator ...")

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        data_format='channels_first',
        preprocessing_function=preprocessing_fcn
    )

    train_generator = train_datagen.flow_from_directory(
        IMAGE_NET_TRAIN_DIR,
        target_size=(227, 227),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=random_seed
    )

    X, y = train_generator.next()

    # -----------------------------------------------------------------------------------
    # Get Activations
    # -----------------------------------------------------------------------------------
    print("Getting Activations ...")
    activations = np.array(feat_extract_layer_cb([X, 0]))
    center_neuron_act = activations[0, :, :, 27, 27]
    max_act = np.max(center_neuron_act)

    for chan_idx in np.arange(center_neuron_act.shape[1]):

        print("Processing kernel {}".format(chan_idx))
        n_subplots = 12
        fig_axis_idx = np.mod(chan_idx, n_subplots)

        if fig_axis_idx == 0:
            f, ax_arr = plt.subplots(n_subplots, 1, sharex=False, squeeze=True)

        ax_arr[fig_axis_idx].hist(
            center_neuron_act[:, chan_idx],
            label='chan {}'.format(chan_idx), bins=100)  # bins=np.arange(0, max_act, 0.1))
        ax_arr[fig_axis_idx].legend()
    
    # Print & Plot Average activation
    avg_act = np.mean(center_neuron_act, axis=0)

    print("Average Activations: ")
    for k_idx in np.arange(center_neuron_act.shape[1]):
        print("{}: {}".format(k_idx, avg_act[k_idx]))

    plt.figure()
    plt.plot(avg_act)
    plt.title("Average activation. Preprocessing {}".format(str(preprocessing_fcn)))
    plt.xlabel("kernel_idx")
    plt.ylabel("Avg Activation")

    # # Plot sorted activations in a single plot
    # plt.figure()
    # for chan_idx in np.arange(center_neuron_act.shape[1]):
    #     plt.plot(sorted(center_neuron_act[:, chan_idx]))

    # # -----------------------------------------------------------------------------------
    # # Save average activation
    # # -----------------------------------------------------------------------------------
    # results_file = './data_generation/average_activation_divide_255_preprocessing.pickle'
    # import pickle
    #
    # with open(results_file, 'wb') as handle:
    #     pickle.dump(avg_act, handle)

    # -----------------------------------------------------------------------------------
    # Debug
    # -----------------------------------------------------------------------------------
    # Manually load an image and check activations
    image_file = './data/sample_images/irregular_shape.jpg'
    test_image = load_image(image_file)
    test_image = alex_net_utils.preprocessing_divide_255(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    activations = np.array(feat_extract_layer_cb([test_image, 0]))
    center_neuron_act = activations[0, :, :, 27, 27]
