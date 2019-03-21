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


def imagenet_training_preprocessing(x):
    """ This preprocessing is for the case the image is already loaded by another function
        Per channel mean values are subtracted
        Expected input is channel first.
    """

    # # Normalize image to range [0, 1]
    # x = keras.preprocessing.image.img_to_array(x, data_format='channels_first') / 255.0

    x[:, :, 2] -= 123.68
    x[:, :, 1] -= 116.779
    x[:, :, 0] -= 103.939
    return x


def zero_one_normalization_preprocessing(x):
    """ This preprocessing is for the case the image is already loaded by another function
        Input image pixels are normalized to range (0, 1)
            Expected input is channel first.
        """
    x = (x - x.min()) / (x.max() - x.min())
    return x


def load_image(img_file):
    """

    :param img_file:
    :return:
    """
    img = keras.preprocessing.image.load_img(img_file, target_size=(227, 227, 3))

    img = keras.preprocessing.image.img_to_array(img, data_format='channels_first')

    return img


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    random_seed = 7
    np.random.seed(random_seed)
    keras.backend.set_image_dim_ordering('th')

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
        preprocessing_function=imagenet_training_preprocessing
        # preprocessing_function = zero_one_normalization_preprocessing
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

    # # Plot sorted activations in a single plot
    # plt.figure()
    # for chan_idx in np.arange(center_neuron_act.shape[1]):
    #     plt.plot(sorted(center_neuron_act[:, chan_idx]))

    # -----------------------------------------------------------------------------------
    # Debug
    # -----------------------------------------------------------------------------------
    # Manually load an image and check activations
    image_file = './data/sample_images/irregular_shape.jpg'
    test_image = load_image(image_file)
    test_image = zero_one_normalization_preprocessing(test_image)

    activations = np.array(feat_extract_layer_cb([test_image, 0]))
    center_neuron_act = activations[0, :, :, 27, 27]
