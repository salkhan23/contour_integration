# -------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------
import keras
from keras.datasets import mnist
import keras.backend as K
import matplotlib.pyplot as plt

import numpy as np

MNIST_NUM_CLASSES = 10


def deprocess_image(x):
    """
    Utility function to convert a tensor into a valid image
    Normalize Tensor: Center on 0., ensure std is 0.1 [Why?]

    REF: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py.
    Unlike the original ref, data is kept within the [0,1] range. Matplotlib.pyplot.imshow prefers
    values in this range rather than [0, 255] which scipy.misc.imshow prefers. The later hangs
     the code until the figure is closed.

    Updated this function to do a more controlled 'linear normalization'

    :param x: image of dimension [r, c, ch]
    :return:
    """
    # x -= x.mean()
    # x /= (x.std() + 1e-5)
    #
    # # Clip to [0, 1]
    # x += 0.5
    # x = np.clip(x, 0, 1)

    # # This is not needed
    # if K.image_data_format() == 'channels_first':  # [ch,r, c]
    #     x = x.transpose((1, 2, 0))  # this is similar to K.permute dimensions but outside keras/TF

    x = (x - x.min()) * 1 / (x.max() - x.min())

    return x


def get_mnist_data(sample_idx=0):
    """
    :param sample_idx: index of sample to return

    :return: data_train, label_train, data_test, label_test, data_sample, label_sample
    """
    (data_train, label_train), (data_test, label_test) = mnist.load_data()
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], data_train.shape[2], 1)
    data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], data_train.shape[2], 1)

    data_train = data_train.astype('float32')
    data_test = data_test.astype('float32')

    data_train /= 255
    data_test /= 255

    label_train = keras.utils.to_categorical(label_train, MNIST_NUM_CLASSES)
    label_test = keras.utils.to_categorical(label_test, MNIST_NUM_CLASSES)

    # a Single sample to test prediction on
    data_sample = data_test[sample_idx].reshape(
        1, data_test[sample_idx].shape[0], data_test[sample_idx].shape[1], data_test[sample_idx].shape[2])
    label_sample = label_test[sample_idx]

    return data_train, label_train, data_test, label_test, data_sample, label_sample


def display_filters(weights, margin=1):
    """
    Display the filters of a layer in a single large image

    :param weights: weight matrix of a model layer
    :param margin: Gap/border between the kernels, filters in the large tiled filters image.

    :return:
    """
    if len(K.int_shape(weights)) == 3:

        if K.image_data_format() == 'channels_last':
            r, c, out_ch = K.int_shape(weights)
            weights = K.reshape(weights, (r, c, 1, out_ch))
        else:
            out_ch, r, c = K.int_shape(weights)
            weights = K.reshape(weights, (out_ch, r, c, 1))
            weights = K.permute_dimensions(weights, [1, 2, 3, 0])

    r, c, in_ch, out_ch = K.int_shape(weights)
    # print("display_filters: [r, c, in_ch, out_ch]", r, c, in_ch, out_ch)

    allowed_in_ch = [1, 3]  # can only display filters where the input dimension is 1 or 3
    if in_ch not in allowed_in_ch:
        raise Exception("Cannot display filters with input channels = %d" % in_ch)

    n = np.int(np.round(np.sqrt(out_ch)))  # Single dimension of tiled image

    width = (n * r) + ((n - 1) * margin)
    height = (n * c) + ((n - 1) * margin)

    tiled_filters = np.zeros((width, height, in_ch))

    # Fill in in composite image with the filters
    for r_idx in range(n):
        for c_idx in range(n):

            filt_idx = (r_idx * n) + c_idx
            if filt_idx >= out_ch:
                break

            print("Processing filter %d" % filt_idx)

            tiled_filters[
                (r + margin) * r_idx: (r + margin) * r_idx + r,
                (c + margin) * c_idx: (c + margin) * c_idx + c,
                :
            ] = deprocess_image(K.eval(weights[:, :, :, filt_idx]))

    # Plot the Composite Figure
    plt.ion()
    plt.figure()

    if 1 == in_ch:
        plt.imshow(tiled_filters[:, :, 0], cmap='seismic')  # force to 2D. Expected by imshow
    else:
        plt.imshow(tiled_filters)
    plt.colorbar()

    return tiled_filters


def display_layer_activations(model, layer_idx, data_sample, margin=1):
    """
    Display the activation volume of the specified layer. Each feature map is displayed in a separate subplot.
    Expected format of layers is [b, r, c, ch]

    :param margin:
    :param model:
    :param layer_idx:
    :param data_sample:

    :return: the whole activation volume
    """

    # Define a function to get the activation volume
    get_layer_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_idx].output]
    )

    # Get the activations in a usable format
    act_volume = np.asarray(get_layer_output(
        [data_sample, 0],  # second input specifies the learning phase 0=output, 1=training
    ))

    # Reshape the activations, the casting above adds another dimension
    act_volume = act_volume.reshape(
        act_volume.shape[1],
        act_volume.shape[2],
        act_volume.shape[3],
        act_volume.shape[4]
    )

    if K.image_data_format() == 'channels_first':
        b, out_ch, r, c = act_volume.shape
    else:
        b, r, c, out_ch = act_volume.shape
    # print("display_layer_activations: shape [b %d,out_ch %d,r %d,c %d]" % (b, out_ch, r, c))

    n = np.int(np.round(np.sqrt(out_ch)))

    # Construct a large image to tile all the activations
    width = (n * r) + ((n - 1) * margin)
    height = (n * c) + ((n - 1) * margin)
    tiled_filters = np.zeros((width, height))

    # Fill in in composite image with the filters
    for r_idx in range(n):
        for c_idx in range(n):

            filt_idx = (r_idx * n) + c_idx
            if filt_idx >= out_ch:
                break

            print("Processing filter %d" % filt_idx)

            if K.image_data_format() == 'channels_first':
                tiled_filters[
                    (r + margin) * r_idx: (r + margin) * r_idx + r,
                    (c + margin) * c_idx: (c + margin) * c_idx + c,
                ] = deprocess_image(act_volume[0, filt_idx, :, :])
            else:
                tiled_filters[
                    (r + margin) * r_idx: (r + margin) * r_idx + r,
                    (c + margin) * c_idx: (c + margin) * c_idx + c,
                ] = deprocess_image(act_volume[0, :, :, filt_idx])

    # Plot the composite figure
    f = plt.figure()
    plt.imshow(tiled_filters, cmap='Greys')  # force to 2D. Expected by imshow
    plt.colorbar()
    f.suptitle("Feature maps of layer @ idx %d: %s." % (layer_idx, model.layers[layer_idx].name))

    return act_volume


def add_noise(images, noise_type, **kwargs):
    """
    # TODO: add support for channel first format

    :param images:
    :param noise_type: ['gaussian', 'pepper']
    :param kwargs: conditional keyword arguments

    if noise_type = gaussian
    var = noise variance. default = 0.1
    mean = mean of the noise. default = 0

    :return: noisy images

    REF: https://stackoverflow.com/questions/22937589/
         how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """

    noise_type = noise_type.lower()
    allowed_noise = ['gaussian', 'pepper']

    if noise_type not in allowed_noise:
        raise Exception("Unknown noise type, %s" % noise_type)

    b, r, c, ch = images.shape

    output = np.zeros_like(images)
    if noise_type == 'gaussian':

        var = 0.1
        mean = 0

        if 'var' in kwargs.keys():
            var = kwargs['var']
        if 'mean' in kwargs.keys():
            mean = kwargs['mean']

        print("Adding Gaussian noise with mean %f, var %f to images of size %s" % (mean, var, images.shape))

        sigma = np.sqrt(var)
        noise = np.random.normal(mean, sigma, (b, r, c, ch))

        output = images + noise

    elif noise_type == 'pepper':
        prob = 0.004

        if 'prob' in kwargs.keys():
            prob = kwargs['prob']

        # num samples to blacken
        num_pepper = int(prob * (b * r * c * ch))

        print("Adding Pepper noise with probability %f to images of size %s" % (prob, images.shape))

        xs = [np.random.randint(0, max(i-1, 1), int(num_pepper)) for i in images.shape]
        xs = np.array(xs)

        # force numpy to create a separate copy of the input
        output = np.copy(images)

        output[xs[0, :], xs[1, :], xs[2, :], xs[3, :]] = 0

    return output


def plot_train_summary(t_summary):
    """

    :param t_summary: Return value of model.fit()
    :return:
    """
    f = plt.figure()

    f. add_subplot(1, 2, 1)
    plt.plot(range(len(t_summary.history['acc'])), t_summary.history['acc'], color='blue', label='Train')
    plt.plot(range(len(t_summary.history['val_acc'])), t_summary.history['val_acc'], color='red', label='Test')
    plt.title('Model Accuracy')
    plt.xlabel("Acc")
    plt.ylabel('Epoch')
    plt.legend(loc='best')

    f.add_subplot(1, 2, 2)
    plt.plot(range(len(t_summary.history['loss'])), t_summary.history['loss'], color='blue')
    plt.plot(range(len(t_summary.history['val_loss'])), t_summary.history['val_loss'], color='red')
    plt.title('Model Loss')
    plt.xlabel("Loss")
    plt.ylabel('Epoch')
