# -------------------------------------------------------------------------------------------------
#  This is the Alex Net model trained on Imagenet.
#
#  Code Ref: https://github.com/heuritech/convnets-keras
#
# Author: Salman Khan
# Date  : 21/07/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


from keras.models import Model
from keras.layers import Input, Activation, Convolution2D, MaxPooling2D
from keras.layers import Dense, Flatten, ZeroPadding2D, merge, Dropout
import keras.backend as K
from keras.layers.core import Lambda

from keras.layers import Conv2D, Concatenate

import utils
reload(utils)


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original Alexnet
    """

    def f(X):
        b, ch, r, c = K.int_shape(X)
        print(b, ch, r, c)

        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(
            K.permute_dimensions(square, (0, 2, 3, 1)), ((0, 0), (half, half)))

        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))

        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, i:i + ch, :, :]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = K.int_shape(X)[axis] // ratio_split

        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def alex_net(weights_path):
    """
    Note: Layer names have to stay the same, to enable loading pre-trained weights

    :param weights_path:
    :return: alexnet model
    """

    inputs = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)

    conv_2_1 = Conv2D(128, (5, 5), activation='relu', name='conv_2_1')(splittensor(ratio_split=2, id_split=0)(conv_2))
    conv_2_2 = Conv2D(128, (5, 5), activation='relu', name='conv_2_2')(splittensor(ratio_split=2, id_split=1)(conv_2))
    conv_2 = Concatenate(axis=1, name='conv_2')([conv_2_1, conv_2_2])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4_1 = Conv2D(192, (3, 3), activation='relu', name='conv_4_1')(splittensor(ratio_split=2, id_split=0)(conv_4))
    conv_4_2 = Conv2D(192, (3, 3), activation='relu', name='conv_4_2')(splittensor(ratio_split=2, id_split=1)(conv_4))
    conv_4 = Concatenate(axis=1, name='conv_4')([conv_4_1, conv_4_2])

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5_1 = Conv2D(128, (3, 3), activation='relu', name='conv_5_1')(splittensor(ratio_split=2, id_split=0)(conv_5))
    conv_5_2 = Conv2D(128, (3, 3), activation='relu', name='conv_5_2')(splittensor(ratio_split=2, id_split=1)(conv_5))
    conv_5 = Concatenate(axis=1, name='conv_5')([conv_5_1, conv_5_2])

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)

    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)

    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(inputs=inputs, outputs=prediction)

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model

if __name__ == "__main__":

    # Model was originally defined with Theano backend.
    K.set_image_dim_ordering('th')
    model = alex_net("trained_models/alexnet_weights.h5")
    model.summary()

    # weights_ch_last = model.layers[1].weights[0]
    # utils.display_filters(weights_ch_last)
    #
    # from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    #
    # img = load_img("cat.7.jpg", target_size=(227,227))
    # x = img_to_array(img)
    # x = np.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])
    #
    # y_hat = model.predict(x, batch_size=1, verbose=1)
    # print("Prediction %s" % np.argmax(y_hat))
    #
    # #utils.display_layer_activations(model, 1, x)
    #
    # layer_idx = 1
    # data_sample = x
    #
    # get_layer_output = K.function(
    #     [model.layers[0].input, K.learning_phase()],
    #     [model.layers[layer_idx].output]
    # )
    #
    # # Get the activations in a usable format
    # act_volume = np.asarray(get_layer_output(
    #     [data_sample, 0],  # second input specifies the learning phase 0=output, 1=training
    # ))
    #
    # # Reshape the activations, the casting above adds another dimension
    # act_volume = act_volume.reshape(
    #     act_volume.shape[1],
    #     act_volume.shape[2],
    #     act_volume.shape[3],
    #     act_volume.shape[4]
    # )
    #
    # max_ch = np.int(np.round(np.sqrt(act_volume.shape[1])))
    #
    # f = plt.figure()
    #
    # for ch_idx in range(act_volume.shape[1]):
    #     f.add_subplot(max_ch, max_ch, ch_idx + 1)
    #     plt.imshow(act_volume[0, ch_idx, :, :], cmap='Greys')
    #
    # f.suptitle("Feature maps of layer @ idx %d: %s" % (layer_idx, model.layers[layer_idx].name))






    # x = img_to_array(img, 'channels_first')
    # x = x.reshape((1,) + x.shape)
    # print(x.shape)
    #
    # inp = Input(shape=(None, None, 3))
    # out1 = Lambda(lambda image: K.image.resize_images(image, (128, 128)))(inp)
    # model2 = Model(input=inp, output=out1)
    #
    #
    # f = misc.imread("cat.7.jpg")
    # out = model2.predict(f[np.newaxis, ...])
    #
    # fig, Axes = plt.subplots(nrows=1, ncols=2)
    # Axes[0].imshow(f)
    # Axes[1].imshow(np.int8(out[0, ...]))
    #
    #
    #
    # # f = np.reshape(f, (1, f.shape[0],f.shape[1],f.shape[2]))
    # # f = K.resize_images(f, 227, 227, 'channels_first')
    # # print(f.shape)
    # # plt.ion()
    # #
    # # plt.imshow(f)
    # #
    # # y_hat = model.predict(f, batch_size=1, verbose=0)
    # # print("Estimate sample", np.argmax(y_hat))



