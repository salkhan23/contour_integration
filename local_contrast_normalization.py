# -------------------------------------------------------------------------------------------------
#  Local Contrast Normalization Layer.
#
#  Ref: First used in Alexnet paper.
#  Implementation Ref: https://github.com/heuritech/convnets-keras
#
# Author: Salman Khan
# Date  : 09/07/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np

import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.core import Lambda

import utils
reload(utils)


def local_response_normalization(x, alpha=1e-4, k=2, beta=0.75, n=5):
    """

    :param x:
    :param alpha:
    :param k:
    :param beta:
    :param n:
    :return:
    """
    if K.image_data_format() == 'channel_first':
        b, ch, r, c = K.int_shape(x)
    else:
        b, r, c, ch = K.int_shape(x)
        x = K.permute_dimensions(x, (0, 3, 1, 2))

    square = K.square(x)
    half = n // 2

    extra_channels = K.spatial_2d_padding(square, ((half, half), (0, 0)))

    scale = k
    for i in range(n):
        scale += alpha * extra_channels[:, i: i+ch, :, :]
    scale ** beta

    output = x / scale
    if K.image_data_format() == 'channels_last':
        output = K.permute_dimensions(output, (0, 2, 3, 1))

    return output


if __name__ == "__main__":

    # Initializations
    BATCH_SIZE = 64
    NUM_CLASSES = 10
    EPOCHS = 12
    IMG_ROWS, IMG_COL = 28, 28

    np.random.seed(7)

    # 1. Load the Data
    # ------------------------------------------------------------------------
    x_train, y_train, x_test, y_test, x_sample, y_sample = utils.get_mnist_data()

    # 2. Define the Model
    # ------------------------------------------------------------------------
    model = Sequential()

    input_dims = (IMG_ROWS, IMG_COL, 1)  # for a single input
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_dims, padding='same'))
    model.add(Lambda(local_response_normalization, output_shape=lambda input_shape: input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=NUM_CLASSES, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,  # Note this is not a function call.
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    train_summary = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test Accuracy', score[1])

    utils.plot_train_summary(train_summary)
