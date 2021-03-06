# ------------------------------------------------------------------------------------------------
# Contour integration layer in an MNIST classifier
#
# In this version of contour integration, a learnable weight matrix is cast on top each pixel in
# the input volume. The weight matrix is shared across a feature map but is not shared across
# feature maps. As such it models lateral connections between similarly oriented neurons in V1.
#
# The hope is that the network automatically learns to add weighted inputs from neighbors that
# are co-planer to its filter,  axial specificity as defined in [Ursino and Cara - 2004 - A
# model of contextual interactions and contour detection in primary visual cortex]
# ------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np

from keras.engine.topology import Layer
from keras.constraints import Constraint

import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.backend as K
import keras

from keras.models import Sequential, save_model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Flatten

import os

import utils

FILENAME = os.path.basename(__file__).split('.')[0] + '.hf'

BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 12

# MNIST Input Image Dimensions
IMG_ROWS, IMG_COL = 28, 28

# Set the random seed for reproducibility
np.random.seed(7)


class ZeroCenter(Constraint):
    def __init__(self, n, ch):
        """
        Add a constraint that the central element of the weigh matrix should be zero. Only lateral connections
        Should be learnt.

        :param n: dimensions of the weight matrix, assuming it is a square
        :param ch: number of channels in the input.
        """

        half_len = n**2 >> 1
        half_mask = K.ones((half_len, 1))
        mask_1d = K.concatenate((half_mask, K.constant([[0]]), half_mask), axis=0)
        mask = K.reshape(mask_1d, (n, n, 1))
        self.mask = K.tile(mask, [1, 1, ch])

    def __call__(self, w):
        w = w * self.mask
        return w


class ContourIntegrationLayer(Layer):

    def __init__(self, n=3,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        """

        :param n:
        :param kernel_initializer:
        :param kernel_regularizer:
        :param kernel_constraint:
        :param kwargs:
        """
        if n & 1 == 0:
            raise Exception("Lateral filter dimension should be an odd number. %d specified" % n)
        self.n = n

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.data_format = K.image_data_format()

        super(ContourIntegrationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Define  any learnable parameters for the layer

        :param input_shape:
        :return:

        """
        if self.data_format == 'channels_last':
            _,  r, c, ch = input_shape
        else:
            raise Exception("Only channel_last data format is supported.")

        self.kernel_shape = (self.n, self.n, ch)

        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=ZeroCenter(self.n, ch),
            trainable=True
        )

        super(ContourIntegrationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        _, r, c, ch = K.int_shape(inputs)
        # print("Call Fcn: Input shape ", r, c, ch)

        # 1. Inputs Formatting
        # Channel First, batch second. This is done to take the unknown batch size into the matrix multiply
        # where it can be handled more easily
        padded_inputs = K.spatial_2d_padding(
            inputs,
            ((self.n / 2, self.n / 2), (self.n / 2, self.n / 2))
        )

        inputs_chan_first = K.permute_dimensions(padded_inputs, [3, 0, 1, 2])
        # print("Call Fcn: inputs_chan_first shape: ", inputs_chan_first.shape)

        # 2. Kernel
        kernel_chan_first = K.permute_dimensions(self.kernel, (2, 0, 1))
        # print("Call Fcn: kernel_chan_first shape", kernel_chan_first.shape)
        k_ch, k_r, k_c = K.int_shape(kernel_chan_first)
        apply_kernel = K.reshape(kernel_chan_first, (k_ch, k_r * k_c, 1))
        # print("Call Fcn: kernel for matrix multiply: ", apply_kernel.shape)

        # 3. Get outputs at each spatial location
        xs = []
        for i in range(r):
            for j in range(c):
                input_slice = inputs_chan_first[:, :, i:i+self.n, j:j+self.n]
                input_slice_apply = K.reshape(input_slice, (ch, -1, self.n**2))

                output_slice = K.batch_dot(input_slice_apply, apply_kernel)
                # Reshape the output slice to put batch first
                output_slice = K.permute_dimensions(output_slice, [1, 0, 2])
                xs.append(output_slice)

        # print("Call Fcn: len of xs", len(xs))
        # print("Call Fcn: shape of each element of xs", xs[0].shape)

        # 4. Reshape the output to correct format
        outputs = K.concatenate(xs, axis=2)
        outputs = K.reshape(outputs, (-1, ch, r, c))  # Break into row and column
        outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])  # Back to batch first
        # print("Call Fcn: shape of output", outputs.shape)

        # 5. Add the lateral and the feed-forward activations
        outputs += inputs

        return outputs


if __name__ == "__main__":

    input_dims = (IMG_ROWS, IMG_COL, 1)  # Input dimensions for a single sample

    # 1. Get Data
    # --------------------------------------------------------------------------
    x_train, y_train, x_test, y_test, x_sample, y_sample = utils.get_mnist_data()

    # 2. Define the model
    # -------------------------------------------
    model = Sequential()
    # First Convolution layer, First sublayer processes feed-forward inputs, second layer adds the
    # lateral connections. Third sublayer adds the activation function.
    # Output = sigma_fcn([W_ff*x + W_l*(W_ff*x)]).
    # Where sigma_fcn is the activation function
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_dims, padding='same'))
    model.add(ContourIntegrationLayer(n=5))
    model.add(Activation('relu'))
    # Rest of the layers.
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=NUM_CLASSES, activation='softmax'))

    # 3. Compile/Train/Save the model
    # -------------------------------------------
    model.compile(
        loss=keras.losses.categorical_crossentropy,  # Note this is not a function call.
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    save_model(model, FILENAME)

    # 4. Evaluate Model accuracy
    # -------------------------------------------
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test Accuracy', score[1])

