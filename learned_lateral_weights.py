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
import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.constraints as constraints
import keras.backend as K

from keras.models import Sequential
from keras.layers import Conv2D, LocallyConnected2D

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12

# MNIST Input Image Dimensions
IMG_ROWS, IMG_COL = 28, 28

# Set the random seed for reproducibility
np.random.seed(7)


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
        self.kernel_constraint = constraints.get(kernel_constraint)
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
            constraint=self.kernel_constraint,
            trainable=True
        )

        super(ContourIntegrationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def get_config(self):
        config = {
            'kernel_shape': self.kernel_shape,
            'data_format': self.data_format,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(ContourIntegrationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        _, r, c, ch = inputs.shape
        print("Call Fcn: Input r, c, ch", r, c, ch)

        # 1. We multiply input slices with the kernel, to retain the input size at the output,
        # the input needs to be padded.
        padded_inputs = K.spatial_2d_padding(
            inputs,
            ((self.n / 2, self.n / 2), (self.n / 2, self.n / 2))
        )
        print("Call Fcn: shape of padded input ", padded_inputs.shape)

        # 2. Tensorflow uses C array indexing - last dimension changes first. Change the
        # dimensions of the input so that row and column are the last two dimensions
        input_col_last = K.permute_dimensions(padded_inputs, (0, 3, 1, 2))
        print("Call Fcn: shape of input_col_last ", input_col_last.shape)

        # 3. Force the central element, (that corresponds to the pixel itself to be zero)
        kernel_col_last = K.permute_dimensions(self.kernel, (2, 0, 1))
        print("Call Fcn: shape of kernel_col_last ", kernel_col_last.shape)

        # mask = K.ones(shape=(self.n, self.n))
        # mask[self.n / 2, self.n / 2] = 0
        # mask = K.reshape(mask, (1, self.n, self.n))
        # mask = K.tile(mask, n=0)
        # print("call Fcn: shape of mask", mask.shape)

        mask = K.var([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        mask = K.tile(mask, n=0)


        #surround_kernel = kernel_col_last *
        #
        # # 3. slice through the inputs to get the part of the input to multiply
        # flattened_kernel = K.reshape(self.kernel, (int(ch) * self.n * self.n, 1))
        # print("Call Fcn: shape of flattened kernel", flattened_kernel.shape)
        #
        # xs = []
        # for i in range(r):
        #     for j in range(c):
        #         input_slice = input_col_last[:, :, i: i+self.n, j: j+self.n]
        #         input_slice = K.reshape(input_slice, (-1, int(ch) * self.n * self.n))
        #         xs.append(input_slice)
        #
        # print("len of xs", len(xs))
        # print("shape of each element of xs", xs[0].shape)




        return inputs


if __name__ == "__main__":

    input_dims = (IMG_ROWS, IMG_COL, 1)  # Input dimensions for a single sample

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_dims, padding='same'))
    model.add(ContourIntegrationLayer(n=3))
    model.summary()
