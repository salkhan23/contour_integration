# -------------------------------------------------------------------------------------------------
#  File defines multiple variants of contour integration layers and some functions to build
#  them onto base alex net models.
#
# Author: Salman Khan
# Date  : 03/09/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Activation, MaxPooling2D, Conv2D, Concatenate
from keras.layers import Dense, Flatten, ZeroPadding2D, Dropout
from keras.preprocessing.image import img_to_array, load_img
from keras.engine.topology import Layer
import keras.backend as K
import keras.activations as activations

from base_models import base_alex_net as alex_net
import alex_net_utils as alex_net_utils
import utils as common_utils

reload(alex_net)
reload(alex_net_utils)
reload(common_utils)


np.random.seed(7)  # Set the random seed for reproducibility


def get_coaligned_masks(weights_type, rf_len, non_overlapping_rfs=True):
    """
    Returns a mask of coaligned (and optionally orthogonal to coaligned) neighbors for
    each of the 96 feature extracting kernels of the first convolutional layer of Alexnet.

    Masks are manually defined.

    :param weights_type: ['enhance', 'suppress', 'enhance_and_suppress']
            enhance  = coaligned neighbors only
            suppress = orthogonal to coaligned neighbors only
            enhance_and_suppress = both.

    :param rf_len: length of RF over which lateral connections are defined

    :param non_overlapping_rfs:[True(default), False]
            True  = include neighbors with non-overlapping visual fields,
            False = include neighbors with overlapping visual fields

    :return: [96 x re_len x rf_len] array of kernels
    """
    half_rf = rf_len // 2

    kernel = np.zeros((96, rf_len, rf_len))  # There are 96 conv layer 1 kernels in alexnet.

    if non_overlapping_rfs:
        # In alexnet, filter size of 11x11 is used with a stride of 4. Therefore every third
        # neighbor has a non-overlapping receptive field.
        step_size = 3
    else:
        step_size = 1

    pre_half_range = range(0, half_rf, step_size)
    post_half_range = range(half_rf + step_size, rf_len, step_size)

    # Vertical_kernel
    kernel[10, pre_half_range, half_rf] = 1
    kernel[10, post_half_range, half_rf] = 1
    if weights_type != 'enhance':  # Suppression values
        kernel[10, half_rf, pre_half_range] = -1
        kernel[10, half_rf, post_half_range] = -1

    # Horizontal kernel
    kernel[5, half_rf, pre_half_range] = 1
    kernel[5, half_rf, post_half_range] = 1
    if weights_type != 'enhance':  # Suppression values
        kernel[5, pre_half_range, half_rf] = -1
        kernel[5, post_half_range, half_rf] = -1

    # Diagonal Kernel (Leaning backwards)
    # kernel[54, (0, 3, 6, 9, 15, 18, 21, 24), (8, 9, 10, 11, 13, 14, 15, 16)] = 1
    kernel[54, (1, 4, 6, 9, 15, 17, 20, 23), (8, 9, 10, 11, 13, 14, 15, 16)] = 1
    # TODO: Add suppression values

    kernel[64, range(0, half_rf, 3), range(0, half_rf, 3)] = 1
    kernel[64, range(half_rf + 3, rf_len, 3), range(half_rf + 3, rf_len, 3)] = 1
    if weights_type != 'enhance':  # Suppression values
        kernel[64, range(rf_len - 1, half_rf, -3), range(0, half_rf, 3)] = -1
        kernel[64, range(half_rf - 3, -1, -3), range(half_rf + 3, rf_len, 3)] = -1

    kernel[67, :, :] = np.copy(kernel[54, :, :])

    kernel[78, (1, 3, 7, 10, 14, 17, 21, 23), (17, 16, 14, 13, 11, 10, 8, 7)] = 1
    # TODO: Add suppression values

    return kernel


def get_non_overlapping_kernels(rf_len):
    """
    For a neuron centered @ (0, 0) returns a surrounding mask identifying neighbors over a
    [rf_len x rf_len] set that have nonoverlapping visual fields.

    Only visually non-overlapping neighbours are connected with lateral connections.

    In Alexnet, convolutional layer 1 has a filter size of 11x11 and uses a stride of 4.
    Therefore every 3rd neighbor is visually non-overlapping.

    :param rf_len:

    :return: [96 x re_len x rf_len] array of kernels
    """
    xx, yy = np.meshgrid(range(0, rf_len, 3), range(0, rf_len, 3))

    mask = np.zeros((rf_len, rf_len))
    mask[xx, yy] = 1
    # set the weight contribution of the center neuron to zero
    mask[rf_len // 2, rf_len // 2] = 0

    kernel = np.repeat(mask[np.newaxis, :, :], 96, axis=0)

    return kernel


class AdditiveContourIntegrationLayer(Layer):
    def __init__(self, weights_type, n=25,  activation=None, **kwargs):
        """
        Linear additive contour integration model where weighted neighbor feed forward
        responses are added to the the feed forward response of neuron. Here feed forward refers
        to the output of the first convolutional layer of Alexnet. Additionally a mask that identifies
        non-overlapping and co-aligned and orthogonal to co-aligned neighbors.

        For each L1 feature map:
        A_L(x,y) = sigma(A_FF(x,y) + sum_over_m_and_n[W(m,n) * M(m, n) * A_FF(x-m,y-m)])

        where
            A_L(x,y) = output of the contour enhancement layer @ (x, y)
            A_FF(x,y) = feed-forward output of conv Layer 1 @ (x, y)
            sigma = non-linearity (currently none)
            m & n are the row and column indices of the contour enhancement layer neurons.
            W(m,n) = weight of contribution from neuron @ (m,n) from (x,y)
            M(m,n) = Manually defined mask that identifies coaligned and orthogonal neurons with
            non-overlapping visual fields.

        :param weights_type: types of weights [enhance, suppress or enhance_and_suppress]
        :param n: size of square contour Integration RF. The number of L1 output neighbors over
                  which lateral connections span (not over the visual space)
        :param kwargs:
        """

        # 7 is the smallest size to include the first visually non-overlapping neighbor
        if n > 55 or n < 7 or (n & 1 == 0):
            raise Exception("Invalid horizontal connections extent." +
                            "Must match dimensions of Conv1 Layer output [7-55] & must be odd.")
        self.n = n

        if weights_type.lower() not in ['enhance', 'suppress', 'enhance_and_suppress']:
            raise Exception("Invalid weight types. Must be [enhance, suppress or enhance_and_suppress]")
        self.weights_type = weights_type.lower()

        self.kernel = get_coaligned_masks(self.weights_type, self.n)
        self.kernel = K.variable(self.kernel)

        self.activation = activations.get(activation)

        super(AdditiveContourIntegrationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        No learnable parameters for this model

        :param input_shape:
        :return:
        """
        super(AdditiveContourIntegrationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :return:
        """
        if K.image_data_format() == 'channels_last':
            _, r, c, ch = K.int_shape(inputs)
        else:
            _, ch, r, c = K.int_shape(inputs)
        # print("Call Fcn: Input shape ", K.int_shape(inputs))

        # 1. Inputs Formatting
        # ----------------------------------
        # Pad the rows and columns to allow full matrix multiplication
        # Note that this function is aware of which dimension the columns and rows are
        padded_inputs = K.spatial_2d_padding(
            inputs,
            ((self.n / 2, self.n / 2), (self.n / 2, self.n / 2))
        )
        # print("Call Fcn: padded_inputs shape ", K.int_shape(padded_inputs))

        # Channel first, batch second. This is done to take the unknown batch size into the matrix multiply
        # where it can be handled more easily
        if K.image_data_format() == 'channels_last':
            inputs_chan_first = K.permute_dimensions(padded_inputs, [3, 0, 1, 2])
        else:
            inputs_chan_first = K.permute_dimensions(padded_inputs, [1, 0, 2, 3])
        # print("Call Fcn: inputs_chan_first shape: ", inputs_chan_first.shape)

        # 2. Kernel Formatting
        # --------------------
        if K.image_data_format() == 'channels_last':
            kernel_chan_first = K.permute_dimensions(self.kernel, (2, 0, 1))
        else:
            kernel_chan_first = self.kernel
        # print("Call Fcn: kernel_chan_first shape", kernel_chan_first.shape)

        # Flatten rows and columns into a single dimension
        k_ch, k_r, k_c = K.int_shape(kernel_chan_first)
        apply_kernel = K.reshape(kernel_chan_first, (k_ch, k_r * k_c, 1))
        # print("Call Fcn: kernel for matrix multiply: ", apply_kernel.shape)

        # 3. Get outputs at each spatial location
        # ----------------------------------------
        xs = []
        for i in range(r):
            for j in range(c):
                input_slice = inputs_chan_first[:, :, i:i + self.n, j:j + self.n]
                input_slice_apply = K.reshape(input_slice, (ch, -1, self.n ** 2))

                output_slice = K.batch_dot(input_slice_apply, apply_kernel)
                # Reshape the output slice to put batch first
                output_slice = K.permute_dimensions(output_slice, [1, 0, 2])
                xs.append(output_slice)

        # print("Call Fcn: len of xs", len(xs))
        # print("Call Fcn: shape of each element of xs", xs[0].shape)

        # Reshape the output to correct format
        outputs = K.concatenate(xs, axis=2)
        outputs = K.reshape(outputs, (-1, ch, r, c))  # Break into row and column

        if K.image_data_format() == 'channels_last':
            outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])  # Back to batch last

        # 4. Add the lateral and the feed-forward activations
        # ------------------------------------------------------
        outputs = self.activation(outputs)
        outputs += inputs
        return outputs


class GaussianMultiplicativeContourIntegrationLayer(Layer):

    def __init__(self, weights_type, n=25, sigma=6.0, activation=None, **kwargs):
        """

        Multiplicative integration model where the weighted sum of neighbor feed forward
        responses is multiplied with the the feed forward response of neuron. Here feed forward refers
        to the output of the first convolutional layer of Alexnet.

        The objective of this multiplication is to model the observed property that contour enhancement
        happens only when there is a signal in the classical RF(when it has a non-zero L1 feed forward output).
        This is not modeled by the AdditiveContourIntegrationLayer.

        Additionally, the relative gain of adding neighbors that are further away from the target neuron
        decreases with distance. This is modeled by casting a gaussian mask centered at the RF to decrease
        the relative gains of far away neighbors.

        Because we are multiplying multiplying feed-forward inputs that can be quite large, two learnable
        parameters are added to scale (alpha) and shift(bias) the contour enhancement gain. Default values of
        this parameters are 1 and 0 for each feature map. However, best fit values, found by matching
        neurophysiology data are available. (TO BE ADDED)

        For each L1 feature map:
            A_L(x,y) = sigma( A_FF(x,y) +
                A_FF(x,y) * alpha * sum_over_m_and_n[G(m,n) * M(m, n) * W(m,n) * A_FF(x-m,y-m) + bias])

        where
            A_L(x,y) = output of the contour enhancement layer @ (x, y)
            A_FF(x,y) = feed-forward output of conv Layer 1 @ (x, y)
            sigma = non-linearity (currently none)
            m & n are the row and column indices of the contour enhancement layer neurons.
            W(m,n) = weight of contribution from neuron @ (m,n) from (x,y)
            M(m,n) = Manually defined mask that identifies coaligned and orthogonal neurons with
            non-overlapping visual fields.

            G(m,n) = The weight W(m,n) is further scaled by a Gaussian (0, sigma)
            alpha = learnable scaling factor. Default = 1.
            bias = learnable shifting factor. Default = 0.

        :param weights_type: types of weights [enhance, suppress or enhance_and_suppress]
        :param n: size of square contour Integration RF. The number of L1 output neighbors over
                  which lateral connections span (not over the visual space). Default=25.
        :param sigma: standard deviation of gaussian mask.
        """

        # 7 is the smallest size to include the first visually non-overlapping neighbor
        if n > 55 or n < 7 or (n & 1 == 0):
            raise Exception("Invalid horizontal connections extent." +
                            "Must match dimensions of Conv1 Layer output [7-55] & must be odd.")
        self.n = n

        if weights_type.lower() not in ['enhance', 'suppress', 'enhance_and_suppress', 'overlap']:
            raise Exception("Invalid weight types. Must be [enhance, suppress or enhance_and_suppress, overlap]")
        self.weights_type = weights_type.lower()

        if self.weights_type == 'overlap':
            self.kernel = get_coaligned_masks(weights_type=self.weights_type, rf_len=self.n, non_overlapping_rfs=False)
        else:
            self.kernel = get_coaligned_masks(self.weights_type, self.n)

        g_kernel = alex_net_utils.get_2d_gaussian_kernel((n, n), sigma)
        self.kernel = np.array([k * g_kernel for k in self.kernel])
        self.kernel = K.variable(self.kernel)

        self.activation = activations.get(activation)

        super(GaussianMultiplicativeContourIntegrationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Two learnable parameters, alpha (96x1x1) and bias (96x1x1)

        :param input_shape:
        :return:
        """
        if K.image_data_format() == 'channels_last':
            _, r, c, ch = input_shape
        else:
            _, ch, r, c = input_shape
        # print("Build Fcn: ", K.image_data_format(), ". shape= ",  input_shape)

        self.bias = self.add_weight(
            shape=(ch, 1, 1),
            initializer='zeros',
            name='bias',
            trainable=True
        )

        self.alpha = self.add_weight(
            shape=(ch, 1, 1),
            initializer='ones',
            name='alpha',
            trainable=True
        )

        super(GaussianMultiplicativeContourIntegrationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs, **kwargs):

        if K.image_data_format() == 'channels_last':
            _, r, c, ch = K.int_shape(inputs)
        else:
            _, ch, r, c = K.int_shape(inputs)
        # print("Call Fcn: ", K.image_data_format(), ". shape= ",  input_shape)

        # 1. Inputs Formatting
        # ----------------------------------
        # Pad the rows and columns to allow full matrix multiplication
        # Note that this function is aware of which dimension the columns and rows are
        padded_inputs = K.spatial_2d_padding(
            inputs,
            ((self.n / 2, self.n / 2), (self.n / 2, self.n / 2))
        )
        # print("Call Fcn: padded_inputs shape ", K.int_shape(padded_inputs))

        # Channel first, batch second. This is done to take the unknown batch size into the matrix multiply
        # where it can be handled more easily
        if K.image_data_format() == 'channels_last':
            inputs_chan_first = K.permute_dimensions(padded_inputs, [3, 0, 1, 2])
        else:
            inputs_chan_first = K.permute_dimensions(padded_inputs, [1, 0, 2, 3])
        # print("Call Fcn: inputs_chan_first shape: ", inputs_chan_first.shape)

        # 2. Kernel Formatting
        # --------------------
        if K.image_data_format() == 'channels_last':

            kernel_chan_first = K.permute_dimensions(self.kernel, (2, 0, 1))
        else:
            kernel_chan_first = self.kernel
        # print("Call Fcn: kernel_chan_first shape", kernel_chan_first.shape)

        # Flatten rows and columns into a single dimension
        k_ch, k_r, k_c = K.int_shape(kernel_chan_first)
        apply_kernel = K.reshape(kernel_chan_first, (k_ch, k_r * k_c, 1))
        # print("Call Fcn: kernel for matrix multiply: ", apply_kernel.shape)

        # 3. Get outputs at each spatial location
        # ----------------------------------------
        xs = []
        for i in range(r):
            for j in range(c):
                input_slice = inputs_chan_first[:, :, i:i + self.n, j:j + self.n]
                input_slice_apply = K.reshape(input_slice, (ch, -1, self.n ** 2))

                output_slice = K.batch_dot(input_slice_apply, apply_kernel) * self.alpha

                # Reshape the output slice to put batch first
                output_slice = K.permute_dimensions(output_slice, [1, 0, 2])
                xs.append(output_slice)

        # print("Call Fcn: len of xs", len(xs))
        # print("Call Fcn: shape of each element of xs", xs[0].shape)

        # Reshape the output to correct format
        outputs = K.concatenate(xs, axis=2)
        outputs = K.reshape(outputs, (-1, ch, r, c))  # Break into row and column

        if K.image_data_format() == 'channels_last':
            outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])  # Back to batch last
            outputs = outputs

        # 4. Add the lateral and the feed-forward activations
        # ------------------------------------------------------
        outputs = outputs * inputs + self.bias
        outputs = self.activation(outputs)

        return outputs + inputs


class MaskedMultiplicativeContourIntegrationLayer(Layer):

    def __init__(self, weights_type, n=25, activation=None, **kwargs):
        """

        Multiplicative integration model where the weighted sum of neighbor feed forward
        responses is multiplied with the the feed forward response of the neuron. Here feed forward refers
        to the output of the first convolutional layer of Alexnet.

        The objective of this multiplication is to model the observed property that contour enhancement
        happens only when there is a signal in the classical RF(when it has a non-zero L1 feed forward output).
        This is not modeled by the AdditiveContourIntegrationLayer.

        Compared to the Gaussian Multiplicative Contour Integration model, no constraint on the weightings
        of neighbors is assumed. Instead, the weights of neighbors are learnt though gradient descent to match
        neurological data (gains). Note a mask is still used to define which of the neighbors are coaligned and
        (orthogonal to coaligned). Initial values of these allowed weights are ones However, best fit values
        found by the optimization process are available. (TO BE ADDED)

        For each L1 feature map:
            A_L(x,y) = sigma(A_FF(x,y) +
               A_FF(x,y) * sum_over_m_and_n[ W(m,n) * M(m,n) * A_FF(x-m,y-m) + bias])

        where
            A_L(x,y) = output of the contour enhancement layer @ (x, y)
            A_FF(x,y) = feed-forward output of conv Layer 1 @ (x, y)
            sigma = non-linearity (currently none)
            m & n are the row and column indices of the contour enhancement layer neurons.
            W(m,n) = weight of contribution from neuron @ (m,n) from (x,y)
            M(m,n) = Manually defined mask that identifies coaligned and orthogonal neurons with
            non-overlapping visual fields.

            bias = learnable shifting factor. Default = 0.

        :param weights_type: types of weights [enhance, suppress or enhance_and_suppress]
        :param n: size of square contour Integration RF. The number of L1 output neighbors over
                  which lateral connections span (not over the visual space). Default=25.
        """

        # 7 is the smallest size to include the first visually non-overlapping neighbor
        if n > 55 or n < 7 or (n & 1 == 0):
            raise Exception("Invalid horizontal connections extent." +
                            "Must match dimensions of Conv1 Layer output [7-55] & must be odd.")
        self.n = n

        if weights_type.lower() not in ['enhance', 'suppress', 'enhance_and_suppress']:
            raise Exception("Invalid weight types. Must be [enhance, suppress or enhance_and_suppress]")
        self.weights_type = weights_type.lower()

        self.mask = get_coaligned_masks(self.weights_type, self.n)
        self.mask = K.variable(self.mask)

        self.activation = activations.get(activation)

        super(MaskedMultiplicativeContourIntegrationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        2 learnable parameters, but first one is a matrix over all L2 neighbors. However, we only care about
        certain neighbors. Those for which the output of raw_kernel * mask is nonzero

        :param input_shape:
        :return:
        """
        if K.image_data_format() == 'channels_last':
            _, r, c, ch = input_shape
            # print("Build Fcn: Channel Last Input shape ", input_shape)
        else:
            _, ch, r, c = input_shape
            # print("Build Fcn: Channel First Input shape ", input_shape)

        self.raw_kernel = self.add_weight(
            shape=(ch, self.n, self.n,),
            # initializer='ones',
            # initializer='glorot_normal',
            initializer='he_normal',  # Recommended for Relu nonlinearity.
            name='raw_kernel',
            trainable=True
        )

        self.bias = self.add_weight(
            shape=(ch, 1, 1),
            initializer='zeros',
            name='bias',
            trainable=True
        )

        super(MaskedMultiplicativeContourIntegrationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs, **kwargs):

        if K.image_data_format() == 'channels_last':
            _, r, c, ch = K.int_shape(inputs)
            # print("Call Fcn: Channel Last Input shape ", K.int_shape(inputs))
        else:
            _, ch, r, c = K.int_shape(inputs)
            # print("Call Fcn: Channel First Input shape ", K.int_shape(inputs))

        # 1. Inputs Formatting
        # ----------------------------------
        # Pad the rows and columns to allow full matrix multiplication
        # Note that this function is aware of which dimension the columns and rows are
        padded_inputs = K.spatial_2d_padding(
            inputs,
            ((self.n / 2, self.n / 2), (self.n / 2, self.n / 2))
        )
        # print("Call Fcn: padded_inputs shape ", K.int_shape(padded_inputs))

        # Channel first, batch second. This is done to take the unknown batch size into the matrix multiply
        # where it can be handled more easily
        if K.image_data_format() == 'channels_last':
            inputs_chan_first = K.permute_dimensions(padded_inputs, [3, 0, 1, 2])
        else:
            inputs_chan_first = K.permute_dimensions(padded_inputs, [1, 0, 2, 3])
        # print("Call Fcn: inputs_chan_first shape: ", inputs_chan_first.shape)

        # 2. Kernel Formatting
        # --------------------
        # mask the kernel to keep only neighbors with overlapping RFs
        self.kernel = self.raw_kernel * self.mask

        if K.image_data_format() == 'channels_last':

            kernel_chan_first = K.permute_dimensions(self.kernel, (2, 0, 1))
        else:
            kernel_chan_first = self.kernel
        # print("Call Fcn: kernel_chan_first shape", kernel_chan_first.shape)

        # Flatten rows and columns into a single dimension
        k_ch, k_r, k_c = K.int_shape(kernel_chan_first)
        apply_kernel = K.reshape(kernel_chan_first, (k_ch, k_r * k_c, 1))
        # print("Call Fcn: kernel for matrix multiply: ", apply_kernel.shape)

        # 3. Get outputs at each spatial location
        # ----------------------------------------
        xs = []
        for i in range(r):
            for j in range(c):
                input_slice = inputs_chan_first[:, :, i:i + self.n, j:j + self.n]
                input_slice_apply = K.reshape(input_slice, (ch, -1, self.n ** 2))

                output_slice = K.batch_dot(input_slice_apply, apply_kernel)

                # Reshape the output slice to put batch first
                output_slice = K.permute_dimensions(output_slice, [1, 0, 2])
                xs.append(output_slice)

        # print("Call Fcn: len of xs", len(xs))
        # print("Call Fcn: shape of each element of xs", xs[0].shape)

        # Reshape the output to correct format
        outputs = K.concatenate(xs, axis=2)
        outputs = K.reshape(outputs, (-1, ch, r, c))  # Break into row and column

        if K.image_data_format() == 'channels_last':
            outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])  # Back to batch last

        # 4. Add the lateral and the feed-forward activations
        # ------------------------------------------------------
        outputs = outputs * inputs + self.bias
        outputs = self.activation(outputs)

        return outputs + inputs


class MultiplicativeContourIntegrationLayer(Layer):

    def __init__(self, n=25, activation=None, **kwargs):
        """

        Multiplicative contour integration model where the weighted sum of neighbor responses
        is multiplied with the the feed forward response of the neuron. Here feed forward refers
        to the output of the first convolutional layer of Alexnet.

        The objective of this multiplication is to model the observed property that contour enhancement
        occurs only when there is a signal in the classical RF. This is not modeled by the
        AdditiveContourIntegrationLayer.

        Compared to the Gaussian multiplicative contour integration layer, no constraint on neighbor
        weights are used. Instead, they are learnt though gradient descent to match neurological data (gains).

        Compared to the masked multiplicative model, the mask used does not identify coaligned
        neighbors. Instead, a generic mask identifying a set of non-overlapping neigbors that see the entire
        visual field is used. The same mask is used for each feature extracting kernel and is modified
        independently by each layer. The learning task is to identify which of the neighbors are coaligned
        and how to combine them to match expected gains.

        For each L1 feature map:
            A_L(x,y) = sigma(A_FF(x,y) +
               A_FF(x,y) * sum_over_m_and_n[ W(m,n) * M(m,n) * A_FF(x-m,y-m) + bias])

        where
            A_L(x,y) = output of the contour enhancement layer @ (x, y)
            A_FF(x,y) = feed-forward output of conv layer 1 @ (x, y)
            sigma = non-linearity (currently none)
            m & n are the row and column indices of the contour enhancement layer neurons.
            W(m,n) = weight of contribution from neuron @ offsets (m,n) from (x,y)
            M(m,n) = Generic mask that identifies a set of neighbor neurons with non-overlapping
            visual fields.

            bias = learnable shifting factor. Default = 0.

        :param n: size of square contour Integration RF. The number of L1 output neighbors over
                  which lateral connections span (not over the visual space). Default=25.
        """

        # 7 is the smallest size to include the first visually non-overlapping neighbor
        if n > 55 or n < 7 or (n & 1 == 0):
            raise Exception("Invalid horizontal connections extent." +
                            "Must match dimensions of Conv1 Layer output [7-55] & must be odd.")
        self.n = n

        self.mask = get_non_overlapping_kernels(self.n)
        self.mask = K.variable(self.mask)

        self.activation = activations.get(activation)

        super(MultiplicativeContourIntegrationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        2 learnable parameters: weights and bias

        The weight matrix is a full rf_len x rf_len matrix. However not all are active. The active weights
        are found by multiplying the raw_kernel with the mask. Nonzero weights are active.

        :param input_shape: [batch_size, row,column, n_channels]

        :return:
        """
        if K.image_data_format() == 'channels_last':
            _, r, c, ch = input_shape
            # print("Build Fcn: Channel Last Input shape ", input_shape)
        else:
            _, ch, r, c = input_shape
            # print("Build Fcn: Channel First Input shape ", input_shape)

        self.raw_kernel = self.add_weight(
            shape=(ch, self.n, self.n,),
            initializer='glorot_normal',
            name='raw_kernel',
            trainable=True
        )

        self.bias = self.add_weight(
            shape=(ch, 1, 1),
            initializer='zeros',
            name='bias',
            trainable=True
        )

        super(MultiplicativeContourIntegrationLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape  # Layer does not change the shape of the input

    def call(self, inputs, **kwargs):

        if K.image_data_format() == 'channels_last':
            _, r, c, ch = K.int_shape(inputs)
            # print("Call Fcn: Channel Last Input shape ", K.int_shape(inputs))
        else:
            _, ch, r, c = K.int_shape(inputs)
            # print("Call Fcn: Channel First Input shape ", K.int_shape(inputs))

        # 1. Inputs Formatting
        # ----------------------------------
        # Pad the rows and columns to allow full matrix multiplication
        # Note that this function is aware of which dimension the columns and rows are
        padded_inputs = K.spatial_2d_padding(
            inputs,
            ((self.n / 2, self.n / 2), (self.n / 2, self.n / 2))
        )
        # print("Call Fcn: padded_inputs shape ", K.int_shape(padded_inputs))

        # Channel first, batch second. This is done to take the unknown batch size into the matrix multiply
        # where it can be handled more easily
        if K.image_data_format() == 'channels_last':
            inputs_chan_first = K.permute_dimensions(padded_inputs, [3, 0, 1, 2])
        else:
            inputs_chan_first = K.permute_dimensions(padded_inputs, [1, 0, 2, 3])
        # print("Call Fcn: inputs_chan_first shape: ", inputs_chan_first.shape)

        # 2. Kernel Formatting
        # --------------------
        # mask the kernel to keep only neighbors with overlapping RFs
        self.kernel = self.raw_kernel * self.mask

        if K.image_data_format() == 'channels_last':

            kernel_chan_first = K.permute_dimensions(self.kernel, (2, 0, 1))
        else:
            kernel_chan_first = self.kernel
        # print("Call Fcn: kernel_chan_first shape", kernel_chan_first.shape)

        # Flatten rows and columns into a single dimension
        k_ch, k_r, k_c = K.int_shape(kernel_chan_first)
        apply_kernel = K.reshape(kernel_chan_first, (k_ch, k_r * k_c, 1))
        # print("Call Fcn: kernel for matrix multiply: ", apply_kernel.shape)

        # 3. Get outputs at each spatial location
        # ----------------------------------------
        xs = []
        for i in range(r):
            for j in range(c):
                input_slice = inputs_chan_first[:, :, i:i + self.n, j:j + self.n]
                input_slice_apply = K.reshape(input_slice, (ch, -1, self.n ** 2))

                output_slice = K.batch_dot(input_slice_apply, apply_kernel)

                # Reshape the output slice to put batch first
                output_slice = K.permute_dimensions(output_slice, [1, 0, 2])
                xs.append(output_slice)

        # print("Call Fcn: len of xs", len(xs))
        # print("Call Fcn: shape of each element of xs", xs[0].shape)

        # Reshape the output to correct format
        outputs = K.concatenate(xs, axis=2)
        outputs = K.reshape(outputs, (-1, ch, r, c))  # Break into row and column

        if K.image_data_format() == 'channels_last':
            outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])  # Back to batch last

        # 4. Add the lateral and the feed-forward activations
        # ------------------------------------------------------
        outputs = outputs * inputs + self.bias
        outputs = self.activation(outputs)

        return outputs + inputs


def build_contour_integration_model(cont_int_type, weights_path=None, **kwargs):
    """
    Build an alexnet model with the specified contour integration layer inserted after the
    first convolutional layer

    :param weights_path:
    :param cont_int_type: the type of contour integration layer to build
    :param kwargs: dictionary contain parameters of the specified contour integration layer

    :return: model
    """

    valid_cont_int_layer_types = ['additive', 'gaussian_multiplicative', 'masked_multiplicative', 'multiplicative']
    if cont_int_type.lower() not in valid_cont_int_layer_types:
        raise Exception("Invalid Contour Integration Layer type! Must be from %s" % valid_cont_int_layer_types)
    cont_int_type = cont_int_type.lower()

    inputs = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(inputs)

    if cont_int_type == 'additive':
        contour_int_layer = AdditiveContourIntegrationLayer(
            name='contour_integration',
            weights_type=kwargs['weights_type'],
            n=kwargs['n'],
            activation=kwargs.get('activation')
        )(conv_1)

    elif cont_int_type == 'gaussian_multiplicative':
        contour_int_layer = GaussianMultiplicativeContourIntegrationLayer(
            name='contour_integration',
            weights_type=kwargs['weights_type'],
            n=kwargs['n'],
            sigma=kwargs['sigma'],
            activation=kwargs.get('activation')
        )(conv_1)

    elif cont_int_type == 'masked_multiplicative':
        contour_int_layer = MaskedMultiplicativeContourIntegrationLayer(
            name='contour_integration',
            weights_type=kwargs['weights_type'],
            n=kwargs['n'],
            activation=kwargs.get('activation')
        )(conv_1)
    else:
        contour_int_layer = MultiplicativeContourIntegrationLayer(
            name='contour_integration',
            n=kwargs['n'],
            activation=kwargs.get('activation')
        )(conv_1)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(contour_int_layer)
    conv_2 = alex_net.crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)

    conv_2_1 = Conv2D(128, (5, 5), activation='relu', name='conv_2_1') \
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_2))
    conv_2_2 = Conv2D(128, (5, 5), activation='relu', name='conv_2_2') \
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_2))
    conv_2 = Concatenate(axis=1, name='conv_2')([conv_2_1, conv_2_2])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = alex_net.crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4_1 = Conv2D(192, (3, 3), activation='relu', name='conv_4_1') \
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_4))
    conv_4_2 = Conv2D(192, (3, 3), activation='relu', name='conv_4_2') \
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_4))
    conv_4 = Concatenate(axis=1, name='conv_4')([conv_4_1, conv_4_2])

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5_1 = Conv2D(128, (3, 3), activation='relu', name='conv_5_1') \
        (alex_net.splittensor(ratio_split=2, id_split=0)(conv_5))
    conv_5_2 = Conv2D(128, (3, 3), activation='relu', name='conv_5_2') \
        (alex_net.splittensor(ratio_split=2, id_split=1)(conv_5))
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


def main(model, fragment, f_idx, l1_act_cb, l2_act_cb):
    """
    This is the main routine for this file. First, the given contour fragment is tiled into
    a test image to form variously oriented contours. This is then passed to the model to
    visualize the activations of the first convolutional layer and contour integration layer

    :param model: contour integration model to use
    :param fragment: contour fragment from which contours are generated
    :param f_idx: target L2 filter index
    :param l2_act_cb: callback to get activations of L1 conv layer
    :param l1_act_cb: callback to get activations of L2 contour integration layer

    :return:
    """

    tgt_conv1_filter = K.eval(model.layers[1].weights[0])
    tgt_conv1_filter = tgt_conv1_filter[:, :, :, f_idx]

    tgt_cont_int_filter = K.eval(model.layers[2].kernel)
    tgt_cont_int_filter = tgt_cont_int_filter[f_idx, :, :]

    contour_test_image = generate_test_contour_image_from_fragment(fragment)

    f = plt.figure()
    ax = plt.subplot2grid((3, 8), (0, 0), colspan=2)
    display_filt = (tgt_conv1_filter - tgt_conv1_filter.min()) * 1 / (tgt_conv1_filter.max() - tgt_conv1_filter.min())
    ax.imshow(display_filt)
    ax.set_title('Conv 1 Filter @ idx %d' % f_idx)

    ax1 = plt.subplot2grid((3, 8), (0, 2), colspan=2)
    ax1.imshow(fragment)
    ax1.set_title('Contour Fragment')

    ax2 = plt.subplot2grid((3, 8), (0, 4), colspan=4, rowspan=3)
    ax2.imshow(contour_test_image)
    ax2.set_title('Test image')

    ax3 = plt.subplot2grid((3, 8), (1, 0), colspan=2)
    ax3.imshow(tgt_cont_int_filter, cmap='seismic')
    ax3.set_title('Cont. Int. Filter @ idx %d' % f_idx)

    # Show individual slices of the conv filter
    tgt_conv_filter_min = tgt_conv1_filter.min()
    tgt_conv_filter_max = tgt_conv1_filter.max()
    ax4 = plt.subplot2grid((3, 8), (2, 0))
    ax4.imshow(tgt_conv1_filter[:, :, 0], cmap='seismic', vmin=tgt_conv_filter_min, vmax=tgt_conv_filter_max)
    ax5 = plt.subplot2grid((3, 8), (2, 1))
    ax5.imshow(tgt_conv1_filter[:, :, 1], cmap='seismic', vmin=tgt_conv_filter_min, vmax=tgt_conv_filter_max)
    ax6 = plt.subplot2grid((3, 8), (2, 2))
    ax_for_cb = \
        ax6.imshow(tgt_conv1_filter[:, :, 2], cmap='seismic', vmin=tgt_conv_filter_min, vmax=tgt_conv_filter_max)
    f.colorbar(ax_for_cb, )

    alex_net_utils.plot_l1_and_l2_activations(contour_test_image, l1_act_cb, l2_act_cb, f_idx)


def generate_test_contour_image_from_fragment(fragment, overlap=4, img_dim=227):
    """
     Generates contours by spatially tiling fragments

    :param img_dim:  dimension of the generated image (square image generated)
    :param overlap:  Number of pixels that overlap between tiled fragments (stride of the convolutional layer)
    :param fragment:
    :return:
    """
    test_image = np.zeros((img_dim, img_dim, 3))

    # Single Point
    add_fragment_at_location(test_image, fragment, 10, 10)

    # VERTICAL: Overlapping fragments
    # # contour_2 = np.zeros(((conv_1_stride * (5 - 1) + 11), 11, 3))
    # # contour_2[:, (0, 3, 4, 5, 9, 10), :] = 1
    # # add_fragment_at_location(test_image, contour_2, 5, 20)
    add_fragment_at_location(test_image, fragment, 5, 20)
    add_fragment_at_location(test_image, fragment, 6, 20)
    add_fragment_at_location(test_image, fragment, 7, 20)
    add_fragment_at_location(test_image, fragment, 8, 20)
    add_fragment_at_location(test_image, fragment, 9, 20)
    test_image = np.clip(test_image, 0, 1)

    # VERTICAL: Non-overlapping fragments
    add_fragment_at_location(test_image, fragment, 16, 20)
    add_fragment_at_location(test_image, fragment, 20, 20)
    add_fragment_at_location(test_image, fragment, 24, 20)
    add_fragment_at_location(test_image, fragment, 28, 20)

    # HORIZONTAL: Overlapping fragments
    add_fragment_at_location(test_image, fragment, 5, 40)
    add_fragment_at_location(test_image, fragment, 5, 41)
    add_fragment_at_location(test_image, fragment, 5, 42)
    add_fragment_at_location(test_image, fragment, 5, 43)
    test_image = np.clip(test_image, 0, 1)

    # HORIZONTAL: Non-overlapping fragments
    add_fragment_at_location(test_image, fragment, 15, 40)
    add_fragment_at_location(test_image, fragment, 15, 44)
    add_fragment_at_location(test_image, fragment, 15, 48)
    add_fragment_at_location(test_image, fragment, 15, 52)

    # HORIZONTAL: Visually Non-overlapping fragments. Spatially adjacent with no gaps
    start_x = 25 * overlap  # Starting at location 25, 40
    start_y = 40 * overlap
    for ii in range(4):
        test_image[start_x: start_x + 11, start_y + ii * 11: start_y + (ii + 1) * 11, :] = fragment

    # DIAGONAL (backward slash): overlapping
    add_fragment_at_location(test_image, fragment, 30, 5)
    add_fragment_at_location(test_image, fragment, 31, 6)
    add_fragment_at_location(test_image, fragment, 32, 7)
    add_fragment_at_location(test_image, fragment, 33, 8)
    add_fragment_at_location(test_image, fragment, 34, 9)
    test_image = np.clip(test_image, 0, 1)

    # DIAGONAL (backward slash): Visually Non-overlapping fragments. Spatially adjacent with no gaps
    start_x = 40 * overlap
    start_y = 5 * overlap
    for ii in range(5):
        test_image[
            start_x + ii * 11: start_x + (ii + 1) * 11,
            start_y + ii * 11: start_y + (ii + 1) * 11,
            :
        ] = fragment

    # DIAGONAL (forward slash): Visually Non-overlapping fragments. Spatially adjacent with no gaps
    start_x = 51 * overlap
    start_y = 30 * overlap
    for ii in range(5):
        test_image[
            start_x - (ii * 11): start_x - (ii * 11) + 11,
            start_y + (ii * 11): start_y + (ii + 1) * 11,
            :
        ] = fragment

    return test_image


def add_fragment_at_location(image, fragment, loc_x, loc_y):
    """

    :param image:
    :param fragment:
    :param loc_x:
    :param loc_y:
    :return:
    """
    stride = 4  # The Stride using the in convolutional layer
    filt_dim_r = fragment.shape[0]
    filt_dim_c = fragment.shape[1]

    print("Fragment Placed at x: %d-%d, y: %d-%d"
          % (loc_x * stride, loc_x * stride + filt_dim_r, loc_y * stride, loc_y * stride + filt_dim_c))

    image[
        loc_x * stride: loc_x * stride + filt_dim_r,
        loc_y * stride: loc_y * stride + filt_dim_c,
        :,
    ] += fragment

    return image


def replace_fragment_at_location(image, fragment, loc_x, loc_y):
    """
    Similar to add_fragment_at_location, but instead of adding the fragment, replaces it

    :param image:
    :param fragment:
    :param loc_x:
    :param loc_y:
    :return:
    """
    stride = 4  # The Stride using the in convolutional layer
    filt_dim_r = fragment.shape[0]
    filt_dim_c = fragment.shape[1]

    print("Fragment Placed at x: %d-%d, y: %d-%d"
          % (loc_x * stride, loc_x * stride + filt_dim_r, loc_y * stride, loc_y * stride + filt_dim_c))

    image[
        loc_x * stride: loc_x * stride + filt_dim_r,
        loc_y * stride: loc_y * stride + filt_dim_c,
        :,
    ] = fragment

    return image


if __name__ == "__main__":
    plt.ion()
    K.clear_session()

    # 1. Build the model
    # ------------------
    K.set_image_dim_ordering('th')  # Model was originally defined with Theano backend.
    print("Building Contour Integration Model...")

    # # Additive Model
    # no_overlap_model = build_contour_integration_model(
    #     "Additive",
    #     "trained_models/AlexNet/alexnet_weights.h5",
    #     weights_type='enhance',
    #     n=7
    # )

    # Gaussian Multiplicative Model
    cont_int_model = build_contour_integration_model(
        "gaussian_multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        weights_type='enhance',
        n=25,
        sigma=6.0
    )

    # # Masked Multiplicative Model
    # no_overlap_model = build_contour_integration_model(
    #     "masked_multiplicative",
    #     "trained_models/AlexNet/alexnet_weights.h5",
    #     weights_type='enhance',
    #     n=25,
    # )

    # # # Multiplicative Model
    # no_overlap_model = build_contour_integration_model(
    #         "multiplicative",
    #         "trained_models/AlexNet/alexnet_weights.h5",
    #         n=25,
    # )

    # callbacks to get activations of L1 & L2. Defined once only to optimize memory usage and
    # prevent the underlying tensorflow graph from growing unnecessarily
    l1_activations_cb = alex_net_utils.get_activation_cb(cont_int_model, 1)
    l2_activations_cb = alex_net_utils.get_activation_cb(cont_int_model, 2)

    # 2. Display Conv 1 and Contour integration layer Kernels
    # -------------------------------------------------------
    weights_ch_last = cont_int_model.layers[1].weights[0]
    common_utils.display_filters(weights_ch_last)

    weights_ch_last = cont_int_model.layers[2].kernel
    common_utils.display_filters(weights_ch_last)

    # 3. Display the activations of a test image
    # ------------------------------------------
    # img = load_img("trained_models/data/sample_images/cat.7.jpg", target_size=(227, 227))
    img = load_img("trained_models/data/sample_images/zahra.jpg", target_size=(227, 227))
    plt.figure()
    plt.imshow(img)
    plt.title('Original Image')

    x = img_to_array(img)
    x = np.reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]])

    y_hat = cont_int_model.predict(x, batch_size=1, verbose=1)
    print("Prediction %s" % np.argmax(y_hat))

    common_utils.display_layer_activations(cont_int_model, 1, x)
    common_utils.display_layer_activations(cont_int_model, 2, x)

    # 4. Contour Enhancement Visualizations
    # ---------------------------------------------------------------------
    # Vertical Filter
    tgt_filt_idx = 10
    frag_1 = np.zeros((11, 11, 3))
    frag_1[:, (0, 3, 4, 5, 9, 10), :] = 1
    main(cont_int_model, frag_1, tgt_filt_idx, l1_activations_cb, l2_activations_cb)

    # Horizontal Filter
    tgt_filt_idx = 5
    frag_2 = np.zeros((11, 11, 3))
    frag_2[0:6, :, :] = 1
    main(cont_int_model, frag_2, tgt_filt_idx, l1_activations_cb, l2_activations_cb)

    # 5. Output of contour enhancement on real images
    # ----------------------------------------------------------------------
    test_real_img = load_img("trained_models/data/sample_images/zahra.jpg", target_size=(227, 227))
    # test_real_img = load_img("trained_models/data/sample_images/cat.7.jpg", target_size=(227, 227))

    tgt_filt_idx = 5
    alex_net_utils.plot_l1_and_l2_activations(test_real_img, l1_activations_cb, l2_activations_cb, tgt_filt_idx)

    tgt_filt_idx = 10
    alex_net_utils.plot_l1_and_l2_activations(test_real_img, l1_activations_cb, l2_activations_cb, tgt_filt_idx)
