# -------------------------------------------------------------------------------------------------
#  Contour Integration model that can use generated contour images
#
# Author: Salman Khan
# Date  : 01/02/18
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras.backend as K
from keras.layers import Input, Conv2D
from keras.engine.topology import Layer
from keras.models import Model

import alex_net_cont_int_models as cont_int_models
import alex_net_utils
import gabor_fits

reload(cont_int_models)
reload(gabor_fits)
reload(alex_net_utils)


class ContourEnhancementGainCalculator(Layer):
    def __init__(self, tgt_kernel_idx, tgt_neuron_loc=(27, 27), **kwargs):
        """
        Calculates the Enhancement gain of the specified neuron at the specified location

        :param tgt_kernel_idx:
        :param tgt_neuron_loc:
        :param kwargs:
        """
        self.tgt_loc = tgt_neuron_loc
        self.tgt_kernel_idx = tgt_kernel_idx

        super(ContourEnhancementGainCalculator, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ContourEnhancementGainCalculator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        feature_activation = inputs[0]
        contour_activation = inputs[1]

        tgt_neuron_gain = \
            feature_activation[:, self.tgt_kernel_idx, self.tgt_loc[0], self.tgt_loc[1]] / \
            (contour_activation[:, self.tgt_kernel_idx, self.tgt_loc[0], self.tgt_loc[1]] + 1e-8)

        return tgt_neuron_gain

    def compute_output_shape(self, input_shape):
        return tuple((input_shape[0][0], 1))


class ContourImageGenerator(object):

    def __init__(self, tgt_filter, tgt_filter_idx):
        """
        Generator contour images for the specified target filter
        :param tgt_filter:
        """
        self.tgt_filter_idx = tgt_filter_idx
        self.tgt_filter = tgt_filter

        self.frag = self.__get_fragment_from_kernel()
        self.frag_len = self.frag.shape[0]

        self.image_len = 227  # size of inputs expected by Alexnet

        self.orientation, self.row_offset = gabor_fits.get_l1_filter_orientation_and_offset(
            self.tgt_filter,
            self.tgt_filter_idx,
            show_plots=False
        )

        # Plot Neurophysiological data Data from [Li-2006]
        with open('.//neuro_data//Li2006.pickle', 'rb') as handle:
            data = pickle.load(handle)

        # Rather than calculate the contour fragment and background tiles locations everytime
        # Calculate them once and store them to be index later

        # Contour Fragment locations for variable contour length
        self.c_lengths = data["contour_len_avg_len"]
        self.cont_tile_loc_no_spacing = []

        for c_len in self.c_lengths:
            tile_locations = alex_net_utils.diagonal_contour_generator(
                frag_len=self.frag_len,
                row_offset=self.row_offset,
                bw_tile_spacing=0,
                cont_len=c_len,
                cont_start_loc=27*4  # (27,27) central location in the output of the feature
                                     # extracting layer. 4 = stride length
            )

            self.cont_tile_loc_no_spacing.append(tile_locations)

        # Contour fragment anf background spacing for variable spacing
        self.frag_spacing = data['contour_separation_avg_rcd']
        self.frag_spacing = np.floor(self.frag_spacing * self.frag_len) - self.frag_len

        self.bg_tile_loc = []
        self.cont_tile_loc_var_spacing = []

        for c_spacing in self.frag_spacing:
            c_spacing = int(c_spacing)

            bg_tile_locations = alex_net_utils.get_background_tiles_locations(
                frag_len=self.frag_len,
                img_len=self.image_len,
                row_offset=self.row_offset,
                space_bw_tiles=c_spacing,
                tgt_n_visual_rf_start=108
            )

            cont_tile_spacing = alex_net_utils.diagonal_contour_generator(
                frag_len=self.frag_len,
                row_offset=self.row_offset,
                bw_tile_spacing=c_spacing,
                cont_len=7,
                cont_start_loc=27*4
            )

            self.bg_tile_loc.append(bg_tile_locations)
            self.cont_tile_loc_var_spacing.append(cont_tile_spacing)

        # Expected gains
        self.expected_gains_lengths = data["contour_len_avg_gain"]
        self.expected_gains_spacing = data["contour_separation_avg_gain"]

    def __get_fragment_from_kernel(self):
        frag = np.copy(self.tgt_filter)
        frag = frag.sum(axis=2)  # collapse all channels
        frag[frag > 0] = 1
        frag[frag <= 0] = 0
        frag *= 255
        frag = np.repeat(frag[:, :, np.newaxis], 3, axis=2)
        return frag

    @staticmethod
    def show_image(img, axis=None):
        """
        Generated images are in the format expected as inputs by the model. The model was
        first designed with a theano backend, and expects channel first. This function reverses
        the transformation, so that the image can be displayed in the format expected by matplotlib

        :param img:
        :param axis:
        :return:
        """
        display_img = np.transpose(img, (1, 2, 0))
        if axis is None:
            _, axis = plt.subplots()

        axis.imshow(display_img)

    def generate(self, batch_size=10, images_type='both'):
        """
        This is the main function of the class, it is a generator that returns the specified
        number of images.

        :param batch_size:
        :param images_type:
        :return:
        """
        images_type = images_type.lower()

        allowed_types = ['both', 'spacing', 'length']

        while 1:

            images = []
            labels = []

            if images_type.lower() not in allowed_types:
                raise Exception("Specified image type is not allowed")

            if images_type == 'both':
                n_c_len_images = batch_size >> 1
                n_c_spacing_images = batch_size - n_c_len_images
            elif images_type == 'length':
                n_c_len_images = batch_size
                n_c_spacing_images = 0
            else:
                n_c_len_images = 0
                n_c_spacing_images = batch_size

            # print("Number of variable length contour images %d, number of variable spacing images %d"
            #       % (n_c_len_images, n_c_spacing_images))

            # Generate the contour length images
            for img_idx in range(n_c_len_images):

                select_idx = np.random.randint(0, high=len(self.c_lengths))

                img = np.zeros((self.image_len, self.image_len, 3))

                # Background Tiles
                img = alex_net_utils.tile_image(
                    img=img,
                    frag=self.frag,
                    insert_locs=self.bg_tile_loc[0],
                    rotate=True,
                    gaussian_smoothing=True
                )

                # Contour Tiles
                img = alex_net_utils.tile_image(
                    img=img,
                    frag=self.frag,
                    insert_locs=self.cont_tile_loc_no_spacing[select_idx],
                    rotate=False,
                    gaussian_smoothing=True
                )

                img = img / 255.0
                img = np.transpose(img, (2, 0, 1))

                images.append(img)
                labels.append(self.expected_gains_lengths[select_idx])

            # Generate the contour spacing images
            for img_idx in range(n_c_spacing_images):

                select_idx = np.random.randint(0, high=len(self.frag_spacing))

                img = np.zeros((self.image_len, self.image_len, 3))

                # Background Tiles
                img = alex_net_utils.tile_image(
                    img=img,
                    frag=self.frag,
                    insert_locs=self.bg_tile_loc[select_idx],
                    rotate=True,
                    gaussian_smoothing=True
                )

                # Contour Tiles
                img = alex_net_utils.tile_image(
                    img=img,
                    frag=self.frag,
                    insert_locs=self.cont_tile_loc_var_spacing[select_idx],
                    rotate=False,
                    gaussian_smoothing=True
                )

                img = img / 255.0
                img = np.transpose(img, (2, 0, 1))

                images.append(img)
                labels.append(self.expected_gains_spacing[select_idx])

            images = np.stack(images, axis=0)
            labels = np.stack(labels, axis=0)

            yield images, labels


if __name__ == "__main__":
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # ---------------------------------------
    tgt_feature_extract_kernel_idx = 54

    # 1. Build the contour Integration Model
    # ---------------------------------------
    input_layer = Input(shape=(3, 227, 227))
    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(input_layer)
    cont_int_layer = cont_int_models.MultiplicativeContourIntegrationLayer(name='cont_int', n=25,)(conv_1)
    cont_gain_calc_layer = \
        ContourEnhancementGainCalculator(tgt_kernel_idx=tgt_feature_extract_kernel_idx)([conv_1, cont_int_layer])

    model = Model(input_layer, outputs=cont_gain_calc_layer)

    model.load_weights("trained_models/AlexNet/alexnet_weights.h5", by_name=True)
    model.layers[1].trainable = False

    # 2. Build the contour Image Generator
    # -------------------------------------
    feature_extract_kernels = K.eval(model.layers[1].weights[0])
    tgt_feature_extract_kernel = feature_extract_kernels[:, :, :, tgt_feature_extract_kernel_idx]

    test_image_generator = ContourImageGenerator(tgt_feature_extract_kernel, tgt_feature_extract_kernel_idx)

    # # Test the generator
    # s = test_image_generator.generate(batch_size=1, images_type='length')
    #
    # generated_images, generated_labels = s.next()
    # for i_idx in range(generated_images.shape[0]):
    #     ContourImageGenerator.show_image(generated_images[i_idx, :, :, :])
    #     plt.title("Gain = %4f" % generated_labels[i_idx])

    # 3. Train the Model
    # -------------------
    model.compile(optimizer='adam', loss='mse')

    history = model.fit_generator(
        generator=test_image_generator.generate(batch_size=10, images_type='both'),
        steps_per_epoch=1,
        epochs=2000,
        # validation_data=train_image_generator,
        # validation_steps=100,
    )

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

