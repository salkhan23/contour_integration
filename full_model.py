# ------------------------------------------------------------------------------------------------
# The complete Alex-Net Contour Integration model
# ------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import pandas as pd

import keras.backend as keras_backend
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Concatenate, \
    Flatten, Dense, Dropout, Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, losses
from keras.callbacks import ModelCheckpoint
from keras.applications import imagenet_utils

from contour_integration_models.alex_net import model_3d as contour_integration_module
import base_models.alex_net as alex_net_module
import learn_cont_int_kernel_3d_model as learning_module

reload(contour_integration_module)
reload(alex_net_module)
reload(learning_module)

IMAGE_NET_TRAIN_DIR = '/media/salman/076d0e17-1483-4b67-ba60-aa8e7efc8edf/SalmanExternal/' \
                     'ImageNet_ILSVRC2012/imagenet-data/train'
IMAGE_NET_VALIDATION_DIR = '/media/salman/076d0e17-1483-4b67-ba60-aa8e7efc8edf/SalmanExternal/' \
                     'ImageNet_ILSVRC2012/imagenet-data/validation'

# IMAGE_NET_TRAIN_DIR = "./data/imagenet-data/train"
# IMAGE_NET_VALIDATION_DIR = './data/imagenet-data/validation'

RESULTS_DIR = "/results/full_model"


class CenteredImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            # x = ((x/255.) - 0.5) * 2.

            x[0, :, :] -= 123.68
            x[1, :, :] -= 116.779
            x[2, :, :] -= 103.939

        return x


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')

    # Input Parameters
    tgt_filt_idx = 0
    rf_size = 35
    inner_leaky_relu_alpha = 0.9
    outer_leaky_relu_alpha = 1.
    l1_reg_loss_weight = 0.0005

    batch_size = 64
    n_epochs = 25  # Alexnet uses 90

    n_train_images = 1300 * 3  # 1200000
    n_test_images = 50 * 3  # 500000

    # Immutable
    contour_integration_layer_weights = \
        '/home/salman/workspace/keras/my_projects/contour_integration/results/beta_rotations_upto30/trained_weights.hf'
    base_model_weights = \
        "trained_models/AlexNet/alexnet_weights.h5"

    # -----------------------------------------------------------------------------------
    # build the model
    # -----------------------------------------------------------------------------------
    print('Building the Model')

    input_layer = Input(shape=(3, 227, 227))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(input_layer)

    contour_integrate_layer = contour_integration_module.ContourIntegrationLayer3D(
        tgt_filt_idx=0,  # not important for full model
        rf_size=rf_size,
        inner_leaky_relu_alpha=inner_leaky_relu_alpha,
        outer_leaky_relu_alpha=outer_leaky_relu_alpha,
        l1_reg_loss_weight=l1_reg_loss_weight,
        name='contour_integration_layer')(conv_1)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(contour_integrate_layer)
    conv_2 = alex_net_module.crosschannelnormalization(name='Contrast_Normalization')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)

    conv_2_1 = Conv2D(128, (5, 5), activation='relu', name='conv_22_1') \
        (alex_net_module.splittensor(ratio_split=2, id_split=0)(conv_2))
    conv_2_2 = Conv2D(128, (5, 5), activation='relu', name='conv_22_2') \
        (alex_net_module.splittensor(ratio_split=2, id_split=1)(conv_2))
    conv_2 = Concatenate(axis=1, name='conv_22')([conv_2_1, conv_2_2])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = alex_net_module.crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_33')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4_1 = Conv2D(192, (3, 3), activation='relu', name='conv_44_1')\
        (alex_net_module.splittensor(ratio_split=2, id_split=0)(conv_4))
    conv_4_2 = Conv2D(192, (3, 3), activation='relu', name='conv_44_2')\
        (alex_net_module.splittensor(ratio_split=2, id_split=1)(conv_4))
    conv_4 = Concatenate(axis=1, name='conv_44')([conv_4_1, conv_4_2])

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5_1 = Conv2D(128, (3, 3), activation='relu', name='conv_55_1')\
        (alex_net_module.splittensor(ratio_split=2, id_split=0)(conv_5))
    conv_5_2 = Conv2D(128, (3, 3), activation='relu', name='conv_55_2')\
        (alex_net_module.splittensor(ratio_split=2, id_split=1)(conv_5))
    conv_5 = Concatenate(axis=1, name='conv_55')([conv_5_1, conv_5_2])

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_11')(dense_1)

    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_22')(dense_2)

    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(3, name='dense_33')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(inputs=input_layer, outputs=prediction)

    # -----------------------------------------------------------------------------------
    # Load weights of the base model and contour integration layer weights
    # -----------------------------------------------------------------------------------
    print('Loading Weights ...')

    model.load_weights(base_model_weights, by_name=True)
    model.load_weights(contour_integration_layer_weights, by_name=True)
    # TODO Verify the weights are loaded correctly.

    # Set first layer and the feature extracting layer as trainable
    do_not_train = ['conv_1', 'contour_integration_layer_weights']
    for layer in model.layers:
        if layer.name in do_not_train:
            layer.trainable = False

    # TODO should the rest of the weights be initialized to zero

    model.summary()
    # TODO: weights seam less, verify same as alexnet without my mods

    # Compile the model
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

    # optimizer_fcn = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False),
    # loss_fcn = losses.mean_squared_error
    # model.compile(optimizer=optimizer_fcn, loss=loss_fcn, metrics=['accuracy'])
    #
    # model.compile(
    #     optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False),
    #     loss=losses.mean_squared_error,
    #     # loss=normalized_loss
    #     metrics=['accuracy']
    # )

    # Clear all untrained kernels
    # ----------------------------
    trained_contour_integration_kernels = [
        5, 10, 19, 20, 21, 22, 48, 49, 51, 59,
        60, 62, 64, 65, 66, 68, 69, 72, 73, 74,
        76, 77, 79, 80, 82
    ]

    learning_module.clear_unlearnt_contour_integration_kernels(model, trained_contour_integration_kernels)

    # for k_idx in np.arange(96):
    #     learning_module.plot_start_n_learnt_contour_integration_kernels(
    #         model,
    #         k_idx,
    #         None,
    #     )
    #     raw_input()

    # -----------------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------------
    print("Building Data Generators ...")

    train_datagen = ImageDataGenerator(
        data_format='channels_first',
        preprocessing_function=imagenet_utils.preprocess_input
    )
    test_datagen = ImageDataGenerator(
        data_format='channels_first',
        preprocessing_function=imagenet_utils.preprocess_input
    )

    # train_datagen = CenteredImageDataGenerator(
    #     data_format='channels_first', featurewise_center=True)
    # test_datagen = CenteredImageDataGenerator(data_format='channels_first', featurewise_center=True)

    train_generator = train_datagen.flow_from_directory(
        IMAGE_NET_TRAIN_DIR,
        target_size=(227, 227),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    validation_generator = test_datagen.flow_from_directory(
        IMAGE_NET_VALIDATION_DIR,
        target_size=(227, 227),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    # Callback
    TEMP_WEIGHT_STORE_FILE = os.path.join(RESULTS_DIR, 'trained_alexnet_weights_lowest_cost.hf')
    checkpoint = ModelCheckpoint(
        TEMP_WEIGHT_STORE_FILE,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        save_weights_only=True,
    )

    print("Start Training ...")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=(n_train_images // batch_size),
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=50,  # todo : Fixme    n_test_images // batch_size
        verbose=1,
        use_multiprocessing=False,
        callbacks=[checkpoint]
    )

    model.save_weights(os.path.join(RESULTS_DIR, "trained_alexnet_weights_final_cost.hf"))

    # Plot Loss
    f, axis = plt.subplots()

    axis.plot(history.history['loss'], label='train_loss_{0}'.format(tgt_filt_idx))
    axis.plot(history.history['val_loss'], label='validation_loss_{0}'.format(tgt_filt_idx))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()

    # -----------------------------------------------------------------------------------
    # Test Predictions of the Model
    # -----------------------------------------------------------------------------------
    print("Testing Final performance")
    test_generator = test_datagen.flow_from_directory(
        directory=IMAGE_NET_VALIDATION_DIR,
        target_size=(227, 227),
        batch_size=1,
        class_mode='categorical',
        shuffle=False,
    )

    test_generator.reset()
    pred = model.predict_generator(test_generator, verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)

    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    # # Change labels to words
    # readable_predictions = imagenet_utils.decode_predictions(pred)

    #
    filenames = test_generator.filenames
    y_true = []
    is_same = np.zeros(len(filenames))

    for f_idx, filename in enumerate(filenames):
        label = os.path.dirname(filename)
        if '/' in label:
            label = label.strip('/')[1]
        y_true.append(label)
        is_same[f_idx] = (y_true[f_idx] == predictions[f_idx])

    # Write Data to File
    results = pd.DataFrame(
        {"Filename": filenames,
         "y_true": y_true,
         "Predictions": predictions,
         "Are Equal": is_same
         }
    )
    results.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)

    print("Final prediction Accuracy {}".format(np.mean(is_same)))
