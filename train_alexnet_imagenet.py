# ------------------------------------------------------------------------------------------------
# Train the base Alexnet Model on Imagenet.
# Similar to the contour integration model, the first feature extracting layer is not trained
# ------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas as pd

import keras.backend as keras_backend
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, losses
from keras.callbacks import ModelCheckpoint
from keras.applications import imagenet_utils

import base_models.alex_net as alex_net_module


reload(alex_net_module)

# IMAGE_NET_TRAIN_DIR = '/media/salman/076d0e17-1483-4b67-ba60-aa8e7efc8edf/SalmanExternal/' \
#                      'ImageNet_ILSVRC2012/imagenet-data/train'
# IMAGE_NET_VALIDATION_DIR = '/media/salman/076d0e17-1483-4b67-ba60-aa8e7efc8edf/SalmanExternal/' \
#                      'ImageNet_ILSVRC2012/imagenet-data/validation'

IMAGE_NET_TRAIN_DIR = "./data/imagenet-data/train"
IMAGE_NET_VALIDATION_DIR = './data/imagenet-data/validation'

RESULTS_DIR = "./results/imagenet_alexnet_model"


def preprocessing_function(x):
    # return imagenet_utils.preprocess_input(x, mode='tf')
    x = (x - x.min()) / (x.max() - x.min())
    return x


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    plt.ion()
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')

    batch_size = 64
    n_epochs = 35  # Alexnet uses 90

    n_train_images = 1200000
    n_test_images = 500000

    # Immutable
    # -----------------------------------
    base_model_weights = \
        "trained_models/AlexNet/alexnet_weights.h5"

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    # -----------------------------------------------------------------------------------
    # build the model
    # -----------------------------------------------------------------------------------
    print('Building the Model')
    model = alex_net_module.alex_net(base_model_weights)

    # Set first feature extracting layer and contour integration layer as untrainable
    do_not_train = ['conv_1', 'contour_integration_layer']
    for layer in model.layers:
        if layer.name in do_not_train:
            layer.trainable = False

    # Compile the model
    # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

    model.compile(
        optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False),
        loss=losses.categorical_crossentropy,
        # loss=normalized_loss
        metrics=['accuracy']
    )

    model.summary()

    # # -----------------------------------------------------------------------------------
    # # Training
    # # -----------------------------------------------------------------------------------
    # print("Building Data Generators ...")
    #
    # train_datagen = ImageDataGenerator(
    #     data_format='channels_first',
    #     preprocessing_function=preprocessing_function
    # )
    # test_datagen = ImageDataGenerator(
    #     data_format='channels_first',
    #     preprocessing_function=preprocessing_function
    # )
    #
    # # train_datagen = CenteredImageDataGenerator(
    # #     data_format='channels_first', featurewise_center=True)
    # # test_datagen = CenteredImageDataGenerator(data_format='channels_first', featurewise_center=True)
    #
    # train_generator = train_datagen.flow_from_directory(
    #     IMAGE_NET_TRAIN_DIR,
    #     target_size=(227, 227),
    #     batch_size=batch_size,
    #     class_mode='categorical',
    #     shuffle=True,
    #     seed=42
    # )
    #
    # validation_generator = test_datagen.flow_from_directory(
    #     IMAGE_NET_VALIDATION_DIR,
    #     target_size=(227, 227),
    #     batch_size=batch_size,
    #     class_mode='categorical',
    #     shuffle=True,
    #     seed=42
    # )
    #
    # # Callback
    # TEMP_WEIGHT_STORE_FILE = os.path.join(RESULTS_DIR, 'trained_alexnet_weights_lowest_cost.hf')
    # checkpoint = ModelCheckpoint(
    #     TEMP_WEIGHT_STORE_FILE,
    #     monitor='val_loss',
    #     verbose=0,
    #     save_best_only=True,
    #     mode='min',
    #     save_weights_only=True,
    # )
    #
    # # -----------------------------------------------------------------------------------
    # print("Start Training ...")
    # start_time = datetime.now()
    #
    # history = model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=(n_train_images // batch_size),
    #     epochs=n_epochs,
    #     validation_data=validation_generator,
    #     validation_steps=50,  # todo : Fixme    n_test_images // batch_size
    #     verbose=1,
    #     use_multiprocessing=False,
    #     callbacks=[checkpoint]
    # )
    #
    # model.save_weights(os.path.join(RESULTS_DIR, "trained_alexnet_weights_final_cost.hf"))
    # print("Training took {}".format(datetime.now() - start_time))
    #
    # # Plot Loss
    # f, axis = plt.subplots()
    #
    # axis.plot(history.history['loss'], label='train_loss')
    # axis.plot(history.history['val_loss'], label='validation_loss')
    # axis.set_xlabel("Epoch")
    # axis.set_ylabel("Loss")
    # axis.legend()
    #
    # # -----------------------------------------------------------------------------------
    # # Test Predictions of the Model
    # # -----------------------------------------------------------------------------------
    # print("Testing Final performance")
    # test_generator = test_datagen.flow_from_directory(
    #     directory=IMAGE_NET_VALIDATION_DIR,
    #     target_size=(227, 227),
    #     batch_size=1,
    #     class_mode='categorical',
    #     shuffle=False,
    # )
    #
    # test_generator.reset()
    # pred = model.predict_generator(test_generator, verbose=1)
    #
    # predicted_class_indices = np.argmax(pred, axis=1)
    #
    # labels = train_generator.class_indices
    # labels = dict((v, k) for k, v in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]
    # # # Change labels to words
    # # readable_predictions = imagenet_utils.decode_predictions(pred)
    #
    # #
    # filenames = test_generator.filenames
    # y_true = []
    # is_same = np.zeros(len(filenames))
    #
    # for f_idx, filename in enumerate(filenames):
    #     label = os.path.dirname(filename)
    #     if '/' in label:
    #         label = label.strip('/')[1]
    #     y_true.append(label)
    #     is_same[f_idx] = (y_true[f_idx] == predictions[f_idx])
    #
    # # Write Data to File
    # results = pd.DataFrame(
    #     {"Filename": filenames,
    #      "y_true": y_true,
    #      "Predictions": predictions,
    #      "Are Equal": is_same
    #      }
    # )
    # results.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    #
    # print("Final prediction Accuracy {}".format(np.mean(is_same)))
