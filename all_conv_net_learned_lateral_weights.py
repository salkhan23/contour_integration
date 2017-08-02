# -------------------------------------------------------------------------------------------------
# All conv net model with contour integration after the first pooling replacement layer.
# For base mode see base_all_conv_net.py
# According to the ref, and as can be seen in the plots of base_all_conv_net.py, Gabors develop after the
# third convolution layer. (the max pooling replacement layer)
#
# Author: Salman Khan
# Date  : 13/07/17
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, GlobalAveragePooling2D, Activation
from keras.optimizers import SGD

import high_dim_kernel_visualization as hd_vis
import learned_lateral_weights
import utils
reload(hd_vis)
reload(learned_lateral_weights)
reload(utils)


NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 100

np.random.seed(7)  # Set the random seed for reproducibility
FILENAME = os.path.basename(__file__).split('.')[0] + '.hf'


if __name__ == "__main__":

    # 1. Import the Data
    # --------------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('Data Set Details: X train shape: %s. X test shape: %s'
          % (str(x_train.shape), str(x_train.shape)))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    sample_idx = 1
    x_sample = x_test[sample_idx].reshape(
        1, x_test[sample_idx].shape[0], x_test[sample_idx].shape[1], x_test[sample_idx].shape[2])
    y_sample = y_test[sample_idx]

    # 2. Build the Model
    # --------------------------------------------------------------------
    all_conv_model = Sequential()
    # all_conv_model.add(Dropout(0.2, input_shape=(None, None, 3)))
    # The input size needs to be specified if contour integration layers are used
    all_conv_model.add(Conv2D(96, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    all_conv_model.add(Conv2D(96, (3, 3), padding='same', activation='relu'))

    all_conv_model.add(Conv2D(96, (3, 3), padding='same', strides=(2, 2), activation='relu'))
    all_conv_model.add(learned_lateral_weights.ContourIntegrationLayer(n=3))
    all_conv_model.add(Activation('relu'))
    all_conv_model.add(Dropout(0.5))

    all_conv_model.add(Conv2D(192, (3, 3), padding='same', activation='relu'))
    all_conv_model.add(Conv2D(192, (3, 3), padding='same', activation='relu'))

    all_conv_model.add(Conv2D(192, (3, 3), padding='same', activation='relu', strides=(2, 2)))
    all_conv_model.add(Dropout(0.5))

    all_conv_model.add(Conv2D(192, (3, 3), padding='same', activation='relu'))
    all_conv_model.add(Conv2D(192, (1, 1), padding='valid', activation='relu'))
    all_conv_model.add(Conv2D(10, (1, 1), padding='valid', activation='relu'))

    all_conv_model.add(GlobalAveragePooling2D())
    all_conv_model.add(Activation('softmax'))
    all_conv_model.summary()

    # Compile the model
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    all_conv_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # 3. Train the Model
    # --------------------------------------------------------------------
    save_file = 'all_conv_net_model.hdf5'

    training_callbacks = []

    # Callback to save model as it trains
    checkpoint = keras.callbacks.ModelCheckpoint(
        save_file,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )
    training_callbacks.append(checkpoint)

    # # Callback to stop training early if loss is not falling
    # training_callbacks.append(keras.callbacks.EarlyStopping(patience=10))

    training_history = all_conv_model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=training_callbacks,
    )

    # 4. Evaluate the Model Accuracy
    # ----------------------------------------------------------------------------
    score = all_conv_model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss: %f' % score[0])
    print('Test Accuracy %f' % score[1])

    # 4. Some plots
    # ----------------------------------------------------------------------------
    plt.ion()
    utils.plot_train_summary(training_history)

    # Unfortunately this cant be used as specifying the input dimension,
    # means that no other size of the input can be used.
    # hd_vis.display_hd_filter_opt_stimuli(all_conv_model, 0, gen_img_row=3, gen_img_col=3, margin=1)
    # hd_vis.display_hd_filter_opt_stimuli(all_conv_model, 1, gen_img_row=5, gen_img_col=5, margin=1)
    # hd_vis.display_hd_filter_opt_stimuli(all_conv_model, 2, gen_img_row=9, gen_img_col=9, margin=1)

    utils.display_layer_activations(all_conv_model, 2, x_sample)
    utils.display_layer_activations(all_conv_model, 3, x_sample)
