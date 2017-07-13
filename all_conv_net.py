# -------------------------------------------------------------------------------------------------
# Implementation of the the All-CNN variant of Model C of
# [Springenberg et. al. - 2015 - Striving for Simplicity: The All Convolutional Net]
# on CIFAR-10
#
# Without Data Augmentation this achieves around 90% accuracy.
#
# Author: Salman Khan
# Date  : 13/07/17
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, GlobalAveragePooling2D, Activation
from keras.optimizers import SGD

from high_dim_kernel_visualization import display_hd_filter_opt_stimuli


NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 2

def plot_train_summary(training_summary):
    """

    :param training_summary: Return value of model.fit()
    :return:
    """
    f = plt.figure()

    f. add_subplot(1, 2, 1)
    plt.plt(range(len(training_summary.history['acc'])), training_summary.history['acc'], color='blue')
    plt.plt(range(len(training_summary.history['val_acc'])), training_summary.history['val_acc'], color='red')
    plt.title('Model Accuracy')
    plt.xlabel("Acc")
    plt.ylabel('Epoch')
    plt.legend(loc='best')

    f.add_subplot(1, 2, 2)
    plt.plt(range(len(training_summary.history['loss'])), training_summary.history['loss'], color='blue')
    plt.plt(range(len(training_summary.history['val_loss'])), training_summary.history['val_loss'], color='red')
    plt.title('Model Accuracy')
    plt.xlabel("Acc")
    plt.ylabel('Epoch')


if __name__ == "__main__":

    # 1. Import the Data
    # --------------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('Data Set Details: X train shape: %s. X test shape: %s' % (str(x_train.shape), str(x_train.shape)))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # 2. Build the Model
    # --------------------------------------------------------------------
    all_conv_model = Sequential()
    all_conv_model.add(Dropout(0.2, input_shape=(None, None, 3)))
    all_conv_model.add(Conv2D(96, (3, 3), padding='same', activation='relu'))
    all_conv_model.add(Conv2D(96, (3, 3), padding='same', activation='relu'))

    all_conv_model.add(Conv2D(96, (3, 3), padding='same', activation='relu', strides=(2, 2)))  # pooling layer
    all_conv_model.add(Dropout(0.5))

    all_conv_model.add(Conv2D(192, (3, 3), padding='same', activation='relu'))
    all_conv_model.add(Conv2D(192, (3, 3), padding='same', activation='relu'))

    all_conv_model.add(Conv2D(192, (3, 3), padding='same', activation='relu', strides=(2, 2)))  # pooling layer
    all_conv_model.add(Dropout(0.5))

    all_conv_model.add(Conv2D(192, (3, 3), padding='same', activation='relu'))
    all_conv_model.add(Conv2D(192, (1, 1), padding='same', activation='relu'))
    all_conv_model.add(Conv2D(10, (1, 1), padding='same', activation='relu'))

    all_conv_model.add(GlobalAveragePooling2D())
    all_conv_model.add(Activation('softmax'))
    all_conv_model.summary()

    # Compile the model
    opt = SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True)
    all_conv_model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # 3. Train the Model
    # --------------------------------------------------------------------
    save_file = 'all_conv_net_model.hdf5'

    training_callbacks = []
    checkpoint = keras.callbacks.ModelCheckpoint(
        save_file,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )
    training_callbacks.append(checkpoint)

    training_summary = all_conv_model.fit(
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
    print('Test Loss:', score[0])
    print('Test Accuracy', score[1])

    # 4. Some plots
    # ----------------------------------------------------------------------------
    plt.ion()
    plot_train_summary()

    display_hd_filter_opt_stimuli(all_conv_model, 1)
