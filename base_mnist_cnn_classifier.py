# -------------------------------------------------------------------------------------------------
# Base CNN based MNISt classifier. This is the model all contour integration models will be
# compared with.
#
# Ref: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
#
# Expected accuracy of 99.2%
#
# Author: Salman Khan
# Date  : 29/06/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import keras
from keras.models import Sequential, save_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

import numpy as np
import os

import utils

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12
IMG_ROWS, IMG_COL = 28, 28

# Set the random seed for reproducibility
np.random.seed(7)

FILENAME = os.path.basename(__file__).split('.')[0] + '.hf'


if __name__ == "__main__":

    # 1. Load the Data
    # ------------------------------------------------------------------------
    x_train, y_train, x_test, y_test, x_sample, y_sample = utils.get_mnist_data()

    # 2. Define the Model
    # ------------------------------------------------------------------------
    model = Sequential()

    # Conv Layer 1
    input_dims = (IMG_ROWS, IMG_COL, 1)  # for a single input
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_dims, padding='same'))
    # Output [IMG_ROWS x IMG_COL x 32]

    # Conv Layer 2
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    # Output [IMG_ROWS,IMG_Col, 64]

    # Pooling and regularization
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output [14,14, 64]
    model.add(Dropout(0.25))

    # Processing to connect with a Dense layer
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))

    # Dense Layer 2
    model.add(Dense(units=NUM_CLASSES, activation='softmax'))

    # 3. Configure Learning Process: Loss function, gradient method to use, and evaluation metrics
    # -------------------------------------------------------------------------
    # Before training a model, you need to configure the learning process, which is done via the compile method.
    # It receives three arguments:

    # LOSS: This is the objective that the model will try to minimize.

    # OPTIMIZER: THe type of methods use to update parameters of the model

    # METRICS = function that is used to judge the performance of your model.
    # It is similar to a loss function, but the results are NOT used to train the model

    model.compile(
        loss=keras.losses.categorical_crossentropy,  # Note this is not a function call.
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    # 4. Training the Model
    # ------------------------------------------------------------------------
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    # 5. Evaluate the Model
    # ------------------------------------------------------------------------
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test Accuracy', score[1])

    # 6. Save the model
    # ------------------------------------------------------------------------
    save_model(model, FILENAME)
