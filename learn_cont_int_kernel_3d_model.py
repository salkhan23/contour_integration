# -------------------------------------------------------------------------------------------------
#  Learn contour enhancement kernels for the 3d model contour integration model
#
# Author: Salman Khan
# Date  : 06/05/18
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pickle
from contour_integration_models.alex_net import model_3d as contour_integration_model
import os

import keras.backend as K
from keras.models import save_model

import image_generator_curve

reload(image_generator_curve)

DATA_DIR = 'data/curved_contours'

if __name__ == '__main__':
    plt.ion()
    K.clear_session()
    K.set_image_dim_ordering('th')

    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    tgt_filter_idx = 10

    image_size = (227, 227, 3)

    batch_size = 10

    # -----------------------------------------------------------------------------------
    # Contour Integration Model
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model.build_contour_integration_model(5)

    # -----------------------------------------------------------------------------------
    # Online Image Generator for training
    # -----------------------------------------------------------------------------------
    train_data_key_loc = './data/curved_contours/filter_{}/data_key.pickle'.format(tgt_filter_idx)

    with open(train_data_key_loc, 'rb') as handle:
        train_data_dict = pickle.load(handle)

    train_image_generator = image_generator_curve.DataGenerator(
        train_data_dict,
        batch_size=batch_size,
        img_size=image_size,
        shuffle=False,
        data_dir=DATA_DIR
    )

    # # Test the generator (sequence) object
    # gen_out = iter(train_image_generator)
    # X, y = gen_out.next()
    #
    # plt.figure()
    # plt.imshow(np.transpose(X[0, ], (1, 2, 0)))

    # -----------------------------------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------------------------------
    history = cont_int_model.fit_generator(
        generator=train_image_generator,
        epochs=10,
        verbose=2,
        # max_q_size=1,
        # workers=1,
    )

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    model_file_name = os.path.join(DATA_DIR, "filter_{}".format(tgt_filter_idx), 'trained_model.hf')
    save_model(cont_int_model, model_file_name)
