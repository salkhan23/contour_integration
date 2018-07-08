# -------------------------------------------------------------------------------------------------
#  Sear
#
# Author: Salman Khan
# Date  : 03/09/17
# -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

import keras.backend as keras_backend
import gc

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


import contour_integration_models.alex_net.model_3d as contour_integration_model_3d
import multi_contour_integration_kernel_training as multi_training
import image_generator_curve
import learn_cont_int_kernel_3d_model

reload(contour_integration_model_3d)
reload(multi_training)
reload(image_generator_curve)
reload(learn_cont_int_kernel_3d_model)


TGT_KERNEL_INDEX = 5
DATA_DIR = './data/curved_contours/filt_matched_frag'


train_data_dict, test_data_dict = multi_training.get_train_n_test_data_dictionaries(
    TGT_KERNEL_INDEX,
    DATA_DIR,
)

train_set = train_data_dict  # Use all the data to train the model

active_train_set = {}
active_test_set = {}
for set_id in train_set:
    active_train_set.update(train_data_dict[set_id])
    active_test_set.update(test_data_dict[set_id])

train_image_generator = image_generator_curve.DataGenerator(
    active_train_set,
    batch_size=32,
    shuffle=True,
)

test_image_generator = image_generator_curve.DataGenerator(
    active_test_set,
    batch_size=32,
    shuffle=True,
)

train_gen_out = iter(train_image_generator)
test_gen_out = iter(test_image_generator)


def f_nn(params):

    print("Exploring Param set {}".format(params))

    if 'l1_reg_loss_weight' in params:
        l1_loss = params['l1_reg_loss_weight']
    else:
        l1_loss = 0.001

    if 'inner_leaky_relu_alpha' in params:
        inner_leaky_relu_alpha = params['inner_leaky_relu_alpha']
    else:
        inner_leaky_relu_alpha = 0.7

    if 'outer_leaky_relu_alpha' in params:
        outer_leaky_relu_alpha = params['outer_leaky_relu_alpha']
    else:
        outer_leaky_relu_alpha = 0.7

    if 'rf_size' in params:
        rf_size = params['rf_size']
    else:
        rf_size = 25

    if 'n_epochs' in params:
        n_epochs = params['n_epochs']
    else:
        n_epochs = 100

    model = contour_integration_model_3d.build_contour_integration_model(
        TGT_KERNEL_INDEX,
        rf_size=rf_size,
        inner_leaky_relu_alpha=inner_leaky_relu_alpha,
        outer_leaky_relu_alpha=outer_leaky_relu_alpha,
        l1_reg_loss_weight=l1_loss,
    )

    model.compile(loss='mse', optimizer='Adam')

    history = model.fit_generator(
        generator=train_image_generator,
        epochs=100,
        steps_per_epoch=n_epochs,
        verbose=0,
        validation_data=test_image_generator,
        validation_steps=10,
        # max_q_size=1,
        workers=8,
    )

    keras_backend.clear_session()
    gc.collect()

    print("Min Loss {}".format(min(history.history['val_loss'])))

    return {'loss': min(history.history['val_loss']), 'status': STATUS_OK}


if __name__ == '__main__':

    plt.ion()
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')

    space = {
        'inner_leaky_relu_alpha': hp.uniform('inner_leaky_relu_alpha', 0.0, 1.0),
        'outer_leaky_relu_alpha': hp.uniform('outer_leaky_relu_alpha', 0.0, 1.0),
        'l1_reg_loss_weight': hp.uniform('l1_reg_loss_weight', 0.0, 0.5),
        # 'n_epochs': hp.randint('n_epochs', 2000),
        # 'rf_size': hp.randint('rf_size', 30),
    }

    trials = Trials()
    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)

    print("Best {}".format(best))

    # -----------------------------------------------------------------------------------
    #  Plot Results
    # -----------------------------------------------------------------------------------
    # 2. Impact of individual variables on Loss
    # ------------------------------------------
    vars_list = trials.vals.keys()
    for var in vars_list:
        values = np.array(trials.vals[var])
        losses = np.array(trials.losses())

        sorted_idxs = np.argsort(values)
        sorted_idxs = [int(x) for x in sorted_idxs]

        plt.figure()
        plt.plot(values[sorted_idxs], losses[sorted_idxs])
        plt.xlabel(var)
        plt.ylabel('loss')
        plt.title("Target Filter index {}".format(TGT_KERNEL_INDEX))
