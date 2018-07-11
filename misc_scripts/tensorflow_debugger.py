# -------------------------------------------------------------------------------------------------
#  Use the Tensorboard Debugger to look at Gradient Flow @ each training step, amongst other things
#
# Author: Salman Khan
# Date  : 06/05/18
# -------------------------------------------------------------------------------------------------
import keras.backend as keras_backend
from tensorflow.python import debug as tf_debug

import alex_net_utils
import contour_integration_models.alex_net.model_3d as contour_integration_model_3d
import image_generator_curve
import learn_cont_int_kernel_3d_model_linear_contours as linear_contour_training
import learn_cont_int_kernel_3d_model
import field_1993_routines


reload(alex_net_utils)
reload(contour_integration_model_3d)
reload(image_generator_curve)
reload(linear_contour_training)
reload(field_1993_routines)
reload(learn_cont_int_kernel_3d_model)


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    keras_backend.clear_session()
    keras_backend.set_image_dim_ordering('th')

    target_kernel_idx = 5
    batch_size = 32
    num_epochs = 20

    data_directory = './data/curved_contours/filt_matched_frag'

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    cont_int_model = contour_integration_model_3d.build_contour_integration_model(
        tgt_filt_idx=0,
        rf_size=25,
        inner_leaky_relu_alpha=0.7,
        outer_leaky_relu_alpha=0.94,
        l1_reg_loss_weight=0.01
    )

    # Wrapper for tensorboard debugger
    # Does not start the program but enters the debugger.
    # Reference: https://www.tensorflow.org/guide/debugger
    sess = keras_backend.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    keras_backend.set_session(sess)

    # -------------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------------
    learn_cont_int_kernel_3d_model.train_contour_integration_kernel(
        model=cont_int_model,
        tgt_filt_idx=target_kernel_idx,
        data_dir=data_directory,
        b_size=batch_size,
        n_epochs=num_epochs,
        steps_per_epoch=10,
    )
