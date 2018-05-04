# -------------------------------------------------------------------------------------------------
#  Visualize model architecture constructed in keras.
#
# Author: Salman Khan
# Date  : 13/11/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.utils import plot_model

from contour_integration_models import alex_net_cont_int_models as cont_int_models
import alex_net_utils

reload(cont_int_models)
reload(alex_net_utils)

np.random.seed(7)  # Set the random seed for reproducibility

if __name__ == "__main__":
    plt.ion()
    K.clear_session()

    # ----------------------------------------------------------------
    # Build the model
    # ----------------------------------------------------------------

    # Gaussian Multiplicative Model
    # Multiplicative Model
    contour_integration_model = cont_int_models.build_contour_integration_model(
        "multiplicative",
        "trained_models/AlexNet/alexnet_weights.h5",
        n=25,
        activation='relu'
    )

    plot_model(contour_integration_model, to_file='model.png', show_shapes=True)



