# -------------------------------------------------------------------------------------------------
#  Evaluate the performance of the models
#
# Author: Salman Khan
# Date  : 04/07/17
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

from keras.models import load_model

from contour_integration_models.mnist_cnn_classifer import learned_lateral_weights
import utils
reload(utils)  # Force Reload utils to pick up latest changes
reload(learned_lateral_weights)

if __name__ == "__main__":

    plt.ion()

    # 1. Get the Data
    # --------------------------------------------
    # Original
    x_train, y_train, x_test, y_test, x_sample, y_sample = utils.get_mnist_data()

    # Noisy
    # x_noisy_test = utils.add_noise(x_test, 'gaussian', var=0.1, mean=10)
    x_noisy_test = utils.add_noise(x_test, 'pepper', prob=0.5)

    # Plot the original image and the noisy image
    image = x_test[0, :, :, :]
    noisy_image = x_noisy_test[0, :, :, :]

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='Greys')
    plt.title('Original Image')
    f.add_subplot(1, 2, 2)
    plt.imshow(noisy_image[:, :, 0], cmap='Greys')
    plt.title('Noisy Image')

    # 2. Load Models
    # -------------------------------------------
    base_model = load_model("base_mnist_cnn_classifier.hf")

    contour_integration_model = load_model(
        "learned_lateral_weights_3x3_overlap.hf",
        custom_objects={'ContourIntegrationLayer': learned_lateral_weights.ContourIntegrationLayer()}
    )
    # contour_integration_model.summary()

    # 3. Evaluate Performances
    # -------------------------------------------
    # Base Model
    score = base_model.evaluate(x_test, y_test, verbose=1)
    print('\nBase Model: Test Loss (Original Data): %f' % score[0])
    print('Base Model: Test Accuracy (Original Data): %f' % score[1])

    score = contour_integration_model.evaluate(x_noisy_test, y_test, verbose=1)
    print('\nBase Model: Test Loss (Noisy Data): %f' % score[0])
    print('Base Model: Test Accuracy (Noisy Data): %f' % score[1])

    # Contour Integration Model
    score = contour_integration_model.evaluate(x_test, y_test, verbose=1)
    print('\nContour Integration Model: Test Loss (Original): %f' % score[0])
    print('Contour Integration Model: Test Accuracy (Original): %f' % score[1])

    score = contour_integration_model.evaluate(x_noisy_test, y_test, verbose=1)
    print('\nContour Integration Model: Test Loss (Noisy): %f' % score[0])
    print('Contour Integration Model: Test Accuracy (Noisy): %f' % score[1])
