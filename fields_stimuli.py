# -------------------------------------------------------------------------------------------------
#  Generate Curved Contours embedded in a sea of distractors. Image generation and tile size is
#  matched to the reference
#
#  "Field, Hayes & Hess - 1993 - Contour Integration by the Human Visual System: Evidence for a
#  local association field "
#
# Author: Salman Khan
# Date  : 25/04/18
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import gabor_fits
import curved_contour_image_generator
import alex_net_utils

reload(gabor_fits)
reload(curved_contour_image_generator)
reload(alex_net_utils)


if __name__ == '__main__':
    plt.ion()

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    image_size = np.array((512, 512, 3))

    fragment_size = np.array([16, 16, 3])

    full_tile_size = np.array([32, 32])

    contour_len = 13
    beta_rotation = 15

    inter_fragment_distance = full_tile_size[0]

    # ----------------------------------------------------------------------------------
    #  Fragment
    # ----------------------------------------------------------------------------------
    fragment_gabor_params = {
        'x0': 0,
        'y0': 0,
        'theta_deg': 90,
        'amp': 1,
        'sigma': 4,
        'lambda1': 8,
        'psi': 0,
        'gamma': 1
    }

    fragment = gabor_fits.get_gabor_fragment(fragment_gabor_params, fragment_size[0:2])

    plt.figure()
    plt.imshow(fragment)
    plt.title("Fragment")

    # Send background to mean value of the fragment
    # bg_value = np.mean(fragment, axis=(0, 1))
    # bg_value = [np.uint8(chan) for chan in bg_value]
    bg_value = curved_contour_image_generator.get_mean_pixel_value_at_boundary(fragment)

    test_image = np.ones(image_size, dtype=np.uint8) * bg_value

    # ----------------------------------------------------------------------------------
    #  Contour Path
    # ----------------------------------------------------------------------------------
    test_image, path_fragment_starts = curved_contour_image_generator.add_contour_path_constant_separation(
        test_image,
        fragment,
        fragment_gabor_params,
        contour_len,
        beta_rotation,
        full_tile_size[0],
    )

    # plt.figure()
    # plt.imshow(test_image)
    # plt.title("Contour")

    curved_contour_image_generator.plot_fragment_rotations(fragment, fragment_gabor_params)

    # ----------------------------------------------------------------------------------
    #  Background Fragments
    # ----------------------------------------------------------------------------------
    test_image, bg_tiles, bg_removed_tiles, bg_relocated_tiles = \
        curved_contour_image_generator.add_background_fragments(
            test_image,
            fragment,
            path_fragment_starts,
            full_tile_size,
            beta_rotation,
            fragment_gabor_params
        )

    plt.figure()
    plt.imshow(test_image)
    plt.title("Contour embedded in a sea of distractors")
