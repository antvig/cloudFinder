from src.image.features import _patch_image, _patch_center, _expend_patch_stat

import numpy as np

from numpy.testing import assert_array_equal

from skimage.morphology import square, rectangle

PATCH_SQUARE_2 = np.array(
    [
        [[[0, 1], [5, 6]], [[1, 2], [6, 7]], [[2, 3], [7, 8]], [[3, 4], [8, 9]]],
        [
            [[5, 6], [10, 11]],
            [[6, 7], [11, 12]],
            [[7, 8], [12, 13]],
            [[8, 9], [13, 14]],
        ],
        [
            [[10, 11], [15, 16]],
            [[11, 12], [16, 17]],
            [[12, 13], [17, 18]],
            [[13, 14], [18, 19]],
        ],
        [
            [[15, 16], [20, 21]],
            [[16, 17], [21, 22]],
            [[17, 18], [22, 23]],
            [[18, 19], [23, 24]],
        ],
    ]
)

PATCH_SQUARE_3 = np.array(
    [
        [
            [[0, 1, 2], [5, 6, 7], [10, 11, 12]],
            [[1, 2, 3], [6, 7, 8], [11, 12, 13]],
            [[2, 3, 4], [7, 8, 9], [12, 13, 14]],
        ],
        [
            [[5, 6, 7], [10, 11, 12], [15, 16, 17]],
            [[6, 7, 8], [11, 12, 13], [16, 17, 18]],
            [[7, 8, 9], [12, 13, 14], [17, 18, 19]],
        ],
        [
            [[10, 11, 12], [15, 16, 17], [20, 21, 22]],
            [[11, 12, 13], [16, 17, 18], [21, 22, 23]],
            [[12, 13, 14], [17, 18, 19], [22, 23, 24]],
        ],
    ]
)


IMAGE = np.arange(5 * 5).reshape(5, 5)


def test_patch_image():

    selem = square(2)

    patch = _patch_image(IMAGE, selem)

    assert_array_equal(patch, PATCH_SQUARE_2)


def test_patch_image_2():

    selem = square(3)

    patch = _patch_image(IMAGE, selem)

    assert_array_equal(patch, PATCH_SQUARE_3)


def test_patch_center():

    patch_center = _patch_center(PATCH_SQUARE_2)

    assert_array_equal(
        patch_center[:, :, 0, 0],
        np.array([[6, 7, 8, 9], [11, 12, 13, 14], [16, 17, 18, 19], [21, 22, 23, 24]]),
    )


def test_patch_center_2():

    patch_center = _patch_center(PATCH_SQUARE_3)

    assert_array_equal(
        patch_center[:, :, 0, 0], np.array([[6, 7, 8], [11, 12, 13], [16, 17, 18]]),
    )


def test_expend_patch_stat():

    A = np.arange(3 * 3).reshape(3, 3)
    selem = rectangle(2, 3)

    A_expend = _expend_patch_stat(A, selem)

    A_expend_expected = np.array(
        [[0, 0, 1, 2, 2], [0, 0, 1, 2, 2], [3, 3, 4, 5, 5], [6, 6, 7, 8, 8]]
    )

    assert_array_equal(A_expend, A_expend_expected)
