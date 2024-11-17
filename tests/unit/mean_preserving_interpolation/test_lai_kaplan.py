"""
Tests of our Lai-Kaplan algorithm implementation and related tools
"""

from __future__ import annotations

import numpy as np
import pytest

from local.mean_preserving_interpolation.lai_kaplan import LaiKaplanArray

lai_kaplan_array_n_elements = pytest.mark.parametrize("n", [3, 4, 5, 10, 15])


@lai_kaplan_array_n_elements
def test_lai_kaplan_array_x_like(n):
    min = 1
    max = n + 1
    stride = 1.0

    n_elements = int((max - min) / stride) + 1

    raw = np.arange(n_elements)

    lkarr = LaiKaplanArray(lai_kaplan_idx_min=min, lai_kaplan_stride=stride, data=raw)

    with pytest.raises(IndexError):
        lkarr[0]

    with pytest.raises(IndexError):
        lkarr[n + 1.5]

    with pytest.raises(IndexError):
        lkarr[n + 2]

    # Don't allow negative indexes, too confusing
    with pytest.raises(IndexError):
        lkarr[-1]

    np.testing.assert_equal(lkarr[:], raw[:])
    np.testing.assert_equal(lkarr[1:], raw[:])
    np.testing.assert_equal(lkarr[: n + 2], raw[:])
    np.testing.assert_equal(lkarr[1 : n + 2], raw[:])

    with pytest.raises(AssertionError):
        # Make sure we're not starting with this value already set
        np.testing.assert_equal(lkarr[min], 2.0)

    lkarr[min] = 2.0
    np.testing.assert_equal(lkarr[min], 2.0)
    np.testing.assert_equal(lkarr.data[0], 2.0)


@lai_kaplan_array_n_elements
def test_lai_kaplan_array_x_extrap_like(n):
    min = 0
    max = n + 2
    stride = 1.0

    n_elements = int((max - min) / stride) + 1

    raw = np.arange(n_elements)

    lkarr = LaiKaplanArray(lai_kaplan_idx_min=min, lai_kaplan_stride=stride, data=raw)

    with pytest.raises(IndexError):
        lkarr[n + 2.5]

    with pytest.raises(IndexError):
        lkarr[n + 3]

    np.testing.assert_equal(lkarr[:], raw[:])
    np.testing.assert_equal(lkarr[0:], raw[:])
    np.testing.assert_equal(lkarr[: n + 3], raw[:])
    np.testing.assert_equal(lkarr[0 : n + 3], raw[:])


@lai_kaplan_array_n_elements
def test_lai_kaplan_array_y_like(n):
    min = 1
    max = n
    stride = 1.0

    n_elements = int((max - min) / stride) + 1

    raw = np.arange(n_elements)

    lkarr = LaiKaplanArray(lai_kaplan_idx_min=min, lai_kaplan_stride=stride, data=raw)

    np.testing.assert_equal(lkarr[:], raw[:])
    np.testing.assert_equal(lkarr[1:], raw[:])
    np.testing.assert_equal(lkarr[: n + 1], raw[:])
    np.testing.assert_equal(lkarr[1 : n + 1], raw[:])


@lai_kaplan_array_n_elements
def test_lai_kaplan_array_y_extrap_like(n):
    min = 0
    max = n + 1
    stride = 1.0

    n_elements = int((max - min) / stride) + 1

    raw = np.arange(n_elements)

    lkarr = LaiKaplanArray(lai_kaplan_idx_min=min, lai_kaplan_stride=stride, data=raw)

    np.testing.assert_equal(lkarr[:], raw[:])
    np.testing.assert_equal(lkarr[0:], raw[:])
    np.testing.assert_equal(lkarr[: n + 2], raw[:])
    np.testing.assert_equal(lkarr[0 : n + 2], raw[:])


@lai_kaplan_array_n_elements
def test_lai_kaplan_array_control_points_x_y_like(n):
    min = 1 / 2
    max = n + 1.5
    stride = 0.5

    n_elements = int((max - min) / stride) + 1

    raw = np.arange(n_elements)

    lkarr = LaiKaplanArray(lai_kaplan_idx_min=min, lai_kaplan_stride=stride, data=raw)

    with pytest.raises(IndexError):
        lkarr[0]

    with pytest.raises(IndexError):
        lkarr[n + 2]

    np.testing.assert_equal(lkarr[1 / 2], raw[0])
    np.testing.assert_equal(lkarr[1], raw[1])
    np.testing.assert_equal(lkarr[2], raw[3])
    np.testing.assert_equal(lkarr[1:2], raw[1:3])

    np.testing.assert_equal(lkarr[:], raw[:])
    np.testing.assert_equal(lkarr[::1], raw[::2])
    np.testing.assert_equal(lkarr[1 / 2 :], raw[:])
    np.testing.assert_equal(lkarr[: n + 2], raw[:])
    np.testing.assert_equal(lkarr[1 / 2 : n + 2], raw[:])


@lai_kaplan_array_n_elements
def test_lai_kaplan_array_control_point_gradients_like(n):
    min = 1
    max = n + 1
    stride = 0.5

    n_elements = int((max - min) / stride) + 1

    raw = np.arange(n_elements)

    lkarr = LaiKaplanArray(lai_kaplan_idx_min=min, lai_kaplan_stride=stride, data=raw)

    assert lkarr[1] == raw[0]
    assert lkarr[3 / 2] == raw[1]
    assert lkarr[3] == raw[4]
    assert lkarr[n + 1] == raw[-1]

    np.testing.assert_equal(lkarr[:], raw[:])
    np.testing.assert_equal(lkarr[::], raw[::])
    np.testing.assert_equal(lkarr[1:2], raw[: int(1 / stride)])
    np.testing.assert_equal(lkarr[1:3], raw[: 2 * int(1 / stride)])

    np.testing.assert_equal(lkarr[1:n], raw[: int((n - 1) / stride)])


@lai_kaplan_array_n_elements
def test_lai_kaplan_array_delta_like(n):
    # Also covers  the 'u' array
    min = 1
    max = n + 1 / 2
    stride = 0.5

    n_elements = int((max - min) / stride) + 1

    raw = np.arange(n_elements)

    lkarr = LaiKaplanArray(lai_kaplan_idx_min=min, lai_kaplan_stride=stride, data=raw)

    assert lkarr[1] == raw[0]
    assert lkarr[3 / 2] == raw[1]
    assert lkarr[n + 1 / 2] == raw[-1]

    np.testing.assert_equal(lkarr[:], raw[:])
    np.testing.assert_equal(lkarr[::], raw[::])
    np.testing.assert_equal(lkarr[1:2], raw[:2])
    np.testing.assert_equal(lkarr[1:3], raw[:4])

    np.testing.assert_equal(lkarr[1 : n + 1 / 2], raw[:-1])
    np.testing.assert_equal(lkarr[1 : n + 1], raw[:])
