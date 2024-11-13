"""
Tests of mean preserving interpolation
"""

from __future__ import annotations

import numpy as np
import pint
import pytest

from local.mean_preserving_interpolation import LaiKaplanArray, mean_preserving_interpolation

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
    # Also covers A
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
    # Also covers u
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


# Test with uneven x-axis?
#    just raise NotImplementedError for now


# Test yearly interpolation
# Test latitudinal gradient interpolation
# The method we're using will mean we need a different bottom layer for each.
# That's fine, a general solution is super computationally expensive (as we have found out).
# Or just use Rymes-Myers for the spatial interpolation


def test_mean_preserving_interpolation():
    Q = pint.get_application_registry().Quantity

    y_in = Q([0, 0, 1, 3, 5, 7, 9.0], "kg")
    y_in = Q([0, 0, 1, 7, 19, 37, 5**3 - 4**3], "kg")
    y_in = Q([0, 0, 0.3, 2, 2.5, 3, 5], "kg")
    # Test with different x spacing
    x_bounds_in = Q(
        np.arange(y_in.size + 1) + 2000,
        "yr",
    )
    print()
    print(f"{y_in=}")
    print(f"{x_bounds_in=}")

    x_bounds_out = Q(np.arange(0, y_in.size + 1 / 12, 1 / 12) + x_bounds_in.m[0], "yr")

    y_out = mean_preserving_interpolation(
        x_bounds_in=x_bounds_in,
        y_in=y_in,
        x_bounds_out=x_bounds_out,
    )
    breakpoint()

    assert False, "Add test of mean-preserving"
    assert False, "Add regression test of units and magnitude"


#
#
# def test_mean_preserving_interpolation_unit_handling():
#     # Should be able to use different units and get same answer (mod units)
#     assert False
