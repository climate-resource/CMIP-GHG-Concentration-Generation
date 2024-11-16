"""
Tests of `local.mean_preserving_interpolation.integrate_groups`
"""

from __future__ import annotations

import numpy as np
import pint
import pint.testing
import pytest

from local.mean_preserving_interpolation.grouping import (
    NonIntersectingBoundsError,
    get_group_averages,
)

Q = pint.get_application_registry().Quantity


@pytest.mark.parametrize(
    "x_bounds, y, group_bounds, exp",
    (
        pytest.param(
            Q(np.arange(13), "yr"),
            Q(np.arange(12), "m"),
            Q(np.arange(0, 13, 3), "yr"),
            Q([3, 12, 21, 30], "yr m") / Q(3, "yr"),
            id="basic",
        ),
        pytest.param(
            Q(np.arange(13), "yr"),
            Q(np.arange(12), "m"),
            Q(np.arange(0, 13 * 12, 3 * 12), "months"),
            Q([3, 12, 21, 30], "yr m") / Q(3, "yr"),
            id="x_bounds_in_different_units",
        ),
        pytest.param(
            Q([0, 4, 10, 20, 25], "yr"),
            Q([1, 2, 3, -1], "m"),
            Q([0, 10, 25], "yr"),
            Q([4 + 12, 30 - 5], "yr m") / Q([10, 15], "yr"),
            id="uneven_x_bounds",
        ),
    ),
)
def test_get_group_averages(x_bounds, y, group_bounds, exp):
    res = get_group_averages(x_bounds, y, group_bounds)

    pint.testing.assert_allclose(res, exp)


def test_get_group_averages_x_bounds_dont_intersect_error():
    """
    Test that we get a sensible error if the bounds don't intersect
    """
    x_bounds = Q([0, 2, 4, 6], "yr")
    group_bounds = Q([0, 3, 6], "yr")

    with pytest.raises(NonIntersectingBoundsError):
        get_group_averages(
            integrand_x_bounds=x_bounds,
            integrand_y="not used",
            group_bounds=group_bounds,
        )
