"""
Regression tests of our mean-preserving interpolation algorithms
"""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing
import pint
import pytest

from local.mean_preserving_interpolation import mean_preserving_interpolation

Q = pint.get_application_registry().Quantity
RNG = np.random.default_rng(seed=4234)


def execute_test_logic(  # noqa: PLR0913
    algorithm: str,
    y_in: numpy.typing.NDArray[np.float64],
    data_regression,
    num_regression,
    x_0: float = 100.0,
    x_in_spacing: float = 1.0,
    res_increase: int = 12,
) -> None:
    x_bounds_in = Q(
        x_0 + np.arange(0.0, x_in_spacing * (y_in.size + 1), x_in_spacing),
        "yr",
    )

    x_bounds_out = Q(
        x_bounds_in.m[0]
        + np.arange(
            0.0,
            x_in_spacing * y_in.size + 1 / (2 * res_increase),
            x_in_spacing / res_increase,
        ),
        "yr",
    )

    assert x_bounds_out.size == (x_bounds_in.size - 1) * res_increase + 1

    y_out = mean_preserving_interpolation(
        x_bounds_in=x_bounds_in,
        y_in=y_in,
        x_bounds_out=x_bounds_out,
        algorithm=algorithm,
    )

    # Check that the output means are correct
    for i, (x_min, x_max) in enumerate(itertools.pairwise(x_bounds_in)):
        y_out_interval = y_out[
            np.where((x_bounds_out >= x_min) & (x_bounds_out < x_max))
        ]
        assert False, "Use product of y and x intervals here rather than np.mean"
        pint.testing.assert_allclose(np.mean(y_out_interval), y_in[i])

    data_regression.check({"y_out_u": str(y_out.u)})
    num_regression.check({"y_out_m": y_out.m})


@pytest.mark.parametrize(
    "y_in, x_0, x_in_spacing",
    (
        pytest.param(Q([0, 0, 1, 3, 5, 7, 9.0], "kg"), 2000.0, 1.0, id="basic"),
        pytest.param(
            Q([0, 0, 1, 3, 5, 7, 9.0], "kg"), 2000.0, 2.0, id="x_spacing_equal_to_2"
        ),
        pytest.param(
            Q([0, 0, 0.3, 2, 2.5, 3, 5], "kg"), 1.0, 1.0, id="x_0_equal_to_one"
        ),
        pytest.param(
            Q(np.arange(50.0) / 20.0 + RNG.random(50), "kg"),
            0.0,
            1.0,
            id="x_0_equal_to_zero",
        ),
    ),
)
@pytest.mark.parametrize("algorithm", ("lazy_linear", "lai_kaplan", "rymes_meyers"))
def test_mean_preserving_interpolation(  # noqa: PLR0913
    algorithm, y_in, x_0, x_in_spacing, data_regression, num_regression
):
    execute_test_logic(
        algorithm=algorithm,
        y_in=y_in,
        data_regression=data_regression,
        num_regression=num_regression,
        x_0=x_0,
        x_in_spacing=x_in_spacing,
    )


@pytest.mark.parametrize(
    "y_in",
    (
        pytest.param(
            Q(np.arange(2022) / 1000.0 + RNG.random(2022), "kg"),
            id="basic",
        ),
    ),
)
@pytest.mark.parametrize("algorithm", ("lai_kaplan", "rymes_meyers"))
def test_mean_preserving_interpolation_long_array(
    algorithm, y_in, data_regression, num_regression
):
    execute_test_logic(
        algorithm=algorithm,
        y_in=y_in,
        data_regression=data_regression,
        num_regression=num_regression,
    )


# To write:
# - tests that include setting a minimum value for outputs
# - tests that include interpolating
#   with different numbers of points per interval (not just 12)
