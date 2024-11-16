"""
Regression tests of our mean-preserving interpolation algorithms
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import pint
import pint.testing
import pytest

from local.mean_preserving_interpolation import (
    MeanPreservingInterpolationAlgorithmLike,
    mean_preserving_interpolation,
)
from local.mean_preserving_interpolation.grouping import get_group_averages
from local.mean_preserving_interpolation.rymes_meyers import RymesMeyersInterpolator

Q = pint.get_application_registry().Quantity
RNG = np.random.default_rng(seed=4234)


def execute_test_logic(  # noqa: PLR0913
    algorithm: str | MeanPreservingInterpolationAlgorithmLike,
    y_in: numpy.typing.NDArray[np.float64],
    x_bounds_in: numpy.typing.NDArray[np.float64],
    x_bounds_out: numpy.typing.NDArray[np.float64],
    data_regression,
    num_regression,
    image_regression,
    tmpdir: Path,
) -> None:
    y_out = mean_preserving_interpolation(
        x_bounds_in=x_bounds_in,
        y_in=y_in,
        x_bounds_out=x_bounds_out,
        algorithm=algorithm,
        verify_output_is_mean_preserving=False,
    )

    # Check that the output means are correct
    y_out_group_averages = get_group_averages(
        integrand_x_bounds=x_bounds_out,
        integrand_y=y_out,
        group_bounds=x_bounds_in,
    )
    pint.testing.assert_allclose(y_in, y_out_group_averages, atol=1e-10)

    data_regression.check({"y_out_u": str(y_out.u)})
    num_regression.check({"y_out_m": y_out.m})
    # Save a plot too (to help easy checking later)
    # Ensure matplot lib does not use a GUI backend (such as Tk).
    matplotlib.use("Agg")

    fig, ax = plt.subplots()

    ax.step(
        x_bounds_in,
        np.hstack([y_in, y_in[-1]]),
        where="post",
        label="input",
    )
    ax.step(
        x_bounds_out,
        np.hstack([y_out, y_out[-1]]),
        where="post",
        label="interpolated",
    )

    ax.legend(loc="upper left")

    plot_filename = tmpdir / "tmp.png"
    fig.savefig(str(plot_filename))

    image_regression.check(plot_filename.read_bytes(), diff_threshold=1.0)

    # Run again, with the verification checks
    # and make sure the result is the same.
    y_out_verify = mean_preserving_interpolation(
        x_bounds_in=x_bounds_in,
        y_in=y_in,
        x_bounds_out=x_bounds_out,
        algorithm=algorithm,
        verify_output_is_mean_preserving=True,
    )

    pint.testing.assert_equal(y_out, y_out_verify)


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
    algorithm,
    y_in,
    x_0,
    x_in_spacing,
    data_regression,
    num_regression,
    image_regression,
    tmpdir,
):
    res_increase = 12

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

    execute_test_logic(
        algorithm=algorithm,
        y_in=y_in,
        x_bounds_in=x_bounds_in,
        x_bounds_out=x_bounds_out,
        data_regression=data_regression,
        num_regression=num_regression,
        image_regression=image_regression,
        tmpdir=Path(tmpdir),
    )


@pytest.mark.parametrize(
    "res_increase, x_in_spacing",
    (
        (2, 2),
        (10, 1),
        (10, 5),
    ),
)
@pytest.mark.parametrize("algorithm", ("lazy_linear", "lai_kaplan", "rymes_meyers"))
def test_mean_preserving_interpolation_res_increase(  # noqa: PLR0913
    algorithm,
    res_increase,
    x_in_spacing,
    data_regression,
    num_regression,
    image_regression,
    tmpdir,
):
    y_in = Q(np.sin(2 * np.pi * np.arange(20) / 8.0), "m")
    x_0 = 100.0
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

    execute_test_logic(
        algorithm=algorithm,
        y_in=y_in,
        x_bounds_in=x_bounds_in,
        x_bounds_out=x_bounds_out,
        data_regression=data_regression,
        num_regression=num_regression,
        image_regression=image_regression,
        tmpdir=Path(tmpdir),
    )


@pytest.mark.parametrize("algorithm", ("lazy_linear", "lai_kaplan", "rymes_meyers"))
def test_mean_preserving_interpolation_uneven_increase(
    algorithm,
    data_regression,
    num_regression,
    image_regression,
    tmpdir,
):
    y_in = Q([0, 1, 10, 20], "m")

    x_bounds_in = Q([0, 1, 6, 12, 24], "month")
    x_bounds_out = Q(
        [0, 1 / 3, 2 / 3, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24], "month"
    )

    execute_test_logic(
        algorithm=algorithm,
        y_in=y_in,
        x_bounds_in=x_bounds_in,
        x_bounds_out=x_bounds_out,
        data_regression=data_regression,
        num_regression=num_regression,
        image_regression=image_regression,
        tmpdir=Path(tmpdir),
    )


@pytest.mark.parametrize(
    "algorithm",
    (
        pytest.param("lazy_linear", marks=pytest.mark.skip(reason="Not implemented")),
        pytest.param(RymesMeyersInterpolator(min_val=-1.0), id="rymes_meyers_-1"),
        pytest.param(RymesMeyersInterpolator(min_val=0.0), id="rymes_meyers_0"),
        "lai_kaplan",
    ),
)
def test_mean_preserving_min_val(
    algorithm,
    data_regression,
    num_regression,
    image_regression,
    tmpdir,
):
    """
    Test support for minimum values

    This also implicitly tests our ability to pass a callable as `algorithm`.
    """
    res_increase = 6
    y_in = Q([0, 0, 0, 1, 4, 9, 10, 12, 8, 4, 0, 0, 2, 6, 5, 0, 0], "m")

    x_0 = 1750.0
    x_in_spacing = 1.0
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

    execute_test_logic(
        algorithm=algorithm,
        y_in=y_in,
        x_bounds_in=x_bounds_in,
        x_bounds_out=x_bounds_out,
        data_regression=data_regression,
        num_regression=num_regression,
        image_regression=image_regression,
        tmpdir=Path(tmpdir),
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
@pytest.mark.parametrize(
    "algorithm",
    (
        "lai_kaplan",
        pytest.param(
            "rymes_meyers",
            marks=pytest.mark.skip(
                reason="Current Rymes-Meyers implementation too slow for long tests"
            ),
        ),
    ),
)
def test_mean_preserving_interpolation_long_array(  # noqa: PLR0913
    algorithm, y_in, data_regression, num_regression, image_regression, tmpdir
):
    execute_test_logic(
        algorithm=algorithm,
        y_in=y_in,
        data_regression=data_regression,
        num_regression=num_regression,
        image_regression=image_regression,
        tmpdir=Path(tmpdir),
    )


# To write:
# - tests that include setting a minimum value for outputs
