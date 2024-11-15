"""
Mean preserving interpolation

This is a surprisingly tricky thing to do.
Hence, this module is surprisingly large.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import pint

MeanPreservingInterpolationAlgorithmLike: TypeAlias = Callable[
    [
        pint.UnitRegistry.Quantity,
        pint.UnitRegistry.Quantity,
        pint.UnitRegistry.Quantity,
    ],
    pint.UnitRegistry.Quantity,
]
"""
Callable that supports being used as a mean-preserving interpolation algorithm
"""


def mean_preserving_interpolation(
    x_bounds_in: pint.UnitRegistry.Quantity,
    y_in: pint.UnitRegistry.Quantity,
    x_bounds_out: pint.UnitRegistry.Quantity,
    algorithm: str | MeanPreservingInterpolationAlgorithmLike = "lai_kaplan",
) -> pint.UnitRegistry.Quantity:
    """
    Perform mean-preserving interpolation

    Parameters
    ----------
    x_bounds_in
        Bounds of the x-range to which each value in `y_in` applies.

    y_in
        y-values for each interval in `x_bounds_in`.

    x_bounds_out
        Bounds of the x-values onto which to interpolate `y_in`.

    algorithm
        Algorithm to use for the interpolation.

        A few default algorithms are supported by their string name, specifically

        - "lai_kaplan", which uses the [Lai-Kaplan](https://doi.org/10.1175/JTECH-D-21-0154.1)
          algorithm
        - "rymes_meyers", which uses the [Rymes-Meyers](https://doi.org/10.1016/S0038-092X(01)00052-4)
          algorithm
        - "lazy_linear", which performs linear interpolation then simply adjusts
          the values in each interval to match the input values' mean.
          This algorithm is really only here for testing purposes.
          We do not recommend using it in production for anything but the simplest cases.

        A callable may also be passed in.
        For example, your own interpolator, created how you would like it
        (rather than using the default initialisation defined in this function).

    Returns
    -------
    :
        Interpolated, mean-preserving values
    """
    if not x_bounds_in.size == y_in.size + 1:
        msg = (
            "`x_bounds_in` should be the bounds of each interval "
            "to which the values in `y_in` apply. "
            "As a result, `x_bounds_in` should be one element longer "
            "than `y_in`. "
            f"However, we received {x_bounds_in.size=} and {y_in.size=}"
        )

    if callable(algorithm):
        algorithm_func = algorithm

    elif isinstance(algorithm, str):
        if algorithm == "lai_kaplan":
            # Use default Lai-Kaplan interpolator
            algorithm_func = LaiKaplanInterpolator()

        elif algorithm == "rymes_meyers":
            # Use default Rymes-Meyers interpolator
            algorithm_func = RymesMeyersInterpolator()

        elif algorithm == "lazy_linear":
            algorithm_func = LazyLinearInterpolator()

        else:
            msg = f"Unknown algorithm supplied: {algorithm=!r}"
            raise NotImplementedError(msg)

    else:
        msg = f"Not supported: {algorithm=}"
        raise NotImplementedError(msg)

    return algorithm_func(
        x_bounds_in,
        y_in,
        x_bounds_out,
    )
