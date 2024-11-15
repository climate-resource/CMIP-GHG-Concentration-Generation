"""
Mean preserving interpolation

This is a surprisingly tricky thing to do.
Hence, this module is surprisingly large.
"""

from __future__ import annotations

from typing import Protocol

import pint
import pint.testing

from local.mean_preserving_interpolation.grouping import get_group_averages
from local.mean_preserving_interpolation.lazy_linear import LazyLinearInterpolator


class MeanPreservingInterpolationAlgorithmLike(Protocol):
    """
    Class that supports being used as a mean-preserving interpolation algorithm
    """

    def __call__(
        self,
        x_bounds_in: pint.UnitRegistry.Quantity,
        y_in: pint.UnitRegistry.Quantity,
        x_bounds_out: pint.UnitRegistry.Quantity,
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

        Returns
        -------
        :
            Interpolated, mean-preserving values
        """


def mean_preserving_interpolation(  # noqa: PLR0913
    x_bounds_in: pint.UnitRegistry.Quantity,
    y_in: pint.UnitRegistry.Quantity,
    x_bounds_out: pint.UnitRegistry.Quantity,
    algorithm: str | MeanPreservingInterpolationAlgorithmLike = "lai_kaplan",
    verify_output_is_mean_preserving: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-10,
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

    verify_output_is_mean_preserving
        Verify that the output is in fact mean-preserving.

        If you are confident in your algorithm's behaviour,
        this can be disabled to increase performance.

    rtol
        If `verify_output_is_mean_preserving` is `True`,
        the relative tolerance to apply during the verification.

    atol
        If `verify_output_is_mean_preserving` is `True`,
        the absolute tolerance to apply during the verification.

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
        msg = f"Not supported: {algorithm=}"  # type: ignore[unreachable] # free to include and just in case
        raise NotImplementedError(msg)

    res = algorithm_func(
        x_bounds_in=x_bounds_in,
        y_in=y_in,
        x_bounds_out=x_bounds_out,
    )

    if verify_output_is_mean_preserving:
        res_group_averages = get_group_averages(
            integrand_x_bounds=x_bounds_out,
            integrand_y=res,
            group_bounds=x_bounds_in,
        )
        pint.testing.assert_allclose(y_in, res_group_averages, atol=atol, rtol=rtol)

    return res
