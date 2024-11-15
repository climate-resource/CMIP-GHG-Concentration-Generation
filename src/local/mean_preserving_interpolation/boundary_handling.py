"""
Handling of the fact that we start with data over an interval, but need data at boundaries sometimes
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, cast

import numpy as np
import pint

from local.optional_dependencies import get_optional_dependency


class GetYAtBoundariesLike(Protocol):
    """
    Class that supports getting y-values at the boundaries
    """

    def __call__(
        self,
        x_bounds: pint.UnitRegistry.Quantity,
        y_in: pint.UnitRegistry.Quantity,
    ) -> pint.UnitRegistry.Quantity:
        """
        Calculate the y-values at the boundaries

        Parameters
        ----------
        x_bounds
            The x-bounds that define the intervals to which each value
            in `y_in` applies.

        y_in
            Average y-value for each interval in `x_bounds`

        Returns
        -------
        :
            y-values at the boundaries for each interval in `x_bounds`.

            In other words, the y-values at the points defined by `x_bounds`,
            rather than the average y-value over each interval.
        """


class BoundaryHandling(StrEnum):
    """
    Options for boundary handling with [`get_y_at_boundaries`][]
    """

    CONSTANT = "constant"
    CUBIC_EXTRAPOLATION = "cubic_extrapolation"


def get_y_at_boundaries(
    x_bounds: pint.UnitRegistry.Quantity,
    y_in: pint.UnitRegistry.Quantity,
    left: BoundaryHandling = BoundaryHandling.CONSTANT,
    right: BoundaryHandling = BoundaryHandling.CUBIC_EXTRAPOLATION,
) -> pint.UnitRegistry.Quantity:
    """
    Get y-values at the boundaries

    For all internal values,
    the boundary values are the average of the interval values
    to either side of the boundary.
    For the external values,
    the boundary values are calculated based on the value of `left` and `right`.

    Parameters
    ----------
    x_bounds
        The x-bounds that define the intervals to which each value
        in `y_in` applies.

    y_in
        Average y-value for each interval in `x_bounds`

    left
        Rule to apply to get the value at the left-hand boundary

    right
        Rule to apply to get the value at the right-hand boundary

    Returns
    -------
    :
        y-values at the boundaries for each interval in `x_bounds`.

        In other words, the y-values at the points defined by `x_bounds`,
        rather than the average y-value over each interval.
    """
    y_val_at_bound_internal = (y_in[1:] + y_in[:-1]) / 2.0

    if left == BoundaryHandling.CONSTANT:
        y_val_at_bound_left = y_in[0]

    if right == BoundaryHandling.CONSTANT:
        y_val_at_bound_right = y_in[-1]

    if any(bh == BoundaryHandling.CUBIC_EXTRAPOLATION for bh in (left, right)):
        scipy_inter = get_optional_dependency("scipy.interpolate")

        x_mid = (x_bounds[1:] + x_bounds[:-1]) / 2.0
        cubic_interpolator = scipy_inter.interp1d(
            x_mid.m,
            y_in.m,
            kind="cubic",
            fill_value="extrapolate",
        )

        if left == BoundaryHandling.CUBIC_EXTRAPOLATION:
            y_val_at_bound_left = cubic_interpolator(x_bounds[0].m) * y_in.u

        if right == BoundaryHandling.CUBIC_EXTRAPOLATION:
            y_val_at_bound_right = cubic_interpolator(x_bounds[-1].m) * y_in.u

    return cast(
        pint.UnitRegistry.Quantity,
        np.hstack([y_val_at_bound_left, y_val_at_bound_internal, y_val_at_bound_right]),
    )
