"""
Calculation of averages for groups within an array
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pint


class NonIntersectingBoundsError(IndexError):
    """
    Raised to signal that the bounds of our input and groups don't intersect

    We don't support such a use case yet,
    because we assume we are working with discrete, piecewise-constant data.
    Altering this would require quite some changes to the algorithms below.
    """

    def __init__(
        self,
        integrand_x_bounds: pint.UnitRegistry.Quantity,
        group_bounds: pint.UnitRegistry.Quantity,
        not_in_integrand_x_bounds: npt.NDArray[bool],
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        integrand_x_bounds
            x-bounds of the input.

        group_bounds
            The bounds of the groups for which we are calculating averages/integrals.

        not_in_integrand_x_bounds
            Array of boolean values.

            We assume that `True` values indicate values in `group_bounds`
            that aren't in `integrand_x_bounds`.
        """
        not_in_integrand_x_bounds_values = group_bounds[not_in_integrand_x_bounds]

        error_msg = (
            "We can only perform our operations if the group boundaries "
            "line up exactly with the x-boundaries. "
            "The following group boundaries "
            f"are not in the integrand's x-boundary values: {not_in_integrand_x_bounds_values}. "
            f"{group_bounds=} {integrand_x_bounds=}"
        )

        super().__init__(error_msg)


def get_group_integrals(
    integrand_x_bounds: pint.UnitRegistry.Quantity,
    integrand_y: pint.UnitRegistry.Quantity,
    group_bounds: pint.UnitRegistry.Quantity,
) -> pint.UnitRegistry.Quantity:
    """
    Get integrals for groups of values within an array

    Parameters
    ----------
    integrand_x_bounds
        The x-bounds of the input

    integrand_y
        The y-values for each interval defined by `integrand_x_bounds`

        These values are assume to be piecewise-constant
        i.e. constant across each domain.

    group_bounds
        The bounds of the groups for which we want the integrals.

    Returns
    -------
    :
        Integrals of the values in `integrand_y` for the groups defined by `group_bounds`.
    """
    not_in_integrand_x_bounds = ~np.in1d(
        group_bounds.m, integrand_x_bounds.to(group_bounds.u).m
    )
    if not_in_integrand_x_bounds.any():
        raise NonIntersectingBoundsError(
            integrand_x_bounds=integrand_x_bounds,
            group_bounds=group_bounds,
            not_in_integrand_x_bounds=not_in_integrand_x_bounds,
        )

    group_boundaries = np.where(
        np.in1d(integrand_x_bounds.m, group_bounds.to(integrand_x_bounds.u).m)
    )

    # The minus one is required to ensure we get the correct integral value.
    # (If the boundary occurs at an index of n
    # we want all the values before that boundary,
    # but not the cumulative value actually on the boundary).
    cumulative_integrals_group_idxs = group_boundaries[0] - 1
    # Drop out any places where the substraction above leads to non-sensical results.
    cumulative_integrals_group_idxs = cumulative_integrals_group_idxs[
        np.where(cumulative_integrals_group_idxs >= 0)
    ]

    # Assumes that y can be treated as a constant
    # over each interval defined by `integrand_x_bounds`
    # for the purposes of the integration.
    x_steps = integrand_x_bounds[1:] - integrand_x_bounds[:-1]

    integrals = x_steps * integrand_y

    cumulative_integrals = np.cumsum(integrals)
    cumulative_integrals_groups = cumulative_integrals[cumulative_integrals_group_idxs]

    res = np.hstack(
        [cumulative_integrals_groups[0], np.diff(cumulative_integrals_groups)]
    )

    return res


def get_group_averages(
    integrand_x_bounds: pint.UnitRegistry.Quantity,
    integrand_y: pint.UnitRegistry.Quantity,
    group_bounds: pint.UnitRegistry.Quantity,
) -> pint.UnitRegistry.Quantity:
    """
    Get averages for groups of values within an array

    Parameters
    ----------
    integrand_x_bounds
        The x-bounds of the input

    integrand_y
        The y-values for each interval defined by `integrand_x_bounds`

        These values are assume to be piecewise-constant
        i.e. constant across each domain.

    group_bounds
        The bounds of the groups for which we want the averages.

    Returns
    -------
    :
        Averages of the values in `integrand_y` for the groups defined by `group_bounds`.
    """
    group_integrals = get_group_integrals(
        integrand_x_bounds=integrand_x_bounds,
        integrand_y=integrand_y,
        group_bounds=group_bounds,
    )
    group_steps = group_bounds[1:] - group_bounds[:-1]

    averages = group_integrals / group_steps

    return averages
