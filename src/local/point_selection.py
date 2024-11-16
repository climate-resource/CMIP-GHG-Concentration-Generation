"""
Point selection
"""

from __future__ import annotations

import numpy as np
import pint
from attrs import define, field


def sort_array_by_x_values(
    inp: tuple[pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity],
) -> tuple[pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity]:
    """
    Sort arrays by their x-values

    Parameters
    ----------
    inp
        Arrays of x-points, y-points

    Returns
    -------
        Arrays of x-points, y-points, sorted so that x-points are increasing.
    """
    # TODO: sort on entry rather than in here
    # which is a surprising side effect.
    x_pool, y_pool = inp
    x_pool_sorted_idx = np.argsort(x_pool)

    return (x_pool[x_pool_sorted_idx], y_pool[x_pool_sorted_idx])


@define
class PointSelector:
    """
    Point selector i.e. an object which, given a pool of points, can select a subset
    """

    window_width: pint.UnitRegistry.Quantity
    """
    Window width
    """

    minimum_data_points_either_side: int
    """
    Minimum number of data points to pick either side of the target point.

    Obviously, if there are no data points to pick on one side of the target,
    less than this will be returned.

    Nicolai's code picks minimum/maximum number either side of obs.
    This is different to what is written in the paper.
    I have followed the code here.
    """

    maximum_data_points_either_side: int
    """
    Maximum number of data points to pick either side of the target point.
    """

    pool: tuple[pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity] = field(
        converter=sort_array_by_x_values
    )
    """
    Pool of x- and y-points to choose from
    """

    def get_points(
        self, target_year: pint.UnitRegistry.Quantity
    ) -> tuple[pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity]:
        """
        Get points

        Parameters
        ----------
        target_year
            Target year, around which to get points

        Returns
        -------
            Selected points
        """
        x_pool, y_pool = self.pool
        selected_x = []
        selected_y = []
        for forward_looking in [True, False]:
            if forward_looking:
                pool = x_pool[np.where(x_pool >= target_year)]
            else:
                pool = x_pool[np.where(x_pool < target_year)]

            if not pool.size:
                # Nothing in this direction, move on
                continue

            pool_abs_distance_from_target = np.abs(pool - target_year)
            within_window = pool[pool_abs_distance_from_target <= self.window_width]

            if within_window.shape[0] >= self.minimum_data_points_either_side:
                select_n_max = self.maximum_data_points_either_side
                select_from = within_window
                select_from_distance_from_target = np.abs(within_window - target_year)

            else:
                select_n_max = self.minimum_data_points_either_side
                select_from = pool
                select_from_distance_from_target = pool_abs_distance_from_target

            selected = select_from._REGISTRY.Quantity(
                np.take_along_axis(
                    select_from.m,
                    np.argsort(select_from_distance_from_target)[:select_n_max],
                    axis=0,
                ),
                select_from.units,
            )

            selected_x.append(selected)
            selected_y.append(y_pool[np.searchsorted(x_pool.to(selected.units).m, selected.m, side="left")])

        selected_x_stacked: pint.UnitRegistry.Quantity = np.hstack(selected_x)  # type: ignore
        selected_y_stacked: pint.UnitRegistry.Quantity = np.hstack(selected_y)  # type: ignore

        out: tuple[pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity] = (
            selected_x_stacked,
            selected_y_stacked,
        )

        return out
