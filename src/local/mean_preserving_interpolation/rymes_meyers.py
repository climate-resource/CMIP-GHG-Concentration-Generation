"""
Rymes-Meyers mean-preserving interpolator

See [Rymes and Myers, Solar Energy 2001](https://doi.org/10.1016/S0038-092X(01)00052-4)
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pint
from attrs import define

from local.mean_preserving_interpolation.boundary_handling import (
    BoundaryHandling,
    GetYAtBoundariesLike,
)
from local.mean_preserving_interpolation.boundary_handling import (
    get_y_at_boundaries as get_y_at_boundaries_default,
)
from local.mean_preserving_interpolation.grouping import (
    get_group_averages,
    get_group_indexes,
)
from local.optional_dependencies import get_optional_dependency


@define
class RymesMeyersInterpolator:
    """
    Rymes-Meyers mean-preserving interpolator

    This uses a recursive algorithm to adjust values to be mean-preserving.
    It produces very smooth results.
    However, it can be quite slow because of its recursive nature.

    See [Rymes and Myers, Solar Energy 2001](https://doi.org/10.1016/S0038-092X(01)00052-4)
    """

    get_y_at_boundaries: GetYAtBoundariesLike = partial(
        get_y_at_boundaries_default,
        left=BoundaryHandling.CONSTANT,
        right=BoundaryHandling.CUBIC_EXTRAPOLATION,
    )
    """
    Function that calculates the y-values at the boundaries from the y-values in each interval
    """

    min_val: pint.UnitRegistry.Quantity | None = None
    """
    Minimum value that can appear in the solution
    """

    min_it: int | None = None
    """
    Minimum number of iterations to perform

    If not provided, we use the length of the output data array.
    """

    max_it: int = int(1e4)
    """
    Maximum number of iterations to perform
    """

    atol: float = 1e-10
    """
    Absolute tolerance for deciding whether the output value means are close to the input means
    """

    rtol: float = 1e-7
    """
    Relative tolerance for deciding whether the output value means are close to the input means
    """

    progress_bar: bool = True
    """
    Whether to show a progress bar for the iterations or not
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
        if self.min_it is None:
            min_it = x_bounds_out.size - 1
        else:
            min_it = self.min_it

        y_at_boundaries = self.get_y_at_boundaries(x_bounds=x_bounds_in, y_in=y_in)

        x_out_mids = (x_bounds_out[1:] + x_bounds_out[:-1]) / 2.0
        y_starting_values = np.interp(x_out_mids, x_bounds_in, y_at_boundaries)

        # Run the algorithm
        current_vals = y_starting_values
        current_vals_group_idexes = get_group_indexes(
            x_bounds=x_bounds_out, group_bounds=x_bounds_in
        )

        adjust_mat = np.zeros((y_starting_values.size, y_starting_values.size))
        rows, cols = np.diag_indices_from(adjust_mat)
        adjust_mat[rows[1:], cols[:-1]] = 1 / 3
        adjust_mat[rows, cols] = 1 / 3
        adjust_mat[rows[:-1], cols[1:]] = 1 / 3

        iterh = range(self.max_it)
        if self.progress_bar:
            tqdman = get_optional_dependency("tqdm.autonotebook")
            iterh = tqdman.tqdm(iterh)

        for i in iterh:
            group_averages = get_group_averages(
                integrand_x_bounds=x_bounds_out,
                integrand_y=current_vals,
                group_bounds=x_bounds_in,
            )

            if (i >= min_it) and self.solution_is_close(
                group_averages=group_averages,
                target=y_in,
            ):
                break

            current_vals = adjust_mat @ current_vals
            current_vals[0] += y_at_boundaries[0] / 3
            current_vals[-1] += y_at_boundaries[-1] / 3

            group_averages = get_group_averages(
                integrand_x_bounds=x_bounds_out,
                integrand_y=current_vals,
                group_bounds=x_bounds_in,
            )
            corrections = y_in - group_averages
            corrections_rep = corrections[(current_vals_group_idexes,)]
            current_vals = current_vals + corrections_rep

            if self.min_val is not None:
                raise NotImplementedError

        else:
            msg = f"Ran out of iterations ({self.max_it=})"
            raise AssertionError(msg)

        return current_vals

    def solution_is_close(
        self,
        group_averages: pint.UnitRegistry.Quantity,
        target: pint.UnitRegistry.Quantity,
    ) -> bool:
        """
        Determine whether a solution is close to the target or not

        Parameters
        ----------
        group_averages
            The group averages of the proposed solution

        target
            The target group averages

        Returns
        -------
        :
            `True` if `group_averages` is close to `target`, otherwise `False`
        """
        try:
            pint.testing.assert_allclose(
                group_averages, target, atol=self.atol, rtol=self.rtol
            )
        except AssertionError:
            # Not close
            return False

        # No error i.e. all close
        return True
