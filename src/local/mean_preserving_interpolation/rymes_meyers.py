"""
Rymes-Meyers mean-preserving interpolator

See [Rymes and Myers, Solar Energy 2001](https://doi.org/10.1016/S0038-092X(01)00052-4)
"""

from __future__ import annotations

from functools import partial

import numpy as np
import numpy.typing as npt
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
    get_group_integrals,
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
        current_vals_group_indexes = get_group_indexes(x_bounds=x_bounds_out, group_bounds=x_bounds_in)

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

            current_vals = self.do_core_iteration(
                current_vals=current_vals,
                current_vals_group_indexes=current_vals_group_indexes,
                adjust_mat=adjust_mat,
                left_bound_val=y_at_boundaries[0],
                right_bound_val=y_at_boundaries[-1],
                x_bounds_target=x_bounds_in,
                target=y_in,
                x_bounds_out=x_bounds_out,
            )

            if self.min_val is not None and (current_vals < self.min_val).any():
                current_vals = self.lower_bound_adjustment(
                    min_val=self.min_val,
                    current_vals=current_vals,
                    current_vals_group_indexes=current_vals_group_indexes,
                    adjust_mat=adjust_mat,
                    left_bound_val=y_at_boundaries[0],
                    right_bound_val=y_at_boundaries[-1],
                    x_bounds_target=x_bounds_in,
                    target=y_in,
                    x_bounds_out=x_bounds_out,
                )

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
            pint.testing.assert_allclose(group_averages, target, atol=self.atol, rtol=self.rtol)
        except AssertionError:
            # Not close
            return False

        # No error i.e. all close
        return True

    @staticmethod
    def do_core_iteration(  # noqa: PLR0913
        current_vals: pint.UnitRegistry.Quantity,
        current_vals_group_indexes: npt.NDArray[np.int_],
        adjust_mat: npt.NDArray[np.float64],
        left_bound_val: float,
        right_bound_val: float,
        x_bounds_target: pint.UnitRegistry.Quantity,
        target: pint.UnitRegistry.Quantity,
        x_bounds_out: pint.UnitRegistry.Quantity,
    ) -> pint.UnitRegistry.Quantity:
        """
        Perform a core iteration

        Parameters
        ----------
        current_vals
            Current values of the solution

        current_vals_group_indexes
            For each value in `current_vals`, the index of the group in `target` it belongs to.

        adjust_mat
            Matrix for adjusting the values

        left_bound_val
            Value to use as the left-bound during the adjustments

        right_bound_val
            Value to use as the right-bound during the adjustments

        x_bounds_target
            The x-bounds of the target

        target
            The target values

        x_bounds_out
            The x-bounds onto which we are interpolating

        Returns
        -------
        :
            The updated solution values based on this iteration
        """
        current_vals = adjust_mat @ current_vals
        current_vals[0] += left_bound_val / 3
        current_vals[-1] += right_bound_val / 3

        group_averages = get_group_averages(
            integrand_x_bounds=x_bounds_out,
            integrand_y=current_vals,
            group_bounds=x_bounds_target,
        )
        corrections = target - group_averages
        current_vals = current_vals + corrections[(current_vals_group_indexes,)]

        return current_vals

    @staticmethod
    def lower_bound_adjustment(  # noqa: PLR0913
        min_val: pint.UnitRegistry.Quantity,
        current_vals: pint.UnitRegistry.Quantity,
        current_vals_group_indexes: npt.NDArray[np.int_],
        adjust_mat: npt.NDArray[np.float64],
        left_bound_val: float,
        right_bound_val: float,
        x_bounds_target: pint.UnitRegistry.Quantity,
        target: pint.UnitRegistry.Quantity,
        x_bounds_out: pint.UnitRegistry.Quantity,
    ) -> pint.UnitRegistry.Quantity:
        """
        Adjust the solution to account for the fact that some values are below a specified minimum

        Parameters
        ----------
        min_val
            Minimum value which is allowed in the solution

        current_vals
            Current values of the solution

        current_vals_group_indexes
            For each value in `current_vals`, the index of the group in `target` it belongs to.

        adjust_mat
            Matrix for adjusting the values

        left_bound_val
            Value to use as the left-bound during the adjustments

        right_bound_val
            Value to use as the right-bound during the adjustments

        x_bounds_target
            The x-bounds of the target

        target
            The target values

        x_bounds_out
            The x-bounds onto which we are interpolating

        Returns
        -------
        :
            The updated solution values based on this iteration
        """
        current_vals[np.where(current_vals < min_val)] = min_val

        group_averages = get_group_averages(
            integrand_x_bounds=x_bounds_out,
            integrand_y=current_vals,
            group_bounds=x_bounds_target,
        )

        # Rymes-Meyers lower-bound "f" correction function
        rm_lb_f_numerator = get_group_integrals(
            integrand_x_bounds=x_bounds_out,
            integrand_y=current_vals - target[(current_vals_group_indexes,)],
            group_bounds=x_bounds_target,
        )
        rm_lb_f_denominator = get_group_integrals(
            integrand_x_bounds=x_bounds_out,
            integrand_y=current_vals - min_val,
            group_bounds=x_bounds_target,
        )

        rm_lb_f = rm_lb_f_numerator / rm_lb_f_denominator

        if np.isnan(rm_lb_f).any():
            # If the numerator is also zeor, set to zero and move on
            rm_lb_f[np.where(rm_lb_f_numerator == 0.0)] = 0.0 * rm_lb_f.u

            if np.isnan(rm_lb_f).any():
                # If still nans, raise
                raise AssertionError()

        group_delta = group_averages - target

        rm_lb_correction = rm_lb_f[(current_vals_group_indexes,)] * (current_vals - min_val)
        # Only apply deltas where we have overshot
        rm_lb_correction[np.where(group_delta <= 0.0)] = 0.0 * group_averages.u

        current_vals = current_vals - rm_lb_correction

        current_vals[np.where(current_vals < min_val)] = min_val

        return current_vals
