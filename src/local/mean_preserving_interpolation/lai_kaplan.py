"""
Lai-Kaplan mean-preserving interpolator

See [Lai and Kaplan, J. Atmos. Oceanic Technol. 2022](https://doi.org/10.1175/JTECH-D-21-0154.1)
"""

from __future__ import annotations

from functools import partial
from typing import Generic, TypeVar

import attrs.validators
import numpy as np
import numpy.typing as npt
import pint
from attrs import define, field

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

T = TypeVar("T")


@define
class LaiKaplanArray(Generic[T]):
    """
    Thin wrapper around numpy arrays to support indexing like in the paper.

    This is sort of like writing a Python array that supports Fortran-style indexing,
    but trying to translate the paper with raw python indexes was too confusing,
    so we wrote this instead.
    """

    lai_kaplan_idx_min: float | int
    """Minimum index"""

    lai_kaplan_stride: float = field(validator=attrs.validators.in_((0.5, 1.0, 1)))
    """Size of stride"""

    data: npt.NDArray[T]
    """Actual data array"""

    @property
    def max_allowed_lai_kaplan_index(self):
        """
        The maximum allowed Lai-Kaplan style index for `self.data`

        Returns
        -------
        :
            The index.
        """
        return (self.data.size - 1) * self.lai_kaplan_stride + self.lai_kaplan_idx_min

    def to_data_index(self, idx_lai_kaplan: int | float | None, is_slice_idx: bool = False) -> int | None:
        """
        Convert a Lai-Kaplan index to the equivalent index for `self.data`

        Parameters
        ----------
        idx_lai_kaplan
            Lai-Kaplan index to translate

        is_slice_idx
            Whether this index is a slice index.

            This is important to ensure we give sensible errors
            about whether the index is too big for `self.data` or not.

        Returns
        -------
        :
            The index for `self.data`.
        """
        if idx_lai_kaplan is None:
            return None

        if idx_lai_kaplan < self.lai_kaplan_idx_min:
            msg = f"{idx_lai_kaplan=} is less than {self.lai_kaplan_idx_min=}"
            raise IndexError(msg)

        idx_data_float = (idx_lai_kaplan - self.lai_kaplan_idx_min) / self.lai_kaplan_stride
        if idx_data_float % 1.0:
            msg = f"{idx_lai_kaplan=} leads to {idx_data_float=}, which is not an int. {self=}"
            raise IndexError(msg)

        idx_data = int(idx_data_float)

        if is_slice_idx:
            max_idx = self.data.size
        else:
            max_idx = self.data.size - 1

        if idx_data > max_idx:
            msg = (
                f"{idx_lai_kaplan=} leads to {idx_data=}, "
                f"which is outside the bounds of `self.data` ({self.data.size=}). "
                f"{self.max_allowed_lai_kaplan_index=}, {self=}"
            )
            raise IndexError(msg)

        return idx_data

    def to_data_step(self, step_lai_kaplan: int | float | None) -> int | None:
        """
        Translate a Lai-Kaplan step into the equivalent step for `self.data`

        Parameters
        ----------
        step_lai_kaplan
            Lai-Kaplan step size

        Returns
        -------
        :
            `self.data` step size
        """
        if step_lai_kaplan is None:
            return None

        step_data_float = step_lai_kaplan / self.lai_kaplan_stride
        if step_data_float % 1.0:
            msg = f"{step_lai_kaplan=} leads to {step_data_float=}, which is not an int. {self=}"
            raise IndexError(msg)

        step_data = int(step_data_float)

        return step_data

    def __getitem__(self, idx_lai_kaplan: int | float | slice) -> T | npt.NDArray[T]:
        """
        Get an item from `self.data` using standard Python indexing

        The trick here is that we can use indexing like in the Lai-Kaplan paper
        and get the correct part of the underlying data array back.
        """
        if isinstance(idx_lai_kaplan, slice):
            idx_data = slice(
                self.to_data_index(idx_lai_kaplan.start, is_slice_idx=True),
                self.to_data_index(idx_lai_kaplan.stop, is_slice_idx=True),
                self.to_data_step(idx_lai_kaplan.step),
            )

        else:
            idx_data = self.to_data_index(idx_lai_kaplan)

        return self.data[idx_data]

    def __setitem__(self, idx_lai_kaplan: int | float | slice, val: T | npt.NDArray[T]) -> None:
        """
        Set an item (or slice) in `self.data` using standard Python indexing

        The trick here is that we can use indexing like in the Lai-Kaplan paper
        and set the correct part of the underlying data array.
        """
        if isinstance(idx_lai_kaplan, slice):
            idx_data = slice(
                self.to_data_index(idx_lai_kaplan.start, is_slice_idx=True),
                self.to_data_index(idx_lai_kaplan.stop, is_slice_idx=True),
                self.to_data_step(idx_lai_kaplan.step),
            )

        else:
            idx_data = self.to_data_index(idx_lai_kaplan)

        self.data[idx_data] = val


@define
class LaiKaplanInterpolator:
    """
    Lai-Kaplan mean-preserving interpolator

    This splits each interval in half,
    then fits a mean-preserving cubic spline across each interval.
    The use of cubic splines means things can go a bit weird, but it is extremely fast.
    The option to specify minimum and maximum bounds for values
    allows you to limit some of the more extreme excursions.
    However, this boundary application is done using a Rymes-Meyers
    (see [`rymes_meyers`][local.mean_preserving_interpolation.rymes_meyers])
    style algorithm, so is much slower.

    See [Lai and Kaplan, J. Atm. Ocn. Tech. 2022](https://doi.org/10.1175/JTECH-D-21-0154.1)
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

    # min_it: int | None = None
    # """
    # Minimum number of iterations to perform
    #
    # If not provided, we use the length of the output data array.
    # """
    #
    # max_it: int = int(1e4)
    # """
    # Maximum number of iterations to perform
    # """
    #
    # progress_bar: bool = True
    # """
    # Whether to show a progress bar for the iterations or not
    # """

    atol: float = 1e-10
    """
    Absolute tolerance for deciding whether the output value means are close to the input means
    """

    rtol: float = 1e-7
    """
    Relative tolerance for deciding whether the output value means are close to the input means
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
        y_at_boundaries = self.get_y_at_boundaries(x_bounds=x_bounds_in, y_in=y_in)

        raise NotImplementedError

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

            current_vals = adjust_mat @ current_vals
            current_vals[0] += y_at_boundaries[0] / 3
            current_vals[-1] += y_at_boundaries[-1] / 3

            group_averages = get_group_averages(
                integrand_x_bounds=x_bounds_out,
                integrand_y=current_vals,
                group_bounds=x_bounds_in,
            )
            corrections = y_in - group_averages
            current_vals = current_vals + corrections[(current_vals_group_indexes,)]

            if self.min_val is not None and (current_vals < self.min_val).any():
                current_vals[np.where(current_vals < self.min_val)] = self.min_val

                group_averages = get_group_averages(
                    integrand_x_bounds=x_bounds_out,
                    integrand_y=current_vals,
                    group_bounds=x_bounds_in,
                )

                # Rymes-Meyers lower-bound "f" correction function
                rm_lb_f_numerator = get_group_integrals(
                    integrand_x_bounds=x_bounds_out,
                    integrand_y=current_vals - y_in[(current_vals_group_indexes,)],
                    group_bounds=x_bounds_in,
                )
                rm_lb_f_denominator = get_group_integrals(
                    integrand_x_bounds=x_bounds_out,
                    integrand_y=current_vals - self.min_val,
                    group_bounds=x_bounds_in,
                )

                rm_lb_f = rm_lb_f_numerator / rm_lb_f_denominator

                if np.isnan(rm_lb_f).any():
                    # If the numerator is also zeor, set to zero and move on
                    rm_lb_f[np.where(rm_lb_f_numerator == 0.0)] = 0.0 * rm_lb_f.u

                    if np.isnan(rm_lb_f).any():
                        # If still nans, raise
                        raise AssertionError()

                group_delta = group_averages - y_in

                rm_lb_correction = rm_lb_f[(current_vals_group_indexes,)] * (current_vals - self.min_val)
                # Only apply deltas where we have overshot
                rm_lb_correction[np.where(group_delta <= 0.0)] = 0.0 * group_averages.u

                current_vals = current_vals - rm_lb_correction

                current_vals[np.where(current_vals < self.min_val)] = self.min_val

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
