"""
Interpolation of binned data
"""

from __future__ import annotations

from typing import cast

import cftime
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata  # type: ignore

from local.binning import BINNING_COLUMNS, LAT_BIN_CENTRES, LON_BIN_CENTRES, VALUE_COLUMN

SPATIAL_BIN_COLUMNS: tuple[str, str] = ("lon_bin", "lat_bin")
"""
Spatial bin columns, including the order.

We need this as a constant so that we can set the dimension order correctly
when creating :obj:`xr.DataArray`.
"""


def get_spatial_dimension_order() -> tuple[str, ...]:
    """
    Get the order of spatial dimensions

    Returns
    -------
        Order of spatial dimensions
    """
    return tuple(v.replace("_bin", "") for v in SPATIAL_BIN_COLUMNS)


def check_data_columns_for_binned_data_interpolation(indf: pd.DataFrame) -> None:
    """
    Check that the data columns match those required for binned data interpolation

    Parameters
    ----------
    indf
        :obj:`pd.DataFrame` to check

    Raises
    ------
    AssertionError
        Required columns are missing
    """
    missing = set(indf.columns).difference(
        {"gas", "lat_bin", "lon_bin", "month", "unit", "value", "year", "weight"}
    )
    if missing:
        msg = f"Missing required columns: {missing=}"
        raise AssertionError(msg)


def combine_weighted_ground_and_satellite_bin_averages(
    bin_averages_ground: pd.DataFrame,
    bin_averages_sat: pd.DataFrame,
    weight_column: str = "weight",
) -> pd.DataFrame:
    """
    Combine ground-network and satellite bin averages using inverse-variance weighting

    Ground-network rows are treated as having a flat weight of one each
    (they're already an equally-weighted average across stations - see
    :func:`local.binning.calculate_bin_averages`); satellite rows carry
    their own combined weight, as produced by
    :func:`local.binning.calculate_bin_averages` when called with
    ``weight_column`` set. Where both sources have a row for the same
    ``(gas, unit, year, month, lat_bin, lon_bin)`` bin, this takes a
    weighted mean of the two instead of leaving both as separate points
    (which is what a plain ``pd.concat`` of the two inputs would do,
    handing spatial interpolation two coincident, un-reconciled points
    for the same bin).

    Parameters
    ----------
    bin_averages_ground
        Ground-network bin averages, as produced by
        :func:`local.binning.calculate_bin_averages` (no ``weight_column``).

    bin_averages_sat
        Satellite bin averages, as produced by
        :func:`local.binning.calculate_bin_averages` called with
        ``weight_column`` set to the same name as ``weight_column`` here.

    weight_column
        Name of the combined-weight column in ``bin_averages_sat``.

    Returns
    -------
        Combined bin averages, with (at most) one row per
        ``(gas, unit, year, month, lat_bin, lon_bin)`` bin.
    """
    ground = bin_averages_ground.copy()
    ground[weight_column] = 1.0

    combined = pd.concat([ground, bin_averages_sat])

    def _weighted_bin_average(group: pd.DataFrame) -> pd.Series[float]:
        return cast(
            "pd.Series[float]",
            pd.Series(
                {
                    VALUE_COLUMN: np.average(group[VALUE_COLUMN], weights=group[weight_column]),
                    weight_column: group[weight_column].sum(),
                }
            ),
        )

    out = combined.groupby(BINNING_COLUMNS)[[VALUE_COLUMN, weight_column]].apply(_weighted_bin_average)

    return out.reset_index()


def get_round_the_world_grid(inv: npt.NDArray[np.float64], is_lon: bool = False) -> npt.NDArray[np.float64]:
    """
    Get the grid required for 'round the world' interpolation

    Parameters
    ----------
    inv
        Input values

    is_lon
        Whether the input values represent longitudes or not

    Returns
    -------
        Grid to use for 'round the world' interpolation
    """
    if is_lon:
        out = np.hstack([inv - 360, inv, inv + 360])

    else:
        out = np.hstack([inv, inv, inv])

    return out


def interpolate(ymdf: pd.DataFrame, value_column: str = "value") -> npt.NDArray[np.float64]:
    """
    Interpolate binned values

    Uses 'round the world' interpolation,
    i.e. longitudes interpolate based on values in both directions.

    Parameters
    ----------
    ymdf
        :obj:`pd.DataFrame` on which to do the interpolation.

    value_column
        The column in ``ymdf`` which contains the values in each bin.

    Returns
    -------
        Interpolated values in each bin, derived from ``ymdf``.
    """
    # Have to be hard-coded to ensure ordering is correct
    spatial_bin_columns = list(SPATIAL_BIN_COLUMNS)

    missing_spatial_cols = [c for c in spatial_bin_columns if c not in ymdf.columns]
    if missing_spatial_cols:
        msg = f"{missing_spatial_cols=}"
        raise AssertionError(msg)

    ymdf_spatial_points = ymdf[spatial_bin_columns].to_numpy()

    lon_grid, lat_grid = np.meshgrid(LON_BIN_CENTRES, LAT_BIN_CENTRES)
    # Malte's trick, duplicate the grids so we can go 'round the world' with interpolation
    lon_grid_interp = get_round_the_world_grid(lon_grid, is_lon=True)
    lat_grid_interp = get_round_the_world_grid(lat_grid)

    points_shift_back = ymdf_spatial_points.copy()
    points_shift_back[:, 0] -= 360
    points_shift_forward = ymdf_spatial_points.copy()
    points_shift_forward[:, 0] += 360
    points_interp = np.vstack(
        [
            points_shift_back,
            ymdf_spatial_points,
            points_shift_forward,
        ]
    )
    values_interp = get_round_the_world_grid(ymdf[value_column].to_numpy())

    res_linear_interp = griddata(
        points=points_interp,
        values=values_interp,
        xi=(lon_grid_interp, lat_grid_interp),
        method="linear",
        # fill_value=12.0
    )
    res_linear = res_linear_interp[:, lon_grid.shape[1] : lon_grid.shape[1] * 2]

    # Have to return the transpose to ensure we match the column order specified above.
    # This is super flaky, alter with care!
    return cast(npt.NDArray[np.float64], res_linear.T)


def to_xarray_dataarray(
    bin_averages_df: pd.DataFrame,
    data: list[npt.NDArray[np.float64]],
    times: list[cftime.datetime],
    name: str,
) -> xr.DataArray:
    """
    Create an :obj:`xr.DataArray` from interpolated values

    Parameters
    ----------
    bin_averages_df
        Initial bin averages :obj:`pd.DataFrame`.
        This is just used to extract metadata, e.g. units.

    data
        Data for the array. We assume that this has already been interpolated
        onto a lat, lon grid defined by {py:const}`LAT_BIN_CENTRES`,
        {py:const}`LON_BIN_CENTRES` and {py:func}`get_spatial_dimension_order`.

    times
        Time axis of the data

    name
        Name of the output :obj:`xr.DataArray`

    Returns
    -------
        Created :obj:`xr.DataArray`
    """
    # lat and lon come from the module scope (not best pattern, not the worst)
    units = bin_averages_df["unit"].unique()
    if len(units) > 1:
        msg = f"Need unit conversion, {units=}"
        raise AssertionError(msg)

    unit = units[0]

    da = xr.DataArray(
        name=name,
        data=np.array(data),
        dims=["time", *get_spatial_dimension_order()],
        coords=dict(
            time=times,
            lat=LAT_BIN_CENTRES,
            lon=LON_BIN_CENTRES,
        ),
        attrs=dict(
            description="Interpolated spatial data",
            units=unit,
        ),
    )

    if da.isnull().any():
        msg = "da should not contain any null values"
        raise AssertionError(msg)

    return da
