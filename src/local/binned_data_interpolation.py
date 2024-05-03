"""
Interpolation of binned data
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

from local.binning import LAT_BIN_CENTRES, LON_BIN_CENTRES

SPATIAL_BIN_COLUMNS: tuple[str, str] = ("lon_bin", "lat_bin")
"""
Spatial bin columns, including the order.

We need this as a constant so that we can set the dimension order correctly
when creating :obj:`xr.DataArray`.
"""


def get_spatial_dimension_order() -> tuple[str, str]:
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
        {"gas", "lat_bin", "lon_bin", "month", "unit", "value", "year"}
    )
    if missing:
        msg = f"Missing required columns: {missing=}"
        raise AssertionError(msg)


def get_round_the_world_grid(inv, is_lon: bool = False) -> npt.NDArray[np.float64]:
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


def interpolate(ymdf: pd.DataFrame, value_column: str = "value") -> pd.DataFrame:
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
    return res_linear.T


def to_xarray_dataarray(
    bin_averages_df: pd.DataFrame,
    data: list[np.ndarray],
    times: np.ndarray,
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

    if da.isnull().any():  # noqa: PD003
        msg = "da should not contain any null values"
        raise AssertionError(msg)

    return da
