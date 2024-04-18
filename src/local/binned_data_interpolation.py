"""
Interpolation of binned data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from local.binning import LAT_BIN_CENTRES, LON_BIN_CENTRES


def check_data_columns_for_binned_data_interpolation(indf: pd.DataFrame) -> None:
    missing = set(indf.columns).difference(
        {"gas", "lat_bin", "lon_bin", "month", "unit", "value", "year"}
    )
    if missing:
        msg = f"Missing required columns: {missing=}"
        raise AssertionError(msg)


def get_round_the_world_grid(inv, is_lon: bool = False):
    if is_lon:
        out = np.hstack([inv - 360, inv, inv + 360])

    else:
        out = np.hstack([inv, inv, inv])

    return out


def interpolate(ymdf: pd.DataFrame, value_column: str = "value") -> pd.DataFrame:
    # Have to be hard-coded to ensure ordering is correct
    spatial_bin_columns = ["lon_bin", "lat_bin"]

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

    return res_linear
