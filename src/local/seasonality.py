"""
Calculations of the seasonality
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from local.xarray_time import convert_time_to_year_month


def calculate_seasonality(
    lon_mean: xr.DataArray,
    global_mean: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate seasonality
    """
    lon_mean_ym = convert_time_to_year_month(lon_mean)
    if lon_mean_ym.isnull().any():  # noqa: PD003
        msg = "Drop out any years with nan data before starting"
        raise AssertionError(msg)

    lon_mean_ym_annual_mean = lon_mean_ym.mean("month")
    lon_mean_ym_monthly_anomalies = lon_mean_ym - lon_mean_ym_annual_mean
    lon_mean_ym_monthly_anomalies_year_average = lon_mean_ym_monthly_anomalies.mean(
        "year"
    )
    seasonality = lon_mean_ym_monthly_anomalies_year_average
    relative_seasonality = seasonality / global_mean.mean("time")

    np.testing.assert_allclose(
        seasonality.mean("month").pint.dequantify(), 0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        relative_seasonality.sum("month").pint.dequantify(), 0.0, atol=1e-13
    )

    return seasonality, relative_seasonality
