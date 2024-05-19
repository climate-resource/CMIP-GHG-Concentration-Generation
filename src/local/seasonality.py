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
    if lon_mean_ym.isnull().any():
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


def calculate_seasonality_change_eofs_pcs(
    lon_mean: xr.DataArray,
    global_mean: xr.DataArray,
) -> tuple[xr.DataArray, xr.Dataset]:
    """
    Calculate EOFs and principal components for seasonality change from longitudinal and global-mean

    Super helpful resource: https://www.ess.uci.edu/~yu/class/ess210b/lecture.5.EOF.all.pdf
    """
    lon_mean_ym = convert_time_to_year_month(lon_mean)
    if lon_mean_ym.isnull().any():
        msg = "Drop out any years with nan data before starting"
        raise AssertionError(msg)

    lon_mean_ym_annual_mean = lon_mean_ym.mean("month")
    lon_mean_ym_monthly_anomalies = lon_mean_ym - lon_mean_ym_annual_mean

    seasonality, _ = calculate_seasonality(lon_mean=lon_mean, global_mean=global_mean)

    seasonality_anomalies = lon_mean_ym_monthly_anomalies - seasonality

    seasonality_anomalies_stacked = seasonality_anomalies.stack(
        {"lat-month": ["lat", "month"]}
    )
    svd_ready = seasonality_anomalies_stacked.transpose(
        "year", "lat-month"
    ).pint.dequantify()

    U, D, Vh = np.linalg.svd(
        svd_ready,
        full_matrices=False,
    )
    # If you take the full SVD, you get back the original matrix
    if not np.allclose(
        svd_ready,
        U @ np.diag(D) @ Vh,
    ):
        msg = "Something wrong with SVD"
        raise AssertionError(msg)

    # Empirical orthogonal functions (each column is an EOF)
    eofs = Vh.T

    # Principal components are the scaling factors on the EOFs
    principal_components = U @ np.diag(D)

    # Similarly, if you use the full EOFs and principal components,
    # you get back the original matrix
    if not np.allclose(
        svd_ready,
        principal_components @ eofs.T,
    ):
        msg = "Something wrong with PC and EOF breakdown"
        raise AssertionError(msg)

    xr_principal_components_keep = xr.DataArray(
        name="principal-components",
        data=principal_components,
        dims=["year", "eof"],
        coords=dict(
            year=svd_ready["year"],
            eof=range(principal_components.shape[1]),
        ),
        attrs=dict(
            description="Principal components for the seasonality change EOFs",
            units="dimensionless",
        ),
    ).pint.quantify()

    xr_eofs_keep = xr.DataArray(
        name="eofs",
        data=eofs,
        dims=["lat-month", "eof"],
        coords={
            "lat-month": svd_ready["lat-month"],
            "eof": range(principal_components.shape[1]),
            "lat": svd_ready["lat"],
            "month": svd_ready["lat"],
        },
        attrs=dict(
            description="EOFs for the seasonality change",
            units=svd_ready.attrs["units"],
        ),
    ).pint.quantify()

    res = xr.merge(
        [xr_eofs_keep, xr_principal_components_keep], combine_attrs="drop_conflicts"
    )

    # One final check
    if not np.allclose(
        svd_ready,
        (res["principal-components"] @ res["eofs"]).pint.dequantify(),
    ):
        msg = "Something wrong with saving as xarray"
        raise AssertionError(msg)

    return seasonality_anomalies, res
