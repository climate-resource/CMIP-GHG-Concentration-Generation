"""
Calculations of the latitudinal gradient
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from local.xarray_time import convert_time_to_year_month


def calculate_eofs_pcs(
    lon_mean: xr.DataArray,
    global_mean: xr.DataArray,
) -> tuple[xr.DataArray, xr.Dataset]:
    """
    Calculate EOFs and principal components from longitudinal and global-mean

    Super helpful resource: https://www.ess.uci.edu/~yu/class/ess210b/lecture.5.EOF.all.pdf
    """
    lat_residuals = lon_mean - global_mean
    lat_residuals_annual_mean = convert_time_to_year_month(lat_residuals).mean("month")

    svd_ready = lat_residuals_annual_mean.transpose("year", "lat").pint.dequantify()
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
            year=lat_residuals_annual_mean["year"],
            eof=range(principal_components.shape[1]),
        ),
        attrs=dict(
            description="Principal components for the latitudinal gradient EOFs",
            units="dimensionless",
        ),
    ).pint.quantify()

    xr_eofs_keep = xr.DataArray(
        name="eofs",
        data=eofs,
        dims=["lat", "eof"],
        coords=dict(
            lat=lat_residuals_annual_mean["lat"],
            eof=range(principal_components.shape[1]),
        ),
        attrs=dict(
            description="EOFs for the latitudinal gradient",
            units=lat_residuals_annual_mean.data.units,
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

    return lat_residuals_annual_mean, res
