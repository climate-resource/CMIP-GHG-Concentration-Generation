"""
Calculations of the seasonality
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from attrs import define

from local.mean_preserving_interpolation import interpolate_annual_mean_to_monthly
from local.regressors import LinearRegressionResult
from local.xarray_time import convert_time_to_year_month


def calculate_seasonality(
    lon_mean: xr.DataArray,
    global_mean: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Calculate seasonality
    """
    lon_mean_ym = convert_time_to_year_month(lon_mean)
    if lon_mean_ym.isnull().any():
        msg = "Drop out any years with nan data before starting"
        raise AssertionError(msg)

    lon_mean_ym_annual_mean = lon_mean_ym.mean("month")
    lon_mean_ym_annual_mean_monthly = lon_mean_ym_annual_mean.groupby("lat", squeeze=False).apply(  # type: ignore
        interpolate_annual_mean_to_monthly,
    )
    lon_mean_ym_monthly_anomalies = lon_mean_ym - lon_mean_ym_annual_mean_monthly

    lon_mean_ym_monthly_anomalies_year_average = lon_mean_ym_monthly_anomalies.mean("year")

    seasonality = lon_mean_ym_monthly_anomalies_year_average
    relative_seasonality = seasonality / global_mean.mean("time")

    # TODO: dial this back down
    # atol = max(1e-6 * global_mean.mean().data.m, 1e-7)
    atol = max(1e-1 * global_mean.mean().data.m, 5e-2)
    np.testing.assert_allclose(seasonality.mean("month").pint.dequantify(), 0.0, atol=atol)
    np.testing.assert_allclose(relative_seasonality.sum("month").pint.dequantify(), 0.0, atol=atol)

    return seasonality, relative_seasonality, lon_mean_ym_monthly_anomalies


def calculate_seasonality_change_eofs_pcs(
    lon_mean: xr.DataArray,
    global_mean: xr.DataArray,
) -> tuple[xr.DataArray, xr.Dataset]:
    """
    Calculate EOFs and principal components for seasonality change from longitudinal and global-mean

    Super helpful resource: https://www.ess.uci.edu/~yu/class/ess210b/lecture.5.EOF.all.pdf
    """
    seasonality, _, lon_mean_ym_monthly_anomalies = calculate_seasonality(
        lon_mean=lon_mean, global_mean=global_mean
    )

    seasonality_anomalies = lon_mean_ym_monthly_anomalies - seasonality

    seasonality_anomalies_stacked = seasonality_anomalies.stack({"lat-month": ["lat", "month"]})
    svd_ready = seasonality_anomalies_stacked.transpose("year", "lat-month").pint.dequantify()

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

    res = xr.merge([xr_eofs_keep, xr_principal_components_keep], combine_attrs="drop_conflicts")

    # One final check
    if not np.allclose(
        svd_ready,
        (res["principal-components"] @ res["eofs"]).pint.dequantify(),
    ):
        msg = "Something wrong with saving as xarray"
        raise AssertionError(msg)

    return seasonality_anomalies, res


@define
class CO2SeasonalityChangeRegression:
    """
    CO2 seasonality change regression

    Holds both the methods for the calculation and the regression result.
    """

    ref_period_start: int = 1850
    """Start of the reference period for the composite's inputs"""

    ref_period_end: int = 1880
    """End of the reference period for the composite's inputs"""

    norm_period_start: int = 2000
    """Start of the period over which the composite should have a value of 1"""

    norm_period_end: int = 2010
    """End of the period over which the composite should have a value of 1"""

    temperature_smoothing_window: int = 15
    """Smoothing window to use for temperature smoothing"""

    temperature_smoothing_min_values: int = 5
    """Minimum number of values to use in the temperature smoothing window"""

    regression_result: LinearRegressionResult | None = None
    """Regression result"""

    def get_normed_ts(self, ts: xr.DataArray) -> xr.DataArray:
        """
        Get normed timeseries

        Parameters
        ----------
        ts
            Timeseries to norm

        Returns
        -------
            Normed timeseries according to self's attributes.
        """
        ts_ref_period_mean = ts.sel(year=range(self.ref_period_start, self.ref_period_end + 1)).mean()
        ts_norm_period_mean = ts.sel(year=range(self.norm_period_start, self.norm_period_end + 1)).mean()

        ts_normed = (ts - ts_ref_period_mean) / (ts_norm_period_mean - ts_ref_period_mean)

        np.testing.assert_allclose(
            1.0,
            ts_normed.sel(year=range(self.norm_period_start, self.norm_period_end + 1))
            .mean()
            .pint.dequantify(),
        )

        return ts_normed

    def get_composite(self, temperatures: xr.DataArray, concentrations: xr.DataArray) -> xr.DataArray:
        """
        Get composite timeseries

        Parameters
        ----------
        temperatures
            Temperatures. These should be raw (i.e. not smoothed).

        concentrations
            Annual-, global-mean CO$_2$ concentrations.

        Returns
        -------
            Composite timeseries to use in the regresssion.
        """
        # The paper and code are super unclear about what was done with temperature smoothing.
        # This guess is as good as any.
        temperatures_smoothed = temperatures.rolling(
            year=self.temperature_smoothing_window,
            center=True,
            min_periods=self.temperature_smoothing_min_values,
        ).mean()
        if temperatures_smoothed.isnull().any():
            msg = "NaN in temperatures_smoothed"
            raise AssertionError(msg)

        temperatures_normed = self.get_normed_ts(temperatures_smoothed)
        concentrations_normed = self.get_normed_ts(concentrations)

        res = (temperatures_normed + concentrations_normed) / 4 + (
            temperatures_normed * concentrations_normed
        ) / 2

        return res
