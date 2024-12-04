# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # C$_8$F$_{18}$-like - Create pieces for gridding
#
# Here we create:
#
# - global-, annual-mean, interpolated to monthly timesteps
# - seasonality, extended over all years
# - latitudinal gradient, interpolated to monthly timesteps
#
# Then we check consistency between these pieces.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import cf_xarray.units
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import pint_xarray
import pooch
import tqdm.autonotebook as tqdman
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.latitudinal_gradient
import local.mean_preserving_interpolation
import local.raw_data_processing
import local.seasonality
import local.xarray_space
import local.xarray_time
from local.config import load_config_from_file
from local.mean_preserving_interpolation.lai_kaplan import (
    LaiKaplanInterpolator,
    get_wall_control_points_y_linear_with_flat_override_on_left,
)

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")
cf_xarray.units.units.define("ppt = ppb / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %%
pint_xarray.setup_registry(openscm_units.unit_registry)

QuantityOSCM = openscm_units.unit_registry.Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_c8f18_like_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "c8f18"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

config_historical_emissions = get_config_for_step_id(
    config=config, step="compile_historical_emissions", step_config_id="only"
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load raw data

# %%
historical_emissions = pd.read_csv(config_historical_emissions.complete_historical_emissions_file)
historical_emissions = historical_emissions[
    historical_emissions["variable"] == f"Emissions|{config_step.gas}"
]
if historical_emissions.empty:
    msg = "No data found for gas, check your config"
    raise AssertionError(msg)
historical_emissions

# %% [markdown]
# There have been no updates since Ivy et al. (2012), so just use CMIP6 data.

# %%
cmip6_concs_hist_fname = pooch.retrieve(
    url="https://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/UoM/UoM-CMIP-1-2-0/atmos/yr/mole-fraction-of-c8f18-in-air/gr1-GMNHSH/v20160830/mole-fraction-of-c8f18-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_0000-2014.nc",
    known_hash="56039beef454a49b1bdc65b9cc9ae9640caff8f0751fbbb23603e41aa465701e",
    progressbar=True,
)
cmip6_concs_hist_fname

# %%
cmip6_concs_ssp245_fname = pooch.retrieve(
    url="https://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/ScenarioMIP/UoM/UoM-MESSAGE-GLOBIOM-ssp245-1-2-1/atmos/yr/mole_fraction_of_c8f18_in_air/gr1-GMNHSH/v20181127/mole-fraction-of-c8f18-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-MESSAGE-GLOBIOM-ssp245-1-2-1_gr1-GMNHSH_2015-2500.nc",
    known_hash="e040d2481ac091cf874f7464e4041be394f0a9af3a34a4c27ca6e937bf228073",
    progressbar=True,
)
cmip6_concs_ssp245_fname

# %%
cmip6_concs_hist = xr.open_dataset(cmip6_concs_hist_fname, decode_times=False)
# Drop out year 0, which breaks everything
cmip6_concs_hist = cmip6_concs_hist.isel(time=range(1, 2015))
cmip6_concs_hist["time"] = cmip6_concs_hist["time"] - 381.5 + 15.5
cmip6_concs_hist["time"].attrs["units"] = "days since 0001-01-01"
cmip6_concs_hist = xr.decode_cf(cmip6_concs_hist, use_cftime=True)
cmip6_concs_hist

# %%
cmip6_concs_ssp245 = xr.open_dataset(cmip6_concs_ssp245_fname, use_cftime=True)
cmip6_concs_ssp245 = cmip6_concs_ssp245.sel(time=cmip6_concs_ssp245["time"].dt.year.isin(range(2015, 2024)))
cmip6_concs_ssp245

# %%
cmip6_concs = xr.concat([cmip6_concs_hist, cmip6_concs_ssp245], "time")
cmip6_concs = cmip6_concs[f"mole_fraction_of_{config_step.gas}_in_air"]
cmip6_concs


# %%
def to_da(inv: xr.DataArray, sector: int) -> xr.DataArray:
    """
    Convert the CMIP6 data to an easy to use `xr.DataArray`
    """
    out = inv.sel(sector=sector).drop_vars("sector")
    out.attrs["units"] = "ppt"
    out.name = None
    out = out.rename({"time": "year"})
    out["year"] = [int(v.year) for v in out["year"].values]
    out = out.pint.quantify()
    out.attrs = {}

    return out


# %%
gm = to_da(cmip6_concs, sector=0)
nh = to_da(cmip6_concs, sector=1)
sh = to_da(cmip6_concs, sector=2)

# %%
global_annual_mean = gm

# %% [markdown]
# ### Seasonality
#
# Assumed zero

# %%
obs_network_seasonality = xr.DataArray(
    np.zeros((12, 12)),
    dims=("lat", "month"),
    coords=dict(month=range(1, 13), lat=local.binning.LAT_BIN_CENTRES),
).pint.quantify("dimensionless")

seasonality_full = global_annual_mean * obs_network_seasonality
np.testing.assert_allclose(
    seasonality_full.mean("month").data.m,
    0.0,
    atol=1e-6,  # Match the tolerance of our mean-preserving interpolation algorithm
)
seasonality_full

# %% [markdown]
# ### Latitudinal gradient
#
# With c8f18, the latitudinal gradient is just linear (incorporating a cosine weighting), so we can do this a bit more simply.

# %%
if config_step.gas != "c8f18":
    raise AssertionError

# %%
lat_bin_weights = np.sin(local.binning.LAT_BIN_BOUNDS * np.pi / 180)
lat_bin_weights = np.diff(lat_bin_weights)
lat_bin_weights

# %% [markdown]
# #### EOF

# %%
n_bins = len(local.binning.LAT_BIN_CENTRES)
lat_grad_eof = np.linspace(-1, 1, n_bins) / lat_bin_weights

nh_slice = slice(int(n_bins / 2), n_bins, 1)
sh_slice = slice(0, int(n_bins / 2), 1)
norm = 2 * np.sum(lat_bin_weights[nh_slice] * lat_grad_eof[nh_slice]) / np.sum(lat_bin_weights[nh_slice])
lat_grad_eof = lat_grad_eof / norm

if not np.isclose((lat_bin_weights * lat_grad_eof).sum(), 0.0):
    msg = "Spatial mean of lat. gradient is clearly wrong"
    raise AssertionError(msg)


if not np.isclose(
    np.sum(lat_bin_weights[nh_slice] * lat_grad_eof[nh_slice]) / np.sum(lat_bin_weights[nh_slice]), 0.5
):
    msg = "Norm of lat. gradient is wrong"
    raise AssertionError(msg)

if not np.isclose(
    np.sum(lat_bin_weights[sh_slice] * lat_grad_eof[sh_slice]) / np.sum(lat_bin_weights[sh_slice]), -0.5
):
    msg = "Norm of lat. gradient is wrong"
    raise AssertionError(msg)

lat_grad_eof = xr.DataArray(
    lat_grad_eof,
    dims=("lat",),
    coords={"lat": local.binning.LAT_BIN_CENTRES},
    attrs={"units": "dimensionless"},
).pint.quantify()
lat_grad_eof

# %%
fig, axes = plt.subplots(ncols=2)
axes[0].plot(local.binning.LAT_BIN_CENTRES, lat_grad_eof)
axes[1].plot(local.binning.LAT_BIN_CENTRES / lat_bin_weights, lat_grad_eof)

# %% [markdown]
# #### Principal components

# %%
lat_grad_pc = nh - sh
lat_grad_pc

# %% [markdown]
# #### Check latitudinal gradient over time
#
# Check that this latitudinal gradient recovers the original NH/SH data.

# %%
lat_grad_checker = lat_grad_pc * lat_grad_eof
lat_grad_checker

# %% [markdown]
# Check that this latitudinal gradient recovers the original NH/SH data.

# %%
for lat_sel, ref in (
    (lat_grad_checker["lat"] >= 0, nh),
    (lat_grad_checker["lat"] < 0, sh),
):
    checker = gm + lat_grad_checker.sel(lat=lat_sel)
    checker = checker.to_dataset(name=config_step.gas)
    checker = checker.cf.add_bounds("lat")
    checker["lat_bounds"].attrs["units"] = "degrees"
    checker = checker.pint.quantify()
    checker = local.xarray_space.calculate_area_weighted_mean_latitude_only(checker, [config_step.gas])[
        config_step.gas
    ]

    xr.testing.assert_allclose(checker, ref)

# %% [markdown]
# #### Regress the latitudinal gradient against emissions

# %%
lat_grad_pc

# %%
historical_emissions = historical_emissions.rename({"time": "year"}, axis="columns")
historical_emissions

# %%
regression_years = np.intersect1d(lat_grad_pc["year"], historical_emissions["year"])
regression_years

# %%
unit = historical_emissions["unit"].unique()
assert len(unit) == 1
unit = unit[0]

historical_emissions_xr = xr.DataArray(
    historical_emissions["value"],
    dims=("year",),
    coords=dict(year=historical_emissions["year"]),
    attrs={"units": unit},
).pint.quantify(unit_registry=openscm_units.unit_registry)
historical_emissions_xr = historical_emissions_xr.sel(
    year=historical_emissions_xr["year"] <= regression_years.max()
)


historical_emissions_xr

# %%
historical_emissions_regression_data = historical_emissions_xr.sel(year=regression_years)

historical_emissions_regression_data

# %%
lat_grad_pc_regression = lat_grad_pc.sel(year=regression_years)
lat_grad_pc_regression

# %%
fig, axes = plt.subplots(ncols=2)
historical_emissions_regression_data.plot(ax=axes[0])
lat_grad_pc_regression.plot(ax=axes[1])
fig.tight_layout()

# %%
# The code below fixes the y-intercept to be zero, so that the latitudinal gradient is zero
# when emissions are zero.
# This is fine for gases like SF6, whose pre-industrial concentrations are zero.
# Have to be more careful when the pre-industrial concentrations are non-zero.
x = QuantityOSCM(
    historical_emissions_regression_data.data.m,
    str(historical_emissions_regression_data.data.units),
)
A = x.m[:, np.newaxis]
y = QuantityOSCM(lat_grad_pc_regression.data.m, str(lat_grad_pc_regression.data.units))

res = np.linalg.lstsq(A, y.m, rcond=None)
m = res[0]
m = QuantityOSCM(m, (y / x).units)
c = QuantityOSCM(0.0, y.units)

latitudinal_gradient_pc0_total_emissions_regression = local.regressors.LinearRegressionResult(m=m, c=c)

fig, ax = plt.subplots()
ax.scatter(x.m, y.m, label="raw data")
ax.plot(x.m, (m * x + c).m, color="tab:orange", label="regression")
ax.plot(
    np.hstack([0, x.m]),
    (m * np.hstack([0, x]) + c).m,
    color="tab:orange",
    label="regression-extended",
)
ax.set_ylabel("PC0")
ax.set_xlabel("PRIMAP emissions")
ax.legend()

latitudinal_gradient_pc0_total_emissions_regression

# %% [markdown]
# ### Calculate global-, annual-mean monthly

# %%
global_annual_mean_monthly = local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly(
    global_annual_mean,
    algorithm=LaiKaplanInterpolator(
        get_wall_control_points_y_from_interval_ys=get_wall_control_points_y_linear_with_flat_override_on_left,
        min_val=global_annual_mean.min().data,
    ),
)
global_annual_mean_monthly

# %%
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))

local.xarray_time.convert_year_month_to_time(global_annual_mean_monthly, calendar="proleptic_gregorian").plot(  # type: ignore
    ax=axes[0]
)
local.xarray_time.convert_year_to_time(global_annual_mean, calendar="proleptic_gregorian").plot.scatter(
    x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[0]
)

local.xarray_time.convert_year_month_to_time(
    global_annual_mean_monthly.sel(year=global_annual_mean_monthly["year"][1:10]),
    calendar="proleptic_gregorian",
).plot(ax=axes[1])  # type: ignore
local.xarray_time.convert_year_to_time(
    global_annual_mean.sel(year=global_annual_mean_monthly["year"][1:10]),
    calendar="proleptic_gregorian",
).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[1])

local.xarray_time.convert_year_month_to_time(
    global_annual_mean_monthly.sel(year=global_annual_mean_monthly["year"][-10:]),
    calendar="proleptic_gregorian",
).plot(ax=axes[2])  # type: ignore
local.xarray_time.convert_year_to_time(
    global_annual_mean.sel(year=global_annual_mean_monthly["year"][-10:]),
    calendar="proleptic_gregorian",
).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[2])

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Latitudinal-gradient, monthly

# %%
lat_grad_pc_monthly = local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly(lat_grad_pc)
lat_grad_pc_monthly

# %%
pcs_annual = lat_grad_pc
pcs_monthly = lat_grad_pc_monthly

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))


local.xarray_time.convert_year_month_to_time(pcs_monthly, calendar="proleptic_gregorian").plot(ax=axes[0])
local.xarray_time.convert_year_to_time(pcs_annual, calendar="proleptic_gregorian").plot.scatter(
    x="time", zorder=3, alpha=0.5, ax=axes[0]
)

local.xarray_time.convert_year_month_to_time(
    pcs_monthly.sel(year=pcs_monthly["year"][1:10]), calendar="proleptic_gregorian"
).plot(ax=axes[1])
local.xarray_time.convert_year_to_time(
    pcs_annual.sel(year=pcs_monthly["year"][1:10]), calendar="proleptic_gregorian"
).plot.scatter(x="time", zorder=3, alpha=0.5, ax=axes[1])

local.xarray_time.convert_year_month_to_time(
    pcs_monthly.sel(year=pcs_monthly["year"][-10:]), calendar="proleptic_gregorian"
).plot(ax=axes[2])
local.xarray_time.convert_year_to_time(
    pcs_annual.sel(year=pcs_monthly["year"][-10:]), calendar="proleptic_gregorian"
).plot.scatter(x="time", zorder=3, alpha=0.5, ax=axes[2])

plt.tight_layout()
plt.show()

# %%
latitudinal_gradient_monthly = lat_grad_eof @ lat_grad_pc_monthly

# Ensure spatial mean is zero
tmp = latitudinal_gradient_monthly
tmp.name = "latitudinal-gradient"
np.testing.assert_allclose(
    local.xarray_space.calculate_global_mean_from_lon_mean(tmp).data.to("ppt").m,
    0.0,
    atol=1e-10,
)

latitudinal_gradient_monthly

# %%
tmp = global_annual_mean_monthly + latitudinal_gradient_monthly
if tmp.min() < 0.0:
    msg = (
        "When combining the global values and the latitudinal gradient, "
        f"the minimum value is less than 0.0. {tmp.min()=}"
    )
    print(msg)
    # raise AssertionError(msg)

    atol_close = 1e-8
    print(
        "Trying with a forced update of the latitudinal gradient "
        f"to be zero where the global-mean is within {atol_close} of zero"
    )
    latitudinal_gradient_monthly_candidate = latitudinal_gradient_monthly.copy(deep=True)
    for yr, yr_da in tqdman.tqdm(global_annual_mean_monthly.groupby("year", squeeze=False)):
        for month, month_da in yr_da.groupby("month", squeeze=False):
            if np.isclose(month_da.data.m, 0.0):
                # print(yr)
                latitudinal_gradient_monthly_candidate.loc[{"year": yr, "month": month}] = 0.0
                continue

            # The latitudinal gradient can't be bigger than the global-mean value,
            # because this leads to negative values.
            # This actually points to an issue in the overall workflow,
            # because what we're actually seeing is a disagreement between the
            # concentration assumptions (i.e. pre-industrial values)
            # and the emissions assumptions (which drive the latitudinal gradient).
            # However, fixing this is an issue for the future.
            # We squeeze even harder, to avoid seasonality breaking things too.
            min_grad_val = latitudinal_gradient_monthly_candidate.loc[{"year": yr, "month": month}].min()
            if np.abs(min_grad_val) > month_da * 0.5:
                shrink_ratio = (0.5 * month_da / np.abs(min_grad_val)).squeeze()
                new_val = (
                    shrink_ratio * latitudinal_gradient_monthly_candidate.loc[{"year": yr, "month": month}]
                )

                msg = (
                    "TODO: fix consistency issue. "
                    f"In {yr:04d}-{month:02d}, "
                    f"the minimum latitudinal gradient value is: {min_grad_val.data}. "
                    f"The global-mean value is {month_da.data}. "
                    f"This makes no sense. For now, force overriding to {new_val.data}."
                )
                print(msg)

                latitudinal_gradient_monthly_candidate.loc[{"year": yr, "month": month}] = new_val

    tmp2 = global_annual_mean_monthly + latitudinal_gradient_monthly_candidate
    if tmp2.min() < 0.0:
        msg = "Even after the force update, " f"the minimum value is less than 0.0. {tmp2.min()=}"
        raise AssertionError(msg)

    print("Updated the latitudinal gradient")
    latitudinal_gradient_monthly = latitudinal_gradient_monthly_candidate

    # Ensure spatial mean is still zero
    tmp = latitudinal_gradient_monthly
    tmp.name = "latitudinal-gradient"
    np.testing.assert_allclose(
        local.xarray_space.calculate_global_mean_from_lon_mean(tmp).data.to("ppb").m,
        0.0,
        atol=1e-10,
    )

latitudinal_gradient_monthly

# %%
local.xarray_time.convert_year_month_to_time(
    latitudinal_gradient_monthly, calendar="proleptic_gregorian"
).plot(hue="lat")

# %% [markdown]
# ### Save

# %%
config_step.global_annual_mean_allyears_monthly_file.parent.mkdir(exist_ok=True, parents=True)
global_annual_mean_monthly.pint.dequantify().to_netcdf(config_step.global_annual_mean_allyears_monthly_file)
global_annual_mean_monthly

# %%
config_step.seasonality_allyears_fifteen_degree_monthly_file.parent.mkdir(exist_ok=True, parents=True)
seasonality_full.pint.dequantify().to_netcdf(config_step.seasonality_allyears_fifteen_degree_monthly_file)
seasonality_full

# %% editable=true slideshow={"slide_type": ""}
config_step.latitudinal_gradient_fifteen_degree_allyears_monthly_file.parent.mkdir(
    exist_ok=True, parents=True
)
latitudinal_gradient_monthly.pint.dequantify().to_netcdf(
    config_step.latitudinal_gradient_fifteen_degree_allyears_monthly_file
)
latitudinal_gradient_monthly
