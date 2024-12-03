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
import pint
import pint_xarray
import pooch
import scmdata
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


# %% [markdown]
# ## Action

# %% [markdown]
# There have been no updates since Ivy et al. (2012), so just use CMIP6 data.

# %%
cmip6_concs_fname = pooch.retrieve(
    url="https://aims3.llnl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6/CMIP/UoM/UoM-CMIP-1-2-0/atmos/mon/mole-fraction-of-c8f18-in-air/gr1-GMNHSH/v20160830/mole-fraction-of-c8f18-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_000001-201412.nc",
    known_hash="e173b8128ab081688a0990e81db31c3ed1ed300cad15b9187e5820a1108d8ad5",
    progressbar=True,
)
cmip6_concs_fname

# %%
cmip6_concs = xr.open_dataset(cmip6_concs_fname, decode_times=False)
# Drop out year 0, which breaks everything
cmip6_concs = cmip6_concs.isel(time=range(12, 24180))
cmip6_concs["time"] = cmip6_concs["time"] - 381.5 + 15.5
cmip6_concs["time"].attrs["units"] = "days since 0001-01-01"
cmip6_concs = xr.decode_cf(cmip6_concs, use_cftime=True)
cmip6_concs = cmip6_concs[f"mole_fraction_of_{config_step.gas}_in_air"]
cmip6_concs

# %%
gm = cmip6_concs.sel(sector=0)
nh = cmip6_concs.sel(sector=1)
sh = cmip6_concs.sel(sector=2)

# %% [markdown]
# With c8f18, the latitudinal gradient is just cos of latitude, so we can do this a bit more simply.

# %%
if config_step.gas != "c8f18":
    raise AssertionError

# %%
np.linspace(-1, 1, len(local.binning.LAT_BIN_CENTRES))

# %%
np.cos(np.pi * local.binning.LAT_BIN_CENTRES / 180)

# %%
local.binning.LAT_BIN_CENTRES

# %%
lat_grad_eof = np.linspace(-1, 1, len(local.binning.LAT_BIN_CENTRES)) / np.cos(np.pi * local.binning.LAT_BIN_CENTRES / 180)
lat_grad_eof = lat_grad_eof / (2 * np.mean(lat_grad_eof[int(lat_grad_eof.size / 2):]))
lat_grad_eof

# %%
plt.plot(local.binning.LAT_BIN_CENTRES, lat_grad_eof)

# %%
plt.plot(local.binning.LAT_BIN_CENTRES / np.cos(np.pi * local.binning.LAT_BIN_CENTRES / 180), lat_grad_eof)

# %%
assert False, "Continue from here"
# Calculate the mean of the lat. gradient
# Normalise the mean of the lat. gradient
# Use the NH and SH data to scale the lat. gradient
# Check that adding lat. gradient to GM
# reproduces NH and SH values.
# so the NH and SH values are matched
# Do the regression against emissions data,
# to recover those values too
# # Done?

# %%
np.mean(lat_grad_eof[:int(lat_grad_eof.size / 2)])


# %%
assert False, "switch to download CMIP6 data, extract lat. gradient, then continue"
assert (
    False
), "switch to using CMIP6 emissions to regress lat. gradient score then use that to extend lat. gradient"
rcmip_concentrations_fname = pooch.retrieve(
    url="https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/rcmip-concentrations-annual-means-v5-1-0.csv",
    known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
    progressbar=True,
)
rcmip_concentrations_fname

# %%
rcmip_concs = scmdata.ScmRun(rcmip_concentrations_fname, lowercase_cols=True)
rcmip_concs

# %%
rcmip_concs_to_use_run = rcmip_concs.filter(region="World", scenario="ssp245", year=range(1, 2022 + 1))
rcmip_concs_to_use_run["variable"] = rcmip_concs_to_use_run["variable"].str.lower()
rcmip_concs_to_use_run = rcmip_concs_to_use_run.filter(variable=f"*{config_step.gas}")
if rcmip_concs_to_use_run.shape[0] != 1:
    raise AssertionError

rcmip_concs_to_use = rcmip_concs_to_use_run.timeseries(time_axis="year")
unit = rcmip_concs_to_use.index.get_level_values("unit")[0]
rcmip_concs_to_use

# %%
global_annual_mean = (
    xr.DataArray(
        rcmip_concs_to_use.values.squeeze(),
        dims=("year",),
        coords=dict(year=rcmip_concs_to_use.columns.values),
        attrs={"units": unit},
    )
    .interp(
        year=np.arange(1, 2022 + 1),
        kwargs={"fill_value": rcmip_concs_to_use.values.squeeze()[0]},
    )
    .pint.quantify()
)

global_annual_mean.plot.line()

global_annual_mean

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
# ### Calculate seasonality
#
# You have to use the global-, annual-mean on a yearly time axis otherwise the time-mean of the seasonality over each year is not zero.

# %%
# Assume zero.
obs_network_seasonality = xr.DataArray(
    np.zeros((12, 12)),
    dims=("lat", "month"),
    coords=dict(month=range(1, 13), lat=local.binning.LAT_BIN_CENTRES),
).pint.quantify("dimensionless")
obs_network_seasonality

# %%
obs_network_seasonality.plot()

# %%
seasonality_full = global_annual_mean * obs_network_seasonality
np.testing.assert_allclose(
    seasonality_full.mean("month").data.m,
    0.0,
    atol=1e-6,  # Match the tolerance of our mean-preserving interpolation algorithm
)

# %%
fig, axes = plt.subplots(ncols=2, sharey=True)
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))

local.xarray_time.convert_year_month_to_time(seasonality_full.sel(year=seasonality_full["year"][-6:])).sel(
    lat=[-82.5, 7.5, 82.5]
).plot(x="time", hue="lat", alpha=0.7, ax=axes[0])

local.xarray_time.convert_year_month_to_time(seasonality_full.sel(year=range(1984, 1986))).sel(
    lat=[-82.5, 7.5, 82.5]
).plot(x="time", hue="lat", alpha=0.7, ax=axes[1])

plt.tight_layout()

# %%
local.xarray_time.convert_year_month_to_time(seasonality_full.sel(year=seasonality_full["year"][-6:])).plot(
    x="time", hue="lat", alpha=0.7, col="lat", col_wrap=3, sharey=True
)

# %%
local.xarray_time.convert_year_month_to_time(seasonality_full, calendar="proleptic_gregorian").plot(
    x="time", hue="lat", alpha=0.7, col="lat", col_wrap=3, sharey=True
)

# %% [markdown]
# ### Latitudinal gradient, monthly

# %% [markdown]
# This obviously needs to be updated to an actual calculation.

# %%
latitudinal_gradient_monthly = xr.DataArray(
    np.zeros((global_annual_mean["year"].shape[0], 12, 12)),
    dims=("year", "month", "lat"),
    coords=dict(
        year=global_annual_mean["year"],
        month=range(1, 13),
        lat=local.binning.LAT_BIN_CENTRES,
    ),
).pint.quantify(unit)
latitudinal_gradient_monthly

# %%
# Ensure spatial mean is zero
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
