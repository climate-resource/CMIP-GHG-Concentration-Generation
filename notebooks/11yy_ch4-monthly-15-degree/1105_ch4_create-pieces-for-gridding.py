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
# # CH$_4$ - Create pieces for gridding
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
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.latitudinal_gradient
import local.mean_preserving_interpolation
import local.raw_data_processing
import local.xarray_space
import local.xarray_time
from local.config import load_config_from_file

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_ch4_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
global_annual_mean: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_step.global_annual_mean_allyears_file
).pint.quantify()
global_annual_mean

# %%
obs_network_seasonality: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_step.observational_network_seasonality_file
).pint.quantify()
obs_network_seasonality

# %%
lat_gradient_eofs_pcs: xr.Dataset = xr.load_dataset(
    config_step.latitudinal_gradient_allyears_pcs_eofs_file
).pint.quantify()
lat_gradient_eofs_pcs

# %% [markdown]
# ### Calculate global-, annual-mean monthly

# %%
# from local.mean_preserving_interpolation import mean_preserving_interpolation

# y_in = global_annual_mean.data

# %%
# y_in.u._REGISTRY.Quantity

# %%
# global_annual_mean.year.to_numpy()

# %%
# x_bounds_in = Quantity(
#     np.hstack([global_annual_mean.year.values, global_annual_mean.year.values[-1] + 1.0]), "yr"
# )
# x_bounds_in

# %%
# x_bounds_out = Quantity(np.round(np.arange(x_bounds_in[0].m, x_bounds_in[-1].m + 1 / 24, 1 / 12), 4), "yr")
# x_bounds_out.to("yr")

# %%
# monthly_vals = mean_preserving_interpolation(
#     x_bounds_in=x_bounds_in,
#     y_in=y_in,
#     x_bounds_out=x_bounds_out,
#     algorithm="lai_kaplan",
#     verify_output_is_mean_preserving=True,
# )
# monthly_vals

# %%
# month_out = (x_bounds_out[1:] + x_bounds_out[:-1]) / 2.0
# month_out

# %%
# import cftime

# N_MONTHS_PER_YEAR = 12
# time = [
#     cftime.datetime(
#         np.floor(time_val),
#         np.round(N_MONTHS_PER_YEAR * (time_val % 1 + 1 / N_MONTHS_PER_YEAR / 2)),
#         1,
#     )
#     for time_val in month_out.m
# ]
# time[:24]

# %%
# out = xr.DataArray(
#     data=monthly_vals,
#     dims=["time"],
#     coords=dict(time=time),
# )
# out

# %%
# from local.xarray_time import convert_time_to_year_month

# %%
# global_annual_mean_monthly = convert_time_to_year_month(out)
# global_annual_mean_monthly

# %%
global_annual_mean_monthly = local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly(
    global_annual_mean
)
global_annual_mean_monthly

# %%
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))

local.xarray_time.convert_year_month_to_time(global_annual_mean_monthly).plot(  # type: ignore
    ax=axes[0]
)
local.xarray_time.convert_year_to_time(global_annual_mean).plot.scatter(
    x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[0]
)

local.xarray_time.convert_year_month_to_time(
    global_annual_mean_monthly.sel(year=global_annual_mean_monthly["year"][1:10])
).plot(ax=axes[1])  # type: ignore
local.xarray_time.convert_year_to_time(
    global_annual_mean.sel(year=global_annual_mean_monthly["year"][1:10])
).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[1])

local.xarray_time.convert_year_month_to_time(
    global_annual_mean_monthly.sel(year=global_annual_mean_monthly["year"][-10:])
).plot(ax=axes[2])  # type: ignore
local.xarray_time.convert_year_to_time(
    global_annual_mean.sel(year=global_annual_mean_monthly["year"][-10:])
).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[2])

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Calculate seasonality
#
# You have to use the global-, annual-mean on a yearly time axis otherwise the time-mean of the seasonality over each year is not zero.

# %%
seasonality_full = global_annual_mean * obs_network_seasonality
atol = (
    # # TODO: dial this back down
    # 1e-6 * global_annual_mean.mean().data.m
    1e-1 * global_annual_mean.mean().data.m
)  # Approximately match the tolerance of our mean-preserving interpolation algorithm
np.testing.assert_allclose(
    seasonality_full.mean("month").data.m,
    0.0,
    atol=atol,
)

# %%
fig, axes = plt.subplots(ncols=2, sharey=True)
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))

local.xarray_time.convert_year_month_to_time(  # type: ignore
    seasonality_full.sel(year=seasonality_full["year"][-6:])
).sel(lat=[-82.5, 7.5, 82.5]).plot(x="time", hue="lat", alpha=0.7, ax=axes[0])

local.xarray_time.convert_year_month_to_time(  # type: ignore
    seasonality_full.sel(year=range(1984, 1986))
).sel(lat=[-82.5, 7.5, 82.5]).plot(x="time", hue="lat", alpha=0.7, ax=axes[1])

plt.tight_layout()

# %%
local.xarray_time.convert_year_month_to_time(  # type: ignore
    seasonality_full.sel(year=seasonality_full["year"][-6:])
).plot(x="time", hue="lat", alpha=0.7, col="lat", col_wrap=3, sharey=True)

# %%
local.xarray_time.convert_year_month_to_time(seasonality_full)["time"]

# %%
local.xarray_time.convert_year_month_to_time(seasonality_full).plot(  # type: ignore
    x="time", hue="lat", alpha=0.7, col="lat", col_wrap=3, sharey=True
)

# %% [markdown]
# ### Latitudinal gradient, monthly
#
# We interpolate the PCs, then apply these to the EOFs.
# This is much faster than interpolating each latitude separately
# and ensures that we preserve a spatial-mean of zero
# in our latitudinal gradient.

# %%
# %pdb

# %%
pcs_monthly = (
    lat_gradient_eofs_pcs["principal-components"]  # type: ignore
    .groupby("eof", squeeze=True)
    .apply(
        local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly,
    )
)
pcs_monthly

# %%
pcs_annual = lat_gradient_eofs_pcs["principal-components"]

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))


local.xarray_time.convert_year_month_to_time(pcs_monthly).plot(ax=axes[0], hue="eof")
local.xarray_time.convert_year_to_time(pcs_annual).plot.scatter(
    x="time", hue="eof", zorder=3, alpha=0.5, ax=axes[0]
)

local.xarray_time.convert_year_month_to_time(pcs_monthly.sel(year=pcs_monthly["year"][1:10])).plot(
    ax=axes[1], hue="eof"
)
local.xarray_time.convert_year_to_time(pcs_annual.sel(year=pcs_monthly["year"][1:10])).plot.scatter(
    x="time", hue="eof", zorder=3, alpha=0.5, ax=axes[1]
)

local.xarray_time.convert_year_month_to_time(pcs_monthly.sel(year=pcs_monthly["year"][-10:])).plot(
    ax=axes[2], hue="eof"
)
local.xarray_time.convert_year_to_time(pcs_annual.sel(year=pcs_monthly["year"][-10:])).plot.scatter(
    x="time", hue="eof", zorder=3, alpha=0.5, ax=axes[2]
)

plt.tight_layout()
plt.show()

# %%
latitudinal_gradient_monthly = pcs_monthly @ lat_gradient_eofs_pcs["eofs"]

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
local.xarray_time.convert_year_month_to_time(latitudinal_gradient_monthly).plot(hue="lat")

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

# %%
config_step.latitudinal_gradient_fifteen_degree_allyears_monthly_file.parent.mkdir(
    exist_ok=True, parents=True
)
latitudinal_gradient_monthly.pint.dequantify().to_netcdf(
    config_step.latitudinal_gradient_fifteen_degree_allyears_monthly_file
)
latitudinal_gradient_monthly
