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
# # C$_4$F$_{10}$-like - Create pieces for gridding
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
import pint
import pint_xarray
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
step: str = "calculate_c4f10_like_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "c4f10"  # config ID to select for this branch

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
global_annual_mean: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_step.global_annual_mean_allyears_file
).pint.quantify()
global_annual_mean

# %%
lat_gradient_eofs_pcs: xr.Dataset = xr.load_dataset(
    config_step.latitudinal_gradient_allyears_pcs_eofs_file
).pint.quantify()
lat_gradient_eofs_pcs

# %% [markdown]
# ### Calculate global-, annual-mean monthly

# %%
global_annual_mean_monthly = local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly(
    global_annual_mean,
    algorithm=LaiKaplanInterpolator(
        get_wall_control_points_y_from_interval_ys=get_wall_control_points_y_linear_with_flat_override_on_left,
        min_val=Quantity(0, "ppt"),
    ),
)
global_annual_mean_monthly

# %%
fig, axes = plt.subplots(ncols=4, figsize=(12, 4))
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
    global_annual_mean_monthly.sel(year=global_annual_mean_monthly["year"].isin(range(1950, 2023))),
    calendar="proleptic_gregorian",
).plot(ax=axes[2])  # type: ignore
local.xarray_time.convert_year_to_time(
    global_annual_mean.sel(year=global_annual_mean["year"].isin(range(1950, 2023))),
    calendar="proleptic_gregorian",
).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[2])

local.xarray_time.convert_year_month_to_time(
    global_annual_mean_monthly.sel(year=global_annual_mean_monthly["year"][-10:]),
    calendar="proleptic_gregorian",
).plot(ax=axes[3])  # type: ignore
local.xarray_time.convert_year_to_time(
    global_annual_mean.sel(year=global_annual_mean_monthly["year"][-10:]),
    calendar="proleptic_gregorian",
).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[3])

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Seasonality
#
# Assumed zero

# %%
obs_network_seasonality = xr.DataArray(
    np.zeros((12, 12)),
    dims=("lat", "month"),
    coords=dict(month=range(1, 13), lat=lat_gradient_eofs_pcs["lat"]),
).pint.quantify("dimensionless")

seasonality_full = global_annual_mean_monthly * obs_network_seasonality
np.testing.assert_allclose(
    seasonality_full.mean("month").data.m,
    0.0,
    atol=1e-6,  # Match the tolerance of our mean-preserving interpolation algorithm
)
seasonality_full

# %% [markdown]
# ### Latitudinal gradient, monthly
#
# We interpolate the PCs, then apply these to the EOFs.
# This is much faster than interpolating each latitude separately
# and ensures that we preserve a spatial-mean of zero
# in our latitudinal gradient.

# %%
pcs_monthly = (
    lat_gradient_eofs_pcs["principal-components"]  # type: ignore
    .groupby("eof", squeeze=True)
    .apply(
        local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly,
        algorithm=LaiKaplanInterpolator(
            get_wall_control_points_y_from_interval_ys=get_wall_control_points_y_linear_with_flat_override_on_left,
            min_val=lat_gradient_eofs_pcs["principal-components"].min().data,
        ),
    )
)
pcs_monthly

# %%
pcs_annual = lat_gradient_eofs_pcs["principal-components"]

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))


local.xarray_time.convert_year_month_to_time(pcs_monthly, calendar="proleptic_gregorian").plot(
    ax=axes[0], hue="eof"
)
local.xarray_time.convert_year_to_time(pcs_annual, calendar="proleptic_gregorian").plot.scatter(
    x="time", hue="eof", zorder=3, alpha=0.5, ax=axes[0]
)

local.xarray_time.convert_year_month_to_time(
    pcs_monthly.sel(year=pcs_monthly["year"][1:10]), calendar="proleptic_gregorian"
).plot(ax=axes[1], hue="eof")
local.xarray_time.convert_year_to_time(
    pcs_annual.sel(year=pcs_monthly["year"][1:10]), calendar="proleptic_gregorian"
).plot.scatter(x="time", hue="eof", zorder=3, alpha=0.5, ax=axes[1])

local.xarray_time.convert_year_month_to_time(
    pcs_monthly.sel(year=pcs_monthly["year"][-10:]), calendar="proleptic_gregorian"
).plot(ax=axes[2], hue="eof")
local.xarray_time.convert_year_to_time(
    pcs_annual.sel(year=pcs_monthly["year"][-10:]), calendar="proleptic_gregorian"
).plot.scatter(x="time", hue="eof", zorder=3, alpha=0.5, ax=axes[2])

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
