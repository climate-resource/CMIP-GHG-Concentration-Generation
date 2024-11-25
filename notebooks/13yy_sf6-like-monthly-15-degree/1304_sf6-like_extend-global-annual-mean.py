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
# # SF$_6$-like - extend the global-, annual-mean
#
# Extend the global-, annual-mean from the observational network back in time.
# For SF$_6$, we do this by combining the values from other data sources
# with an assumption about when zero was reached.

# %% [markdown]
# ## Imports

# %%
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cf_xarray.units
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import pint_xarray
import scipy.optimize
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.global_mean_extension
import local.latitudinal_gradient
import local.mean_preserving_interpolation
import local.raw_data_processing
import local.seasonality
import local.xarray_space
import local.xarray_time
from local.config import load_config_from_file

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")
cf_xarray.units.units.define("ppt = ppb / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %%
QuantityOSCM = openscm_units.unit_registry.Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_sf6_like_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "c3f8"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Helper functions


# %%
def get_col_assert_single_value(idf: pd.DataFrame, col: str) -> Any:
    """Get a column's value, asserting that it only has one value"""
    res = idf[col].unique()
    if len(res) != 1:
        raise AssertionError

    return res[0]


# %%
@contextmanager
def axes_vertical_split(
    ncols: int = 2,
) -> Iterator[tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Get two split axes, formatting after exiting the context"""
    fig, axes = plt.subplots(ncols=2)
    if isinstance(axes, matplotlib.axes.Axes):
        raise TypeError(type(axes))

    yield (axes[0], axes[1])

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### Load data

# %%
global_mean_supplement_files = local.global_mean_extension.get_global_mean_supplement_files(
    gas=config_step.gas, config=config
)
global_mean_supplement_files

# %%
if global_mean_supplement_files:
    raise NotImplementedError
    # global_mean_supplements_l = []
    # for f in global_mean_supplement_files:
    #     try:
    #         global_mean_supplements_l.append(
    #             local.raw_data_processing.read_and_check_global_mean_supplementing_columns(
    #                 f
    #             )
    #         )
    #     except Exception as exc:
    #         msg = f"Error reading {f}"
    #         raise ValueError(msg) from exc
    #
    # global_mean_supplements = pd.concat(global_mean_supplements_l)
    # # TODO: add check of gas names to processed data checker
    # # global_mean_supplements["gas"] = global_mean_supplements["gas"].str.lower()
    # assert global_mean_supplements["gas"].unique().tolist() == [config_step.gas]

else:
    global_mean_supplements = None

global_mean_supplements

# %%
global_annual_mean_obs_network = xr.load_dataarray(  # type: ignore
    config_step.observational_network_global_annual_mean_file
).pint.quantify()
global_annual_mean_obs_network

# %%
lat_grad_eofs_allyears = xr.load_dataset(
    config_step.latitudinal_gradient_allyears_pcs_eofs_file
).pint.quantify()
lat_grad_eofs_allyears

# %% [markdown]
# #### Create annual-mean latitudinal gradient
#
# This can be combined with our hemispheric information if needed.

# %%
allyears_latitudinal_gradient = (
    lat_grad_eofs_allyears["principal-components"] @ lat_grad_eofs_allyears["eofs"]
)
allyears_latitudinal_gradient

# %% [markdown]
# ### Define some important constants

# %%
if not config.ci:
    out_years = np.arange(1, global_annual_mean_obs_network["year"].max() + 1)

else:
    out_years = np.arange(1750, global_annual_mean_obs_network["year"].max() + 1)

out_years

# %%
obs_network_years = global_annual_mean_obs_network["year"]
obs_network_years

# %%
use_extensions_years = np.setdiff1d(out_years, obs_network_years)
use_extensions_years

# %% [markdown]
# ### Extend global-, annual-mean
#
# This happens in a few steps.

# %%
global_annual_mean_composite = global_annual_mean_obs_network.copy()

# %% [markdown]
# #### Use other global-mean sources

# %%
if global_mean_supplements is not None:
    raise NotImplementedError
# if (
#     global_mean_supplements is not None
#     and "global" in global_mean_supplements["region"].tolist()
# ):
#     # Add in the global stuff here
#     msg = (
#         "Add some other global-mean sources handling here "
#         "(including what to do in overlap/join periods)"
#     )
#     raise NotImplementedError(msg)

# %% [markdown]
# #### Use other spatial sources

# %%
if global_mean_supplements is not None:
    raise NotImplementedError
# if (
#     global_mean_supplements is not None
#     and (~global_mean_supplements["lat"].isnull()).any()
# ):
#     # Add in the latitudinal information here
#     msg = (
#         "Add some other spatial sources handling here. "
#         "That will need to use the gradient information too "
#         "(including what to do in overlap/join periods)"
#     )
#     raise NotImplementedError(msg)

# %% [markdown]
# #### Use pre-industrial value and time

# %%
pre_ind_value = config_step.pre_industrial.value
pre_ind_year = config_step.pre_industrial.year

# %%
if (global_annual_mean_composite["year"] <= pre_ind_year).any():
    pre_ind_years = global_annual_mean_composite["year"][
        np.where(global_annual_mean_composite["year"] <= pre_ind_year)
    ]
    msg = f"You have data before your pre-industrial year, please check. {pre_ind_years=}"
    raise AssertionError(msg)

# %%
pre_ind_part = (
    global_annual_mean_composite.pint.dequantify()
    .interp(
        year=out_years[np.where(out_years <= pre_ind_year)],
        kwargs={"fill_value": pre_ind_value.to(global_annual_mean_composite.data.units).m},
    )
    .pint.quantify()
)

# %%
global_annual_mean_composite = xr.concat([pre_ind_part, global_annual_mean_composite], "year")
global_annual_mean_composite

# %%
SHOW_AFTER_YEAR = 1950
with axes_vertical_split() as axes:
    global_annual_mean_composite.plot.line(ax=axes[0])
    global_annual_mean_composite.plot.scatter(ax=axes[0], alpha=1.0, color="tab:orange", marker="x")

    global_annual_mean_composite.sel(
        year=global_annual_mean_composite["year"][
            np.where(global_annual_mean_composite["year"] >= SHOW_AFTER_YEAR)
        ]
    ).plot.line(ax=axes[1], color="tab:blue")
    global_annual_mean_composite.sel(
        year=global_annual_mean_composite["year"][
            np.where(global_annual_mean_composite["year"] >= SHOW_AFTER_YEAR)
        ]
    ).plot.scatter(ax=axes[1], alpha=1.0, color="tab:orange", marker="x")


# %% [markdown]
# ### Interpolate between to fill missing values
#
# Here we fit a sigmoidal function to our gap.
# We raise errors if the sigmoid won't fit nicely.


# %%
def sigmoid(x, a, c, x_0, x_delta):
    """
    Sigmoidal function
    """
    res = a / (1 + np.exp(-(x - x_0) / x_delta)) + c

    return res


# %%
out_years_still_missing = np.setdiff1d(out_years, global_annual_mean_composite.year)
if np.diff(out_years_still_missing).max() > 1:
    msg = "More than one gap"
    raise AssertionError(msg)

print(f"Filling in from {out_years_still_missing[0]} to {out_years_still_missing[-1]}")

# %%
years_pre_gap = np.arange(out_years_still_missing[0] - 10, out_years_still_missing[0])
years_pre_gap

# %%
years_post_gap = np.arange(
    out_years_still_missing[-1] + 1,
    min(out_years_still_missing[-1] + 10, global_annual_mean_composite.year[-1] + 1),
)
years_post_gap

# %%
fit_years = np.hstack([years_pre_gap, years_post_gap])
fit_years

# %%
fit_values = global_annual_mean_composite.sel(year=fit_years).data.m
fit_values

# %%
pre_ind_value_m = pre_ind_value.m
pre_ind_value_m

# %%
centre_year = (years_pre_gap[-1] + years_post_gap[0]) / 2.0
centre_year

# %%
width = (years_post_gap[0] - years_pre_gap[-1]) / 2.0
width

# %%
increase = (fit_values[-1] - fit_values[0]) / 2.0
increase


# %%
def transform_yr_to_scipy(y):
    """
    Transform a year value to a value that can be used in scipy's fitting

    This is needed because scipy's fitting works best if all parameter values are ~1.0.
    """
    return (y - centre_year) / width


def transform_yr_from_scipy(y):
    """
    Transform a year value to a value on the raw scale

    This is needed because scipy's fitting works best if all parameter values are ~1.0.
    """
    return y * width + centre_year


# %%
fit_years_scipy = transform_yr_to_scipy(fit_years)
fit_years_scipy


# %%
def transform_val_to_scipy(v):
    """
    Transform a value to a value that can be used in scipy's fitting

    This is needed because scipy's fitting works best if all parameter values are ~1.0.
    """
    return (v - pre_ind_value_m) / (2 * increase)


def transform_val_from_scipy(v):
    """
    Transform a value to a value on the raw scale

    This is needed because scipy's fitting works best if all parameter values are ~1.0.
    """
    return v * 2 * increase + pre_ind_value_m


# %%
fit_values_scipy = transform_val_to_scipy(fit_values)
fit_values_scipy

# %%
fig, ax = plt.subplots()
ax.scatter(fit_years_scipy, fit_values_scipy)

# %%
p_res = scipy.optimize.curve_fit(
    sigmoid,
    fit_years_scipy,
    fit_values_scipy,
)
p_fit = p_res[0]
p_fit

# %%
compare_vals = transform_val_from_scipy(sigmoid(transform_yr_to_scipy(fit_years), *p_fit))
interp_vals = transform_val_from_scipy(sigmoid(transform_yr_to_scipy(out_years_still_missing), *p_fit))

# %%
fig, ax = plt.subplots()
ax.scatter(fit_years, fit_values)
ax.plot(years_pre_gap, transform_val_from_scipy(sigmoid(transform_yr_to_scipy(years_pre_gap), *p_fit)))
ax.plot(years_post_gap, transform_val_from_scipy(sigmoid(transform_yr_to_scipy(years_post_gap), *p_fit)))
ax.plot(out_years_still_missing, interp_vals)

np.testing.assert_allclose(
    compare_vals, fit_values, rtol=1e-2, atol=increase * 1e-4, err_msg="The fit is clearly not good"
)

# %%
global_annual_mean_composite = (
    global_annual_mean_composite.pint.dequantify().interp(year=out_years, method="linear").pint.quantify()
)
global_annual_mean_composite.loc[{"year": out_years_still_missing}] = (
    interp_vals * global_annual_mean_composite.data.u
)
global_annual_mean_composite

# %%
SHOW_AFTER_YEAR = 1850
with axes_vertical_split() as axes:
    global_annual_mean_composite.plot.line(ax=axes[0])
    global_annual_mean_composite.plot.scatter(ax=axes[0], alpha=1.0, color="tab:orange", marker="x")

    global_annual_mean_composite.sel(
        year=global_annual_mean_composite["year"][
            np.where(global_annual_mean_composite["year"] >= SHOW_AFTER_YEAR)
        ]
    ).plot.line(ax=axes[1], color="tab:blue")
    global_annual_mean_composite.sel(
        year=global_annual_mean_composite["year"][
            np.where(global_annual_mean_composite["year"] >= SHOW_AFTER_YEAR)
        ]
    ).plot.scatter(ax=axes[1], alpha=1.0, color="tab:orange", marker="x")

# %% [markdown]
# ### Create full field over all years

# %%
allyears_full_field = global_annual_mean_composite + allyears_latitudinal_gradient
allyears_full_field

# %% [markdown]
# #### Observational network
#
# We start with the field we have from the observational network.

# %%
obs_network_full_field = allyears_latitudinal_gradient + global_annual_mean_obs_network
obs_network_full_field

# %% [markdown]
# #### Check our full field calculation
#
# If we have got this right the field will:
#
# - be decomposable into:
#   - a global-mean timeseries (with dims (year,))
#   - a latitudinal gradient (with dims (year, lat))
#     that has a spatial-mean of zero.
#     This latitudinal gradient should match the latitudinal
#     gradient we calculated earlier.

# %%
tmp = allyears_full_field.copy()
tmp.name = "allyears_global_annual_mean"
allyears_global_annual_mean = local.xarray_space.calculate_global_mean_from_lon_mean(tmp)

# Round appropriately
allyears_global_annual_mean[np.isclose(allyears_global_annual_mean.data.m, 0.0)] = 0.0

if (allyears_global_annual_mean.data.m < 0.0).any():
    msg = "Values less than zero"
    raise AssertionError(msg)

allyears_global_annual_mean

# %% [markdown]
# The residual between our full field and our annual, global-mean
# should just be the latitudinal gradient we started with.

# %%
check = allyears_full_field - allyears_global_annual_mean
xr.testing.assert_allclose(check, allyears_latitudinal_gradient)

# %%
tmp = allyears_latitudinal_gradient.copy()
tmp.name = "tmp"
np.testing.assert_allclose(
    local.xarray_space.calculate_global_mean_from_lon_mean(tmp).data.to("ppb").m,
    0.0,
    atol=1e-10,
)

# %% [markdown]
# ## Save

# %%
config_step.global_annual_mean_allyears_file.parent.mkdir(exist_ok=True, parents=True)
allyears_global_annual_mean.pint.dequantify().to_netcdf(config_step.global_annual_mean_allyears_file)
allyears_global_annual_mean
