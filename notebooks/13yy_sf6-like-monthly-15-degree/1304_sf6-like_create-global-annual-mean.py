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
# # SF$_6$-like - create the global-, annual-mean
#
# Where available, we use data from WMO/Western et al. (2024)/Velders et al. (2022)
# to set our global-mean.
# Then we extend it back in time as needed.
# For SF$_6$-like gases, we do this by combining the values from other data sources
# with an assumption about when zero was reached.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cf_xarray.units
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import openscm_units
import pandas as pd
import pint
import pint_xarray
import scipy.optimize  # type: ignore
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.global_mean_extension
import local.harmonisation
import local.latitudinal_gradient
import local.mean_preserving_interpolation
import local.raw_data_processing
import local.seasonality
import local.xarray_space
import local.xarray_time
from local.config import load_config_from_file

# %% editable=true slideshow={"slide_type": ""}
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")
cf_xarray.units.units.define("ppt = ppb / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %% editable=true slideshow={"slide_type": ""}
QuantityOSCM = openscm_units.unit_registry.Quantity

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_sf6_like_monthly_fifteen_degree_pieces"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "cfc12"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Helper functions


# %% editable=true slideshow={"slide_type": ""}
def get_col_assert_single_value(idf: pd.DataFrame, col: str) -> Any:
    """Get a column's value, asserting that it only has one value"""
    res = idf[col].unique()
    if len(res) != 1:
        raise AssertionError

    return res[0]


# %% editable=true slideshow={"slide_type": ""}
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


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Load data

# %% editable=true slideshow={"slide_type": ""}
global_mean_supplement_config = local.global_mean_extension.get_global_mean_supplement_config(
    gas=config_step.gas, config=config
)

# %% editable=true slideshow={"slide_type": ""}
if global_mean_supplement_config:
    global_mean_supplement_file = [global_mean_supplement_config.processed_data_file]
    global_mean_data = pd.read_csv(global_mean_supplement_file[0])
    global_mean_data = global_mean_data[global_mean_data["gas"] == config_step.gas]

    local.dependencies.save_dependency_into_db(
        db=config.dependency_db,
        gas=config_step.gas,
        dependency_short_name=global_mean_supplement_config.source_info.short_name,
    )

    print(global_mean_data)

# %% editable=true slideshow={"slide_type": ""}
global_annual_mean_obs_network = xr.load_dataarray(  # type: ignore
    config_step.observational_network_global_annual_mean_file
).pint.quantify()
global_annual_mean_obs_network

# %% editable=true slideshow={"slide_type": ""}
lat_grad_eofs_allyears = xr.load_dataset(
    config_step.latitudinal_gradient_allyears_pcs_eofs_file
).pint.quantify()
lat_grad_eofs_allyears

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Create annual-mean latitudinal gradient
#
# This can be combined with our hemispheric information if needed.

# %% editable=true slideshow={"slide_type": ""}
allyears_latitudinal_gradient = (
    lat_grad_eofs_allyears["principal-components"] @ lat_grad_eofs_allyears["eofs"]
)
allyears_latitudinal_gradient

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Define some important constants

# %%
if global_mean_supplement_config:
    max_year = min(
        2023,
        max(
            global_annual_mean_obs_network["year"].max(),
            global_mean_data["year"].max(),
        ),
    )
else:
    max_year = global_annual_mean_obs_network["year"].max()

# %% editable=true slideshow={"slide_type": ""}
if not config.ci:
    out_years = np.arange(1, max_year + 1)

else:
    out_years = np.arange(1750, max_year + 1)

out_years

# %% editable=true slideshow={"slide_type": ""}
obs_network_years = global_annual_mean_obs_network["year"]
obs_network_years

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Initialise

# %% editable=true slideshow={"slide_type": ""}
if global_mean_supplement_config:
    if global_mean_data["year"].max() >= np.max(out_years):
        # Use this global-mean up until our latest year
        tmp = global_mean_data[global_mean_data["year"].isin(out_years)]
        unit = global_mean_data["unit"].unique()
        if len(unit) != 1:
            raise AssertionError
        unit = unit[0]

        global_annual_mean_composite = xr.DataArray(
            tmp["value"], dims=("year",), coords={"year": tmp["year"]}, attrs={"units": unit}
        ).pint.quantify()

    else:
        # Use observations where we have them,
        # then join with the global-mean supplement.
        if config_step.gas not in ["cf4", "c2f6", "c3f8"]:
            msg = "Check this carefully before using"
            raise AssertionError(msg)

        tmp = global_mean_data[global_mean_data["year"].isin(out_years)]
        unit = global_mean_data["unit"].unique()
        if len(unit) != 1:
            raise AssertionError
        unit = unit[0]

        harm_year = float(global_annual_mean_obs_network.year.min())
        harm_value = float(global_annual_mean_obs_network.pint.to(unit).sel(year=harm_year).data.m)

        harmonised = local.harmonisation.get_harmonised_timeseries(
            ints=tmp.set_index(["gas", "year", "unit"])["value"].unstack("year"),
            harm_value=harm_value,
            harm_units=unit,
            harm_year=harm_year,
            n_transition_years=100,
        )
        if harmonised.isnull().any().any():
            raise AssertionError

        harmonised = harmonised.stack("year").to_frame("value").reset_index()

        fig, ax = plt.subplots()

        global_annual_mean_obs_network.plot(ax=ax, label="Obs network")
        ax.plot(
            tmp["year"],
            tmp["value"],
            label="raw",
        )
        ax.plot(
            harmonised["year"],
            harmonised["value"],
            label="harmonised",
        )

        ax.legend()

        global_annual_mean_composite = xr.DataArray(
            np.hstack([harmonised["value"], global_annual_mean_obs_network.pint.to(unit).data.m]),
            dims=("year",),
            coords={"year": np.hstack([harmonised["year"], global_annual_mean_obs_network.year])},
            attrs={"units": unit},
        ).pint.quantify()

else:
    global_annual_mean_composite = global_annual_mean_obs_network.copy()

global_annual_mean_composite

# %% editable=true slideshow={"slide_type": ""}
use_extensions_years = np.setdiff1d(out_years, global_annual_mean_composite.year)
use_extensions_years

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Extend global-, annual-mean
#
# This happens in a few steps.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Use pre-industrial value and time

# %% editable=true slideshow={"slide_type": ""}
pre_ind_value = config_step.pre_industrial.value
pre_ind_value

# %% editable=true slideshow={"slide_type": ""}
pre_ind_year = config_step.pre_industrial.year
pre_ind_year

# %% editable=true slideshow={"slide_type": ""}
if (global_annual_mean_composite["year"] <= pre_ind_year).any():
    pre_ind_years = global_annual_mean_composite["year"][
        np.where(global_annual_mean_composite["year"] <= pre_ind_year)
    ]
    msg = f"You have data before your pre-industrial year, please check. {pre_ind_years=}"
    raise AssertionError(msg)

# %% editable=true slideshow={"slide_type": ""}
pre_ind_part = (
    global_annual_mean_composite.pint.dequantify()
    .interp(
        year=out_years[np.where(out_years <= pre_ind_year)],
        kwargs={"fill_value": pre_ind_value.to(global_annual_mean_composite.data.units).m},
    )
    .pint.quantify()
)

# %% editable=true slideshow={"slide_type": ""}
global_annual_mean_composite = xr.concat([pre_ind_part, global_annual_mean_composite], "year")
global_annual_mean_composite

# %% editable=true slideshow={"slide_type": ""}
SHOW_AFTER_YEAR = min(1950, pre_ind_year - 20)
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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Interpolate between to fill missing values
#
# Here we fit a sigmoidal function to our gap.
# We raise errors if the sigmoid won't fit nicely.

# %% editable=true slideshow={"slide_type": ""}
out_years_still_missing = np.setdiff1d(out_years, global_annual_mean_composite.year)
if out_years_still_missing.size > 1 and np.diff(out_years_still_missing).max() > 1:
    msg = f"More than one gap {out_years_still_missing}"
    raise AssertionError(msg)

print(f"Filling in from {out_years_still_missing[0]} to {out_years_still_missing[-1]}")

# %%
if global_annual_mean_composite.sel(year=out_years_still_missing[0] - 1) == global_annual_mean_composite.sel(
    year=out_years_still_missing[-1] + 1
):
    # Data is already at pre-industrial so we can just keep things flat
    years_pre_gap = np.arange(out_years_still_missing[0] - 3, out_years_still_missing[0])
    years_post_gap = np.arange(
        out_years_still_missing[-1] + 1,
        min(out_years_still_missing[-1] + 2, global_annual_mean_composite.year[-1] + 1),
    )

else:
    # Will need to actually interpolate
    years_pre_gap = np.arange(out_years_still_missing[0] - 3, out_years_still_missing[0])
    years_post_gap = np.arange(
        out_years_still_missing[-1] + 1,
        min(out_years_still_missing[-1] + 4, global_annual_mean_composite.year[-1] + 1),
    )

print(f"{years_pre_gap=}")
print(f"{years_post_gap=}")

# %% editable=true slideshow={"slide_type": ""}
fit_years = np.hstack([years_pre_gap, years_post_gap])
fit_years

# %% editable=true slideshow={"slide_type": ""}
fit_values = global_annual_mean_composite.sel(year=fit_years).data.m
fit_values

# %% editable=true slideshow={"slide_type": ""}
fig, ax = plt.subplots()
ax.scatter(fit_years, fit_values)

# %% editable=true slideshow={"slide_type": ""}
width = (years_post_gap[0] - years_pre_gap[-1]) / 2.0
width

# %% editable=true slideshow={"slide_type": ""}
increase = (fit_values[-1] - fit_values[0]) / 2.0
increase


# %% editable=true slideshow={"slide_type": ""}
def quartic(x: npt.NDArray[Any], a: float, b: float) -> npt.NDArray[Any]:
    """
    Quartic function, locked to make fitting easier
    """
    res = a * ((x - pre_ind_year) / width) ** 4 + b * ((x - pre_ind_year) / width) ** 2

    return res  # type: ignore


# %% editable=true slideshow={"slide_type": ""}
def transform_val_to_scipy(v: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Transform a value to a value that can be used in scipy's fitting

    This is needed because scipy's fitting works best if all parameter values are ~1.0.
    """
    if increase > 0.0:
        return (v - pre_ind_value.m) / (2 * increase)  # type: ignore

    return np.zeros_like(v)


def transform_val_from_scipy(v: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Transform a value to a value on the raw scale

    This is needed because scipy's fitting works best if all parameter values are ~1.0.
    """
    return v * 2 * increase + pre_ind_value.m  # type: ignore


# %% editable=true slideshow={"slide_type": ""}
fit_values_scipy = transform_val_to_scipy(fit_values)
fit_values_scipy

# %% editable=true slideshow={"slide_type": ""}
fig, ax = plt.subplots()
ax.scatter(fit_years, fit_values_scipy)

# %% editable=true slideshow={"slide_type": ""}
p_res = scipy.optimize.curve_fit(
    quartic,
    fit_years,
    fit_values_scipy,
    bounds=(0.0, np.inf),
)
p_fit = p_res[0]
p_fit

# %% editable=true slideshow={"slide_type": ""}
compare_vals = transform_val_from_scipy(quartic(years_post_gap, *p_fit))
interp_vals = transform_val_from_scipy(quartic(out_years_still_missing, *p_fit))

# %% editable=true slideshow={"slide_type": ""}
if (interp_vals < global_annual_mean_composite.sel(year=years_pre_gap[-1]).data.m).any() or (
    interp_vals > 1.1 * global_annual_mean_composite.sel(year=years_post_gap[0]).data.m
).any():
    print("quadratic spline doesn't work")
    workable_interpolation = False
else:
    workable_interpolation = True

# %% editable=true slideshow={"slide_type": ""}
if not workable_interpolation:
    interpolator = scipy.interpolate.interp1d(fit_years, fit_values, kind="quadratic")
    compare_vals = interpolator(years_post_gap)
    interp_vals = interpolator(out_years_still_missing)
    if (interp_vals < global_annual_mean_composite.sel(year=years_pre_gap[-1]).data.m).any() or (
        interp_vals > global_annual_mean_composite.sel(year=years_post_gap[0]).data.m
    ).any():
        print("quadratic spline doesn't work")
    else:
        workable_interpolation = True

# %% editable=true slideshow={"slide_type": ""}
if not workable_interpolation:
    interpolator = scipy.interpolate.CubicSpline(fit_years, fit_values)
    compare_vals = interpolator(years_post_gap)
    interp_vals = interpolator(out_years_still_missing)
    if (interp_vals < global_annual_mean_composite.sel(year=years_pre_gap[-1]).data.m).any() or (
        interp_vals > global_annual_mean_composite.sel(year=years_post_gap[0]).data.m
    ).any():
        print("cubic spline doesn't work")
    else:
        workable_interpolation = True

# %% editable=true slideshow={"slide_type": ""}
if not workable_interpolation:
    msg = "Did not find workable interpolation"
    raise AssertionError(msg)

# %% editable=true slideshow={"slide_type": ""}
fig, ax = plt.subplots()
ax.scatter(fit_years, fit_values)
ax.scatter(years_post_gap, compare_vals)
ax.plot(out_years_still_missing, interp_vals)
ax.axhline(pre_ind_value.m, color="k", linestyle="--")

np.testing.assert_allclose(
    compare_vals,
    global_annual_mean_composite.sel(year=years_post_gap).data.m,
    rtol=2e-1,
    atol=increase * 1e-1,
    err_msg="The fit is clearly not good",
)

# %% editable=true slideshow={"slide_type": ""}
global_annual_mean_composite = (
    global_annual_mean_composite.pint.dequantify().interp(year=out_years, method="linear").pint.quantify()
)
global_annual_mean_composite.loc[{"year": out_years_still_missing}] = (
    interp_vals * global_annual_mean_composite.data.u
)
global_annual_mean_composite

# %% editable=true slideshow={"slide_type": ""}
SHOW_AFTER_YEAR = min(1950, pre_ind_year - 20)
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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Create full field over all years

# %% editable=true slideshow={"slide_type": ""}
allyears_full_field = global_annual_mean_composite + allyears_latitudinal_gradient
allyears_full_field

# %% [markdown] editable=true slideshow={"slide_type": ""}
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
allyears_full_field_renamed = allyears_full_field.copy()
allyears_full_field_renamed.name = "allyears_global_annual_mean"
allyears_global_annual_mean = local.xarray_space.calculate_global_mean_from_lon_mean(
    allyears_full_field_renamed
)

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
xr.testing.assert_allclose(check, allyears_latitudinal_gradient.sel(year=check.year))

# %% editable=true slideshow={"slide_type": ""}
allyears_latitudinal_gradient_renamed = allyears_latitudinal_gradient.copy()
allyears_latitudinal_gradient_renamed.name = "tmp"
np.testing.assert_allclose(
    local.xarray_space.calculate_global_mean_from_lon_mean(allyears_latitudinal_gradient_renamed)
    .data.to("ppb")
    .m,
    0.0,
    atol=1e-10,
)

# %% [markdown]
# ## Save

# %% editable=true slideshow={"slide_type": ""}
config_step.global_annual_mean_allyears_file.parent.mkdir(exist_ok=True, parents=True)
allyears_global_annual_mean.pint.dequantify().to_netcdf(config_step.global_annual_mean_allyears_file)
allyears_global_annual_mean
