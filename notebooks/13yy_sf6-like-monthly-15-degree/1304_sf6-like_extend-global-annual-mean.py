# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
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

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

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
step_config_id: str = "sf6"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)


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
global_mean_supplement_files = (
    local.global_mean_extension.get_global_mean_supplement_files(
        gas=config_step.gas, config=config
    )
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
pre_ind_value = config_step.pre_industrial.value  # Quantity(0, "ppt")
pre_ind_year = config_step.pre_industrial.year  # 1950

# %%
if (global_annual_mean_composite["year"] <= pre_ind_year).any():
    pre_ind_years = global_annual_mean_composite["year"][
        np.where(global_annual_mean_composite["year"] <= pre_ind_year)
    ]
    msg = (
        f"You have data before your pre-industrial year, please check. {pre_ind_years=}"
    )
    raise AssertionError(msg)

# %%
pre_ind_part = (
    global_annual_mean_composite.pint.dequantify()
    .interp(
        year=out_years[np.where(out_years <= pre_ind_year)],
        kwargs={
            "fill_value": pre_ind_value.to(global_annual_mean_composite.data.units).m
        },
    )
    .pint.quantify()
)

# %%
global_annual_mean_composite = xr.concat(
    [pre_ind_part, global_annual_mean_composite], "year"
)
global_annual_mean_composite

# %%
SHOW_AFTER_YEAR = 1950
with axes_vertical_split() as axes:
    global_annual_mean_composite.plot.line(ax=axes[0])
    global_annual_mean_composite.plot.scatter(
        ax=axes[0], alpha=1.0, color="tab:orange", marker="x"
    )

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

# %%
global_annual_mean_composite = (
    global_annual_mean_composite.pint.dequantify()
    .interp(year=out_years, method="cubic")
    .pint.quantify()
)
global_annual_mean_composite

# %%
SHOW_AFTER_YEAR = 1950
with axes_vertical_split() as axes:
    global_annual_mean_composite.plot.line(ax=axes[0])
    global_annual_mean_composite.plot.scatter(
        ax=axes[0], alpha=1.0, color="tab:orange", marker="x"
    )

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
allyears_global_annual_mean = local.xarray_space.calculate_global_mean_from_lon_mean(
    tmp
)
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
allyears_global_annual_mean.pint.dequantify().to_netcdf(
    config_step.global_annual_mean_allyears_file
)
allyears_global_annual_mean
