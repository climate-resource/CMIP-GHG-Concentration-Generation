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
# # CO$_2$ - extend the global-, annual-mean
#
# Extend the global-, annual-mean back in time.
# For CO$_2$, we do this by combining the values from ice cores etc.
# and our latitudinal gradient information.

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

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %%
QuantityOSCM = openscm_units.unit_registry.Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_co2_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_smooth_law_dome_data = get_config_for_step_id(
    config=config, step="smooth_law_dome_data", step_config_id=config_step.gas
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
global_annual_mean_obs_network = xr.load_dataarray(  # type: ignore
    config_step.observational_network_global_annual_mean_file
).pint.quantify()
global_annual_mean_obs_network

# %%
lat_grad_eofs_allyears = xr.load_dataset(
    config_step.latitudinal_gradient_allyears_pcs_eofs_file
).pint.quantify()
lat_grad_eofs_allyears

# %%
smooth_law_dome = pd.read_csv(config_smooth_law_dome_data.smoothed_median_file)
smooth_law_dome = smooth_law_dome[smooth_law_dome["gas"] == config_step.gas]
smooth_law_dome["source"] = "law_dome"
smooth_law_dome

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

# %% [markdown]
# #### Define some constants

# %%
law_dome_lat = get_col_assert_single_value(smooth_law_dome, "latitude")
law_dome_lat

# %%
law_dome_lat_nearest = float(
    lat_grad_eofs_allyears.sel(lat=law_dome_lat, method="nearest")["lat"]
)
law_dome_lat_nearest

# %%
conc_unit = get_col_assert_single_value(smooth_law_dome, "unit")
conc_unit

# %% [markdown]
# #### Create annual-mean latitudinal gradient
#
# This is then combined with our ice core information.

# %%
allyears_latitudinal_gradient = (
    lat_grad_eofs_allyears["principal-components"] @ lat_grad_eofs_allyears["eofs"]
)
allyears_latitudinal_gradient

# %% [markdown]
# ### Create full field over all years

# %% [markdown]
# #### Observational network
#
# We start with the field we have from the observational network.

# %%
obs_network_full_field = allyears_latitudinal_gradient + global_annual_mean_obs_network
obs_network_full_field

# %% [markdown]
# #### Law Dome
#
# Then we use the Law Dome data.
# We calculate the offset by ensuring that the value in Law Dome's bin
# matches our smoothed Law Dome timeseries in the years in which we have Law Dome data
# and don't have the observational network.

# %%
smooth_law_dome_to_use = smooth_law_dome[
    smooth_law_dome["year"] < float(obs_network_full_field["year"].min())
]
law_dome_da = xr.DataArray(
    data=smooth_law_dome_to_use["value"],
    dims=["year"],
    coords=dict(year=smooth_law_dome_to_use["year"]),
    attrs=dict(units=conc_unit),
).pint.quantify()
law_dome_da

# %%
offset = law_dome_da - allyears_latitudinal_gradient.sel(
    lat=law_dome_lat, method="nearest"
)
offset

# %%
law_dome_years_full_field = allyears_latitudinal_gradient + offset
law_dome_years_full_field

# %% [markdown]
# #### Join back together

# %%
mostyears_full_field = xr.concat(
    [law_dome_years_full_field, obs_network_full_field], "year"
)

mostyears_full_field

# %%
mostyears_full_field.plot(hue="lat")

# %% [markdown]
# #### Check our full field calculation
#
# There's a lot of steps above, if we have got this right the field will:
#
# - have an annual-average that matches:
#    - our smoothed Law Dome in the Law Dome latitude
#      (for the years of the smoothed Law Dome timeseries)
#
# - be decomposable into:
#   - a global-mean timeseries (with dims (year,))
#   - a latitudinal gradient (with dims (year, lat))
#     that has a spatial-mean of zero.
#     This latitudinal gradient should match the latitudinal
#     gradient we calculated earlier.

# %%
if not config.ci:
    np.testing.assert_allclose(
        mostyears_full_field.sel(lat=law_dome_lat, method="nearest")
        .sel(year=smooth_law_dome_to_use["year"].values)
        .data.to(conc_unit)
        .m,
        smooth_law_dome_to_use["value"],
    )
else:
    law_dome_compare_years = smooth_law_dome_to_use["year"].values[
        np.isin(smooth_law_dome_to_use["year"].values, out_years)  # type: ignore
    ]
    np.testing.assert_allclose(
        mostyears_full_field.sel(lat=law_dome_lat, method="nearest")
        .sel(year=law_dome_compare_years)
        .data.to(conc_unit)
        .m,
        smooth_law_dome_to_use[
            np.isin(smooth_law_dome_to_use["year"], law_dome_compare_years)
        ]["value"],
    )

# %%
tmp = mostyears_full_field.copy()
tmp.name = "mostyears_global_annual_mean"
mostyears_global_annual_mean = local.xarray_space.calculate_global_mean_from_lon_mean(
    tmp
)
mostyears_global_annual_mean

# %% [markdown]
# #### Extending back to year 1
#
# We simply assume that global-mean concentrations are constant
# before the start of the Law Dome record.

# %%
back_extend_years = np.setdiff1d(
    out_years[np.where(out_years < mostyears_global_annual_mean["year"].values[-1])],
    mostyears_global_annual_mean["year"],
)
back_extend_years

# %%
if back_extend_years.size > 0:
    tmp = mostyears_global_annual_mean.sel(
        year=[mostyears_global_annual_mean["year"][0]]
    )
    back_extended_global_annual_mean = (
        mostyears_global_annual_mean.pint.dequantify()
        .interp(year=back_extend_years, kwargs={"fill_value": tmp.data[0].m})
        .pint.quantify()
    )
    allyears_global_annual_mean = (
        mostyears_global_annual_mean.pint.dequantify()
        .interp(year=out_years, kwargs={"fill_value": tmp.data[0].m})
        .pint.quantify()
    )

    back_extended_full_field = (
        allyears_latitudinal_gradient.sel(year=back_extend_years)
        + back_extended_global_annual_mean
    )
    allyears_full_field = xr.concat(
        [back_extended_full_field, mostyears_full_field], "year"
    )


else:
    allyears_full_field = mostyears_full_field
    allyears_global_annual_mean = mostyears_global_annual_mean

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
