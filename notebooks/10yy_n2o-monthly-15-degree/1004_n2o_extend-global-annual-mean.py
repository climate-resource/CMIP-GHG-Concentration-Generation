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
# # N$_2$O - extend the global-, annual-mean
#
# Extend the global-, annual-mean back in time.
# For N$_2$O, we do this by combining the values from ice cores etc.
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

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %%
QuantityOSCM = openscm_units.unit_registry.Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_n2o_monthly_fifteen_degree_pieces"

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

config_smooth_ghosh_et_al_2023_data = get_config_for_step_id(
    config=config, step="smooth_ghosh_et_al_2023_data", step_config_id="only"
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
smooth_ghosh_et_al = pd.read_csv(config_smooth_ghosh_et_al_2023_data.smoothed_file)
smooth_ghosh_et_al["source"] = "ghosh_et_al_2023"
smooth_ghosh_et_al

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
conc_unit = get_col_assert_single_value(smooth_ghosh_et_al, "unit")
conc_unit

# %% [markdown]
# #### Create annual-mean latitudinal gradient

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
# #### Ghosh et al. 2023
#
# Then we use the Ghosh et al. 2023 data.
# The latitudinal gradient's mean is by construction zero,
# so we don't need to calculate any offset
# (unlike for other gases).

# %%
smooth_ghosh_to_use = smooth_ghosh_et_al[
    (smooth_ghosh_et_al["year"] < float(obs_network_full_field["year"].min()))
    & (smooth_ghosh_et_al["year"] >= 1)
]
ghosh_da = xr.DataArray(
    data=smooth_ghosh_to_use["value"],
    dims=["year"],
    coords=dict(year=smooth_ghosh_to_use["year"]),
    attrs=dict(units=conc_unit),
).pint.quantify()
ghosh_da

# %%
ghosh_years_full_field = ghosh_da + allyears_latitudinal_gradient
ghosh_years_full_field

# %% [markdown]
# #### Join back together

# %%
allyears_full_field = xr.concat([ghosh_years_full_field, obs_network_full_field], "year")

allyears_full_field

# %%
allyears_full_field.plot(hue="lat")

# %% [markdown]
# #### Check our full field calculation
#
# There's a lot of steps above, if we have got this right the field will:
#
# - have an annual-average that matches:
#    - our smoothed Ghosh et al. data
#
# - be decomposable into:
#   - a global-mean timeseries (with dims (year,))
#   - a latitudinal gradient (with dims (year, lat))
#     that has a spatial-mean of zero.
#     This latitudinal gradient should match the latitudinal
#     gradient we calculated earlier.

# %%
allyears_global_annual_mean = local.xarray_space.calculate_global_mean_from_lon_mean(allyears_full_field)

# %%
ghosh_compare_years = smooth_ghosh_to_use["year"].values[
    np.isin(smooth_ghosh_to_use["year"].values, out_years)  # type: ignore
]
np.testing.assert_allclose(
    allyears_global_annual_mean.sel(year=ghosh_compare_years).data.to(conc_unit).m,
    smooth_ghosh_to_use[np.isin(smooth_ghosh_to_use["year"], ghosh_compare_years)]["value"],
)

# %%
if not config.ci:
    if float(allyears_full_field["year"].min()) != 1.0:
        raise AssertionError

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

# %%
allyears_global_annual_mean.plot()

# %%
join_year = int(obs_network_years.min())
fig, ax = plt.subplots()
allyears_global_annual_mean.sel(year=range(join_year - 20, join_year + 21)).plot(ax=ax)
# as smooth a transition as we could hope for I think
ax.axvline(join_year)

# %% [markdown]
# ## Save

# %%
config_step.global_annual_mean_allyears_file.parent.mkdir(exist_ok=True, parents=True)
allyears_global_annual_mean.pint.dequantify().to_netcdf(config_step.global_annual_mean_allyears_file)
allyears_global_annual_mean
