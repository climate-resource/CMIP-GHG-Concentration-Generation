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
import scipy.interpolate  # type: ignore
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.harmonisation
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

config_retrieve_and_process_menking_et_al_2025_data = get_config_for_step_id(
    config=config, step="retrieve_and_process_menking_et_al_2025_data", step_config_id="only"
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
menking_et_al = pd.read_csv(config_retrieve_and_process_menking_et_al_2025_data.processed_data_file)
menking_et_al = menking_et_al[menking_et_al["gas"] == config_step.gas]
menking_et_al["source"] = "menking_et_al_2025"
menking_et_al

# %% [markdown]
# Extend Meking data back to year 1.

# %%
if menking_et_al["year"].min() > 1:
    if menking_et_al["year"].min() > 10:  # noqa: PLR2004
        raise NotImplementedError

    extrap_years = np.arange(1, menking_et_al["year"].min())
    # Use the last 5 years or data to do a linear extrapolation
    loc = menking_et_al["year"] < menking_et_al["year"].min() + 5

    extrap_values = scipy.interpolate.make_interp_spline(
        x=menking_et_al[loc]["year"], y=menking_et_al[loc]["value"], k=1
    )(extrap_years)

    extrap_df = menking_et_al.iloc[: extrap_years.size, :].copy()
    extrap_df["year"] = extrap_years
    extrap_df["value"] = extrap_values
    menking_et_al_full = pd.concat([extrap_df, menking_et_al]).sort_values("year").reset_index(drop=True)

else:
    menking_et_al_full = menking_et_al

menking_et_al_full

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
conc_unit = get_col_assert_single_value(menking_et_al_full, "unit")
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
# #### Menking et al. 2025
#
# Then we use the Menking et al. 2025 data.
# The Menking et al. data for N2O is a global-mean
# and the latitudinal gradient's mean is by construction zero,
# so we don't need to calculate any offset
# (unlike for other gases where these assumptions don't hold).

# %% [markdown]
# ##### Harmonise
#
# Firstly, we harmonise the Meking et al. 2025 data with the observational record
# to avoid jumps as we transition between the two.

# %%
join_year = int(obs_network_years.min())
join_year

# %%
menking_et_al_harmonised = (
    local.harmonisation.get_harmonised_timeseries(  # type: ignore
        ints=menking_et_al_full.set_index(["year", "unit", "gas", "source"])["value"].unstack("year"),
        harm_units=conc_unit,
        harm_value=float(global_annual_mean_obs_network.pint.to(conc_unit).sel(year=join_year).data.m),
        harm_year=join_year,
        n_transition_years=100,
    )
    .stack()
    .to_frame("value")
    .reset_index()
)
menking_et_al_harmonised

# %%
fig, ax = plt.subplots()

global_annual_mean_obs_network.plot(ax=ax, label="Obs network")
ax.plot(
    menking_et_al_full["year"],
    menking_et_al_full["value"],
    label="Menking et al., raw",
)
ax.plot(
    menking_et_al_harmonised["year"],
    menking_et_al_harmonised["value"],
    label="Menking et al., harmonised",
)
ax.legend()
ax.set_xlim((1850, 2030))

# %%
menking_da = xr.DataArray(
    data=menking_et_al_harmonised["value"],
    dims=["year"],
    coords=dict(year=menking_et_al_harmonised["year"]),
    attrs=dict(units=conc_unit),
).pint.quantify()
menking_da

# %%
menking_years_full_field = menking_da + allyears_latitudinal_gradient
menking_years_full_field

# %% [markdown]
# #### Join back together

# %%
allyears_full_field = xr.concat([menking_years_full_field, obs_network_full_field], "year")

allyears_full_field

# %%
allyears_full_field.plot(hue="lat")

# %% [markdown]
# #### Check our full field calculation
#
# There's a lot of steps above, if we have got this right the field will:
#
# - have an annual-average that matches:
#    - the Menking et al. data
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
if not config.ci:
    np.testing.assert_allclose(
        allyears_global_annual_mean.sel(year=menking_et_al_harmonised["year"].values).data.to(conc_unit).m,
        menking_et_al_harmonised["value"],
    )
else:
    compare_years = np.intersect1d(allyears_global_annual_mean.year, menking_et_al_harmonised["year"])
    np.testing.assert_allclose(
        allyears_global_annual_mean.sel(year=compare_years).data.to(conc_unit).m,
        menking_et_al_harmonised[menking_et_al_harmonised["year"].isin(compare_years)]["value"],
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
    local.xarray_space.calculate_global_mean_from_lon_mean(tmp).data.to("ppb").m,  # type: ignore
    0.0,
    atol=1e-10,
)

# %%
allyears_global_annual_mean.plot()  # type: ignore

# %% [markdown]
# ## Save

# %%
config_step.global_annual_mean_allyears_file.parent.mkdir(exist_ok=True, parents=True)
allyears_global_annual_mean.pint.dequantify().to_netcdf(config_step.global_annual_mean_allyears_file)
allyears_global_annual_mean
