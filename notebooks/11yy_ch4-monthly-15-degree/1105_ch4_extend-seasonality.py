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
# # CH$_4$ - extend seasonality
#
# Extend the seasonality over the entire time period. For CH$_4$, we assume a relative seasonality hence we do this just by extending the seasonality with the global-mean CH$_4$ concentration.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import openscm_units
import pint
import pint_xarray  # noqa: F401
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.latitudinal_gradient
import local.raw_data_processing
import local.seasonality
import local.xarray_space
import local.xarray_time
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)

Quantity = pint.get_application_registry().Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_ch4_monthly_15_degree"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
seasonality = xr.load_dataarray(
    config_step.observational_network_seasonality_file
).pint.quantify()
seasonality

# %%
global_annual_mean_monthly = xr.load_dataarray(
    config_step.global_annual_mean_monthly_file
).pint.quantify()
global_annual_mean_monthly

# %% [markdown]
# ### Calculate global-, annual-mean

# %%
global_annual_mean = global_annual_mean_monthly.mean("month")
global_annual_mean

# %% [markdown]
# ### Calculate seasonality

# %%
seasonality_full = global_annual_mean * seasonality

fig, axes = plt.subplots(ncols=2, sharey=True)
local.xarray_time.convert_year_month_to_time(
    seasonality_full.sel(year=range(2017, 2023))
).sel(lat=[-82.5, 7.5, 82.5]).plot(x="time", hue="lat", alpha=0.7, ax=axes[0])

local.xarray_time.convert_year_month_to_time(
    seasonality_full.sel(year=range(1984, 1986))
).sel(lat=[-82.5, 7.5, 82.5]).plot(x="time", hue="lat", alpha=0.7, ax=axes[1])

plt.tight_layout()

# %%
local.xarray_time.convert_year_month_to_time(
    seasonality_full.sel(year=range(2015, 2023))
).plot(x="time", hue="lat", alpha=0.7, col="lat", col_wrap=3, sharey=True)

# %%
local.xarray_time.convert_year_month_to_time(seasonality_full).plot(
    x="time", hue="lat", alpha=0.7, col="lat", col_wrap=3, sharey=True
)

# %% [markdown]
# ### Save

# %%
config_step.seasonality_file.parent.mkdir(exist_ok=True, parents=True)
seasonality_full.pint.dequantify().to_netcdf(config_step.seasonality_file)
seasonality_full
