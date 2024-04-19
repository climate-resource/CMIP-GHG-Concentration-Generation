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
import openscm_units
import pint
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
tmp = xr.load_dataarray(
    config_step.observational_network_global_annual_mean_file
).pint.quantify()
tmp

# %%
local.xarray_time.convert_year_month_to_time(seasonality_full).plot(
    x="time", hue="lat", alpha=0.7, col="lat", col_wrap=3, sharey=True
)

# %% [markdown]
# ### Save

# %%
assert False, "Work out saving"

# %%
config_step.latitudinal_gradient_file.parent.mkdir(exist_ok=True, parents=True)
lat_gradient_extended.pint.dequantify().to_netcdf(config_step.latitudinal_gradient_file)
lat_gradient_extended
