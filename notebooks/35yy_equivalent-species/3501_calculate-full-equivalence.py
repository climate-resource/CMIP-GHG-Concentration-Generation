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

# %% [markdown]
# # Calculate full equivalence
#
# Here we calculate a full equivalent dataset.

# %% [markdown]
# ## Imports

# %%

import cf_xarray.units
import pint_xarray
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")
cf_xarray.units.units.define("ppt = ppb / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "equivalent_species"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "cfc11eq"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_grid_crunching_included_gases = [
    get_config_for_step_id(
        config=config,
        step="crunch_grids",
        step_config_id=gas,
    )
    for gas in config_step.equivalent_component_gases
]


# %% [markdown]
# ## Action

# %% [markdown]
# ### Calculate equivalents

# %%
equivalents = {}
for key, attr_to_grab in (("fifteen_degree", "fifteen_degree_monthly_file"),):
    total = None

    for crunch_gas_config in config_grid_crunching_included_gases:
        break

    equivalents[key] = total
    # Set metadata about components etc. here

# %%
loaded = xr.load_dataarray(  # type: ignore
    getattr(crunch_gas_config, attr_to_grab)
).pint.quantify()

# %%
if total is None:
    total = loaded
else:
    total += loaded

# %% [markdown]
# ### Save

# %%
config_step.fifteen_degree_monthly_file.parent.mkdir(exist_ok=True, parents=True)
equivalents["fifteen_degree"].pint.dequantify().to_netcdf(
    config_step.fifteen_degree_monthly_file
)
equivalents["fifteen_degree"]

# %%
config_step.half_degree_monthly_file.parent.mkdir(exist_ok=True, parents=True)
equivalents["half_degree"].pint.dequantify().to_netcdf(
    config_step.half_degree_monthly_file
)
equivalents["half_degree"]

# %%
config_step.gmnhsh_mean_monthly_file.parent.mkdir(exist_ok=True, parents=True)
equivalents["gmnhsh"].pint.dequantify().to_netcdf(config_step.gmnhsh_mean_monthly_file)
equivalents["gmnhsh"]

# %%
config_step.gmnhsh_mean_annual_file.parent.mkdir(exist_ok=True, parents=True)
equivalents["gmnhsh_annual_mean"].pint.dequantify().to_netcdf(
    config_step.gmnhsh_mean_annual_file
)
equivalents["gmnhsh_annual_mean"]
