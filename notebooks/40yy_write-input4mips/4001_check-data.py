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

# %% [markdown]
# # Check data
#
# Here we perform some final data checks before writing our input4MIPs files.
# These are for checks that we haven't put elsewhere/want to apply to every gas.

# %% [markdown]
# ## Imports

# %%
import datetime
from pathlib import Path

import cf_xarray.units
import pint_xarray
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")
cf_xarray.units.units.define("ppt = ppb / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "write_input4mips"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "hfc152a"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

if "eq" in config_step.gas:
    config_crunch_grids = get_config_for_step_id(
        config=config,
        step="crunch_equivalent_species",
        step_config_id=config_step.gas,
    )

else:
    config_crunch_grids = get_config_for_step_id(
        config=config,
        step="crunch_grids",
        step_config_id=config_step.gas,
    )


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
fifteen_degree_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.fifteen_degree_monthly_file
).pint.quantify()

# half_degree_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
#     config_crunch_grids.half_degree_monthly_file
# ).pint.quantify()

global_mean_monthly_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.global_mean_monthly_file
).pint.quantify()

hemispheric_mean_monthly_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.hemispheric_mean_monthly_file
).pint.quantify()

global_mean_annual_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.global_mean_annual_mean_file
).pint.quantify()

hemispheric_mean_annual_data_raw: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_crunch_grids.hemispheric_mean_annual_mean_file
).pint.quantify()


# %%
all_data_arrays = {
    "fifteen_degree": fifteen_degree_data_raw,
    # "half_degree": half_degree_data_raw,
    "global_mean_monthly": global_mean_monthly_data_raw,
    "hemispheric_mean_monthly": hemispheric_mean_monthly_data_raw,
    "global_mean_annual": global_mean_annual_data_raw,
    "hemispheric_mean_annual": hemispheric_mean_annual_data_raw,
}

# %% [markdown]
# ## Check the number of raw arrays

# %%
if len(all_data_arrays) != 5:  # noqa: PLR2004
    raise AssertionError(len(all_data_arrays))

# %% [markdown]
# ## Make sure there are no negative values

# %%
neg_vals = []
for name, arr in all_data_arrays.items():
    if (arr < 0).any():
        neg_vals.append(f"{name} {arr.min().data=:.10f}")

if neg_vals:
    msg = f"Negative values in {neg_vals}"
    raise AssertionError(msg)

# %%
with open(config_step.complete_file_check_data, "w") as fh:
    fh.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

config_step.complete_file_check_data
