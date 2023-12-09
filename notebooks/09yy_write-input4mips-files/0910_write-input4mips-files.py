# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Write files for input4MIPs
#
# - read already processed data off disk
# - set metadata that is universal
# - infer other metadata from data
#     - this is one to speak to Paul about, surely there already tools for this...
# - create complete set
# - write
#
# CSIRO notebook: https://github.com/climate-resource/csiro-hydrogen-esm-inputs/blob/main/notebooks/300_projected_concentrations/330_write-input4MIPs-files.py

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import cf_xarray  # noqa: F401 # required to add cf accessors
import pint_xarray
import xarray as xr
from carpet_concentrations.input4MIPs.dataset import (
    Input4MIPsDataset,
    Input4MIPsMetadata,
)
from openscm_units import unit_registry

from local.config import load_config_from_file
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "write_input4mips"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %%
config_grid = get_config_for_step_id(config=config, step="grid", step_config_id="only")

# %% [markdown]
# ## Action

# %% [markdown]
# ## Set-up unit registry

# %%
pint_xarray.accessors.default_registry = pint_xarray.setup_registry(unit_registry)

# %% [markdown]
# ## Load data
#
# In future, this should load all the gridded data, having already been crunched to global-means, annual-means etc.

# %%
raw_gridded = xr.open_dataset(
    config_grid.processed_data_file, use_cftime=True
).pint.quantify()
raw_gridded

# %%
assert (
    False
), "Introduce infer_metadata or other function to help auto-fill, see also notes at top of notebook"

# %%
Input4MIPsMetadata

# %%
# Input4MIPsDataset?

# %%
raw_gridded.loc[dict(region="World", scenario="historical")][
    ["Atmospheric Concentrations|CO2"]
]

# %%
source_version = "0.1.0"
metadata_junk = Input4MIPsMetadata(
    contact="zebedee.nicholls@climate-resource.com",
    dataset_category="GHGConcentrations",
    frequency="mon",
    further_info_url="TBD",  # TODO: point to website or paper or something
    grid_label="{grid_label}",  # TODO: check this against controlled vocabulary
    institution="Climate Resource, Northcote, Victoria [TODO postcode], Australia [TODO check if this should be CSIRO instead]",
    institution_id="CR",
    nominal_resolution="{nominal_resolution}",
    realm="atmos",
    source_id="{scenario}",
    source_version=source_version,
    target_mip="ScenarioMIP",
    title="{equal-to-source_id}",
    Conventions="CF-1.6",
    activity_id="input4MIPs",
    mip_era="CMIP6",
    source=f"CR {source_version}",
)

# %%
input4mips_ds = Input4MIPsDataset.from_metadata_autoadd_bounds_to_dimensions(
    raw_gridded.loc[dict(region="World", scenario="historical")][
        ["Atmospheric Concentrations|CO2"]
    ],
    dimensions=("lat", "time"),
    metadata=metadata_junk,
)
input4mips_ds

# %%
input4mips_ds.write(Path("test-input4mips"))

# %%
Input4MIPsDataset(raw_gridded).from_metadata_autoadd_bounds_to_dimensions

# %%
Input4MIPsDataset(raw_gridded).write(".")

# %%
