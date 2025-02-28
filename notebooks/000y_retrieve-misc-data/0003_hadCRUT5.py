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
# # HadCRUT
#
# Retrieve data from [HadCRUT5](https://www.metoffice.gov.uk/hadobs/hadcrut5/index.html).
#
# Use the global-mean file from HadCRUT5 analysis (i.e. including better infilling).

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import openscm_units
import pint
import pooch
from pydoit_nb.checklist import generate_directory_checklist
from pydoit_nb.config_handling import get_config_for_step_id

import local.dependencies
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve_misc_data"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

# %% [markdown]
# ## Action

# %%
url_source = config_step.hadcrut5.download_url
fname = url_source.url.split("/")[-1]

fnames = pooch.retrieve(
    url=url_source.url,
    known_hash=url_source.known_hash,
    fname=fname,
    path=config_step.hadcrut5.raw_dir,
    progressbar=True,
)

# %%
generate_directory_checklist(config_step.hadcrut5.raw_dir)

# %%
import xarray as xr

comment = xr.load_dataset(
    config_step.hadcrut5.raw_dir / config_step.hadcrut5.download_url.url.split("/")[-1]
).attrs["comment"]  # ["tas_mean"].plot.line()
if "1961-1990 climatology" in comment:
    ref_start_year = 1961
    ref_end_year = 1990
else:
    msg = "Unexpected reference period"
    raise AssertionError(msg)

# %%
local.dependencies.save_source_info_to_db(
    db=config.dependency_db,
    source_info=config_step.hadcrut5.source_info,
)
