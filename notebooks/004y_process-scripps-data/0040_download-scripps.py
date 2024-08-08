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
# # Scripps - download
#
# Download data from the [Scripps CO$_2$ program](https://scrippsco2.ucsd.edu/).

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from pathlib import Path

import openscm_units
import pint
import pooch
from pydoit_nb.checklist import generate_directory_checklist
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "retrieve_and_process_scripps_data"

# %% [markdown] editable=true slideshow={"slide_type": ""}
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

# %% [markdown]
# ## Action

# %% [markdown]
# ### Download merged ice core data
#
# We probably won't use this directly, but it is handy to have as a comparison point.

# %%
pooch.retrieve(
    url=config_step.merged_ice_core_data.url,
    known_hash=config_step.merged_ice_core_data.known_hash,
    fname=config_step.merged_ice_core_data.url.split("/")[-1],
    path=config_step.raw_dir,
    progressbar=True,
)

# %% [markdown]
# ### Download station data

# %%
for scripps_source in config_step.station_data:
    outfile = pooch.retrieve(
        url=scripps_source.url_source.url,
        known_hash=scripps_source.url_source.known_hash,
        fname=scripps_source.url_source.url.split("/")[-1],
        path=config_step.raw_dir,
        progressbar=True,
    )
    assert scripps_source.station_code in str(outfile)

# %%
generate_directory_checklist(config_step.raw_dir)
