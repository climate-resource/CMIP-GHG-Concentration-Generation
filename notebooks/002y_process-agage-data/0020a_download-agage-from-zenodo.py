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
# # AGAGE - download from Zenodo
#
# Download GAGE data from https://zenodo.org/records/14892947

# %% [markdown]
# ## Imports

# %%
import shutil
from pathlib import Path

import openscm_units
import pint
import pooch
import tqdm.auto
from pydoit_nb.complete import write_complete_file
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve_and_extract_agage_data"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "ch4_gc-md_monthly"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

# %% [markdown]
# ### Download

# %%
url_source = config_step.download_url_zenodo
url_source

# %%
extracted_files = pooch.retrieve(
    url=url_source.url, known_hash=url_source.known_hash, progressbar=True, processor=pooch.Untar()
)
extracted_files[:3]

# %% [markdown]
# ## Put extracted files in the right place

# %%
to_move = [f for f in extracted_files if "data/raw/agage/agage" in f and f.endswith("mon.txt")]
config_step.raw_dir.mkdir(exist_ok=True, parents=True)
for f in tqdm.auto.tqdm(to_move):
    shutil.copy2(f, config_step.raw_dir / Path(f).name)

# %%
write_complete_file(config_step.download_complete_file)
