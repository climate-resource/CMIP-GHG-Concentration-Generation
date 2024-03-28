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
# # Natural earth shape files
#
# Download natural earth shape files to help with later plotting.

# %% [markdown]
# ## Imports

# %%

import pooch
from pydoit_nb.checklist import generate_directory_checklist
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

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
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %% [markdown]
# ## Action

# %%
for url_source in config_step.natural_earth.download_urls:
    fname = url_source.url.split("/")[-1]

    fnames = pooch.retrieve(
        url=url_source.url,
        known_hash=url_source.known_hash,
        fname=fname,
        path=config_step.natural_earth.raw_dir,
        processor=pooch.Unzip(extract_dir=config_step.natural_earth.raw_dir),
        progressbar=True,
    )

# %%
generate_directory_checklist(config_step.natural_earth.raw_dir)
