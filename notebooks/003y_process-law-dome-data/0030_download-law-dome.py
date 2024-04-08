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
# # Law dome ice core - download
#
# Download data from the [Law Dome ice core dataset](https://data.csiro.au/collection/csiro%3A37077v2),
# specifically [this DOI](https://doi.org/10.25919/5bfe29ff807fb).
#
# This notebook doesn't actually do any downloading.
# Instead, we have included the data in the repository.
# All we do here is check we have the expected data.
#
# The reason for including the data in the repository is to faciliate automated testing of the workflow.
# We cannot download the data automatically from CSIRO, because they do not provide a persistent download link
# (and we felt that reverse engineering the link was not in the spirit of things,
# and also likely not possible because CSIRO rotates the permission tokens anyway).
# If you want CSIRO's data, **do not use the data in this repository**.
# Instead, go and download it from [the record](https://doi.org/10.25919/5bfe29ff807fb)
# and then cite the original references therein.

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import openscm_units
import pint
from doit.dependency import get_file_md5
from pydoit_nb.checklist import generate_directory_checklist
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "retrieve_and_process_law_dome_data"

# %% [markdown] editable=true slideshow={"slide_type": ""}
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
# ### Check we have the intended files

# %%
for fp, expected_md5 in config_step.files_md5_sum.items():
    actual_md5 = get_file_md5(fp)
    if not expected_md5 == actual_md5:
        error_msg = (
            f"Unexpected MD5 hash for {fp}. " f"{expected_md5=} " f"{actual_md5=} "
        )
        raise AssertionError(error_msg)

# %%
generate_directory_checklist(config_step.raw_dir)
