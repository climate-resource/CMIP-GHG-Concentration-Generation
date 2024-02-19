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
# # Law dome
#
# Retrieve data from the [Law Dome ice core dataset](https://data.csiro.au/collection/csiro%3A37077v2).

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from doit.dependency import get_file_md5
from pydoit_nb.checklist import generate_directory_checklist
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve"

# %% [markdown] editable=true slideshow={"slide_type": ""}
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
print(f"DOI for dataset: {config_step.law_dome.doi}")

# %% [markdown]
# ## Retrieve the data
#
# **Note**: We would like to automate this using the command below. However, the keys are rotated.
#
# ```
# !AWS_ACCESS_KEY_ID=ADROIMK8WFURFMTPLD9A AWS_SECRET_ACCESS_KEY=woGxqR3TD5gmn1/ICDEp9G8iQLhm968IqHtV0rF0 aws s3 cp --endpoint-url https://s3.data.csiro.au --recursive s3://dapprd/000037077v001/ {config_step.law_dome.raw_dir}
# ```
#
# As a result, we just include the data in the repository instead (not perfect, but fine for now and the DOI is above for clarity).

# %% [markdown]
# ### Check we have the intended files

# %%
for fp, expected_md5 in config_step.law_dome.files_md5_sum.items():
    actual_md5 = get_file_md5(fp)
    if not expected_md5 == actual_md5:
        error_msg = (
            f"Unexpected MD5 hash for {fp}. " f"{expected_md5=} " f"{actual_md5=} "
        )
        raise AssertionError(error_msg)

# %%
generate_directory_checklist(config_step.law_dome.raw_dir)
