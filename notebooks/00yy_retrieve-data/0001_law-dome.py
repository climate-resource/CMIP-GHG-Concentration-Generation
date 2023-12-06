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

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from doit.dependency import get_file_md5

from local.config import load_config_from_file
from local.pydoit_nb.config_handling import get_config_for_step_id

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
# To-do: automate this. For now I have just downloaded the data by hand from the above DOI.
#
# Automation lead: Found this, which is useable (although not sure if keys are rotated are not, if not then can download from DOI by hand)
#
# ```sh
# # TODO: translate into call below and define CSIRO_RAW_DIR appropriately
# AWS_ACCESS_KEY_ID=Q6EU674V7FXBPCRHRL21 AWS_SECRET_ACCESS_KEY=HqgiwG1pgdojWw3kcT63Pwaaq6ebH+zIR50IsMq7 aws s3 cp --endpoint-url https://s3.data.csiro.au --recursive s3://dapprd/000037077v001/ {config_branch.law_dome.raw_files_root_dir}
# ```

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
