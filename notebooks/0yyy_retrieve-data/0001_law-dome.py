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

# %%
from doit.dependency import get_file_md5

from local.config import get_config_for_branch_id, load_config_from_file

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
branch: str = "retrieve"

# %% [markdown]
# ## Parameters

# %%
config_file: str = "../../dev-config-absolute.yaml"  # config file
branch_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_branch = get_config_for_branch_id(
    config=config, branch=branch, branch_config_id=branch_config_id
)

# %% [markdown]
# ## Action

# %%
print(f"DOI for dataset: {config_branch.law_dome.doi}")

# %% [markdown]
# ### Check we're using the intended files

# %%
for fp, expected_md5 in config_branch.law_dome.files_md5_sum.items():
    actual_md5 = get_file_md5(fp)
    if not expected_md5 == actual_md5:
        error_msg = (
            f"Unexpected MD5 hash for {fp}. " f"{expected_md5=} " f"{actual_md5=} "
        )
        raise AssertionError(error_msg)
