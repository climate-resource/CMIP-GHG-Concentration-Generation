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
# # Create clean table
#
# Ready for easy plotting

# %%
import pandas as pd

from local.config import get_config_for_branch_id, load_config_from_file

# %%
branch: str = "figures"

# %%
config_file: str = "../../dev-config-absolute.yaml"  # config file
branch_config_id: str = "only"  # config ID to select for this branch

# %%
config = load_config_from_file(config_file)
config_branch = get_config_for_branch_id(
    config=config, branch=branch, branch_config_id=branch_config_id
)

# %%
config

# %%
all_dat = []

for covariance_variation in config.covariance:
    draws = pd.read_csv(covariance_variation.draw_file)
    draws["label"] = f"covariance_{covariance_variation.branch_config_id}"
    all_dat.append(draws)

for constraint_variation in config.constraint:
    draws = pd.read_csv(constraint_variation.draw_file)
    draws["label"] = f"constraint_{constraint_variation.branch_config_id}"
    all_dat.append(draws)

all_dat = pd.concat(all_dat, axis="rows")
all_dat

# %%
config_branch.draw_comparison_table.parent.mkdir(parents=True, exist_ok=True)
all_dat.to_csv(config_branch.draw_comparison_table, index=False)
config_branch.draw_comparison_table
