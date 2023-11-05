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
# # Covariance plotting
#
# Quick checks of the covariance against each other

# %%
import pandas as pd
import seaborn as sns

from local.config import get_config_for_branch_id, load_config_from_file

# %%
branch: str = "covariance_plotting"

# %%
config_file: str = "../../dev-config-absolute.yaml"  # config file
branch_config_id: str = "only"  # config ID to select for this branch

# %%
config = load_config_from_file(config_file)
config_branch = get_config_for_branch_id(
    config=config, branch=branch, branch_config_id=branch_config_id
)

# %%
all_dat = []
for covariance_variation in config.covariance:
    draws = pd.read_csv(covariance_variation.draw_file)
    draws["branch_config_id"] = covariance_variation.branch_config_id
    all_dat.append(draws)

all_dat = pd.concat(all_dat, axis="rows")
all_dat

# %%
sns.scatterplot(data=all_dat, x="x", y="y", hue="branch_config_id", alpha=0.5)
