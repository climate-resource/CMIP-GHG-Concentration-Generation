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

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import seaborn as sns  # type: ignore

from local.config import get_config_for_step_id, load_config_from_file

# %% [markdown]
# ## Define step this notebook belongs to

# %%
step: str = "covariance_plotting"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this step

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Make plots
#
# Make quick plots to check difference between covariance draws.

# %% editable=true slideshow={"slide_type": ""}
all_dat: list[pd.DataFrame] = []
for covariance_variation in config.covariance:
    draws = pd.read_csv(covariance_variation.draw_file)
    draws["step_config_id"] = covariance_variation.step_config_id
    all_dat.append(draws)

all_dat_df = pd.concat(all_dat, axis="rows")  # type: ignore # pandas being silly
all_dat_df

# %%
sns.scatterplot(data=all_dat_df, x="x", y="y", hue="step_config_id", alpha=0.5)
