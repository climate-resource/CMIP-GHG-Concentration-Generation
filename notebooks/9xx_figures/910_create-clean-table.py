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

# %% [markdown]
# ## Imports

# %%
import pandas as pd

from local.config import get_config_for_step_id, load_config_from_file

# %% [markdown]
# ## Define step this notebook belongs to

# %%
step: str = "figures"

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
# ### Make clean table

# %% editable=true slideshow={"slide_type": ""}
all_dat: list[pd.DataFrame] = []

for covariance_variation in config.covariance:
    draws = pd.read_csv(covariance_variation.draw_file)
    draws["label"] = f"covariance_{covariance_variation.step_config_id}"
    all_dat.append(draws)

for constraint_variation in config.constraint:
    draws = pd.read_csv(constraint_variation.draw_file)
    draws["label"] = f"constraint_{constraint_variation.step_config_id}"
    all_dat.append(draws)

all_dat_df = pd.concat(all_dat, axis="rows")  # type: ignore # pandas being silly
all_dat_df

# %% editable=true slideshow={"slide_type": ""}
config_step.draw_comparison_table.parent.mkdir(parents=True, exist_ok=True)
all_dat_df.to_csv(config_step.draw_comparison_table, index=False)
config_step.draw_comparison_table
