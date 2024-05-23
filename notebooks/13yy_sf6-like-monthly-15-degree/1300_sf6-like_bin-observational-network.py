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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # SF$_6$-like - binning
#
# Bin the observational network.

# %% [markdown]
# ## Imports

# %%
import openscm_units
import pandas as pd
import pint
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.observational_network_binning
import local.raw_data_processing
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_sf6_like_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
config_file: str = "../../ci-config-absolute.yaml"  # config file
step_config_id: str = "hfc134a"  # config ID to select for this branch

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
# ### Load data

# %%
obs_network_input_files = (
    local.observational_network_binning.get_obs_network_binning_input_files(
        gas=config_step.gas, config=config
    )
)
obs_network_input_files

# %%
all_data_l = []
for f in obs_network_input_files:
    try:
        all_data_l.append(local.raw_data_processing.read_and_check_binning_columns(f))
    except Exception as exc:
        msg = f"Error reading {f}"
        raise ValueError(msg) from exc

all_data = pd.concat(all_data_l)
# TODO: add check of gas names to processed data checker
all_data["gas"] = all_data["gas"].str.lower()
all_data = all_data[all_data["gas"] == config_step.gas]
all_data

# %% [markdown]
# ## Bin and average data
#
# - all measurements from a station are first averaged for the month
# - then average over all stations
#     - stations get equal weight
#     - flask/in situ networks (i.e. different measurement methods/techniques)
#       are treated as separate stations i.e. get equal weight
# - this order is best as you have a better chance of avoiding giving different times more weight by accident
#     - properly equally weighting all times in the month would be very hard,
#       because you'd need to interpolate to a super fine grid first (one for future research)


# %%
all_data_with_bins = local.binning.add_lat_lon_bin_columns(all_data)
all_data_with_bins

# %%
print(local.binning.get_network_summary(all_data_with_bins))

# %%
bin_averages = local.binning.calculate_bin_averages(all_data_with_bins)
bin_averages

# %% [markdown]
# ### Save

# %%
local.binned_data_interpolation.check_data_columns_for_binned_data_interpolation(
    bin_averages
)
assert set(bin_averages["gas"]) == {config_step.gas}

# %%
config_step.processed_bin_averages_file.parent.mkdir(exist_ok=True, parents=True)
bin_averages.to_csv(config_step.processed_bin_averages_file, index=False)
bin_averages

# %%
config_step.processed_all_data_with_bins_file.parent.mkdir(exist_ok=True, parents=True)
all_data_with_bins.to_csv(config_step.processed_all_data_with_bins_file, index=False)
all_data_with_bins
