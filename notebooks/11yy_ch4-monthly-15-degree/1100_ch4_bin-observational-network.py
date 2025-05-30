# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # CH$_4$ - binning
#
# Bin the observational network.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import openscm_units
import pandas as pd
import pint
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.dependencies
import local.raw_data_processing
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_ch4_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

config_process_noaa_surface_flask_data = get_config_for_step_id(
    config=config,
    step="process_noaa_surface_flask_data",
    step_config_id=config_step.gas,
)
config_process_noaa_in_situ_data = get_config_for_step_id(
    config=config,
    step="process_noaa_in_situ_data",
    step_config_id=config_step.gas,
)
config_process_agage_data_gc_md = get_config_for_step_id(
    config=config,
    step="retrieve_and_extract_agage_data",
    step_config_id=f"{config_step.gas}_gc-md_monthly",
)
config_process_ale_data = get_config_for_step_id(
    config=config, step="retrieve_and_extract_ale_data", step_config_id="monthly"
)
config_process_gage_data = get_config_for_step_id(
    config=config, step="retrieve_and_extract_gage_data", step_config_id="monthly"
)

# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
all_data_l = []
for f, dep_short_names in [
    (
        config_process_noaa_surface_flask_data.processed_monthly_data_with_loc_file,
        local.dependencies.load_source_info_short_names(
            config_process_noaa_surface_flask_data.source_info_short_names_file
        ),
    ),
    (
        config_process_noaa_in_situ_data.processed_monthly_data_with_loc_file,
        local.dependencies.load_source_info_short_names(
            config_process_noaa_in_situ_data.source_info_short_names_file
        ),
    ),
    (
        config_process_agage_data_gc_md.processed_monthly_data_with_loc_file,
        local.dependencies.load_source_info_short_names(
            config_process_agage_data_gc_md.source_info_short_names_file
        ),
    ),
    (
        config_process_ale_data.processed_monthly_data_with_loc_file,
        [config_process_ale_data.source_info.short_name],
    ),
    (
        config_process_gage_data.processed_monthly_data_with_loc_file,
        [config_process_gage_data.source_info.short_name],
    ),
]:
    try:
        all_data_l.append(local.raw_data_processing.read_and_check_binning_columns(f))
    except Exception as exc:
        msg = f"Error reading {f}"
        raise ValueError(msg) from exc

    for dsn in dep_short_names:
        local.dependencies.save_dependency_into_db(
            db=config.dependency_db,
            gas=config_step.gas,
            dependency_short_name=dsn,
        )

all_data = pd.concat(all_data_l)
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
local.binned_data_interpolation.check_data_columns_for_binned_data_interpolation(bin_averages)
assert set(bin_averages["gas"]) == {config_step.gas}

# %%
config_step.processed_bin_averages_file.parent.mkdir(exist_ok=True, parents=True)
bin_averages.to_csv(config_step.processed_bin_averages_file, index=False)
bin_averages
