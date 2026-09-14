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
# # CO$_2$ - binning
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
import local.raw_data_processing
import local.train_test_split
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_co2_monthly_fifteen_degree_pieces"

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

# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %% editable=true slideshow={"slide_type": ""}
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
# all_data

# %% [markdown]
# all_data_with_sat = pd.concat([all_data, df])
# all_data_with_sat

# %% [markdown]
# ### Train/test split
#
# If `test_split_fraction` is set, randomly hold out that fraction of NOAA
# surface-flask rows before binning, so the training pipeline only ever
# sees the remaining rows. NOAA in-situ rows (the smaller, continuously-
# operating "backbone" of the network) are never eligible for holdout.
#
# The split is stratified at the (year, month, lat_bin, lon_bin) level -
# i.e. by spatial bin, since that's the actual input granularity to
# `local.binned_data_interpolation.interpolate`'s `griddata` call. Within a
# bin, flask rows can only be held out down to the point of fully removing
# them *if* an in-situ row is also present in that bin (which alone keeps
# it populated); otherwise at least 1 flask row is always kept in training.
# This means the split can reduce how many stations get averaged together
# within a bin, but never empties out a bin that used to have data, so it
# can't shrink a month's spatial coverage/convex hull below what the full
# dataset already achieves - see `local.train_test_split.stratified_test_split`
# for the full rationale. A plain unstratified random sample of rows
# doesn't have this guarantee - it can silently empty thinly-covered bins,
# which cascades into `griddata` failing to fill the grid for that month,
# that month being dropped entirely, and (if enough months drop out of the
# same year) into non-uniformly-spaced years, which breaks later steps that
# assume uniform year spacing (e.g. the seasonality decomposition's
# mean-preserving interpolation). The held-out rows are saved to
# `held_out_test_data_file` for out-of-sample evaluation (see
# `notebooks/evaluation/`) and never touch any other pipeline step.

# %%
if config_step.test_split_fraction is not None:
    all_data = all_data.reset_index(drop=True)
    all_data_with_bins_for_split = local.binning.add_lat_lon_bin_columns(all_data)
    eligible_for_test = (all_data_with_bins_for_split["network"].str.upper() == "NOAA") & (
        all_data_with_bins_for_split["measurement_method"] == "flask"
    )
    train_data, test_data = local.train_test_split.stratified_test_split(
        all_data_with_bins_for_split,
        eligible=eligible_for_test,
        test_fraction=config_step.test_split_fraction,
        seed=config_step.test_split_seed,
    )
    all_data = train_data.drop(columns=["lat_bin", "lon_bin"])
    test_data = test_data.drop(columns=["lat_bin", "lon_bin"])

    config_step.held_out_test_data_file.parent.mkdir(exist_ok=True, parents=True)
    test_data.to_csv(config_step.held_out_test_data_file, index=False)
    print(
        f"Held out {len(test_data)} of {len(all_data) + len(test_data)} rows for testing, "
        f"wrote to {config_step.held_out_test_data_file}"
    )

    # Informational only (not read by anything else in the pipeline) - lets
    # `notebooks/evaluation/` report the achieved holdout rate against both
    # eligible (flask) rows and all rows (flask + backbone), without having
    # to re-derive the pre-split counts itself.
    test_split_stats_file = config_step.held_out_test_data_file.with_name(
        f"{config_step.gas}_observational-network_test-holdout-stats.csv"
    )
    pd.DataFrame(
        [
            {
                "n_test": len(test_data),
                "n_eligible_flask_total": int(eligible_for_test.sum()),
                "n_all_total": len(all_data_with_bins_for_split),
            }
        ]
    ).to_csv(test_split_stats_file, index=False)

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


# %% jupyter={"outputs_hidden": true}
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

# %%
