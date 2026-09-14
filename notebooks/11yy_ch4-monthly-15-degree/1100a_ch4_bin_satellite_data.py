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

# %% [markdown]
# # CH$_4$ - binning for sat data
#
# Bin the scaled satellite data.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import numpy as np
import openscm_units
import pandas as pd
import pint
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.raw_data_processing
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "calculate_ch4_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %%
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

config_process_scaled_sat_data = get_config_for_step_id(
    config=config,
    step="process_scaled_sat_data",
    step_config_id=config_step.gas,
)

# %%
config_process_scaled_sat_data.interim_data_path

# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
# If satellite data isn't switched on for this gas, we still need to write
# an (empty) interim file, since downstream steps always expect one to exist.
if config_step.include_satellite_data:
    import xarray as xr

    sat_data = xr.open_dataset(config_process_scaled_sat_data.scaled_data_path)

    sat_data["xch4"] = sat_data["xch4"] * 1e9

    if config_step.weight_satellite_data:
        stderr_rchi2_col = "xch4_scaled_stderr_rchi2"
        sat_data[stderr_rchi2_col] = sat_data[stderr_rchi2_col] * 1e9

        sat_df = sat_data[["xch4", stderr_rchi2_col]].to_dataframe().reset_index()
        sat_df = sat_df.rename(columns={"xch4": "value"})

        # Inverse-variance weight from the satellite retrieval's rchi2-calibrated
        # standard error. Ground-network stations are not weighted (each gets an
        # implicit weight of one) - see notebooks/diagnostics/CO2/uncertainty_co2.py
        # for why that isn't extended to ground-network uncertainty columns too
        # (they're not directly comparable across sources, and aren't plumbed
        # through to this point in the pipeline in the first place).
        sat_df["weight"] = 1.0 / (sat_df[stderr_rchi2_col] ** 2)
        sat_df = sat_df.drop(columns=[stderr_rchi2_col])
    else:
        sat_df = sat_data["xch4"].to_dataframe(name="value").reset_index()

    # Extract year and month
    sat_df["year"] = sat_df["time"].dt.year
    sat_df["month"] = sat_df["time"].dt.month

    # Rename coordinates
    sat_df = sat_df.rename(columns={"lat": "latitude", "lon": "longitude"})

    # Create station/site_code strings
    coord_string = (
        "[" + sat_df["longitude"].round(6).astype(str) + ", " + sat_df["latitude"].round(6).astype(str) + "]"
    )

    sat_df["station"] = coord_string
    sat_df["site_code"] = coord_string
    sat_df["site_code_filename"] = coord_string

    # Add constant columns
    sat_df["gas"] = "ch4"
    sat_df["reporting_id"] = "MonthlyData"
    sat_df["unit"] = "ppb"
    sat_df["surf_or_ship"] = "satellite"
    sat_df["source"] = "satellite"
    sat_df["network"] = "OBS4MIPs"
    sat_df["measurement_method"] = "satellite"

    # Reorder columns
    column_order = [
        "gas",
        "reporting_id",
        "year",
        "month",
        "latitude",
        "longitude",
        "value",
        "unit",
        "site_code_filename",
        "site_code",
        "surf_or_ship",
        "source",
        "network",
        "station",
        "measurement_method",
    ]
    if config_step.weight_satellite_data:
        column_order = [*column_order, "weight"]
    sat_df = sat_df[column_order]
    dropna_subset = ["value", "weight"] if config_step.weight_satellite_data else ["value"]
    sat_df = sat_df.dropna(subset=dropna_subset)
    if config_step.weight_satellite_data:
        sat_df = sat_df[np.isfinite(sat_df["weight"])]

    sat_df_with_bins = local.binning.add_lat_lon_bin_columns(sat_df)
    bin_averages = local.binning.calculate_bin_averages(
        sat_df_with_bins, weight_column="weight" if config_step.weight_satellite_data else None
    )

    assert set(bin_averages["gas"]) == {config_step.gas}

else:
    bin_averages = pd.DataFrame(columns=["gas", "lat_bin", "lon_bin", "month", "unit", "value", "year"])

bin_averages

# %% [markdown]
# ### Save

# %%
local.binned_data_interpolation.check_data_columns_for_binned_data_interpolation(bin_averages)

# %%
config_process_scaled_sat_data.interim_data_path.parent.mkdir(exist_ok=True, parents=True)

# %%
bin_averages.to_csv(config_process_scaled_sat_data.interim_data_path, index=False)
bin_averages
