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
# # CO$_2$ - binning for sat data
#
# Bin the scaled satellite data.

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
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "calculate_co2_monthly_fifteen_degree_pieces"

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

    sat_data["xco2"] = sat_data["xco2"] * 1e6

    sat_df = sat_data["xco2"].to_dataframe(name="value").reset_index()

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
    sat_df["gas"] = "co2"
    sat_df["reporting_id"] = "MonthlyData"
    sat_df["unit"] = "ppm"
    sat_df["surf_or_ship"] = "satellite"
    sat_df["source"] = "satellite"
    sat_df["network"] = "OBS4MIPs"
    sat_df["measurement_method"] = "satellite"

    # Reorder columns
    sat_df = sat_df[
        [
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
    ]
    sat_df = sat_df.dropna(subset=["value"])

    sat_df_with_bins = local.binning.add_lat_lon_bin_columns(sat_df)
    bin_averages = local.binning.calculate_bin_averages(sat_df_with_bins)

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

# %%

# %%

# %%
