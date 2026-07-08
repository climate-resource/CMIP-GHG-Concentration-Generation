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

# %%
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
import xarray as xr

sat_data = xr.open_dataset(config_process_scaled_sat_data.scaled_data_path)

# %%
sat_data["xco2"] = sat_data["xco2"] * 1e6

# %%

df = sat_data["xco2"].to_dataframe(name="value").reset_index()

# Extract year and month
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month

# Rename coordinates
df = df.rename(columns={"lat": "latitude", "lon": "longitude"})

# Create station/site_code strings
coord_string = "[" + df["longitude"].round(6).astype(str) + ", " + df["latitude"].round(6).astype(str) + "]"

df["station"] = coord_string
df["site_code"] = coord_string
df["site_code_filename"] = coord_string

# Add constant columns
df["gas"] = "co2"
df["reporting_id"] = "MonthlyData"
df["unit"] = "ppm"
df["surf_or_ship"] = "satellite"
df["source"] = "satellite"
df["network"] = "OBS4MIPs"
df["measurement_method"] = "satellite"

# Reorder columns
df = df[
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
df = df.dropna(subset=["value"])
df

# %%
df_with_bins = local.binning.add_lat_lon_bin_columns(df)
df_with_bins

# %%
bin_averages = local.binning.calculate_bin_averages(df_with_bins)
bin_averages

# %% [markdown]
# ### Save

# %%
local.binned_data_interpolation.check_data_columns_for_binned_data_interpolation(bin_averages)
assert set(bin_averages["gas"]) == {config_step.gas}

# %%
config_process_scaled_sat_data.interim_data_path.parent.mkdir(exist_ok=True, parents=True)

# %%
bin_averages.to_csv(config_process_scaled_sat_data.interim_data_path, index=False)
bin_averages

# %%

# %%

# %%
