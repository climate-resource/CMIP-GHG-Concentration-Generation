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
# # NOAA - process in-situ
#
# Process data from NOAA's in-situ network.
#
# To-do's:
#
# - work out why MLO only starts in 1974 when https://scrippsco2.ucsd.edu/data/atmospheric_co2/primary_mlo_co2_record.html says it starts in 1958...

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.axes
import matplotlib.pyplot as plt
import openscm_units
import pandas as pd
import pint
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

import local.raw_data_processing
from local.config import load_config_from_file
from local.noaa_processing import PROCESSED_DATA_COLUMNS

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "process_noaa_in_situ_data"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "ch4"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

config_retrieve = get_config_for_step_id(config=config, step="retrieve_misc_data", step_config_id="only")
config_retrieve_noaa = get_config_for_step_id(
    config=config,
    step="retrieve_and_extract_noaa_data",
    step_config_id=f"{config_step.gas}_in-situ",
)

# %% [markdown]
# ## Action

# %% editable=true slideshow={"slide_type": ""}
df_months = pd.read_csv(config_retrieve_noaa.interim_files["monthly_data"])

# %% editable=true slideshow={"slide_type": ""}
df_months

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Extract data
#
# Nice and easy as this data already has everything we need.

# %%
monthly_dfs_with_loc = df_months[PROCESSED_DATA_COLUMNS]

# %% [markdown]
# Hack around duplicate entries for 2023 MKO.

# %%
potential_dups_locator = (
    (monthly_dfs_with_loc["year"] == 2023)  # noqa: PLR2004
    & (monthly_dfs_with_loc["month"] == 7)  # noqa: PLR2004
    & (monthly_dfs_with_loc["site_code_filename"] == "mko")
)
potential_dups = monthly_dfs_with_loc[potential_dups_locator]
if potential_dups.shape[0] > 1:
    print("Removing dups by simply averaging. Not perfect. Better solution TBD")
    averaged = potential_dups.groupby(list(set(potential_dups.columns) - {"value"})).mean().reset_index()
    if averaged.shape[0] > 1:
        msg = "Still a problem after averaging"
        raise AssertionError(msg)

    monthly_dfs_with_loc = pd.concat([monthly_dfs_with_loc[~potential_dups_locator], averaged]).reset_index(
        drop=True
    )

# %% editable=true slideshow={"slide_type": ""}
duplicate_cols = ["gas", "year", "month", "site_code_filename"]
duplicate_entries = monthly_dfs_with_loc[duplicate_cols][
    monthly_dfs_with_loc[duplicate_cols].duplicated(keep=False)
]
assert (
    duplicate_entries.shape[0] == 0
), f"Duplicate entries for a station in a month\n{monthly_dfs_with_loc[monthly_dfs_with_loc[duplicate_cols].duplicated(keep=False)]}"
monthly_dfs_with_loc

# %% editable=true slideshow={"slide_type": ""}
# Handy check to see if all months have at least some data
pd.MultiIndex.from_product([range(1972, 2022 + 1), range(1, 13)]).difference(  # type: ignore
    monthly_dfs_with_loc.set_index(["year", "month"]).index.drop_duplicates()
)

# %% [markdown]
# ### Plot

# %%
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
colours = (c for c in ["tab:blue", "tab:green", "tab:red", "tab:pink", "tab:brown"])
markers = (m for m in ["o", "x", ".", ",", "v"])

for station, station_df in tqdman.tqdm(monthly_dfs_with_loc.groupby("site_code_filename"), desc="Stations"):
    print(station_df)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    if isinstance(axes, matplotlib.axes.Axes):
        raise TypeError(type(axes))

    colour = next(colours)
    marker = next(markers)

    countries.plot(color="lightgray", ax=axes[0])

    station_df[["longitude", "latitude"]].drop_duplicates().plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        ax=axes[0],
        alpha=0.5,
        label=station,
        color=colour,
        zorder=3,
        s=100,
        marker=marker,
    )

    pdf = station_df.copy()
    pdf["year-month"] = pdf["year"] + pdf["month"] / 12
    pdf.plot(
        x="year-month",
        y="value",
        kind="scatter",
        ax=axes[1],
        label=station,
        color=colour,
        marker=marker,
        alpha=0.4,
    )

    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-90, 90])

    axes[1].legend()

    plt.tight_layout()
    plt.show()

# %%
fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))

colours = (c for c in ["tab:blue", "tab:green", "tab:red", "tab:pink", "tab:brown"])
markers = (m for m in ["o", "x", ".", ",", "v"])

countries.plot(color="lightgray", ax=axes[0])

for station, station_df in tqdman.tqdm(monthly_dfs_with_loc.groupby("site_code_filename"), desc="Stations"):
    colour = next(colours)
    marker = next(markers)

    station_df[["longitude", "latitude"]].drop_duplicates().plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        ax=axes[0],
        alpha=0.5,
        label=station,
        color=colour,
        zorder=3,
        s=100,
        marker=marker,
    )

    pdf = station_df.copy()
    pdf["year-month"] = pdf["year"] + pdf["month"] / 12
    pdf.plot(
        x="year-month",
        y="value",
        kind="scatter",
        ax=axes[1],
        label=station,
        color=colour,
        marker=marker,
        alpha=0.4,
    )

axes[0].set_xlim([-180, 180])
axes[0].set_ylim([-90, 90])

axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Prepare and check output

# %%
out = monthly_dfs_with_loc.copy()
out["network"] = "NOAA"
out["station"] = out["site_code"].str.lower()
out["measurement_method"] = out["source"]

out

# %%
local.raw_data_processing.check_processed_data_columns_for_spatial_binning(out)

# %% [markdown]
# ### Save

# %%
assert set(out["gas"]) == {config_step.gas}
out.to_csv(config_step.processed_monthly_data_with_loc_file, index=False)
out
