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
# # NOAA - overview
#
# Overview of all NOAA data.

# %% [markdown]
# ## Imports

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "plots"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
# config_step = get_config_for_step_id(
#     config=config, step=step, step_config_id=step_config_id
# )

gas_configs = {
    f"{gas}_{source}": get_config_for_step_id(
        config=config, step=step, step_config_id=gas
    )
    for gas, source, step in (
        ("co2", "in-situ", "process_noaa_in_situ_data"),
        ("ch4", "in-situ", "process_noaa_in_situ_data"),
        ("co2", "surface-flask", "process_noaa_surface_flask_data"),
        ("ch4", "surface-flask", "process_noaa_surface_flask_data"),
        ("n2o", "surface-flask", "process_noaa_surface_flask_data"),
        ("sf6", "surface-flask", "process_noaa_surface_flask_data"),
    )
}

config_retrieve = get_config_for_step_id(
    config=config, step="retrieve", step_config_id="only"
)

# %% [markdown]
# ## Action

# %% editable=true slideshow={"slide_type": ""}
full_df = pd.concat(
    [
        pd.read_csv(c.processed_monthly_data_with_loc_file)
        for c in tqdman.tqdm(gas_configs.values())
    ]
)
full_df

# %%
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir
    / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
source_colours = {
    "insitu": "tab:blue",
    "flask": "tab:brown",
}
surf_ship_markers = {
    "surface": "o",
    "shipboard": "x",
}
zoom_ts_start_year = 2019

for gas, gdf in tqdman.tqdm(
    full_df.groupby("gas"),
    desc="Gases",
):
    labels = []
    fig, axes = plt.subplot_mosaic(
        [
            ["map", "map"],
            ["full_ts", "zoom_ts"],
        ],
        figsize=(8, 5),
        layout="constrained",
    )
    countries.plot(color="lightgray", ax=axes["map"])

    unit_arr = gdf["unit"].unique()
    assert len(unit_arr) == 1
    unit = str(unit_arr[0])

    for (station, source, surf_or_ship), station_df in tqdman.tqdm(
        gdf.groupby(["site_code_filename", "source", "surf_or_ship"]),
        desc=f"{gas} stations",
        leave=False,
    ):
        label = f"{source} {surf_or_ship}"

        station_df[["longitude", "latitude"]].round(0).drop_duplicates().plot(
            x="longitude",
            y="latitude",
            kind="scatter",
            ax=axes["map"],
            alpha=0.3 if source == "flask" else 1.0,
            zorder=2 if source == "flask" else 3,
            label=label if label not in labels else None,
            color=source_colours[source],
            # s=100,
            marker=surf_ship_markers[surf_or_ship],
        )

        pdf = station_df.copy()
        pdf["year-month"] = pdf["year"] + pdf["month"] / 12
        pdf.plot(
            x="year-month",
            y="value",
            kind="scatter",
            ax=axes["full_ts"],
            label=label if label not in labels else None,
            color=source_colours[source],
            marker=surf_ship_markers[surf_or_ship],
            alpha=0.3 if source == "flask" else 1.0,
            zorder=2 if source == "flask" else 3,
        )

        pdf[pdf["year"] >= zoom_ts_start_year].plot(
            x="year-month",
            y="value",
            kind="scatter",
            ax=axes["zoom_ts"],
            label=label if label not in labels else None,
            color=source_colours[source],
            marker=surf_ship_markers[surf_or_ship],
            alpha=0.3 if source == "flask" else 1.0,
            zorder=2 if source == "flask" else 3,
        )

        labels.append(label)
        # break

    axes["map"].set_xlim((-180.0, 180.0))
    axes["map"].set_ylim((-90.0, 90.0))
    axes["map"].legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    axes["full_ts"].set_ylabel(unit)
    axes["zoom_ts"].set_ylabel(unit)
    axes["full_ts"].legend().remove()
    axes["zoom_ts"].legend().remove()

    plt.suptitle(str(gas))
    # plt.tight_layout()
    plt.show()

# %% [markdown]
# Could probably do something cool here with interactivity if we had more time.
