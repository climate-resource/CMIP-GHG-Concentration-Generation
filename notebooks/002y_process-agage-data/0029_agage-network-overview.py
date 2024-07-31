# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # AGAGE - overview
#
# Overview of all AGAGE data.

# %% [markdown]
# ## Imports

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import openscm_units
import pandas as pd
import pint
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "plot_input_data_overviews"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_retrieve_and_extract_gage_data = get_config_for_step_id(
    config=config, step="retrieve_and_extract_gage_data", step_config_id="monthly"
)

config_retrieve_and_extract_ale_data = get_config_for_step_id(
    config=config, step="retrieve_and_extract_ale_data", step_config_id="monthly"
)

config_retrieve = get_config_for_step_id(
    config=config, step="retrieve_misc_data", step_config_id="only"
)

# %% [markdown]
# ## Action

# %%
full_df = pd.concat(
    [
        pd.read_csv(
            config_retrieve_and_extract_gage_data.processed_monthly_data_with_loc_file
        ),
        pd.read_csv(
            config_retrieve_and_extract_ale_data.processed_monthly_data_with_loc_file
        ),
        *[
            pd.read_csv(c.processed_monthly_data_with_loc_file)
            for c in config.retrieve_and_extract_agage_data
        ],
    ]
)
# TODO: define column cols
common_cols = [
    "month",
    "year",
    "gas",
    "value",
    "unit",
    "latitude",
    "longitude",
    "source",
    "site_code",
]
full_df = full_df[common_cols]
# Need better checks on this earlier
full_df["gas"] = full_df["gas"].str.lower()
full_df

# %%
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir
    / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
station_colours = {
    "CGO": "tab:blue",  # Cape Grim
    "ADR": "tab:green",  # Adrigole, Ireland
    "MHD": "tab:green",  # Macehead, Ireland
    "ORG": "tab:purple",  # Cape Meares, Oregon
    "RPB": "tab:cyan",  # Ragged Point, Barbados
    "SMO": "tab:olive",  # Cape Matatula, Samoa
    "THD": "tab:purple",  # Trinidad Head (?)
    "CMN": "lime",  # ?
    "GSN": "magenta",  # ?
    "JFJ": "tab:orange",  # Jungfrauchjoch, Austria
    "TAC": "red",  # ?
    "TOB": "darkblue",  # ? Germany somewhere I think
    "ZEP": "magenta",  # Zeppelin
}
source_markers = {
    "AGAGE": "o",
    "GAGE": "x",
    "ALE": "+",
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

    for (station, source), station_df in tqdman.tqdm(
        gdf.groupby(["site_code", "source"]),
        desc=f"{gas} stations",
        leave=False,
    ):
        label = f"{source} {station}"

        station_df[["longitude", "latitude"]].round(0).drop_duplicates().plot(
            x="longitude",
            y="latitude",
            kind="scatter",
            ax=axes["map"],
            # alpha=0.3 if source == "flask" else 1.0,
            # zorder=2 if source == "flask" else 3,
            label=label if label not in labels else None,
            color=station_colours[station],
            # s=100,
            marker=source_markers[source],
        )

        pdf = station_df.copy()
        pdf["year-month"] = pdf["year"] + pdf["month"] / 12
        pdf.plot(
            x="year-month",
            y="value",
            kind="scatter",
            ax=axes["full_ts"],
            label=label if label not in labels else None,
            color=station_colours[station],
            marker=source_markers[source],
            alpha=0.3,
        )

        pdf[pdf["year"] >= zoom_ts_start_year].plot(
            x="year-month",
            y="value",
            kind="scatter",
            ax=axes["zoom_ts"],
            label=label if label not in labels else None,
            color=station_colours[station],
            marker=source_markers[source],
            alpha=0.3,
        )

        labels.append(label)
    #     # break

    axes["map"].set_xlim((-180.0, 180.0))
    axes["map"].set_ylim((-90.0, 90.0))
    axes["map"].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), ncols=2)

    axes["full_ts"].set_ylabel(unit)
    axes["zoom_ts"].set_ylabel(unit)
    axes["full_ts"].legend().remove()
    axes["zoom_ts"].legend().remove()

    plt.suptitle(str(gas))
    # plt.tight_layout()
    plt.show()
    # break
