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
# # Law dome - overview
#
# Overview of all Law Dome data.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "plot_input_data_overviews"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_retrieve = get_config_for_step_id(
    config=config, step="retrieve_misc_data", step_config_id="only"
)

config_process = get_config_for_step_id(
    config=config, step="retrieve_and_process_law_dome_data", step_config_id="only"
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %% editable=true slideshow={"slide_type": ""}
full_df = pd.read_csv(config_process.processed_data_with_loc_file)
full_df

# %% editable=true slideshow={"slide_type": ""}
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir
    / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %% editable=true slideshow={"slide_type": ""}
zoom_ts_start_year = 1950
for gas, gdf in tqdman.tqdm(
    full_df.groupby("gas"),
    desc="Gases",
):
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

    gdf[["longitude", "latitude"]].round(0).drop_duplicates().plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        ax=axes["map"],
        label="Observing location",
    )

    pdf = gdf.copy()
    pdf["year-month"] = pdf["year"] + pdf["month"] / 12
    pdf.plot(
        x="year-month",
        y="value",
        kind="scatter",
        ax=axes["full_ts"],
    )

    pdf[pdf["year"] >= zoom_ts_start_year].plot(
        x="year-month",
        y="value",
        kind="scatter",
        ax=axes["zoom_ts"],
    )

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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Could probably do something cool here with interactivity if we had more time.
