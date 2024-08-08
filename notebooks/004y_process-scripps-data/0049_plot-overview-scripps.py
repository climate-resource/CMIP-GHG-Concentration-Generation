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
# # Scripps - overview
#
# Overview of all Scripps data.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from pathlib import Path

import geopandas as gpd
import matplotlib.axes
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
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_retrieve_misc = get_config_for_step_id(
    config=config, step="retrieve_misc_data", step_config_id="only"
)

config_process = get_config_for_step_id(
    config=config, step="retrieve_and_process_scripps_data", step_config_id="only"
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %%
monthly_df_with_loc = pd.read_csv(config_process.processed_data_with_loc_file)

# %% editable=true slideshow={"slide_type": ""}
countries = gpd.read_file(
    config_retrieve_misc.natural_earth.raw_dir
    / config_retrieve_misc.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
for (station, source), sdf in tqdman.tqdm(
    monthly_df_with_loc.groupby(["station_code", "source"]),
):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    if isinstance(axes, matplotlib.axes.Axes):
        raise TypeError(type(axes))

    countries.plot(color="lightgray", ax=axes[0])

    label = f"{station} {source}"

    axes[0].scatter(
        x=sdf["longitude"], y=sdf["latitude"], alpha=0.4, label=label, zorder=2
    )

    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-90, 90])

    pdf = sdf.copy()
    pdf["year-month"] = pdf["year"] + pdf["month"] / 12
    axes[1].scatter(
        x=pdf["year-month"],
        y=pdf["value"],
        alpha=0.4,
        label=f"{label} monthly data",
        zorder=2,
    )

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    plt.show()

# %%
fig, axes = plt.subplots(ncols=2, figsize=(12, 8))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))

countries.plot(color="lightgray", ax=axes[0])

for (station, source), sdf in tqdman.tqdm(
    monthly_df_with_loc.groupby(["station_code", "source"]),
):
    # if station != "mlo":
    #     continue
    # print(f"Examining {station} ")
    label = f"{station} {source}"

    axes[0].scatter(
        x=sdf["longitude"], y=sdf["latitude"], alpha=0.4, label=label, zorder=2
    )

    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-90, 90])

    pdf = sdf.copy()
    pdf["year-month"] = pdf["year"] + pdf["month"] / 12
    axes[1].scatter(
        x=pdf["year-month"],
        y=pdf["value"],
        alpha=0.4,
        label=f"{label} monthly data",
        zorder=2,
    )

    # break

axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.show()
