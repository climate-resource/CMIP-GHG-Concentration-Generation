# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
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

from local.config import load_config_from_file
from local.pydoit_nb.config_handling import get_config_for_step_id

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

config_process_noaa_surface_flask_data_co2 = get_config_for_step_id(
    config=config, step="process_noaa_surface_flask_data", step_config_id="co2"
)
config_process_noaa_in_situ_data_co2 = get_config_for_step_id(
    config=config, step="process_noaa_in_situ_data", step_config_id="co2"
)
config_retrieve = get_config_for_step_id(
    config=config, step="retrieve", step_config_id="only"
)

# %% [markdown]
# ## Action

# %%
df_flask = pd.read_csv(
    config_process_noaa_surface_flask_data_co2.processed_monthly_data_with_loc_file
)
df_in_situ = pd.read_csv(
    config_process_noaa_in_situ_data_co2.processed_monthly_data_with_loc_file
)

# %%
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir
    / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
full_df = pd.concat([df_flask, df_in_situ])
full_df

# %%
source_colours = {
    "insitu": "tab:blue",
    "flask": "tab:brown",
}
surf_ship_markers = {
    "surface": "o",
    "shipboard": "x",
}

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

countries.plot(color="lightgray", ax=axes[0])

labels = []
for (station, source, surf_or_ship), station_df in tqdman.tqdm(
    full_df.groupby(["site_code_filename", "source", "surf_or_ship"]),
    desc="Stations",
):
    label = f"{source} {surf_or_ship}"

    station_df[["longitude", "latitude"]].round(0).drop_duplicates().plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        ax=axes[0],
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
        ax=axes[1],
        label=label if label not in labels else None,
        color=source_colours[source],
        marker=surf_ship_markers[surf_or_ship],
        alpha=0.3 if source == "flask" else 1.0,
        zorder=2 if source == "flask" else 3,
    )
    labels.append(label)
    # break

axes[0].set_xlim([-180, 180])
axes[0].set_ylim([-90, 90])

axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# Could probably do something cool here with interactivity if we had more time.
