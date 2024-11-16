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
# # EPICA - overview
#
# Overview of the EPICA data.

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
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

config_retrieve_misc = get_config_for_step_id(config=config, step="retrieve_misc_data", step_config_id="only")

config_process = get_config_for_step_id(
    config=config, step="retrieve_and_process_epica_data", step_config_id="only"
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %%
monthly_df_with_loc = pd.read_csv(config_process.processed_data_with_loc_file)

# %% editable=true slideshow={"slide_type": ""}
countries = gpd.read_file(
    config_retrieve_misc.natural_earth.raw_dir / config_retrieve_misc.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))

countries.plot(color="lightgray", ax=axes[0])

axes[0].scatter(
    x=monthly_df_with_loc["longitude"],
    y=monthly_df_with_loc["latitude"],
    alpha=0.4,
    zorder=2,
)

axes[0].set_xlim([-180, 180])
axes[0].set_ylim([-90, 90])

axes[1].scatter(
    x=monthly_df_with_loc["year"],
    y=monthly_df_with_loc["value"],
    alpha=0.4,
    zorder=2,
)


axes[2].scatter(
    x=monthly_df_with_loc["year"],
    y=monthly_df_with_loc["value"],
    alpha=0.4,
    zorder=2,
)

axes[2].set_xlim([0, 2020])
axes[2].set_ylim([600, 750])

plt.tight_layout()
plt.show()
