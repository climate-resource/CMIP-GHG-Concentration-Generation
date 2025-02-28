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
# # NEEM - process
#
# Process data from the NEEM dataset.

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from pathlib import Path

import openscm_units
import pandas as pd
import pint
from pydoit_nb.config_handling import get_config_for_step_id

import local.dependencies
from local.config import load_config_from_file

# %% editable=true slideshow={"slide_type": ""}
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "retrieve_and_process_neem_data"

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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Read and process data

# %%
with open(config_step.raw_dir / config_step.download_url.url.split("/")[-1]) as fh:
    raw = fh.read()

assert "Methane [ppbv] (CH4)" in raw
units = "ppb"
assert "LATITUDE: 77.450000 * LONGITUDE: -51.060000" in raw
lat = 77.450000
lon = -51.060000


read_df = pd.read_csv(
    config_step.raw_dir / config_step.download_url.url.split("/")[-1],
    skiprows=37,
    header=0,
    delimiter="\t",
)
read_df = read_df.rename(
    {
        "CH4 [ppbv] (5 yr medians, see abstract)": "value",
        "Age [a AD/CE] (Gas age (yr CE) (constant del...)": "year",
    },
    axis="columns",
)
read_df["unit"] = units
# read_df["year"] = year_now - (read_df["Age [ka BP]"] * 1000)
read_df["latitude"] = lat
read_df["longitude"] = lon
read_df["gas"] = "ch4"
read_df = read_df[["year", "value", "unit", "latitude", "longitude", "gas"]]
read_df

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Save

# %% editable=true slideshow={"slide_type": ""}
config_step.processed_data_with_loc_file.parent.mkdir(exist_ok=True, parents=True)
read_df.to_csv(config_step.processed_data_with_loc_file, index=False)
config_step.processed_data_with_loc_file

# %%
local.dependencies.save_source_info_to_db(
    db=config.dependency_db,
    source_info=config_step.source_info,
)
