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
# # WMO 2022 ozone assessment - process
#
# Process data from the 2022 WMO ozone assessment.

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from pathlib import Path

import openscm_units
import pandas as pd
import pint
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %% editable=true slideshow={"slide_type": ""}
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "retrieve_and_process_wmo_2022_ozone_assessment_ch7_data"

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

# %% editable=true slideshow={"slide_type": ""}
# Created with:
# `# {k: k.lower().replace("-", "") for k in wmo_ch7_df_raw.columns}`
wmo_variable_normalisation_map = {
    "CFC-11": "cfc11",
    "CFC-12": "cfc12",
    "CFC-113": "cfc113",
    "CFC-114": "cfc114",
    "CFC-115": "cfc115",
    "CCl4": "ccl4",
    "CH3CCl3": "ch3ccl3",
    "HCFC-22": "hcfc22",
    "HCFC-141b": "hcfc141b",
    "HCFC-142b": "hcfc142b",
    "halon-1211": "halon1211",
    "halon-1202": "halon1202",
    "halon-1301": "halon1301",
    "halon-2402": "halon2402",
    "CH3Br": "ch3br",
    "CH3Cl": "ch3cl",
}

# %% editable=true slideshow={"slide_type": ""}
assumed_unit = "ppt"

# %% editable=true slideshow={"slide_type": ""}
wmo_ch7_df_raw = pd.read_excel(config_step.raw_data)
wmo_ch7_df = wmo_ch7_df_raw.rename({"Year": "year", **wmo_variable_normalisation_map}, axis="columns")

wmo_ch7_df

# %% [markdown] editable=true slideshow={"slide_type": ""}
# WMO data is start of year, yet we want mid-year values, hence do the below.

# %% editable=true slideshow={"slide_type": ""}
wmo_ch7_df = wmo_ch7_df.set_index(["year"])
tmp = (wmo_ch7_df.iloc[:-1, :].values + wmo_ch7_df.iloc[1:, :].values) / 2.0
wmo_ch7_df = wmo_ch7_df.iloc[:-1, :]
wmo_ch7_df.iloc[:, :] = tmp
wmo_ch7_df = wmo_ch7_df.reset_index()

wmo_ch7_df

# %% editable=true slideshow={"slide_type": ""}
wmo_ch7_df = wmo_ch7_df.set_index("year")
wmo_ch7_df.columns.name = "gas"
wmo_ch7_df = wmo_ch7_df.stack().to_frame("value").reset_index()
wmo_ch7_df["unit"] = assumed_unit
wmo_ch7_df

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Save

# %% editable=true slideshow={"slide_type": ""}
config_step.processed_data_file.parent.mkdir(exist_ok=True, parents=True)
wmo_ch7_df.to_csv(config_step.processed_data_file, index=False)
config_step.processed_data_file
