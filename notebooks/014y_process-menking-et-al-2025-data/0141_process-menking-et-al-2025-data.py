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
# # Menking et al., 2025 - process
#
# Process data from Menking et al., 2025 (in prep.).
# Raw data is included in the repository
# because the paper isn't published yet.
# If you are using this repo,
# please go and find the paper and cite it.
# If you can't find the paper,
# please feel free to
# [raise an issue](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/issues/new).

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import hashlib
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
step: str = "retrieve_and_process_menking_et_al_2025_data"

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

# %%
# Check against the known hash
with open(config_step.raw_data_file, "rb") as fh:
    if hashlib.md5(fh.read()).hexdigest() != config_step.expected_hash:  # noqa: S324
        raise AssertionError

# %%
raw = pd.read_excel(config_step.raw_data_file, sheet_name="Sheet1", skiprows=6)
raw

# %%
clean_l = []
for yr_col, value_col, latitude in (
    ("Year (CE)", "CO2 (ppm)", -66.0),  # roughly Law Dome latitude, near enough
    ("Year (CE).1", "N2O (ppb)", 0.0),  # global-mean spline
):
    toks = value_col.split(" ")
    gas = toks[0].strip().lower()
    unit = toks[1].replace("(", "").replace(")", "").strip().lower()

    tmp = raw[[yr_col, value_col]].rename({yr_col: "year", value_col: "value"}, axis="columns")
    tmp["unit"] = unit
    tmp["gas"] = gas
    tmp["latitude"] = latitude

    clean_l.append(tmp)

clean = pd.concat(clean_l).dropna()
clean["year"] = clean["year"].astype(int)
clean

# %%
clean.set_index(["year", "unit", "gas"]).unstack("year")

# %% [markdown]
# ## Save

# %%
config_step.processed_data_file.parent.mkdir(exist_ok=True, parents=True)
clean.to_csv(config_step.processed_data_file, index=False)
config_step.processed_data_file
