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
# # Ghosh et al., 2023 - process
#
# Process data from [Ghosh et al., 2023](https://doi.org/10.1029/2022JD038281).
# Raw data is included in the repository
# because you can't download it an automated way.
# If you are using this repo,
# please go and download your own copy
# from https://www.usap-dc.org/view/dataset/601693
# to help the authors own tracking statistics
# as a way of saying thank you.

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
step: str = "retrieve_and_process_ghosh_et_al_2023_data"

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
raw = pd.read_excel(config_step.raw_data_file, sheet_name="Inversion (N2O)")
raw

# %%
if raw.iloc[0, 0] != "Median (column D) is the best estimate for the atmospheric history":
    raise AssertionError

best_estimate_col = 3

# %%
cleaner = pd.read_excel(
    config_step.raw_data_file, sheet_name="Inversion (N2O)", skiprows=3
)  # .rename({"Unnamed: 0": "year"}, axis="columns").iloc[:, [0, best_estimate_col]].set_index("year")

unit = cleaner.loc[:, "N2O, median"].iloc[0]
if unit != "ppb":
    raise AssertionError

# Drop out the units row
clean = cleaner.iloc[1:, :]
clean = clean.rename({"Cal year CE": "year", "N2O, median": "value"}, axis="columns")[["year", "value"]]
clean["unit"] = unit
clean["gas"] = "n2o"

clean

# %% [markdown]
# ## Save

# %%
config_step.processed_data_file.parent.mkdir(exist_ok=True, parents=True)
clean.to_csv(config_step.processed_data_file, index=False)
config_step.processed_data_file
