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
# # Western et al., 2024 - process
#
# Process data from [Western et al., 2024](https://doi.org/10.1038/s41558-024-02038-7).

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
step: str = "retrieve_and_process_western_et_al_2024_data"

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
assumed_unit = "ppt"

# %%
files = list(config_step.raw_dir.rglob("*.csv"))
if len(files) != 1:
    raise AssertionError

raw_data_file = files[0]
raw_data_file

# %%
raw = pd.read_csv(raw_data_file, skiprows=1)
raw

# %%
western_variable_normalisation_map = {
    "HCFC-22": "hcfc22",
    "HCFC-141b": "hcfc141b",
    "HCFC-142b": "hcfc142b",
}

# %%
clean = raw.rename({"Year": "year", **western_variable_normalisation_map}, axis="columns")
clean = clean.set_index("year")
clean

# %%
# Western data from this file is start of year, yet we want mid-year values, hence do the below
tmp = ((clean.iloc[:-1, :].values + clean.iloc[1:, :].values) / 2.0).copy()
clean = clean.iloc[:-1, :]
clean.iloc[:, :] = tmp
clean

# %%
clean.columns.name = "gas"
clean = clean.stack().to_frame("value").reset_index()  # type: ignore
clean["unit"] = assumed_unit
clean

# %% [markdown]
# ## Save

# %%
config_step.processed_data_file.parent.mkdir(exist_ok=True, parents=True)
clean.to_csv(config_step.processed_data_file, index=False)
config_step.processed_data_file

# %%
local.dependencies.save_source_info_to_db(
    db=config.dependency_db,
    source_info=config_step.source_info,
)
