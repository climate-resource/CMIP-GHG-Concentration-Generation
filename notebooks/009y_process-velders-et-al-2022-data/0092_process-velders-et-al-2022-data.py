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
# # Velders et al., 2022 - process
#
# Process data from [Velders et al., 2022](https://doi.org/10.5194/acp-22-6087-2022).

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import io
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
step: str = "retrieve_and_process_velders_et_al_2022_data"

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
expected_species = [
    "HFC-32",
    "HFC-125",
    "HFC-134a",
    "HFC-143a",
    "HFC-152a",
    "HFC-227ea",
    "HFC-236fa",
    "HFC-245fa",
    "HFC-365mfc",
    "HFC-43-10mee",
]

# %% [markdown]
# Read the temporary file, while the Zenodo record is not updated.

# %%
with open(config_step.raw_data_file_tmp) as fh:
    raw = tuple(fh.readlines())

# %%
start_idx = 74
block_length = 512
expected_n_blocks = 10

clean_l = []
for i in range(expected_n_blocks):
    start = start_idx + i * (block_length + 1)

    block = io.StringIO("\n".join(raw[start : start + block_length]))
    species_df = pd.read_csv(block, sep=r"\s+")

    gas = species_df["Species"].unique()
    if len(gas) != 1:
        raise AssertionError
    gas = gas[0]

    keep = species_df[["Year", "Mix_tot"]].rename({"Year": "year", "Mix_tot": str(gas)}, axis="columns")
    keep = keep.set_index("year")
    # display(keep)

    clean_l.append(keep)

clean = pd.concat(clean_l, axis="columns")
if set(clean.columns) != set(expected_species):
    raise AssertionError

clean

# %%
# files = list(config_step.raw_dir.rglob("*.xlsx"))
# if len(files) != 1:
#     raise AssertionError

# raw_data_file = files[0]
# raw_data_file

# %%
# # Doesn't matter whether we use upper or lower as we're just getting historical data
# raw_excel = pd.read_excel(raw_data_file, sheet_name="Upper", header=None)

# %%
# start_idx = 4
# block_length = 112
# expected_n_blocks = 10

# clean_l = []
# for i in range(expected_n_blocks):
#     start = start_idx + i * (block_length + 1)
#     species_df = raw_excel.iloc[start : start + block_length]
#     species_df = species_df.dropna(how="all", axis="columns")
#     species_df.columns = species_df.iloc[0, :]  # type: ignore
#     species_df = species_df.iloc[1:, :]

#     gas = species_df["Species"].unique()
#     if len(gas) != 1:
#         raise AssertionError
#     gas = gas[0]

#     keep = species_df[["Year", "Mix_tot"]].rename({"Year": "year", "Mix_tot": str(gas)}, axis="columns")
#     keep = keep.set_index("year")
#     # display(keep)

#     clean_l.append(keep)

# clean = pd.concat(clean_l, axis="columns")
# if set(clean.columns) != set(expected_species):
#     raise AssertionError

# clean

# %%
# Velders data is start of year, yet we want mid-year values, hence do the below
tmp = (clean.iloc[:-1, :].values + clean.iloc[1:, :].values) / 2.0
clean = clean.iloc[:-1, :]
clean.iloc[:, :] = tmp
clean

# %%
velders_variable_normalisation_map = {
    "HFC-32": "hfc32",
    "HFC-125": "hfc125",
    "HFC-134a": "hfc134a",
    "HFC-143a": "hfc143a",
    "HFC-152a": "hfc152a",
    "HFC-227ea": "hfc227ea",
    "HFC-236fa": "hfc236fa",
    "HFC-245fa": "hfc245fa",
    "HFC-365mfc": "hfc365mfc",
    "HFC-43-10mee": "hfc4310mee",
}

# %%
clean = clean.rename(velders_variable_normalisation_map, axis="columns")
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
