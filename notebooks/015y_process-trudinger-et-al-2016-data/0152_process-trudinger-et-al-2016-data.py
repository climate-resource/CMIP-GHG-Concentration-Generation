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
# # Trudinger et al., 2016 - process
#
# Process data from [Trudinger et al., 2016](https://doi.org/10.5194/acp-16-11733-2016).

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from pathlib import Path

import matplotlib.pyplot as plt
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
step: str = "retrieve_and_process_trudinger_et_al_2016_data"

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
files = list(config_step.raw_dir.rglob("*.xlsx"))
if len(files) > 1:
    raise NotImplementedError(files)

file = files[0]

# %%
clean_l = []
for sheet in ["InvE2", "InvEF"]:
    raw = pd.read_excel(file, sheet_name=sheet, skiprows=1)
    conc_start = raw.columns.tolist().index(
        f"Annual values of mole fraction (ppt) in the high latitudes of each hemisphere from inversion {sheet} "
    )
    tmp = raw.iloc[:, conc_start:]

    columns_new_start = tmp.iloc[:2, 1:].T
    columns_new_start.columns = ["gas", "hemisphere_id"]  # type: ignore
    columns_new = pd.MultiIndex.from_frame(columns_new_start)

    long = tmp.iloc[2:, :]
    # long.columns = long.iloc[0, :]
    long = long.iloc[1:, :]
    long.columns = ["year", *long.columns[1:]]  # type: ignore
    long = long.set_index("year")
    long.columns = columns_new

    units = tmp.iloc[2, 1:].unique().tolist()
    if len(units) > 1:
        raise AssertionError(units)

    unit = units[0].replace("(", "").replace(")", "")

    sheet_out = long.stack(["gas", "hemisphere_id"], future_stack=True).to_frame("value").reset_index()  # type: ignore
    sheet_out["unit"] = unit
    sheet_out["inversion_method"] = sheet
    sheet_out["gas"] = sheet_out["gas"].str.lower()

    clean_l.append(sheet_out)

clean = pd.concat(clean_l)
clean

# %%
clean_gm = (
    clean.set_index(clean.columns.difference(["value"]).tolist())["value"]
    .groupby(clean.columns.difference(["value", "hemisphere_id"]).tolist())
    .mean()
    .reset_index()
)
clean_gm["hemisphere_id"] = "global-mean"
clean_incl_gm = pd.concat([clean, clean_gm])

# %%
pdf = clean_incl_gm.pivot_table(
    values="value",
    index=["gas", "inversion_method", "hemisphere_id", "unit"],
    columns=["year"],
)

for gas, gdf in pdf.groupby("gas"):
    gdf.T.plot()
    plt.show()

# %% [markdown]
# Only use the InvEF inversion and the global-mean values in further processing (for now?).

# %%
out_start_year = clean_incl_gm[
    (clean_incl_gm["hemisphere_id"] == "global-mean") & (clean_incl_gm["inversion_method"] == "InvEF")
]
out_start_year

# %%
helper = out_start_year.set_index(out_start_year.columns.difference(["value"]).tolist()).unstack("year")
out = (
    (
        (helper + helper.shift(periods=-1, axis="columns"))  # type: ignore
        / 2.0
    )
    .dropna(axis="columns")
    .stack("year", future_stack=True)
    .reset_index()
)
out

# %% [markdown]
# Trudinger et al. values are start of year, but we want mid-year.

# %% [markdown]
# ## Save

# %%
config_step.processed_data_file.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(config_step.processed_data_file, index=False)
config_step.processed_data_file

# %%
local.dependencies.save_source_info_to_db(
    db=config.dependency_db,
    source_info=config_step.source_info,
)
