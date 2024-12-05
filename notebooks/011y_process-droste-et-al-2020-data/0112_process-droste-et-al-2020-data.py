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
# # Droste et al., 2022 - process
#
# Process data from [Droste et al., 2020](https://doi.org/10.5194/acp-20-4787-2020).

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
step: str = "retrieve_and_process_droste_et_al_2020_data"

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
files

# %%
out_l = []
for file in files:
    if file.name.startswith("best-fits_CG"):
        # Cape grim lat
        lat = -40.6833

    elif file.name.startswith("best-fits_TAC"):
        # Talconeston, UK lat
        lat = 52.5127

    else:
        raise NotImplementedError(file)

    raw = pd.read_csv(file)
    raw = raw[
        [
            "Date",
            "cC4F8",
            "nC4F10",
            "nC5F12",
            # 'iC6F14',  # not using for now
            "nC6F14",
            "nC7F16",
        ]
    ]

    raw = raw.rename(
        {
            "cC4F8": "cc4f8",
            "nC4F10": "c4f10",
            "nC5F12": "c5f12",
            "nC6F14": "c6f14",
            "nC7F16": "c7f16",
        },
        axis="columns",
    )
    raw["year"] = raw["Date"].apply(lambda x: int(x.split("/")[-1]))
    raw["month"] = raw["Date"].apply(lambda x: int(x.split("/")[1]))
    raw = raw.drop("Date", axis="columns")
    annual_mean = raw.groupby("year")[["cc4f8", "c4f10", "c5f12", "c6f14", "c7f16"]].mean()

    annual_mean.columns.name = "gas"
    annual_mean = annual_mean.stack().to_frame("value").reset_index()  # type: ignore
    annual_mean["unit"] = assumed_unit
    annual_mean["lat"] = lat

    out_l.append(annual_mean)

clean = pd.concat(out_l)
clean

# %% [markdown]
# ## Save

# %%
config_step.processed_data_file.parent.mkdir(exist_ok=True, parents=True)
clean.to_csv(config_step.processed_data_file, index=False)
config_step.processed_data_file
