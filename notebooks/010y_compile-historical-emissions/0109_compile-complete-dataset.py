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

# %% [markdown]
# # Compile historical emissions - complete dataset
#
# Put all the data together.
#
# At the moment, this just uses the RCMIP data.
# In future, it will actually compile data from previous steps.

# %% [markdown]
# ## Imports

# %%

import openscm_units
import pandas as pd
import pint
import pooch
import scmdata
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "compile_historical_emissions"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %% [markdown]
# ## Action

# %%
rcmip_emissions_fname = pooch.retrieve(
    url="https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
    path=config_step.complete_historical_emissions_file.parent,
    progressbar=True,
)
rcmip_emissions_fname

# %%
rcmip_all = scmdata.ScmRun(rcmip_emissions_fname, lowercase_cols=True)
rcmip_cmip6_historical = (
    rcmip_all.filter(region="World", scenario="ssp245")
    .filter(variable=["*Montreal Gases*", "*F-Gases*"])
    .resample("YS")
)
rcmip_cmip6_historical


# %%
def rename_variable(v: str) -> str:
    """
    Re-name variable to our internal conventions
    """
    toks = v.split("|")
    return "|".join([toks[0], toks[-1].lower()])


def fix_units(u: str) -> str:
    """
    Fix units so that we can handle them
    """
    return u


# %%
out: pd.DataFrame = rcmip_cmip6_historical.long_data(time_axis="year")  # type: ignore
out["variable"] = out["variable"].apply(rename_variable)
out["unit"] = out["unit"].apply(fix_units)
out

# %%
# Make sure all our units are understood by OpenSCM-units
for unit in out["unit"].unique():
    openscm_units.unit_registry.Quantity(1, unit)

# %%
sorted(out["unit"].unique())

# %%
sorted(out["variable"].unique())

# %%
config_step.complete_historical_emissions_file.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(config_step.complete_historical_emissions_file, index=False)
config_step.complete_historical_emissions_file
