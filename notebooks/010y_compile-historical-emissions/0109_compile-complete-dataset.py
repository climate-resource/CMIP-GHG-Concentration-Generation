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
# # Compile historical emissions - complete dataset
#
# Put all the data together.
#
# At the moment, this just uses the RCMIP data.
# In future, it will actually compile data from previous steps.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import openscm_units
import pandas as pd
import pint
import pooch
import scmdata
from pydoit_nb.config_handling import get_config_for_step_id

import local.dependencies
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
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

# %% [markdown]
# ## Action

# %%
config_step.complete_historical_emissions_file.parent.mkdir(exist_ok=True, parents=True)
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

# %%
source_info_short_names = []
for si in (
    local.dependencies.SourceInfo(
        short_name="Nicholls et al., 2020",
        licence="CC BY 4.0",
        reference=(
            "Nicholls, Z. R. J., Meinshausen, M., Lewis, J., ..., Tsutsui, J., and Xie, Z.: "
            "Reduced Complexity Model Intercomparison Project Phase 1: "
            "introduction and evaluation of global-mean temperature response, "
            "Geosci. Model Dev., 13, 5175-5190, "
            "https://doi.org/10.5194/gmd-13-5175-2020, 2020."
        ),
        doi="https://doi.org/10.5194/gmd-13-5175-2020",
        url="https://doi.org/10.5194/gmd-13-5175-2020",
        resource_type="publication-article",
    ),
):
    local.dependencies.save_source_info_to_db(db=config.dependency_db, source_info=si)
    source_info_short_names.append(si.short_name)

# %%
with open(config_step.source_info_short_names_file, "w") as fh:
    fh.write(";".join(source_info_short_names))
