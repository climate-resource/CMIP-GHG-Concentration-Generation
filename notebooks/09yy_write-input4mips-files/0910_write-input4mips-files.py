# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Write files for input4MIPs
#
# - read already processed data off disk
# - set metadata that is universal
# - infer other metadata from data
#     - this is one to speak to Paul about, surely there already tools for this...
# - create complete set
# - write
#
# CSIRO notebook: https://github.com/climate-resource/csiro-hydrogen-esm-inputs/blob/main/notebooks/300_projected_concentrations/330_write-input4MIPs-files.py

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import cf_xarray  # noqa: F401 # required to add cf accessors
import pint_xarray
import xarray as xr
from carpet_concentrations.input4MIPs.dataset import (
    Input4MIPsDataset,
)
from openscm_units import unit_registry

from local.config import load_config_from_file
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "write_input4mips"

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

# %%
config_grid = get_config_for_step_id(config=config, step="grid", step_config_id="only")

# %% [markdown]
# ## Action

# %% [markdown]
# ## Set-up unit registry

# %%
pint_xarray.accessors.default_registry = pint_xarray.setup_registry(unit_registry)

# %% [markdown]
# ## Load data
#
# In future, this should load all the gridded data, having already been crunched to global-means, annual-means etc.

# %%
raw_gridded = xr.open_dataset(
    config_grid.processed_data_file, use_cftime=True
).pint.quantify()
raw_gridded

# %%
assert (
    False
), "Introduce infer_metadata or other function to help auto-fill, see also notes at top of notebook"

# %%
# raw_gridded.loc[dict(region="World", scenario="historical")][
#     ["Atmospheric Concentrations|CO2"]
# ]

# %%
# TODO: pull this from config
source_version = "0.1.0"

# %%
metadata_universal = dict(
    contact="zebedee.nicholls@climate-resource.com;malte.meinshausen@climate-resource.com",
    # dataset_category="GHGConcentrations",
    # frequency="mon",
    further_info_url="TBD TODO",
    # grid_label="{grid_label}",  # TODO: check this against controlled vocabulary
    # nominal_resolution="{nominal_resolution}",
    institution="Climate Resource, Fitzroy, Victoria 3065, Australia",
    institution_id="CR",
    # realm="atmos",
    # source_id="{scenario}",
    source_version=source_version,
    # target_mip="ScenarioMIP",
    # title="{equal-to-source_id}",
    # Conventions="CF-1.6",
    activity_id="input4MIPs",
    mip_era="CMIP6Plus",
    source=f"CR {source_version}",
)

metadata_universal

# %%
import json
import urllib.request

cv_experiment_id_url = "https://raw.githubusercontent.com/WCRP-CMIP/CMIP6_CVs/master/CMIP6_experiment_id.json"

with urllib.request.urlopen(cv_experiment_id_url) as url:
    cv_experiment_id = json.load(url)

cv_experiment_id["version_metadata"]

# %%
import numpy as np


def lat_fifteen_deg(ds: xr.Dataset) -> bool:
    return np.allclose(
        ds.lat.values,
        np.array(
            [-82.5, -67.5, -52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5, 67.5, 82.5]
        ),
    )


def get_grid_label_nominal_resolution(ds: xr.Dataset) -> dict[str, str]:
    scenario = list(np.unique(ds["scenario"].values))
    assert len(scenario) == 1
    scenario = scenario[0]

    ds = ds.loc[{"scenario": scenario}]
    dims = ds.dims

    grid_label = None
    nominal_resolution = None
    if "lon" not in dims:
        if "lat" in dims:
            if lat_fifteen_deg(ds) and list(dims) == ["lat", "time"]:
                grid_label = "gn-15x360deg"
                nominal_resolution = "2500km"

        elif "sector" in dims:
            # TODO: more stable handling of dims and whether bounds
            # have already been added or not
            if inp["sector"].size == 3 and list(sorted(dims)) == [  # noqa: PLR2004
                "bounds",
                "sector",
                "time",
            ]:
                grid_label = "gr1-GMNHSH"
                nominal_resolution = "10000 km"

    if any([v is None for v in [grid_label, nominal_resolution]]):
        raise NotImplementedError(  # noqa: TRY003
            f"Could not determine grid_label for data: {ds}"
        )

    target_mip = cv_experiment_id["experiment_id"][scenario]["activity_id"]
    assert len(target_mip) == 1
    target_mip = target_mip[0]

    # This seems wrong logic?
    source_id = scenario

    return {
        "grid_label": grid_label,
        "nominal_resolution": nominal_resolution,
        "target_mip": target_mip,
        "source_id": source_id,
        # This seems like a misunderstanding too?
        "title": scenario,
    }


def infer_metadata_from_dataset(ds: xr.Dataset) -> dict[str, str]:
    out = {**get_grid_label_nominal_resolution(ds)}

    return out


# %%
ds = raw_gridded.loc[dict(region="World")][["Atmospheric Concentrations|CO2"]]

# grid_label="{grid_label}",  # TODO: check this against controlled vocabulary
# nominal_resolution="{nominal_resolution}",
# source_id="{scenario}",
# title="{equal-to-source_id}",
# target_mip="ScenarioMIP",
# dataset_category="GHGConcentrations",
# realm="atmos",
# Conventions="CF-1.6",
# frequency="mon",
infer_metadata_from_dataset(ds)

# %%

input4mips_ds = Input4MIPsDataset.from_metadata_autoadd_bounds_to_dimensions(
    ds,
    dimensions=tuple(ds.dims.keys()),
    metadata=metadata_junk,
)
input4mips_ds

# %%
input4mips_ds.write(Path("test-input4mips"))

# %%
Input4MIPsDataset(raw_gridded).from_metadata_autoadd_bounds_to_dimensions

# %%
Input4MIPsDataset(raw_gridded).write(".")

# %%
