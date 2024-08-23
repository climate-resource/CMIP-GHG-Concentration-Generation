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
# # Create draft Zenodo deposition
#
# Here we reserve a DOI for this version on Zenodo.

# %% [markdown]
# ## Imports

# %%
import os
from pathlib import Path

import cf_xarray.units
import pint_xarray
import requests.exceptions
from dotenv import load_dotenv
from openscm_zenodo.zenodo import ZenodoDomain, ZenodoInteractor, get_reserved_doi
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
load_dotenv()

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")
cf_xarray.units.units.define("ppt = ppb / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "create_zenodo_deposition"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)


# %% [markdown]
# ## Action

# %%
zenoodo_interactor = ZenodoInteractor(
    token=os.environ["ZENODO_TOKEN"],
    zenodo_domain=ZenodoDomain.production.value,
)
latest_deposition_id = zenoodo_interactor.get_latest_deposition_id(
    any_deposition_id=config_step.any_deposition_id,
)
latest_deposition_id

# %%
new_deposition_id = zenoodo_interactor.create_new_version_from_latest(
    latest_deposition_id=latest_deposition_id
).json()["id"]
new_deposition_id

# %%
try:
    zenoodo_interactor.remove_all_files(deposition_id=new_deposition_id)
except requests.exceptions.HTTPError:
    # assume all files already removed and move on
    pass

# %%
metadata = zenoodo_interactor.get_metadata(latest_deposition_id)
metadata["metadata"]["prereserve_doi"] = True  # type: ignore

update_metadata_response = zenoodo_interactor.update_metadata(
    deposition_id=new_deposition_id,
    metadata=metadata,
)

reserved_doi = get_reserved_doi(update_metadata_response)
reserved_doi

# %%
with open(config_step.reserved_zenodo_doi_file, "w") as fh:
    fh.write(reserved_doi)
