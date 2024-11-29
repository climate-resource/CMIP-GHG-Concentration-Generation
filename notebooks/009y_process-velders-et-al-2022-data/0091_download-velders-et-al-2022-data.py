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
# # Velders et al., 2022 - download
#
# Download data from [Velders et al., 2022](https://doi.org/10.5194/acp-22-6087-2022).

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import shutil
from pathlib import Path

import openscm_units
import pint
import pooch
from pydoit_nb.complete import write_complete_file
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
raw_data_file_l = pooch.retrieve(
    url=config_step.zenodo_record.url,
    known_hash=config_step.zenodo_record.known_hash,
    processor=pooch.Unzip(
        members=["veldersguus-HFC-scenarios-2022-859d44c/HFC_Current_Policy_2022_Scenario.xlsx"]
    ),
    progressbar=True,
)
if len(raw_data_file_l) != 1:
    raise AssertionError

raw_data_file = Path(raw_data_file_l[0])

config_step.raw_dir.mkdir(parents=True, exist_ok=True)
out_file = config_step.raw_dir / raw_data_file.name
shutil.copyfile(raw_data_file, out_file)

out_file

# %%
write_complete_file(config_step.download_complete_file)
