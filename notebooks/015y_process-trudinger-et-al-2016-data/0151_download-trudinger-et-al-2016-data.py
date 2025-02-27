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
# # Trudinger et al., 2016 - download
#
# Download data from [Trudinger et al., 2016](https://doi.org/10.5194/acp-16-11733-2016).

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
raw_data_files_l = pooch.retrieve(
    url=config_step.supplement.url,
    known_hash=config_step.supplement.known_hash,
    processor=pooch.Unzip(members=["Supplementary_Data_Trudinger_et_al.xlsx"]),
    progressbar=True,
)
if isinstance(raw_data_files_l, Path):
    raise TypeError

for raw_data_file in [Path(f) for f in raw_data_files_l]:
    config_step.raw_dir.mkdir(parents=True, exist_ok=True)
    out_file = config_step.raw_dir / raw_data_file.name
    shutil.copyfile(raw_data_file, out_file)

    print(out_file)

# %%
write_complete_file(config_step.download_complete_file)
