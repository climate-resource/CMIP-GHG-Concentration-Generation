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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # CH$_4$ - extend the latitudinal gradient and global-, annual-mean
#
# Extend the latitudinal gradient and global-, annual-mean back in time.
# For CH$_4$, we do this by combining the values from ice cores etc.
# and our latitudinal gradient information.

# %% [markdown]
# ## Imports

# %%

import cf_xarray.units
import openscm_units
import pint
import pint_xarray
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

Quantity = pint.get_application_registry().Quantity

# %%
QuantityOSCM = openscm_units.unit_registry.Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_ch4_monthly_15_degree"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_smooth_law_dome_data = get_config_for_step_id(
    config=config, step="smooth_law_dome_data", step_config_id="ch4"
)

config_process_epica = get_config_for_step_id(
    config=config, step="retrieve_and_process_epica_data", step_config_id="only"
)

config_process_neem = get_config_for_step_id(
    config=config, step="retrieve_and_process_neem_data", step_config_id="only"
)

config_retrieve_misc = get_config_for_step_id(
    config=config, step="retrieve_misc_data", step_config_id="only"
)


# %% [markdown]
# ## Action
