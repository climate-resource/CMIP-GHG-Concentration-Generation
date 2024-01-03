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
# # NOAA
#
# Download data from the [NOAA Global Monitoring Laboratory (GML) Carbon Cycle Greenhouse Gases (CCGG) research area](https://gml.noaa.gov/ccgg/flask.html), specifically the [data page](https://gml.noaa.gov/ccgg/data/).
#
# For simplicity, here we just refer to this as the NOAA network. This is sort of line with what is done in [Forster et al., 2023](https://essd.copernicus.org/articles/15/2295/2023/essd-15-2295-2023.pdf), who call it the "NOAA Global Monitoring Laboratory (GML)" (which appears to be the name of the top-level program). Puzzlingly, this network seems to also be referred to as the [Global Greenhouse Gas Reference Network (GGGRN)](https://gml.noaa.gov/ccgg/data/) (TODO: ask someone who knows what the difference between the acronyms is meant to mean).
#
# To-do:
#
# - read 0010 and extract any insights from there
# - add in handling of in situ measurements too (in situ and flask measurements treated as different stations in M17)
# - parameterise notebook so we can do same for CH4, N2O and SF6 observations

# %% [markdown]
# ## Imports

# %%
import pooch

from local.config import load_config_from_file
from local.pydoit_nb.checklist import generate_directory_checklist
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve"

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
for url_source in config_step.noaa_network.download_urls:
    pooch.retrieve(
        url=url_source.url,
        known_hash=url_source.known_hash,
        fname=url_source.url.split("/")[-1],
        path=config_step.noaa_network.raw_dir,
        progressbar=True,
    )

# %%
generate_directory_checklist(config_step.noaa_network.raw_dir)
