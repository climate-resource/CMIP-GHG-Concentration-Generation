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
# # Quick calculate global-mean
#
# This isn't how we will do the calculations in the end, but it is a quick route to having a global-mean value with which we can then test the formats etc.

# %% [markdown]
# ## Imports

# %%
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from scmdata.run import BaseScmRun, run_append

from local.config import load_config_from_file
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "quick-crunch"

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
config_process = get_config_for_step_id(
    config=config, step="process", step_config_id="only"
)

# %% [markdown]
# ## Action

# %%
gggrn_global_mean = BaseScmRun(config_process.gggrn.processed_file_global_mean)
gggrn_global_mean

# %%
csiro_law_dome = BaseScmRun(config_process.law_dome.processed_file)
csiro_law_dome

# %%
for vdf in run_append(
    [
        csiro_law_dome,
        gggrn_global_mean,
    ]
).groupby("variable"):
    sources = []
    for source, sdf in vdf.timeseries().groupby("source"):
        sdf = sdf.copy().dropna(how="all", axis="columns")
        yearly_in_interp = (
            BaseScmRun(sdf.copy())
            .interpolate(
                [
                    dt.datetime(year, 1, 1)
                    for year in range(
                        sdf.columns.min().year, sdf.columns.max().year + 1
                    )
                ]
            )
            .timeseries()
        )
        sources.append(yearly_in_interp)

    sources = pd.concat(sources)
    avg = sources.groupby(["region", "scenario", "unit", "variable"]).mean()
    avg["source"] = "average"
    avg = BaseScmRun(avg)

    vdf.append(avg).filter(year=range(1, 2030)).lineplot(hue="source", style="variable")
    plt.show()

    vdf.append(avg).filter(year=range(1900, 2030)).lineplot(
        hue="source", style="variable"
    )
    plt.show()
    # break
