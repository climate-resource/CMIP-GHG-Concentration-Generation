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
step: str = "quick_crunch"

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
out_list = []
for vdf in run_append(
    [
        csiro_law_dome,
        gggrn_global_mean,
    ]
).groupby("variable"):
    sources = []
    for source, sdf in vdf.timeseries().groupby("source"):
        sdf_clean = sdf.copy().dropna(how="all", axis="columns")
        yearly_in_interp = (
            BaseScmRun(sdf_clean.copy())
            .interpolate(
                [
                    dt.datetime(year, m, 1)
                    for year in range(
                        sdf_clean.columns.min().year, sdf_clean.columns.max().year + 1
                    )
                    for m in range(1, 12 + 1)
                ]
            )
            .timeseries()
        )
        sources.append(yearly_in_interp)

    sources = pd.concat(sources)
    avg = sources.groupby(["region", "scenario", "unit", "variable"]).mean()
    avg["source"] = "average"
    avg = BaseScmRun(avg).interpolate(
        [dt.datetime(year, m, 1) for year in range(1, 2023) for m in range(1, 12 + 1)]
        + [
            dt.datetime(year, m, 1)
            for year in range(2023, 2023 + 1)
            for m in range(1, 8 + 1)
        ],
        extrapolation_type="constant",
    )

    fig, axes = plt.subplots(ncols=2, sharey=False, figsize=(10, 4))

    vdf.append(avg).lineplot(
        hue="source", style="variable", ax=axes[0], time_axis="seconds since 1970-01-01"
    )

    vdf.append(avg).filter(year=range(1980, 2030)).lineplot(
        hue="source", style="variable", ax=axes[1]
    )
    axes[1].legend().remove()
    plt.show()

    out_list.append(avg)
    # break

out = run_append(out_list)

# %%
assert not out.timeseries().isna().any().any()
out

# %%
config_step.processed_data_file_global_means.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(config_step.processed_data_file_global_means)
config_step.processed_data_file_global_means
