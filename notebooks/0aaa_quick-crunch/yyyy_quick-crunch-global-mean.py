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

# %% editable=true slideshow={"slide_type": ""}
step: str = "quick_crunch"

# %% [markdown] editable=true slideshow={"slide_type": ""}
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

# %% editable=true slideshow={"slide_type": ""}
config_process = get_config_for_step_id(
    config=config, step="process", step_config_id="only"
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %%
gggrn_global_mean = BaseScmRun(config_process.gggrn.processed_file_global_mean)
gggrn_global_mean

# %%
csiro_law_dome = BaseScmRun(config_process.law_dome.processed_file)
csiro_law_dome


# %%
def get_interp_year_month_dts(df: pd.DataFrame) -> list[dt.datetime]:
    """
    Get :obj:`dt.datetime` at the start of each year-month combination to use for interpolation

    Parameters
    ----------
    df
        :obj:`pd.DataFrame` for which to determine the timesteps to interpolate

    Returns
    -------
        Timesteps to interpolate
    """
    col_min = df.columns.min()
    col_max = df.columns.max()

    year_max = col_max.year
    month_max = col_max.month
    interp_times = [
        dt.datetime(y, m, 1)
        for y in range(col_min.year, year_max)
        for m in range(1, 12 + 1)
    ]

    current_month = 1
    while current_month <= month_max:
        interp_times.append(dt.datetime(year_max, current_month, 1))

        current_month += 1

    return interp_times


# %%
out_list = []
for vdf in run_append(
    [
        csiro_law_dome,
        gggrn_global_mean,
    ]
).groupby("variable"):
    sources_list = []
    for source, sdf in vdf.timeseries().groupby("source"):
        sdf_clean = sdf.copy().dropna(how="all", axis="columns")
        interp_steps = get_interp_year_month_dts(sdf_clean)
        yearly_in_interp = (
            BaseScmRun(sdf_clean.copy()).interpolate(interp_steps).timeseries()
        )
        sources_list.append(yearly_in_interp)

    sources = pd.concat(sources_list)
    avg = sources.groupby(["region", "scenario", "unit", "variable"]).mean()
    avg["source"] = "average"
    avg_run = BaseScmRun(avg)

    if not config.ci:
        # Interpolate onto the full timespan of interest i.e. back to year 1
        tmp: pd.DataFrame = avg.timeseries().copy()  # type: ignore
        tmp.columns = avg.time_points.as_cftime()
        tmp.loc[:, type(tmp.columns.values[0])(1, 1, 1)] = 0
        interp_years = get_interp_year_month_dts(tmp)

        avg_run = avg_run.interpolate(
            interp_years,
            extrapolation_type="constant",
        )

    fig, axes = plt.subplots(ncols=2, sharey=False, figsize=(10, 4))

    vdf.append(avg_run).lineplot(  # type: ignore
        hue="source", style="variable", ax=axes[0], time_axis="seconds since 1970-01-01"
    )

    vdf.append(avg_run).filter(year=range(1980, 2030)).lineplot(  # type: ignore
        hue="source", style="variable", ax=axes[1]
    )
    axes[1].legend().remove()
    plt.show()

    print(avg_run)
    out_list.append(avg_run)
    # break

out = run_append(out_list)

if config.ci:
    # Just do dumb resampling to get rid of nans
    out = out.resample("MS")
else:
    # Drop any years for which we don't have data for all variables
    out = BaseScmRun(out.timeseries().dropna(axis="columns"))

assert not out.timeseries().isna().any().any(), (
    out.timeseries().isna().any(axis="columns")
)
out

# %%
config_step.processed_data_file_global_means.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(config_step.processed_data_file_global_means)
config_step.processed_data_file_global_means
