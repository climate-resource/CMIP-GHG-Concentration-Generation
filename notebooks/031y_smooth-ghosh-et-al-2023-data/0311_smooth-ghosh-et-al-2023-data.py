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
# # Ghosh et al., 2023 - smoothing
#
# Smooth the Ghosh et al., 2023 data.
#
# Just applies a very basic smoothing spline
# because the underlying data is so dense already.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import scipy.interpolate  # type: ignore
from openscm_units import unit_registry
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
ur = unit_registry
pint.set_application_registry(ur)  # type: ignore
Q = ur.Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "smooth_ghosh_et_al_2023_data"

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

config_process_ghosh_et_al = get_config_for_step_id(
    config=config, step="retrieve_and_process_ghosh_et_al_2023_data", step_config_id="only"
)

# %% [markdown]
# ## Action

# %%
gas_df = pd.read_csv(config_process_ghosh_et_al.processed_data_file)
gas_df = gas_df.sort_values("year")
gas_df

# %%
gas_unit = gas_df["unit"].unique()
if len(gas_unit) > 1:
    msg = f"More than one unit {gas_unit=}"
    raise ValueError(msg)
gas_unit = gas_unit[0]

gas_name = gas_df["gas"].unique()
if len(gas_name) > 1:
    msg = f"More than one name {gas_name=}"
    raise ValueError(msg)
gas_name = gas_name[0]

x_raw = Q(gas_df["year"].values, "yr")
y_raw = Q(gas_df["value"].values, gas_unit)  # type: ignore

# %%
plt.scatter(x_raw.m, y_raw.m)
plt.xlabel(str(x_raw.units))
plt.ylabel(str(y_raw.units))

# %% [markdown]
# ## Create the smoothing spline

# %%
smoothing_spline = scipy.interpolate.make_smoothing_spline(
    x_raw.m,
    y_raw.m,
    # lam=None,  # let defaults do their thing
)

# %%
x_annual = np.arange(np.ceil(gas_df["year"].min()), np.floor(gas_df["year"].max()) + 1.0) * x_raw.u
y_smoothed = smoothing_spline(x_annual.m) * y_raw.u
y_smoothed

# %% [markdown]
# Plot

# %%
fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(16, 4))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))

for ax, xlim in zip(axes, ((0, 2025), (1750, 2025), (1950, 2025))):
    ax.scatter(
        x_raw.m,
        y_raw.m,
        alpha=0.6,
        s=20,
        zorder=2,
    )

    ax.plot(
        x_annual.m,
        y_smoothed.m,
        color="tab:green",
        alpha=0.8,
        linewidth=3,
    )

    ax.set_xlabel("year")
    ax.set_xlim(xlim)

ax.set_ylabel(y_smoothed.units)


# %% [markdown]
# ## Write output


# %%
smoothed_df = pd.DataFrame(
    y_smoothed.m,
    columns=["value"],
    index=pd.Index(x_annual.m, name="year"),
)
smoothed_df["unit"] = gas_unit
smoothed_df = smoothed_df.reset_index()
smoothed_df["gas"] = gas_name
smoothed_df

# %%
config_step.smoothed_file.parent.mkdir(exist_ok=True, parents=True)
smoothed_df.to_csv(config_step.smoothed_file, index=False)
config_step.smoothed_file
