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
# # CH$_4$ - interpolate observational network
#
# Interpolate the observational network data onto our grid.

# %% [markdown]
# ## Imports

# %%
import cftime
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.raw_data_processing
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_ch4_monthly_fifteen_degree_pieces"

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


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
bin_averages = pd.read_csv(config_step.processed_bin_averages_file)
bin_averages

# %% [markdown]
# ## Interpolate

# %%
MIN_POINTS_FOR_SPATIAL_INTERPOLATION = 4

# %%
times_l = []
interpolated_dat_l = []
year_month_plot = (
    (1983, 1),
    (1984, 1),
    (1984, 3),
    (2000, 4),
    (2022, 12),
    (2023, 1),
    (2023, 12),
)

for (year, month), ymdf in tqdman.tqdm(bin_averages.groupby(["year", "month"])):
    if ymdf.shape[0] < MIN_POINTS_FOR_SPATIAL_INTERPOLATION:
        msg = f"Not enough data ({ymdf.shape[0]} data points) for {year=}, {month=}, not performing spatial interpolation"
        print(msg)
        continue

    times_l.append(cftime.datetime(year, month, 15))

    interpolated_ym = local.binned_data_interpolation.interpolate(ymdf)
    show_plot = False
    if np.isnan(interpolated_ym).any():
        msg = f"Nan data after interpolation for {year=}, {month=}, not including spatial interpolation in output"
        print(msg)
        show_plot = True

    else:
        interpolated_dat_l.append(interpolated_ym)

    if (year, month) in year_month_plot or show_plot:
        # This will break if we ever change our internal gridding logic, ok for now.
        lon_grid, lat_grid = np.meshgrid(
            local.binning.LON_BIN_CENTRES,
            local.binning.LAT_BIN_CENTRES,
        )

        plt.pcolormesh(lon_grid, lat_grid, interpolated_ym.T, shading="auto")
        plt.plot(ymdf["lon_bin"], ymdf["lat_bin"], "ok", label="input point")
        plt.legend()
        plt.colorbar()
        # plt.axis("equal")
        plt.ylim([-90, 90])
        plt.title(f"{year} {month}")
        plt.show()

# %%
out = local.binned_data_interpolation.to_xarray_dataarray(
    name=config_step.gas,
    bin_averages_df=bin_averages,
    data=interpolated_dat_l,
    times=times_l,
)
out

# %% [markdown]
# ### Save

# %%
config_step.observational_network_interpolated_file.parent.mkdir(
    exist_ok=True, parents=True
)
out.to_netcdf(config_step.observational_network_interpolated_file)
out
