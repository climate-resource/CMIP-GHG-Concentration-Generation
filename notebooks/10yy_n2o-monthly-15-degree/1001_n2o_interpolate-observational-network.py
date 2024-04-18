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
# # N$_2$O - interpolate observational network
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
step: str = "calculate_n2o_monthly_15_degree"

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

# %%
MIN_POINTS_FOR_SPATIAL_INTERPOLATION = 3

# %%
times_l = []
interpolated_dat_l = []
year_month_plot = (
    # (1984, 1),
    # (1984, 3),
    # (1990, 1),
    # (1990, 2),
    (1988, 4),
    (1989, 4),
    (1990, 4),
    (1991, 4),
    (1992, 4),
    (1993, 4),
    (1994, 4),
    (1995, 4),
    (1996, 4),
    (1997, 4),
    (1998, 4),
    (2000, 4),
    (2022, 12),
    (2023, 1),
)

for (year, month), ymdf in tqdman.tqdm(bin_averages.groupby(["year", "month"])):
    if ymdf.shape[0] < MIN_POINTS_FOR_SPATIAL_INTERPOLATION:
        msg = f"Not enough data ({ymdf.shape[0]} data points) for {year=}, {month=}, not performing spatial interpolation"
        print(msg)
        continue

    times_l.append(cftime.datetime(year, month, 15))

    interpolated_ym = local.binned_data_interpolation.interpolate(ymdf)
    interpolated_dat_l.append(interpolated_ym)

    if (year, month) in year_month_plot:
        lon_grid, lat_grid = np.meshgrid(
            local.binned_data_interpolation.LON_BIN_CENTRES,
            local.binned_data_interpolation.LAT_BIN_CENTRES,
        )

        plt.pcolormesh(lon_grid, lat_grid, interpolated_ym, shading="auto")
        plt.plot(ymdf["lon_bin"], ymdf["lat_bin"], "ok", label="input point")
        plt.legend()
        plt.colorbar()
        # plt.axis("equal")
        plt.ylim([-90, 90])
        plt.title(f"{year} {month}")
        plt.show()

# %% [markdown]
# ### Save

# %%
assert False, "Add data from NOAA HATS https://gml.noaa.gov/hats/combined/N2O.html"
assert False, "Convert to xarray and save as netCDF"

# %%
local.binned_data_interpolation.check_data_columns_for_binned_data_interpolation(
    bins_interpolated
)
assert set(bins_interpolated["gas"]) == {config_step.gas}

# %%
config_step.processed_bin_averages_file.parent.mkdir(exist_ok=True, parents=True)
bins_interpolated.to_csv(config_step.processed_bin_averages_file, index=False)
bins_interpolated
