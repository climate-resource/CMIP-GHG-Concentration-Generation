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

# %% [markdown]
# ## Imports

# %%
import cf_xarray.units
import matplotlib.pyplot as plt
import pint
import pint_xarray
import tqdm.autonotebook as tqdman
import xarray as xr

import local.binned_data_interpolation
import local.binning
import local.latitudinal_gradient
import local.mean_preserving_interpolation
import local.raw_data_processing
import local.seasonality
import local.xarray_space
import local.xarray_time
from local.config import load_config_from_file

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

Quantity = pint.get_application_registry().Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "crunch_other_grids"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
# config_step = get_config_for_step_id(
#     config=config, step=step, step_config_id=step_config_id
# )

# config_monthly_15_degree_step = get_config_for_step_id(
#     config=config, step=f"calculate_{config_step.gas}_monthly_15_degree", step_config_id="only"
# )
config_monthly_15_degree_step = config.calculate_ch4_monthly_15_degree[0]


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
monthly_15_degree = xr.load_dataarray(
    config_monthly_15_degree_step.processed_data_file
).pint.quantify()
monthly_15_degree

# %% [markdown]
# ### Demonstrate mean-preserving interpolation
#
# TODO: write some formal unit tests for this.

# %%
time_demo = (
    monthly_15_degree.mean("month").sel(lat=7.5).sel(year=range(1950, 2023)).squeeze()
)
time_demo_monthly = (
    local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly(time_demo)
)

# %%
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

local.xarray_time.convert_year_month_to_time(time_demo_monthly).plot(ax=axes[0])
local.xarray_time.convert_year_to_time(time_demo).plot.scatter(
    x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[0]
)

local.xarray_time.convert_year_month_to_time(
    time_demo_monthly.sel(year=time_demo_monthly["year"][-10:])
).plot(ax=axes[1])
local.xarray_time.convert_year_to_time(
    time_demo.sel(year=time_demo_monthly["year"][-10:])
).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[1])

local.xarray_time.convert_year_month_to_time(
    time_demo_monthly.sel(year=time_demo_monthly["year"][1:10])
).plot(ax=axes[2])
local.xarray_time.convert_year_to_time(
    time_demo.sel(year=time_demo_monthly["year"][1:10])
).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[2])

plt.tight_layout()
plt.show()

# %%
lat_15_degree_demo = monthly_15_degree.sel(year=2022, month=6)
lat_05_degree_demo = (
    local.mean_preserving_interpolation.interpolate_lat_15_degree_to_half_degree(
        lat_15_degree_demo
    )
)

# %%
fig, ax = plt.subplots(ncols=1, figsize=(12, 4))

lat_15_degree_demo.plot.scatter(x="lat", color="tab:orange", zorder=3, alpha=0.5, ax=ax)
lat_05_degree_demo.plot(ax=ax)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 0.5 degree resolution

# %%
monthly_05_degree_l = []
for time, time_da in tqdman.tqdm(
    local.xarray_time.convert_year_month_to_time(
        monthly_15_degree.sel(year=range(2021, 2023))
    ).groupby("time", squeeze=False)
):
    time_intp = (
        local.mean_preserving_interpolation.interpolate_lat_15_degree_to_half_degree(
            time_da
        )
    )
    time_intp.assign_coords(time=time)
    monthly_05_degree_l.append(time_intp)

monthly_05_degree = xr.concat(monthly_05_degree_l, "time")
monthly_05_degree

# %% [markdown]
# ### Global-, northern-hemisphere- and southern-hemisphere-means

# %%
# assert False, "Skip this until we've discussed sensible convention with Paul"

# %%
# global_mean = local.xarray_space.calculate_global_mean_from_lon_mean(monthly_15_degree)
# global_mean

# %%
# nh_mean = local.xarray_space.calculate_global_mean_from_lon_mean(monthly_15_degree.sel(lat=monthly_15_degree["lat"]>0))
# nh_mean

# %%
# sh_mean = local.xarray_space.calculate_global_mean_from_lon_mean(monthly_15_degree.sel(lat=monthly_15_degree["lat"]<0))
# sh_mean

# %%
# global_mean.mean("month")

# %% [markdown]
# ### Save

# %%
