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
# # Grid data
#
# Here we create our gridded data based on the gridding pieces
# created in earlier steps.
# We create four data products:
#
# - 15&deg; latitudinal, monthly
# - 0.5&deg; latitudinal, monthly
# - global-, northern hemisphere-mean, southern-hemisphere mean, monthly
# - global-, northern hemisphere-mean, southern-hemisphere mean, annual-mean

# %% [markdown]
# ## Imports

# %%
import cf_xarray.units
import matplotlib.pyplot as plt
import pint_xarray
import tqdm.autonotebook as tqdman
import xarray as xr
from carpet_concentrations.gridders.latitude_seasonality_gridder import (
    LatitudeSeasonalityGridder,
)
from pydoit_nb.config_handling import get_config_for_step_id

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

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "crunch_grids"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "ch4"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_gridding_pieces_step = get_config_for_step_id(
    config=config,
    step=f"calculate_{config_step.gas}_monthly_fifteen_degree_half_degree",
    step_config_id="only",
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
global_annual_mean_monthly = xr.load_dataarray(
    config_gridding_pieces_step.global_annual_mean_allyears_monthly_file
).pint.quantify()
global_annual_mean_monthly.name = config_step.gas
global_annual_mean_monthly

# %%
seasonality_monthly = xr.load_dataarray(
    config_gridding_pieces_step.seasonality_allyears_monthly_file
).pint.quantify()
seasonality_monthly.name = "seasonality"
seasonality_monthly

# %%
lat_grad_fifteen_degree_monthly = xr.load_dataarray(
    config_gridding_pieces_step.latitudinal_gradient_allyears_monthly_file
).pint.quantify()
lat_grad_fifteen_degree_monthly.name = "latitudinal_gradient"
lat_grad_fifteen_degree_monthly

# %%
lat_grad_half_degree_monthly = xr.load_dataarray(
    config_gridding_pieces_step.latitudinal_gradient_half_degree_allyears_monthly_file
).pint.quantify()
lat_grad_half_degree_monthly.name = "latitudinal_gradient"
lat_grad_half_degree_monthly

# %% [markdown]
# ### 15 &deg; monthly file

# %%
gridding_values = (
    xr.merge([seasonality_monthly.copy(), lat_grad_fifteen_degree_monthly])
    .cf.add_bounds("lat")
    .pint.quantify({"lat_bounds": "deg"})
)
gridding_values

# %%
fifteen_degree_data = LatitudeSeasonalityGridder(gridding_values).calculate(
    global_annual_mean_monthly.to_dataset()
)[config_step.gas]
fifteen_degree_data

# %%
fifteen_degree_data_time_axis = local.xarray_time.convert_year_month_to_time(
    fifteen_degree_data
)

# %%
print("Colour mesh plot")
fifteen_degree_data_time_axis.plot.pcolormesh(
    x="time", y="lat", cmap="rocket_r", levels=100
)
plt.show()

# %%
print("Contour plot fewer levels")
fifteen_degree_data_time_axis.plot.contour(
    x="time", y="lat", cmap="rocket_r", levels=30
)
plt.show()

# %%
print("Concs at different latitudes")
fifteen_degree_data_time_axis.sel(lat=[-87.5, 0, 87.5], method="nearest").plot.line(
    hue="lat", alpha=0.4
)
plt.show()

# %%
print("Flying carpet")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
tmp = fifteen_degree_data_time_axis.copy()
tmp = tmp.assign_coords(time=tmp["time"].dt.year + tmp["time"].dt.month / 12)
(
    tmp.isel(time=range(-150, 0)).plot.surface(
        x="time",
        y="lat",
        ax=ax,
        cmap="rocket_r",
        levels=30,
        # alpha=0.7,
    )
)
ax.view_init(15, -135, 0)  # type: ignore
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 0.5 &deg; monthly file

# %%
lat_grad_half_degree_monthly

# %%
gridding_values_half_degree = (
    xr.merge([seasonality_monthly.copy(), lat_grad_half_degree_monthly])
    .cf.add_bounds("lat")
    .pint.quantify({"lat_bounds": "deg"})
)
gridding_values_half_degree

# %%

# %%
half_degree_data = LatitudeSeasonalityGridder(gridding_values_half_degree).calculate(
    global_annual_mean_monthly.to_dataset()
)[config_step.gas]
half_degree_data

# %%
print("Annual-mean concs at different latitudes")
gridded_time_axis[config_step.gas].sel(lat=[-87.5, 0, 87.5], method="nearest").groupby(
    "time.year"
).mean().plot.line(hue="lat", alpha=0.4)
plt.show()

# %%
print("Annual-, global-mean")
fifteen_degree_data_time_axis[config_step.gas].groupby("time.year").mean().mean(
    "lat"
).plot()
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
