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
from pathlib import Path

import cf_xarray.units
import matplotlib.pyplot as plt
import numpy as np
import pint_xarray
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
cf_xarray.units.units.define("ppt = ppb / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "crunch_grids"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "hfc152a"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

if config_step.gas in ("co2", "ch4", "n2o"):
    step = f"calculate_{config_step.gas}_monthly_fifteen_degree_pieces"
    step_config_id_gridding_pieces_step = "only"

elif config_step.gas in (
    "c4f10",
    "c5f12",
    "c6f14",
    "c7f16",
    "cc4f8",
):
    step = "calculate_c4f10_like_monthly_fifteen_degree_pieces"
    step_config_id_gridding_pieces_step = config_step.gas

elif config_step.gas in ("c8f18",):
    step = "calculate_c8f18_like_monthly_fifteen_degree_pieces"
    step_config_id_gridding_pieces_step = config_step.gas

else:
    step = "calculate_sf6_like_monthly_fifteen_degree_pieces"
    step_config_id_gridding_pieces_step = config_step.gas

config_gridding_pieces_step = get_config_for_step_id(
    config=config,
    step=step,
    step_config_id=step_config_id_gridding_pieces_step,
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
global_annual_mean_monthly: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_gridding_pieces_step.global_annual_mean_allyears_monthly_file
).pint.quantify()
global_annual_mean_monthly.name = config_step.gas
global_annual_mean_monthly

# %%
seasonality_monthly: xr.DataArray = (
    xr.load_dataarray(  # type: ignore
        config_gridding_pieces_step.seasonality_allyears_fifteen_degree_monthly_file
    )
    .pint.quantify()
    .sel(year=global_annual_mean_monthly.year)
)
seasonality_monthly.name = "seasonality"
seasonality_monthly

# %%
lat_grad_fifteen_degree_monthly: xr.DataArray = (
    xr.load_dataarray(  # type: ignore
        config_gridding_pieces_step.latitudinal_gradient_fifteen_degree_allyears_monthly_file
    )
    .pint.quantify()
    .sel(year=global_annual_mean_monthly.year)
)
lat_grad_fifteen_degree_monthly.name = "latitudinal_gradient"
lat_grad_fifteen_degree_monthly

# %% [markdown]
# ### 15&deg; monthly file

# %%
atol_seasonality_shift = max(1e-8, global_annual_mean_monthly.max().data.m * 7.5e-5)
atol_seasonality_shift

# %%
seasonality_monthly_use = seasonality_monthly.copy()
seasonality_monthly_use_month_mean = seasonality_monthly_use.mean("month")
max_shift = np.abs(seasonality_monthly_use_month_mean.data.m).max()
if np.isclose(seasonality_monthly_use_month_mean.data.m, 0.0, atol=atol_seasonality_shift).all():
    # Force the data to zero. This is a bit of a hack, but also basically fine.
    print(f"Applying max shift of {max_shift}")
    seasonality_monthly_use = seasonality_monthly_use - seasonality_monthly_use_month_mean

else:
    msg = f"Absolute value of max shift would be {max_shift}"
    raise AssertionError(msg)

seasonality_monthly_use.mean("month")

# %%
gridding_values = (
    xr.merge([seasonality_monthly_use.copy(), lat_grad_fifteen_degree_monthly])
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
if fifteen_degree_data.min() < 0.0:
    msg = f"{fifteen_degree_data.min()=} {np.unique(fifteen_degree_data.idxmin('year'))=}"
    raise AssertionError(msg)

# %%
fifteen_degree_data_time_axis = local.xarray_time.convert_year_month_to_time(fifteen_degree_data)

# %%
print("Colour mesh plot")
fifteen_degree_data_time_axis.plot.pcolormesh(x="time", y="lat", cmap="magma_r", levels=100)
plt.show()

# %%
print("Contour plot fewer levels")
fifteen_degree_data_time_axis.plot.contour(x="time", y="lat", cmap="magma_r", levels=30)
plt.show()

# %%
print("Concs at different latitudes")
fifteen_degree_data_time_axis.sel(lat=[-87.5, 0, 87.5], method="nearest").plot.line(hue="lat", alpha=0.4)
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
        cmap="magma_r",
        levels=30,
        # alpha=0.7,
    )
)
ax.view_init(15, -135, 0)  # type: ignore
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 0.5&deg; monthly file

# %%
# try:
#     process_map_res: list[xr.DataArray] = process_map(  # type: ignore
#         local.mean_preserving_interpolation.interpolate_time_slice_parallel_helper,
#         local.xarray_time.convert_year_month_to_time(
#             fifteen_degree_data
#             # .sel(year=range(1981, 2023))
#         )
#         .pint.dequantify()
#         .groupby("time", squeeze=False),
#         max_workers=6,
#         chunksize=24,
#     )
#     interpolation_successful = True
#     print(len(process_map_res))
# except AssertionError:
#     interpolation_successful = False

# if not interpolation_successful:
#     for degrees_freedom_scalar in np.arange(2.0, 5.1, 0.25):
#         print(f"Trying {degrees_freedom_scalar=}")
#         try:
#             process_map_res: list[xr.DataArray] = process_map(  # type: ignore
#                 partial(
#                     local.mean_preserving_interpolation.interpolate_time_slice_parallel_helper,
#                     degrees_freedom_scalar=degrees_freedom_scalar,
#                 ),
#                 local.xarray_time.convert_year_month_to_time(
#                     fifteen_degree_data
#                     # .sel(year=range(1981, 2023))
#                 )
#                 .pint.dequantify()
#                 .groupby("time", squeeze=False),
#                 max_workers=6,
#                 chunksize=24,
#             )
#             print(f"Run succeeded with {degrees_freedom_scalar=}")
#             break
#         except AssertionError:
#             print(f"Run failed with {degrees_freedom_scalar=}")
#             continue

#     else:
#         msg = "Mean-preserving interpolation failed, consider increasing degrees_freedom_scalar"
#         raise AssertionError(msg)

# len(process_map_res)

# half_degree_data_l = []
# for map_res in tqdman.tqdm(process_map_res):
#     half_degree_data_l.append(map_res[1].assign_coords(time=map_res[0]))

# half_degree_data = local.xarray_time.convert_time_to_year_month(
#     xr.concat(half_degree_data_l, "time")
# )
# half_degree_data.name = fifteen_degree_data.name
# half_degree_data

# np.testing.assert_allclose(
#     fifteen_degree_data.transpose("year", "month", "lat").data.m,
#     half_degree_data.groupby_bins("lat", bins=local.binning.LAT_BIN_BOUNDS)  # type: ignore
#     .apply(local.xarray_space.calculate_global_mean_from_lon_mean)
#     .transpose("year", "month", "lat_bins")
#     .data.m,
#     atol=5e-6,  # Tolerance of our mean-preserving algorithm
# )

# print("Flying carpet")
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(projection="3d")
# tmp = local.xarray_time.convert_year_month_to_time(half_degree_data).copy()
# tmp = tmp.assign_coords(time=tmp["time"].dt.year + tmp["time"].dt.month / 12)
# (
#     tmp.isel(time=range(-150, 0)).plot.surface(
#         x="time",
#         y="lat",
#         ax=ax,
#         cmap="magma_r",
#         levels=30,
#         # alpha=0.7,
#     )
# )
# ax.view_init(15, -135, 0)  # type: ignore
# plt.tight_layout()
# plt.show()

# %% [markdown]
# ### Global-mean

# %%
global_mean_monthly = local.xarray_space.calculate_global_mean_from_lon_mean(fifteen_degree_data)
global_mean_monthly

# %%
print("Global-mean")
local.xarray_time.convert_year_month_to_time(global_mean_monthly).plot()  # type: ignore
plt.show()

# %% [markdown]
# ### Hemispheric-means

# %%
hemispheric_means_l = []
for lat_use, lat_sel in (
    (-45.0, fifteen_degree_data["lat"] < 0),
    (45.0, fifteen_degree_data["lat"] > 0),
):
    tmp = local.xarray_space.calculate_global_mean_from_lon_mean(fifteen_degree_data.sel(lat=lat_sel))
    tmp = tmp.assign_coords(lat=lat_use)
    hemispheric_means_l.append(tmp)

hemispheric_means_monthly = xr.concat(hemispheric_means_l, "lat")
hemispheric_means_monthly

# %%
print("Global-mean")
local.xarray_time.convert_year_month_to_time(hemispheric_means_monthly).plot(hue="lat")
plt.show()

# %% [markdown]
# ### Global-, hemispheric-means, annual-means

# %%
global_mean_annual_mean = global_mean_monthly.mean("month")
hemispheric_means_annual_mean = hemispheric_means_monthly.mean("month")

# %%
print("Annual-, global-mean")
global_mean_annual_mean.plot()  # type: ignore
plt.show()

hemispheric_means_annual_mean.plot()
plt.show()

# %% [markdown]
# ### Save

# %%
config_step.fifteen_degree_monthly_file.parent.mkdir(exist_ok=True, parents=True)
fifteen_degree_data.pint.dequantify().to_netcdf(config_step.fifteen_degree_monthly_file)
fifteen_degree_data

# %%
# config_step.half_degree_monthly_file.parent.mkdir(exist_ok=True, parents=True)
# half_degree_data.pint.dequantify().to_netcdf(config_step.half_degree_monthly_file)
# half_degree_data

# %%
config_step.global_mean_monthly_file.parent.mkdir(exist_ok=True, parents=True)
global_mean_monthly.pint.dequantify().to_netcdf(config_step.global_mean_monthly_file)
global_mean_monthly

# %%
config_step.hemispheric_mean_monthly_file.parent.mkdir(exist_ok=True, parents=True)
hemispheric_means_monthly.pint.dequantify().to_netcdf(config_step.hemispheric_mean_monthly_file)
hemispheric_means_monthly

# %%
config_step.global_mean_annual_mean_file.parent.mkdir(exist_ok=True, parents=True)
global_mean_annual_mean.pint.dequantify().to_netcdf(config_step.global_mean_annual_mean_file)
global_mean_annual_mean

# %%
config_step.hemispheric_mean_annual_mean_file.parent.mkdir(exist_ok=True, parents=True)
hemispheric_means_annual_mean.pint.dequantify().to_netcdf(config_step.hemispheric_mean_annual_mean_file)
hemispheric_means_annual_mean
