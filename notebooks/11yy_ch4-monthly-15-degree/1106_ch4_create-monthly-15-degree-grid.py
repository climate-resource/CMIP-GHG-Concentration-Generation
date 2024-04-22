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
# # CH$_4$ - create monthly 15&deg; gridded file
#
# We have all the pieces now, the only remaining step is mean-preserving interpolation.

# %% [markdown]
# ## Imports

# %%
import carpet_concentrations.gridders
import cf_xarray.units
import matplotlib.pyplot as plt
import pint
import pint_xarray
import xarray as xr
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

Quantity = pint.get_application_registry().Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_ch4_monthly_15_degree"

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
global_annual_mean_monthly = xr.load_dataarray(
    config_step.global_annual_mean_monthly_file
).pint.quantify()
global_annual_mean_monthly.name = config_step.gas
global_annual_mean_monthly

# %%
latitudinal_gradient = xr.load_dataarray(
    config_step.latitudinal_gradient_file
).pint.quantify()
latitudinal_gradient.name = "latitudinal_gradient"
latitudinal_gradient

# %%
seasonality = xr.load_dataarray(config_step.seasonality_file).pint.quantify()
seasonality.name = "seasonality"
seasonality

# %% [markdown]
# ### Create full grids

# %%
gridding_values = (
    xr.merge(
        [
            seasonality,
            latitudinal_gradient,
        ]
    )
    .cf.add_bounds("lat")
    .pint.quantify({"lat_bounds": "degrees_north"})
)

# %%
gridded = carpet_concentrations.gridders.LatitudeSeasonalityGridder(
    gridding_values=gridding_values
).calculate(global_annual_mean_monthly.to_dataset())
gridded

# %%
gridded_time_axis = local.xarray_time.convert_year_month_to_time(gridded)

# %%
print("Annual-, global-mean")
gridded_time_axis[config_step.gas].groupby("time.year").mean().mean("lat").plot()
plt.show()

# %%
print("Colour mesh plot")
gridded_time_axis[config_step.gas].plot.pcolormesh(
    x="time", y="lat", cmap="rocket_r", levels=100
)
plt.show()

# %%
print("Contour plot fewer levels")
gridded_time_axis[config_step.gas].plot.contour(
    x="time", y="lat", cmap="rocket_r", levels=30
)
plt.show()

# %%
print("Concs at different latitudes")
gridded_time_axis[config_step.gas].sel(
    lat=[-87.5, 0, 87.5], method="nearest"
).plot.line(hue="lat", alpha=0.4)
plt.show()

# %%
print("Annual-mean concs at different latitudes")
gridded_time_axis[config_step.gas].sel(lat=[-87.5, 0, 87.5], method="nearest").groupby(
    "time.year"
).mean().plot.line(hue="lat", alpha=0.4)
plt.show()

# %%
print("Flying carpet")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
tmp = gridded_time_axis[config_step.gas].copy()
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
# ### Save

# %%
out = gridded[config_step.gas]
out

# %%
config_step.processed_data_file.parent.mkdir(exist_ok=True, parents=True)
out.pint.dequantify().to_netcdf(config_step.processed_data_file)
out
