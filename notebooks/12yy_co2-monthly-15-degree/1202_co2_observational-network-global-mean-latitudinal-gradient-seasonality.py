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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # CO$_2$ - calculate observational network global- annual-mean, latitudinal gradient and seasonality
#
# Calculate global-mean, latitudinal gradient and seasonality
# based on the observational network.
# We also perform an EOF analysis for the latitudinal gradient.
# We also perform an EOF analysis for the seasonality change.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import cf_xarray.units
import matplotlib.axes
import matplotlib.pyplot as plt
import pint_xarray
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.latitudinal_gradient
import local.raw_data_processing
import local.seasonality
import local.xarray_space
import local.xarray_time
from local.config import load_config_from_file

# %%
cf_xarray.units.units.define("ppm = 1 / 1000000")
cf_xarray.units.units.define("ppb = ppm / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_co2_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
interpolated_spatial: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_step.observational_network_interpolated_file
).pint.quantify()
interpolated_spatial

# %% [markdown]
# ### Drop out any years that have nan
#
# These break everything later on

# %%
interpolated_spatial_nan_free = local.xarray_time.convert_year_month_to_time(
    local.xarray_time.convert_time_to_year_month(interpolated_spatial).dropna("year")
)
interpolated_spatial_nan_free

# %% [markdown]
# ### Longitudinal-mean

# %%
lon_mean = interpolated_spatial_nan_free.mean(dim="lon")
lon_mean.plot(hue="lat")  # type: ignore

# %% [markdown]
# ### Global-, annual-mean

# %%
global_mean = local.xarray_space.calculate_global_mean_from_lon_mean(lon_mean)
global_mean.plot()  # type: ignore

# %%
global_annual_mean = global_mean.groupby("time.year").mean()
global_annual_mean.plot()  # type: ignore

# %% [markdown]
# ### Latitudinal gradient

# %%
(
    lat_residuals_annual_mean,
    lat_gradient_full_eofs_pcs,
) = local.latitudinal_gradient.calculate_eofs_pcs(lon_mean, global_mean)
lat_gradient_full_eofs_pcs

# %%
fig, ax = plt.subplots()
n_eofs_to_show = 3

for i in range(n_eofs_to_show):
    ax.plot(
        lat_gradient_full_eofs_pcs["principal-components"].sel(eof=i).data.m,
        label=f"Principal component {i}",
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
fig, ax = plt.subplots()

for i in range(3):
    ax.plot(
        lat_gradient_full_eofs_pcs["eofs"].sel(eof=i).data.m,
        lat_gradient_full_eofs_pcs["lat"].data,
        label=f"EOF {i}",
        zorder=2 - i / 10,
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
fig, ax = plt.subplots()

for i in range(3):
    ax.plot(
        lat_gradient_full_eofs_pcs["principal-components"].sel(eof=i).isel(year=1)
        @ lat_gradient_full_eofs_pcs["eofs"].sel(eof=i),
        lat_gradient_full_eofs_pcs["lat"],
        label=f"EOF {i}",
        zorder=2 - i / 10,
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
lat_gradient_eofs_pcs = lat_gradient_full_eofs_pcs.sel(eof=range(config_step.lat_gradient_n_eofs_to_use))
lat_gradient_eofs_pcs

# %%
latitudinal_anomaly_from_eofs = lat_gradient_eofs_pcs["principal-components"] @ lat_gradient_eofs_pcs["eofs"]

for year in latitudinal_anomaly_from_eofs["year"]:
    if year % 5:
        continue

    fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)
    if isinstance(axes, matplotlib.axes.Axes):
        raise TypeError(type(axes))

    selected = lat_residuals_annual_mean.sel(year=year)
    axes[0].plot(selected.data.m, lat_residuals_annual_mean.lat)
    axes[0].set_title("Full anomaly")

    selected_eofs = latitudinal_anomaly_from_eofs.sel(year=year)
    axes[1].plot(selected_eofs.data.m, latitudinal_anomaly_from_eofs.lat)
    axes[1].set_title("Anomaly from EOFs")

    axes[2].plot(selected.data.m - selected_eofs.data.m, lat_residuals_annual_mean.lat)
    axes[2].set_title("Difference")

    axes[0].set_ylim([-90, 90])

    plt.suptitle(str(int(year)))
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Seasonality

# %%
seasonality, _, lon_mean_ym_monthly_anomalies = local.seasonality.calculate_seasonality(
    lon_mean=lon_mean,
    global_mean=global_mean,
)

# %%
lon_mean_ym_monthly_anomalies.plot.line(hue="lat", col="year", col_wrap=3)

# %%
seasonality.plot.line(hue="lat")

# %%
(
    seasonality_anomalies,
    seasonality_change_full_eofs_pcs,
) = local.seasonality.calculate_seasonality_change_eofs_pcs(lon_mean, global_mean)
seasonality_change_full_eofs_pcs

# %%
local.xarray_time.convert_year_month_to_time(seasonality_anomalies).plot.line(col="lat", col_wrap=3)
seasonality_anomalies

# %%
n_eofs_to_show = 3
(seasonality_change_full_eofs_pcs["principal-components"].sel(eof=range(n_eofs_to_show)).plot.line(hue="eof"))

# %%
(
    seasonality_change_full_eofs_pcs["eofs"]
    .unstack("lat-month")
    .sel(eof=range(n_eofs_to_show))
    .plot.line(hue="eof", col="lat", col_wrap=3)
)

# %%
eof_based_anomalies = local.xarray_time.convert_year_month_to_time(
    (
        seasonality_change_full_eofs_pcs["principal-components"] * seasonality_change_full_eofs_pcs["eofs"]
    ).unstack("lat-month")
)
(eof_based_anomalies.sel(eof=range(n_eofs_to_show)).plot.line(hue="eof", col="lat", col_wrap=3))

# %%
seasonality_change_eofs_pcs = seasonality_change_full_eofs_pcs.sel(
    eof=range(config_step.seasonality_change_n_eofs_to_use),
)
seasonality_change_eofs_pcs

# %%
seasonality_change_from_eofs = xr.dot(
    seasonality_change_eofs_pcs["principal-components"],
    seasonality_change_eofs_pcs["eofs"],
    dim="eof",
)
seasonality_change_from_eofs

# %%
(seasonality_change_eofs_pcs["principal-components"].plot.line(hue="eof"))

# %%
(seasonality_change_eofs_pcs["eofs"].unstack("lat-month").plot.pcolormesh(y="lat", x="month", col="eof"))

# %%
for year in seasonality_change_from_eofs["year"]:
    if year % 5:
        continue

    for lat in [-37.5, 7.5, 52.5, 67.5]:
        fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
        if isinstance(axes, matplotlib.axes.Axes):
            raise TypeError(type(axes))

        selected = seasonality_anomalies.sel(year=year, lat=lat)
        axes[0].plot(selected.data.m)
        axes[0].set_title("Full anomaly")

        selected_eofs = seasonality_change_from_eofs.sel(year=year, lat=lat)
        axes[1].plot(selected_eofs.data.m)
        axes[1].set_title("Anomaly from EOFs")

        axes[2].plot(selected.data.m - selected_eofs.data.m)
        axes[2].axhline(0, color="k", alpha=0.3)
        axes[2].set_title("Difference")

        # axes[0].set_ylim([-90, 90])

        plt.suptitle(f"{float(year)=} {lat=}")
        plt.tight_layout()
        plt.show()

        # break
    # break

# %% [markdown]
# ### Save

# %%
config_step.observational_network_global_annual_mean_file.parent.mkdir(exist_ok=True, parents=True)
global_annual_mean.pint.dequantify().to_netcdf(config_step.observational_network_global_annual_mean_file)
global_annual_mean

# %%
config_step.observational_network_latitudinal_gradient_eofs_file.parent.mkdir(exist_ok=True, parents=True)
lat_gradient_eofs_pcs.pint.dequantify().to_netcdf(
    config_step.observational_network_latitudinal_gradient_eofs_file
)
lat_gradient_eofs_pcs

# %% [markdown]
# Use absolute seasonality plus seasonality change for CO2

# %%
config_step.observational_network_seasonality_file.parent.mkdir(exist_ok=True, parents=True)
seasonality.pint.dequantify().to_netcdf(config_step.observational_network_seasonality_file)
seasonality

# %%
config_step.observational_network_seasonality_change_eofs_file.parent.mkdir(exist_ok=True, parents=True)
to_save = seasonality_change_eofs_pcs.unstack("lat-month")
to_save.pint.dequantify().to_netcdf(config_step.observational_network_seasonality_change_eofs_file)
to_save
