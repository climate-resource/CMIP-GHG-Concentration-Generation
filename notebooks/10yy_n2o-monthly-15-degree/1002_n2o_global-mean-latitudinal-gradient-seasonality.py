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
# # N$_2$O - calculate global- annual-mean, latitudinal gradient and seasonality
#
# Calculate global-mean, latitudinal gradient and seasonality.

# %% [markdown]
# ## Imports

# %%
import cf_xarray.units
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

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_n2o_monthly_fifteen_degree_pieces"

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
interpolated_spatial = xr.load_dataarray(  # type: ignore
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

# %% [markdown]
# ### Longitudinal-mean

# %%
lon_mean = interpolated_spatial_nan_free.mean(dim="lon")
lon_mean.plot(hue="lat")

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
    full_eofs_pcs,
) = local.latitudinal_gradient.calculate_eofs_pcs(lon_mean, global_mean)
full_eofs_pcs

# %%
fig, ax = plt.subplots()

for i in range(3):
    ax.plot(
        full_eofs_pcs["principal-components"].sel(eof=i).data.m,
        label=f"Principal component {i}",
    )
    if i > 3:  # noqa: PLR2004
        break

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
fig, ax = plt.subplots()

for i in range(3):
    ax.plot(
        full_eofs_pcs["eofs"].sel(eof=i).data.m,
        full_eofs_pcs["lat"].data,
        label=f"EOF {i}",
        zorder=2 - i / 10,
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
fig, ax = plt.subplots()

for i in range(3):
    ax.plot(
        full_eofs_pcs["principal-components"].sel(eof=i).isel(year=1)
        @ full_eofs_pcs["eofs"].sel(eof=i),
        full_eofs_pcs["lat"],
        label=f"EOF {i}",
        zorder=2 - i / 10,
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
eofs_pcs = full_eofs_pcs.sel(eof=range(config_step.lat_gradient_n_eofs_to_use))
eofs_pcs

# %%
latitudinal_anomaly_from_eofs = eofs_pcs["principal-components"] @ eofs_pcs["eofs"]

for year in latitudinal_anomaly_from_eofs["year"]:
    if year % 5:
        continue

    fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)

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
(
    seasonality,
    relative_seasonality,
    lon_mean_ym_monthly_anomalies,
) = local.seasonality.calculate_seasonality(
    lon_mean=lon_mean,
    global_mean=global_mean,
)

# %%
lon_mean_ym_monthly_anomalies.plot.line(hue="lat", col="year", col_wrap=3)

# %%
seasonality.plot.line(hue="lat")

# %%
relative_seasonality.plot.line(hue="lat")

# %% [markdown]
# ### Save

# %%
config_step.observational_network_global_annual_mean_file.parent.mkdir(
    exist_ok=True, parents=True
)
global_annual_mean.pint.dequantify().to_netcdf(
    config_step.observational_network_global_annual_mean_file
)
global_annual_mean

# %%
config_step.observational_network_latitudinal_gradient_eofs_file.parent.mkdir(
    exist_ok=True, parents=True
)
eofs_pcs.pint.dequantify().to_netcdf(
    config_step.observational_network_latitudinal_gradient_eofs_file
)
eofs_pcs

# %%
# Use relative seasonality for CH4
config_step.observational_network_seasonality_file.parent.mkdir(
    exist_ok=True, parents=True
)
relative_seasonality.pint.dequantify().to_netcdf(
    config_step.observational_network_seasonality_file
)
relative_seasonality
