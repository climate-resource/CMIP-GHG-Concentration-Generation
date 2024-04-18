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
# # CH$_4$ - calculate global-mean, latitudinal gradient and seasonality
#
# Calculate global-mean, latitudinal gradient and seasonality.

# %% [markdown]
# ## Imports

# %%
import cf_xarray.units
import matplotlib.pyplot as plt
import numpy as np
import pint_xarray
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.raw_data_processing
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
interpolated_spatial = xr.load_dataarray(
    config_step.interpolated_observational_network_file
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
lon_mean

# %% [markdown]
# ### Global-mean

# %%
global_mean = local.xarray_space.calculate_global_mean_from_lon_mean(lon_mean)
global_mean.plot()
# variable

# %% [markdown]
# ### Smoothed global-mean

# %%
# TBD

# %% [markdown]
# ### Latitudinal gradient

# %%
lat_residuals = lon_mean - global_mean
lat_residuals

# %%
lat_residuals_annual_mean = local.xarray_time.convert_time_to_year_month(
    lat_residuals
).mean("month")
# lat_residuals_annual_mean

# %%
lat_residuals_annual_mean.pint.dequantify()

# %%
# Super helpful: https://www.ess.uci.edu/~yu/class/ess210b/lecture.5.EOF.all.pdf
svd_ready = lat_residuals_annual_mean.transpose("year", "lat").pint.dequantify()
U, D, Vh = np.linalg.svd(
    svd_ready,
    full_matrices=False,
)

# If you take the full SVD, you get back the original matrix
assert np.allclose(
    svd_ready,
    U @ np.diag(D) @ Vh,
)

# Principal components are the scaling factors on the EOFs
principal_components = U @ np.diag(D)

# Empirical orthogonal functions
eofs = Vh.T

# Similarly, if you use the full EOFs and principal components,
# you get back the original matrix
assert np.allclose(
    svd_ready,
    principal_components @ eofs.T,
)

# %%
fig, ax = plt.subplots()

for i, pc in enumerate(principal_components.T):
    ax.plot(pc, label=f"Principal component {i + 1}")
    if i > 3:
        break

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
fig, ax = plt.subplots()

for i in range(3):
    ax.plot(
        eofs[:, i],
        lat_residuals_annual_mean.lat,
        label=f"EOF {i + 1}",
        zorder=2 - i / 10,
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
fig, ax = plt.subplots()

for i in range(3):
    ax.plot(
        eofs[:, i] * principal_components[0, i],
        lat_residuals_annual_mean.lat,
        label=f"EOF {i + 1}",
        zorder=2 - i / 10,
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %% [markdown]
# TODO: split the below into a function

# %%
N_EOFS_TO_USE = 2

# %%
xr_principal_components_keep = xr.DataArray(
    name=f"{config_step.gas}_latitudinal-gradient_principal-components",
    data=principal_components[:, :N_EOFS_TO_USE],
    dims=["year", "eof"],
    coords=dict(year=lat_residuals_annual_mean["year"], eof=range(N_EOFS_TO_USE)),
    attrs=dict(
        description="Principal components for the latitudinal gradient EOFs",
        units="dimensionless",
    ),
).pint.quantify()
xr_principal_components_keep

# %%
xr_eofs_keep = xr.DataArray(
    name=f"{config_step.gas}_latitudinal-gradient_eofs",
    data=eofs[:, :N_EOFS_TO_USE],
    dims=["lat", "eof"],
    coords=dict(lat=lat_residuals_annual_mean["lat"], eof=range(N_EOFS_TO_USE)),
    attrs=dict(
        description="EOFs for the latitudinal gradient",
        units=lat_residuals_annual_mean.data.units,
    ),
).pint.quantify()
xr_eofs_keep

# %%
xr.merge([xr_eofs_keep, xr_principal_components_keep], combine_attrs="drop_conflicts")

# %%
assert False, "Clean this up too"

# %%
latitudinal_anomaly_from_eofs = xr_principal_components_keep @ xr_eofs_keep

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

    plt.suptitle(int(year))
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Seasonality

# %%
lon_mean_ym = local.xarray_time.convert_time_to_year_month(lon_mean)
if lon_mean_ym.isnull().any():
    msg = "Drop out any years with nan data before starting"
    raise AssertionError(msg)

lon_mean_ym_annual_mean = lon_mean_ym.mean("month")
lon_mean_ym_monthly_anomalies = lon_mean_ym - lon_mean_ym_annual_mean
lon_mean_ym_monthly_anomalies_year_average = lon_mean_ym_monthly_anomalies.mean("year")
seasonality = lon_mean_ym_monthly_anomalies_year_average
np.testing.assert_allclose(
    seasonality.mean("month").pint.to("ppb").pint.dequantify(), 0.0, atol=1e-13
)
seasonality.plot.line(hue="lat")
seasonality

# %%
seasonality.plot.line(hue="lat")

# %% [markdown]
# ### Save

# %%
assert False, "Save global-mean"
assert False, "Save global-mean smoothed"
assert False, "Save latitudinal gradient EOF and PC"
assert False, "Save seasonality"

# %%
local.binned_data_interpolation.check_data_columns_for_binned_data_interpolation(
    bins_interpolated
)
assert set(bins_interpolated["gas"]) == {config_step.gas}

# %%
config_step.processed_bin_averages_file.parent.mkdir(exist_ok=True, parents=True)
bins_interpolated.to_csv(config_step.processed_bin_averages_file, index=False)
bins_interpolated
