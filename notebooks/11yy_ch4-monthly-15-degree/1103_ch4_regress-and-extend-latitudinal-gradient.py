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
# # CH$_4$ - regress and extend latitudinal gradient
#
# Extend the latitudinal gradient back in time. For CH$_4$, we do this by keeping principal component 2 constant and by extending principal component 1 back in time based on a regression with fossil and industrial CH$_4$ emissions.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pint
import pint_xarray
import primap2
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
pint.set_application_registry(openscm_units.unit_registry)

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

config_retrieve_misc = get_config_for_step_id(
    config=config, step="retrieve_misc_data", step_config_id="only"
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
lat_grad_eofs = xr.load_dataset(config_step.observational_network_latitudinal_gradient_eofs_file).pint.quantify()
lat_grad_eofs

# %% [markdown]
# ### Extend EOF one
#
# (Zero-indexing, hence this is the second EOF)

# %%
# Quick assertion that things are as expected
if len(lat_grad_eofs["eof"]) != 2:
    raise AssertionError("Rethink")

# %%
new_years = np.arange(1, lat_grad_eofs["year"].max() + 1)
new_years

# %%
current_pc2 = lat_grad_eofs["principal-components"].sel(eof=1).data.m
new_pc2 = np.hstack([
    current_pc2[0] * np.ones(new_years.size - current_pc2.size),
    current_pc2,
])

if new_pc2.size != new_years.size:
    msg = "PC2 extrapolation led to wrong size"
    raise AssertionError(msg)

new_pc2

# %% [markdown]
# ### Extend EOF zero
#
# (Zero-indexing, hence this is the first EOF)

# %%
current_pc1 = lat_grad_eofs["principal-components"].sel(eof=0)

# %%
primap_full = primap2.open_dataset(config_retrieve_misc.primap.raw_dir / config_retrieve_misc.primap.download_url.url.split("/")[-1])
primap_full

# %%
fig, axes = plt.subplots(ncols=2)

primap_fossil_ch4_emissions = (
    local
    .xarray_time
    .convert_time_to_year_month(primap_full)
    .pr.loc[
        {"category": "M.0.EL", "scenario": "HISTTP", "area": "EARTH", "month": 1}
    ][config_step.gas.upper()]
    .squeeze()
    .pint.to("MtCH4 / yr")
)
primap_regression_data = (
    primap_fossil_ch4_emissions
    .sel(year=current_pc1["year"])

)
primap_regression_data.plot(ax=axes[0])

current_pc1.plot(ax=axes[1])
plt.tight_layout()

# %%
x = Quantity(primap_regression_data.data, str(primap_regression_data.data.units))
A = np.vstack([x.m, np.ones(x.size)]).T
y = Quantity(current_pc1.data, str(current_pc1.data.units))

# %%
res = np.linalg.lstsq(A, y.m, rcond=None)
m, c = res[0]
m = Quantity(m, (y / x).units)
c = Quantity(c, y.units)

fig, ax = plt.subplots()
ax.scatter(x, y, label="raw data")
ax.plot(x, m * x + c, color="tab:orange", label="regression")
ax.set_ylabel("PC")
ax.set_xlabel("PRIMAP emissions")
ax.legend()

# %%
fig, ax = plt.subplots()
primap_regression_data_extended = primap_fossil_ch4_emissions
primap_regression_data_extended.plot(ax=ax)
ax.set_ylim(ymin=0)

# %%
m * primap_regression_data_extended.data + c

# %%
m

# %%
c

# %%
c

# %%
- grab PRIMAP data
- get out M0EL CH4
- get years that match our PC years
- regress (CH4 on x, PC on y)
- assume anthropogenic emissions zero before the start of PRIMAP so we can extend emissions back in time


# %%

# %%
assert False

# %%
new_pcs = np.vstack([
    new_pc1,
    new_pc2,
]).T

# Check that we didn't break the data as we put everything back together
xr.testing.assert_equal(
   xr_new_pcs.sel(year=lat_grad_eofs["year"]),
   lat_grad_eofs["principal-components"],
)

xr_new_pcs = xr.DataArray(
    name="principal-components",
    data=new_pcs,
    dims=["year", "eof"],
    coords=dict(
        year=new_years,
        eof=range(new_pcs.shape[1]),
    ),
    attrs=dict(
        description=(
            "Principal components for the latitudinal gradient EOFs, "
            "extended to cover the whole time period"
        ),
        units=lat_grad_eofs["principal-components"].data.units,
    ),
).pint.quantify()
xr_new_pcs

# %%

    # %%
    xr_principal_components_keep = xr.DataArray(
        name="principal-components",
        data=principal_components,
        dims=["year", "eof"],
        coords=dict(
            year=lat_residuals_annual_mean["year"],
            eof=range(principal_components.shape[1]),
        ),
        attrs=dict(
            description="Principal components for the latitudinal gradient EOFs",
            units="dimensionless",
        ),
    ).pint.quantify()


# %%
lat_grad_eofs["principal-components"]

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
# ### Global-mean

# %%
global_mean = local.xarray_space.calculate_global_mean_from_lon_mean(lon_mean)
global_mean.plot()

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
        label=f"Principal component {i + 1}",
    )
    if i > 3:
        break

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# %%
fig, ax = plt.subplots()

for i in range(3):
    ax.plot(
        full_eofs_pcs["eofs"].sel(eof=i).data.m,
        full_eofs_pcs["lat"].data,
        label=f"EOF {i + 1}",
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
        label=f"EOF {i + 1}",
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

    plt.suptitle(int(year))
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Seasonality

# %%
seasonality, relative_seasonality = local.seasonality.calculate_seasonality(
    lon_mean=lon_mean,
    global_mean=global_mean,
)

# %%
seasonality.plot.line(hue="lat")

# %%
relative_seasonality.plot.line(hue="lat")

# %% [markdown]
# ### Save

# %%
config_step.observational_network_global_mean_file.parent.mkdir(
    exist_ok=True, parents=True
)
global_mean.pint.dequantify().to_netcdf(
    config_step.observational_network_global_mean_file
)
global_mean

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
