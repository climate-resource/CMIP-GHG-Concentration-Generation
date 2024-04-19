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
lat_grad_eofs = xr.load_dataset(
    config_step.observational_network_latitudinal_gradient_eofs_file
).pint.quantify()
lat_grad_eofs

# %%
new_years = np.arange(1, lat_grad_eofs["year"].max() + 1)
new_years

# %% [markdown]
# ### Extend EOF one
#
# (Zero-indexing, hence this is the second EOF)

# %%
# Quick assertion that things are as expected
if len(lat_grad_eofs["eof"]) != 2:
    raise AssertionError("Rethink")

# %%
new_pc1 = lat_grad_eofs["principal-components"].sel(eof=1).copy()
new_pc1 = new_pc1.pint.dequantify().interp(
    year=new_years, kwargs={"fill_value": new_pc1.data[0].m}
)
new_pc1

# %% [markdown]
# ### Extend EOF zero
#
# (Zero-indexing, hence this is the first EOF)

# %%
current_pc1 = lat_grad_eofs["principal-components"].sel(eof=0)

# %%
primap_full = primap2.open_dataset(
    config_retrieve_misc.primap.raw_dir
    / config_retrieve_misc.primap.download_url.url.split("/")[-1]
)
primap_full

# %%
fig, axes = plt.subplots(ncols=2)

primap_fossil_ch4_emissions = (
    local.xarray_time.convert_time_to_year_month(primap_full)
    .sel(
        **{
            "category (IPCC2006_PRIMAP)": "M.0.EL",
            "scenario (PRIMAP-hist)": "HISTTP",
            "area (ISO3)": "EARTH",
            "month": 1,
        }
    )[config_step.gas.upper()]
    .squeeze()
    .pint.to("MtCH4 / yr")
    .reset_coords(drop=True)
)
primap_regression_data = primap_fossil_ch4_emissions.sel(year=current_pc1["year"])
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
ax.scatter(x.m, y.m, label="raw data")
ax.plot(x.m, (m * x + c).m, color="tab:orange", label="regression")
ax.set_ylabel("PC")
ax.set_xlabel("PRIMAP emissions")
ax.legend()

# %% [markdown]
# Extend with constant back in time so that latitudinal gradient is also constant back in time.

# %%
fig, axes = plt.subplots(ncols=2)
primap_regression_data_extended = primap_fossil_ch4_emissions.copy()
primap_regression_data_extended = (
    primap_regression_data_extended.pint.dequantify().interp(
        year=new_years, kwargs={"fill_value": primap_regression_data_extended.data[0].m}
    )
)
primap_regression_data_extended.plot(ax=axes[0])
primap_regression_data_extended.sel(year=range(1700, 2023)).plot(ax=axes[1])
axes[0].set_ylim(ymin=0)
axes[1].set_ylim(ymin=0)
plt.tight_layout()

# %%
new_pc0 = m * primap_regression_data_extended.pint.quantify() + c
new_pc0.name = new_pc1.name
new_pc0 = new_pc0.assign_coords(eof=0)
new_pc0

# %%
new_pcs = xr.concat([new_pc0, new_pc1], "eof")
new_pcs.attrs = dict(
    description=(
        "Principal components for the latitudinal gradient EOFs, "
        "extended to cover the whole time period"
    ),
    units=lat_grad_eofs["principal-components"].data.units,
)
new_pcs.plot(hue="eof")
new_pcs

# %%
res = xr.merge([new_pcs, lat_grad_eofs["eofs"]], combine_attrs="drop")
res.attrs = {
    "description": "EOFs and PCs for the latitudinal gradient, extended to cover the whole time period"
}
res

# %%
# lat_gradient_extended = new_pcs @ lat_grad_eofs["eofs"]
# lat_gradient_extended.name = "latitudinal_gradient"
# lat_gradient_extended
# fig, axes = plt.subplots(ncols=2)
# lat_gradient_extended.sel(
#     year=np.hstack([
#         1,
#         1900,
#         np.arange(1700, 1901, 100),
#         2020
#     ])
# ).plot(y="lat", hue="year", alpha=0.7, ax=axes[0])

# lat_gradient_extended.sel(
#     year=np.hstack([
#         np.arange(1900, 2021, 10), 2022,
#     ])
# ).plot(y="lat", hue="year", alpha=0.7, ax=axes[1])
# plt.tight_layout()

# %% [markdown]
# ### Save

# %%
config_step.latitudinal_gradient_eofs_extended_file.parent.mkdir(
    exist_ok=True, parents=True
)
res.pint.dequantify().to_netcdf(config_step.latitudinal_gradient_eofs_extended_file)
res
