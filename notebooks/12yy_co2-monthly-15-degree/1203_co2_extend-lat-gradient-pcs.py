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
# # CO$_2$ - extend the latitudinal gradient principal components
#
# Extend the latitudinal gradient's principal components back in time.
# For CO$_2$, we do this by using a regression against emissions.

# %% [markdown]
# ## Imports

# %%
from collections.abc import Iterator
from contextlib import contextmanager

import cf_xarray.units
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pint
import pint_xarray
import primap2  # type: ignore
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.config
import local.latitudinal_gradient
import local.mean_preserving_interpolation
import local.raw_data_processing
import local.regressors
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

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %%
QuantityOSCM = openscm_units.unit_registry.Quantity

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
# ### Helper functions


# %%
@contextmanager
def axes_vertical_split(
    ncols: int = 2,
) -> Iterator[tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Get two split axes, formatting after exiting the context"""
    fig, axes = plt.subplots(ncols=ncols)
    yield axes
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### Load data

# %%
global_annual_mean_obs_network: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_step.observational_network_global_annual_mean_file
).pint.quantify()
global_annual_mean_obs_network

# %%
lat_grad_eofs_obs_network = xr.load_dataset(
    config_step.observational_network_latitudinal_gradient_eofs_file
).pint.quantify()
lat_grad_eofs_obs_network

# %% [markdown]
# ### Define some important constants

# %%
if not config.ci:
    out_years = np.arange(1, lat_grad_eofs_obs_network["year"].max() + 1)

else:
    out_years = np.arange(1750, lat_grad_eofs_obs_network["year"].max() + 1)

out_years

# %%
obs_network_years = lat_grad_eofs_obs_network["year"]
obs_network_years

# %%
use_extensions_years = np.setdiff1d(out_years, obs_network_years)
use_extensions_years

# %% [markdown]
# ### Extend PC one
#
# (Zero-indexing, hence this is the second PC)
#
# This is returned to zero in 1850 from the start of the observational network period.

# %%
# Quick assertion that things are as expected
exp_n_eofs = 2
if len(lat_grad_eofs_obs_network["eof"]) != exp_n_eofs:
    raise AssertionError("Rethink")

# %%
pc1 = lat_grad_eofs_obs_network["principal-components"].sel(eof=1).copy()

zero_year = 1850
tmp = pc1.sel(year=pc1["year"][0])
tmp = tmp.assign_coords(year=zero_year)
tmp.values = 0.0

allyears_pc1 = (
    xr.concat([tmp, pc1], "year")
    .pint.dequantify()
    .interp(year=out_years, kwargs={"fill_value": 0})
)

with axes_vertical_split() as axes:
    allyears_pc1.plot(ax=axes[0])
    allyears_pc1.sel(year=range(1950, 2023)).plot(ax=axes[1])

allyears_pc1

# %% [markdown]
# ### Extend PC zero
#
# (Zero-indexing, hence this is the first PC)
#
# This happens in a few steps.

# %% [markdown]
# #### Use a regression against emissions to fill in the gap
#
# There is a gap between the observational network period
# and the optimised ice core period.
# We fill this using a regression against emissions.

# %%
primap_full = primap2.open_dataset(
    config_retrieve_misc.primap.raw_dir
    / config_retrieve_misc.primap.download_url.url.split("/")[-1]
)

primap_fossil_co2_emissions = (
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
    .pint.to("GtCO2 / yr")
    .reset_coords(drop=True)
)

primap_fossil_co2_emissions

# %%
primap_regression_data = primap_fossil_co2_emissions.sel(
    year=lat_grad_eofs_obs_network["year"]
)
primap_regression_data

# %%
pc0_obs_network = lat_grad_eofs_obs_network["principal-components"].sel(eof=0)
pc0_obs_network

# %%
with axes_vertical_split() as axes:
    primap_regression_data.plot(ax=axes[0])
    pc0_obs_network.plot(ax=axes[1])

# %%
x = QuantityOSCM(primap_regression_data.data, str(primap_regression_data.data.units))
A = np.vstack([x.m, np.ones(x.size)]).T
y = QuantityOSCM(pc0_obs_network.data, str(pc0_obs_network.data.units))

res = np.linalg.lstsq(A, y.m, rcond=None)
m, c = res[0]
m = QuantityOSCM(m, (y / x).units)
c = QuantityOSCM(c, y.units)

latitudinal_gradient_pc0_co2_fossil_emissions_regression = (
    local.regressors.LinearRegressionResult(m=m, c=c)
)

fig, ax = plt.subplots()
ax.scatter(x.m, y.m, label="raw data")
ax.plot(x.m, (m * x + c).m, color="tab:orange", label="regression")
ax.plot(
    np.hstack([0, x.m]),
    (m * np.hstack([0, x]) + c).m,
    color="tab:orange",
    label="regression-extended",
)
ax.set_ylabel("PC0")
ax.set_xlabel("PRIMAP emissions")
ax.legend()

# %% [markdown]
# Extend PRIMAP emissions back to year 1, assuming constant before 1750.

# %%
primap_fossil_co2_emissions_extension_years = np.union1d(
    np.setdiff1d(out_years, pc0_obs_network["year"]),
    primap_fossil_co2_emissions["year"],
)
primap_fossil_co2_emissions_extension_years

# %%
primap_fossil_co2_emissions_extended = primap_fossil_co2_emissions.copy()
primap_fossil_co2_emissions_extended = (
    primap_fossil_co2_emissions_extended.pint.dequantify().interp(
        year=primap_fossil_co2_emissions_extension_years,
        kwargs={"fill_value": primap_fossil_co2_emissions.data[0].m},
    )
)

with axes_vertical_split() as axes:
    primap_fossil_co2_emissions_extended.plot(ax=axes[0])
    primap_fossil_co2_emissions_extended.sel(year=range(1950, 2023)).plot(ax=axes[1])

primap_fossil_co2_emissions_extended

# %%
years_to_fill_with_regression = np.setdiff1d(
    primap_fossil_co2_emissions_extended["year"],
    pc0_obs_network["year"],
)

years_to_fill_with_regression

# %%
pc0_emissions_extended = (
    m
    * primap_fossil_co2_emissions_extended.sel(
        year=years_to_fill_with_regression
    ).pint.quantify(unit_registry=openscm_units.unit_registry)
    + c
)
pc0_emissions_extended = pc0_emissions_extended.assign_coords(eof=0)
pc0_emissions_extended = pc0_emissions_extended.assign_coords(eof=0)
pc0_emissions_extended

# %% [markdown]
# #### Concatenate the pieces of PC0
#
# Join:
#
# - extended based on regression with emissions
# - raw values derived from the observational network

# %%
allyears_pc0 = xr.concat(
    [
        pc0_emissions_extended,
        pc0_obs_network,
    ],
    "year",
)

with axes_vertical_split() as axes:
    allyears_pc0.plot(ax=axes[0])

    pc0_emissions_extended.plot(ax=axes[1])
    pc0_obs_network.plot(ax=axes[1])

allyears_pc0

# %% [markdown]
# ### Join the PCs back together

# %%
allyears_pcs = (
    xr.concat([allyears_pc0, allyears_pc1], "eof").pint.dequantify().pint.quantify()
)
allyears_pcs

# %% [markdown]
# ### Join PCs and EOFs back together

# %%
allyears_pcs.name = "principal-components"
out = xr.merge([allyears_pcs, lat_grad_eofs_obs_network["eofs"]])
out

# %%
(out["principal-components"] @ out["eofs"]).sel(year=2022).plot()  # type: ignore

# %%
(out["principal-components"] @ out["eofs"]).sel(year=1980).plot()  # type: ignore

# %%
(out["principal-components"] @ out["eofs"]).sel(year=out["year"].min()).plot()  # type: ignore

# %% [markdown]
# Quick check that our output matches the observational network in the years they overlap.

# %%
xr.testing.assert_allclose(
    (out["principal-components"] @ out["eofs"]).sel(
        year=lat_grad_eofs_obs_network["year"]
    ),
    lat_grad_eofs_obs_network["principal-components"]
    @ lat_grad_eofs_obs_network["eofs"],
)

# %% [markdown]
# ## Save

# %%
config_step.latitudinal_gradient_allyears_pcs_eofs_file.parent.mkdir(
    exist_ok=True, parents=True
)
out.pint.dequantify().to_netcdf(config_step.latitudinal_gradient_allyears_pcs_eofs_file)
out

# %%
config_step.latitudinal_gradient_pc0_co2_fossil_emissions_regression_file.parent.mkdir(
    exist_ok=True, parents=True
)
with open(
    config_step.latitudinal_gradient_pc0_co2_fossil_emissions_regression_file, "w"
) as fh:
    fh.write(
        local.config.converter_yaml.dumps(
            latitudinal_gradient_pc0_co2_fossil_emissions_regression
        )
    )

latitudinal_gradient_pc0_co2_fossil_emissions_regression
