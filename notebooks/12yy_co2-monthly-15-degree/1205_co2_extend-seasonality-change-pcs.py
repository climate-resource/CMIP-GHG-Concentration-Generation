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
# # CO$_2$ - extend the seasonality change principal components
#
# Extend the seasonality change's principal components back in time.
# For CO$_2$, we do this by using a regression against
# temperatures and global-mean concentrations.

# %% [markdown]
# ## Imports

# %%
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cf_xarray.units
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pint
import pint_xarray
import xarray as xr
from attrs import evolve
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

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %%
opscm_reg = pint_xarray.setup_registry(openscm_units.unit_registry)
QuantityOSCM = opscm_reg.Quantity

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

config_retrieve_misc = get_config_for_step_id(config=config, step="retrieve_misc_data", step_config_id="only")


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
    if isinstance(axes, matplotlib.axes.Axes):
        raise TypeError(type(axes))

    yield (axes[0], axes[1])

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### Load data

# %%
global_annual_mean: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_step.global_annual_mean_allyears_file
).pint.quantify()
global_annual_mean

# %%
hadcrut_temperatures = xr.load_dataset(
    config_retrieve_misc.hadcrut5.raw_dir / config_retrieve_misc.hadcrut5.download_url.url.split("/")[-1]
)["tas_mean"].pint.quantify()
hadcrut_temperatures = (
    local.xarray_time.convert_time_to_year_month(hadcrut_temperatures)
    .sel(month=7)
    .drop_vars(["month", "latitude", "longitude", "realization"])
)
hadcrut_temperatures

# %%
seasonality_change_eofs_obs_network = xr.load_dataset(
    config_step.observational_network_seasonality_change_eofs_file
).pint.quantify()
seasonality_change_eofs_obs_network

# %% [markdown]
# ### Define some important constants

# %%
if not config.ci:
    out_years = np.arange(1, seasonality_change_eofs_obs_network["year"].max() + 1)

else:
    out_years = np.arange(1750, seasonality_change_eofs_obs_network["year"].max() + 1)

out_years

# %%
obs_network_years = seasonality_change_eofs_obs_network["year"]
obs_network_years

# %%
use_extensions_years = np.setdiff1d(out_years, obs_network_years)
use_extensions_years

# %% [markdown]
# ### Extend PC zero
#
# (Zero-indexing, hence this is the first PC)
#
# This happens in a few steps.

# %%
# Quick assertion that things are as expected
exp_n_eofs = 1
if len(seasonality_change_eofs_obs_network["eof"]) != exp_n_eofs:
    raise AssertionError("Rethink")

# %% [markdown]
# #### Use a regression against concentrations and temperatures to fill in the gap
#
# There is a gap between the observational network period
# and the optimised ice core period.
# We fill this using a regression against global-mean concentrations and temperatures.

# %%
regressor = local.seasonality.CO2SeasonalityChangeRegression()
regressor

# %%
regression_timeseries = regressor.get_composite(
    temperatures=hadcrut_temperatures, concentrations=global_annual_mean
)
regression_timeseries

# %%
pc0_obs_network = seasonality_change_eofs_obs_network["principal-components"].sel(eof=0)
pc0_obs_network

# %%
regression_timeseries_same_years = regression_timeseries.sel(year=pc0_obs_network["year"])
regression_timeseries_same_years

# %%
with axes_vertical_split() as axes:
    regression_timeseries_same_years.pint.dequantify().plot(ax=axes[0])
    pc0_obs_network.plot(ax=axes[1])

# %%
x = QuantityOSCM(
    regression_timeseries_same_years.data.m,
    str(regression_timeseries_same_years.data.units),
)
A = np.vstack([x.m, np.ones(x.size)]).T
y = QuantityOSCM(pc0_obs_network.data.m, str(pc0_obs_network.data.units))

res = np.linalg.lstsq(A, y.m, rcond=None)
m, c = res[0]
m = QuantityOSCM(m, (y / x).units)
c = QuantityOSCM(c, y.units)

latitudinal_gradient_pc0_composite_regression = local.regressors.LinearRegressionResult(m=m, c=c)

fig, ax = plt.subplots()
ax.scatter(x.m, y.m, label="raw data")
ax.plot(x.m, (m * x + c).m, color="tab:orange", label="regression")
ax.set_ylabel("PC0")
ax.set_xlabel("Composite timeseries")
ax.legend()

# %%
regressor_incl_result = evolve(regressor, regression_result=latitudinal_gradient_pc0_composite_regression)
regressor_incl_result

# %% [markdown]
# Extend the composite back to year 1, assuming constant before 1850.

# %%
composite_extension_years = np.union1d(
    np.setdiff1d(out_years, pc0_obs_network["year"]),
    regression_timeseries["year"],
)
composite_extension_years

# %%
regression_timeseries_extended = regression_timeseries.copy()
regression_timeseries_extended = regression_timeseries_extended.pint.dequantify().interp(
    year=composite_extension_years,
    kwargs={"fill_value": regression_timeseries.data[0].m},
)

with axes_vertical_split() as axes:
    regression_timeseries_extended.plot(ax=axes[0])  # type: ignore
    regression_timeseries_extended.sel(year=range(1950, 2023)).plot(ax=axes[1])  # type: ignore

regression_timeseries_extended

# %%
years_to_fill_with_regression = np.setdiff1d(
    regression_timeseries_extended["year"],
    pc0_obs_network["year"],
)

years_to_fill_with_regression

# %%
pc0_regression_extended = (
    m
    * regression_timeseries_extended.sel(year=years_to_fill_with_regression).pint.quantify(
        unit_registry=opscm_reg
    )
    + c
)
pc0_regression_extended = pc0_regression_extended.assign_coords(eof=0)
pc0_regression_extended

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
        pc0_regression_extended,
        pc0_obs_network,
    ],
    "year",
)

with axes_vertical_split() as axes:
    allyears_pc0.plot(ax=axes[0])
    axes[0].set_xlim((1750, 2030))

    pc0_regression_extended.plot(ax=axes[1])
    pc0_obs_network.plot(ax=axes[1])
    axes[1].set_xlim((1900, 2030))

allyears_pc0

# %% [markdown]
# ### Join the PCs back together

# %%
allyears_pcs = xr.concat([allyears_pc0], "eof").pint.dequantify().pint.quantify()
allyears_pcs

# %% [markdown]
# ### Join PCs and EOFs back together

# %%
allyears_pcs.name = "principal-components"
out = xr.merge([allyears_pcs, seasonality_change_eofs_obs_network["eofs"]])
out

# %%
(out["principal-components"] @ out["eofs"]).sel(year=2022).plot()  # type: ignore

# %%
(out["principal-components"] @ out["eofs"]).sel(year=2000).plot()  # type: ignore

# %%
(out["principal-components"] @ out["eofs"]).sel(year=1980).plot()  # type: ignore

# %%
(out["principal-components"] @ out["eofs"]).sel(year=out["year"].min()).plot()  # type: ignore

# %% [markdown]
# Quick check that our output matches the observational network in the years they overlap.

# %%
xr.testing.assert_allclose(
    (out["principal-components"] @ out["eofs"]).sel(year=seasonality_change_eofs_obs_network["year"]),
    seasonality_change_eofs_obs_network["principal-components"] @ seasonality_change_eofs_obs_network["eofs"],
)

# %% [markdown]
# ## Save

# %%
config_step.seasonality_change_allyears_pcs_eofs_file.parent.mkdir(exist_ok=True, parents=True)
out.pint.dequantify().to_netcdf(config_step.seasonality_change_allyears_pcs_eofs_file)
out

# %%
config_step.seasonality_change_temperature_co2_conc_regression_file.parent.mkdir(exist_ok=True, parents=True)
with open(config_step.seasonality_change_temperature_co2_conc_regression_file, "w") as fh:
    fh.write(local.config.converter_yaml.dumps(regressor_incl_result))

regressor_incl_result
