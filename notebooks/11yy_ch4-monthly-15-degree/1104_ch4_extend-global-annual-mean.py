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
# # CH$_4$ - extend the global-, annual-mean
#
# Extend the global-, annual-mean back in time.
# For CH$_4$, we do this by combining the values from ice cores etc.
# and our latitudinal gradient information.

# %% [markdown]
# ## Imports

# %%
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import cf_xarray.units
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import openscm_units
import pandas as pd
import pint
import pint_xarray
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.harmonisation
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

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %%
QuantityOSCM = openscm_units.unit_registry.Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_ch4_monthly_fifteen_degree_pieces"

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

config_smooth_law_dome_data = get_config_for_step_id(
    config=config, step="smooth_law_dome_data", step_config_id=config_step.gas
)

config_process_epica = get_config_for_step_id(
    config=config, step="retrieve_and_process_epica_data", step_config_id="only"
)

config_process_neem = get_config_for_step_id(
    config=config, step="retrieve_and_process_neem_data", step_config_id="only"
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Helper functions


# %%
def get_col_assert_single_value(idf: pd.DataFrame, col: str) -> str | float:
    """Get a column's value, asserting that it only has one value"""
    res = idf[col].unique()
    if len(res) != 1:
        raise AssertionError

    return cast(str | float, res[0])


# %%
@contextmanager
def axes_vertical_split(
    ncols: int = 2,
) -> Iterator[tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Get two split axes, formatting after exiting the context"""
    fig, axes = plt.subplots(ncols=2)
    if isinstance(axes, matplotlib.axes.Axes):
        raise TypeError(type(axes))

    yield (axes[0], axes[1])

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
lat_grad_eofs_allyears = xr.load_dataset(
    config_step.latitudinal_gradient_allyears_pcs_eofs_file
).pint.quantify()
lat_grad_eofs_allyears

# %%
smooth_law_dome = pd.read_csv(config_smooth_law_dome_data.smoothed_median_file)
smooth_law_dome = smooth_law_dome[smooth_law_dome["gas"] == config_step.gas]
smooth_law_dome["source"] = "law_dome"
smooth_law_dome

# %%
neem_data = pd.read_csv(config_process_neem.processed_data_with_loc_file)
neem_data["year"] = neem_data["year"].round(0)
neem_data["source"] = "neem"
neem_data.sort_values("year")

# %%
epica_data = pd.read_csv(config_process_epica.processed_data_with_loc_file)
epica_data["source"] = "epica"
epica_data.sort_values("year")

# %% [markdown]
# ### Define some important constants

# %%
if not config.ci:
    out_years: npt.NDArray[np.int64] = np.arange(1, global_annual_mean_obs_network["year"].max() + 1)

else:
    out_years = np.arange(1750, global_annual_mean_obs_network["year"].max() + 1)

out_years

# %%
obs_network_years = global_annual_mean_obs_network["year"]
obs_network_years

# %%
use_extensions_years = np.setdiff1d(out_years, obs_network_years)
use_extensions_years

# %% [markdown]
# ### Extend global-, annual-mean
#
# This happens in a few steps.

# %% [markdown]
# #### Define some constants

# %%
law_dome_lat = get_col_assert_single_value(smooth_law_dome, "latitude")
law_dome_lat

# %%
law_dome_lat_nearest = float(lat_grad_eofs_allyears.sel(lat=law_dome_lat, method="nearest")["lat"])
law_dome_lat_nearest

# %%
neem_lat = get_col_assert_single_value(neem_data, "latitude")
neem_lat

# %%
neem_lat_nearest = float(lat_grad_eofs_allyears.sel(lat=neem_lat, method="nearest")["lat"])
neem_lat_nearest

# %%
epica_lat = get_col_assert_single_value(epica_data, "latitude")
epica_lat

# %%
epica_lat_nearest = float(lat_grad_eofs_allyears.sel(lat=epica_lat, method="nearest")["lat"])
epica_lat_nearest

# %%
conc_unit = get_col_assert_single_value(smooth_law_dome, "unit")
conc_unit

# %%
neem_unit = get_col_assert_single_value(neem_data, "unit")
if neem_unit != conc_unit:
    raise AssertionError

neem_unit

# %%
epica_unit = get_col_assert_single_value(epica_data, "unit")
if epica_unit != conc_unit:
    raise AssertionError

epica_unit

# %% [markdown]
# #### Create annual-mean latitudinal gradient
#
# This is then combined with our ice core information.

# %%
allyears_latitudinal_gradient = (
    lat_grad_eofs_allyears["principal-components"] @ lat_grad_eofs_allyears["eofs"]
)
allyears_latitudinal_gradient

# %% [markdown]
# ### Create full field over all years

# %% [markdown]
# #### Observational network
#
# We start with the field we have from the observational network.

# %%
obs_network_full_field = allyears_latitudinal_gradient + global_annual_mean_obs_network
obs_network_full_field

# %% [markdown]
# #### Law Dome
#
# Then we use the Law Dome data.
# We calculate the offset by ensuring that the value in Law Dome's bin
# matches our smoothed Law Dome timeseries in the years in which we have Law Dome data
# and don't have the observational network.

# %% [markdown]
# ##### Harmonise
#
# Firstly, we harmonise the Law Dome data with the observational record
# to avoid jumps as we transition between the two.

# %%
join_year = int(obs_network_years.min())
join_year

# %%
smooth_law_dome_harmonised = (
    local.harmonisation.get_harmonised_timeseries(  # type: ignore
        ints=smooth_law_dome.set_index(["year", "unit", "gas", "source"])["value"].unstack("year"),
        harm_units=conc_unit,  # type: ignore
        harm_value=float(
            obs_network_full_field.sel(lat=law_dome_lat, method="nearest")
            .pint.to(conc_unit)
            .sel(year=join_year)
            .data.m
        ),
        harm_year=join_year,
        n_transition_years=100,
    )
    .stack()
    .to_frame("value")
    .reset_index()
)
smooth_law_dome_harmonised

# %%
fig, ax = plt.subplots()

obs_network_full_field.sel(lat=law_dome_lat, method="nearest").plot(ax=ax, label="Obs network")
ax.plot(
    smooth_law_dome["year"],
    smooth_law_dome["value"],
    label="Smoothed Law Dome",
)
ax.plot(
    smooth_law_dome_harmonised["year"],
    smooth_law_dome_harmonised["value"],
    label="Law Dome, harmonised",
    alpha=0.4,
)
ax.legend()
ax.set_xlim((1970, 2030))

# %%
law_dome_da = xr.DataArray(
    data=smooth_law_dome_harmonised["value"],
    dims=["year"],
    coords=dict(year=smooth_law_dome_harmonised["year"]),
    attrs=dict(units=conc_unit),
).pint.quantify()
law_dome_da

# %%
offset = law_dome_da - allyears_latitudinal_gradient.sel(lat=law_dome_lat, method="nearest")
offset

# %%
law_dome_years_full_field = allyears_latitudinal_gradient + offset
law_dome_years_full_field

# %% [markdown]
# #### EPICA
#
# Then we add a couple of data points from EPICA to round out the Law Dome timeseries
# so we can have a timeseries that goes back to year 1.
# There are probably better ways to do this, but this is fine for now.

# %%
# Make sure that we have epica data before our start year,
# so that the interpolation will have something to join with.
epica_data_pre_start_year = -50
epica_data_to_add = epica_data[
    (epica_data["year"] > epica_data_pre_start_year)
    & (epica_data["year"] < smooth_law_dome_harmonised["year"].min())
].sort_values(by="year")
epica_data_to_add

# %%
epica_lat

# %% [markdown]
# We simply linearly interpolate the EPICA data to get a yearly timeseries over the period of interest.
# We make sure that the interpolated values match our current values at the year in which Law Dome starts
# to avoid having a jump.
# This isn't perfect and could be investigated further, but will do for now.

# %%
law_dome_start_year = law_dome_da["year"].min()

if not config.ci:
    years_use_epica = np.arange(1, law_dome_start_year)
else:
    years_use_epica = np.empty(0)

years_use_epica

# %%
if years_use_epica.size > 0:
    harmonisation_value = float(
        law_dome_years_full_field.sel(year=law_dome_start_year).sel(lat=epica_lat, method="nearest").data.m
    )
    harmonisation_value

# %%
if years_use_epica.size > 0:
    fig, ax = plt.subplots()

    epica_da = (
        xr.DataArray(
            data=np.hstack([epica_data_to_add["value"], harmonisation_value]),
            dims=["year"],
            coords=dict(year=np.hstack([epica_data_to_add["year"], law_dome_start_year])),
            attrs=dict(units=conc_unit),
        )
        .interp(year=years_use_epica)
        .pint.quantify()
    )

    epica_da.pint.dequantify().plot(ax=ax, label="interpolated")
    epica_data[
        (epica_data["year"] > -1000) & (epica_data["year"] < 200)  # noqa: PLR2004
    ].plot.scatter(x="year", y="value", ax=ax, color="tab:orange", label="EPICA raw")

    ax.legend()

    epica_da

# %%
if years_use_epica.size > 0:
    offset_epica = epica_da - allyears_latitudinal_gradient.sel(lat=epica_lat, method="nearest")
    epica_years_full_field = allyears_latitudinal_gradient + offset_epica
    epica_years_full_field

# %% [markdown]
# #### Join back together

# %%
if years_use_epica.size > 0:
    allyears_full_field = xr.concat(
        [epica_years_full_field, law_dome_years_full_field, obs_network_full_field],
        "year",
    )

else:
    if not config.ci:
        msg = "Should be using EPICA"
        raise AssertionError(msg)

    allyears_full_field = xr.concat([law_dome_years_full_field, obs_network_full_field], "year")

allyears_full_field

# %%
allyears_full_field.plot(hue="lat")

# %% [markdown]
# #### Check our full field calculation
#
# There's a lot of steps above, if we have got this right the field will:
#
# - have an annual-average that matches:
#    - NEEM in the NEEM latitude (for the years of NEEM observations)
#    - our smoothed Law Dome in the Law Dome latitude
#      (for the years of the smoothed Law Dome timeseries)
#    - EPICA in the EPICA latitude (for the years of EPICA observations)
#
# - be decomposable into:
#   - a global-mean timeseries (with dims (year,))
#   - a latitudinal gradient (with dims (year, lat))
#     that has a spatial-mean of zero.
#     This latitudinal gradient should match the latitudinal
#     gradient we calculated earlier.

# %%
if not config.ci:
    np.testing.assert_allclose(
        allyears_full_field.sel(lat=neem_lat, method="nearest")
        .sel(year=neem_data["year"].values)
        .data.to(conc_unit)
        .m,
        neem_data["value"],
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        allyears_full_field.sel(lat=law_dome_lat, method="nearest")
        .sel(year=smooth_law_dome_harmonised["year"].values)
        .data.to(conc_unit)
        .m,
        smooth_law_dome_harmonised["value"],
    )
else:
    neem_compare_years = neem_data["year"].values[
        np.isin(neem_data["year"].values, out_years)  # type: ignore
    ]
    np.testing.assert_allclose(
        allyears_full_field.sel(lat=neem_lat, method="nearest")
        .sel(year=neem_compare_years)
        .data.to(conc_unit)
        .m,
        neem_data[np.isin(neem_data["year"], neem_compare_years)]["value"],
        rtol=1e-3,
    )
    law_dome_compare_years = smooth_law_dome_harmonised["year"].values[
        np.isin(smooth_law_dome_harmonised["year"].values, out_years)  # type: ignore
    ]
    np.testing.assert_allclose(
        allyears_full_field.sel(lat=law_dome_lat, method="nearest")
        .sel(year=law_dome_compare_years)
        .data.to(conc_unit)
        .m,
        smooth_law_dome_harmonised[np.isin(smooth_law_dome_harmonised["year"], law_dome_compare_years)][
            "value"
        ],
    )

if years_use_epica.size > 0:
    np.testing.assert_allclose(
        allyears_full_field.sel(lat=epica_lat, method="nearest")
        .sel(year=epica_da["year"].values)
        .data.to(conc_unit)
        .m,
        epica_da.data.m,
    )

elif not config.ci:
    msg = "Should be using EPICA"
    raise AssertionError(msg)

# %%
allyears_full_field

# %%
tmp = allyears_full_field.copy()
tmp.name = "allyears_global_annual_mean"
allyears_global_annual_mean = local.xarray_space.calculate_global_mean_from_lon_mean(tmp)
allyears_global_annual_mean

# %%
fig, ax = plt.subplots()
allyears_global_annual_mean.sel(year=range(join_year - 50, 2023)).plot(ax=ax)  # type: ignore
ax.axvline(join_year, linestyle="--", color="gray")
ax.grid()

# %% [markdown]
# The residual between our full field and our annual, global-mean
# should just be the latitudinal gradient we started with.

# %%
check = allyears_full_field - allyears_global_annual_mean
xr.testing.assert_allclose(check, allyears_latitudinal_gradient)

# %%
tmp = allyears_latitudinal_gradient.copy()
tmp.name = "tmp"
np.testing.assert_allclose(
    local.xarray_space.calculate_global_mean_from_lon_mean(tmp).data.to("ppb").m,
    0.0,
    atol=1e-10,
)

# %% [markdown]
# ## Save

# %%
config_step.global_annual_mean_allyears_file.parent.mkdir(exist_ok=True, parents=True)
allyears_global_annual_mean.pint.dequantify().to_netcdf(config_step.global_annual_mean_allyears_file)
allyears_global_annual_mean
