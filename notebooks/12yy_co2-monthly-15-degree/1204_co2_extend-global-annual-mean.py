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
# # CO$_2$ - extend the global-, annual-mean
#
# Extend the global-, annual-mean back in time.
# For CO$_2$, we do this by combining the values from ice cores etc.
# and our latitudinal gradient information.

# %% [markdown]
# ## Imports

# %%
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cf_xarray.units
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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

config_retrieve_and_process_scripps_data = get_config_for_step_id(
    config=config, step="retrieve_and_process_scripps_data", step_config_id="only"
)

config_retrieve_and_process_menking_et_al_2025_data = get_config_for_step_id(
    config=config, step="retrieve_and_process_menking_et_al_2025_data", step_config_id="only"
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Helper functions


# %%
def get_col_assert_single_value(idf: pd.DataFrame, col: str) -> Any:
    """Get a column's value, asserting that it only has one value"""
    res = idf[col].unique()
    if len(res) != 1:
        raise AssertionError

    return res[0]


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
global_annual_mean_obs_network = xr.load_dataarray(  # type: ignore
    config_step.observational_network_global_annual_mean_file
).pint.quantify()
global_annual_mean_obs_network

# %%
lat_grad_eofs_allyears = xr.load_dataset(
    config_step.latitudinal_gradient_allyears_pcs_eofs_file
).pint.quantify()
lat_grad_eofs_allyears

# %%
mauna_loa_merged = pd.read_csv(
    config_retrieve_and_process_scripps_data.merged_ice_core_data_processed_data_file
)
mauna_loa_merged

# %%
menking_et_al = pd.read_csv(config_retrieve_and_process_menking_et_al_2025_data.processed_data_file)
menking_et_al = menking_et_al[menking_et_al["gas"] == config_step.gas]
menking_et_al["source"] = "menking_et_al_2025"
if menking_et_al["year"].min() > 1:
    raise AssertionError

menking_et_al

# %% [markdown]
# ### Define some important constants

# %%
if not config.ci:
    out_years = np.arange(1, global_annual_mean_obs_network["year"].max() + 1)

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
menking_et_al_lat = get_col_assert_single_value(menking_et_al, "latitude")
menking_et_al_lat

# %%
menking_et_al_nearest = float(lat_grad_eofs_allyears.sel(lat=menking_et_al_lat, method="nearest")["lat"])
menking_et_al_nearest

# %%
conc_unit = get_col_assert_single_value(menking_et_al, "unit")
conc_unit

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
# #### Mauna Loa
#
# We start by adding in data from the Mauna Loa - Law Dome merged ice core record.
# We use this data from the first full year of Mauna Loa data.

# %%
mauna_loa_start = 1959

# %% [markdown]
# ##### Harmonise
#
# Firstly, we harmonise the Mauna Loa data with the observational record
# to avoid jumps as we transition between the two.

# %%
join_year = int(obs_network_years.min())
join_year

# %%
mauna_loa_use_years = np.arange(
    mauna_loa_start + 0.5, join_year + 1
)  # keep an extra year for harmonisation to work
mauna_loa_use_years

# %%
mauna_loa_law_dome_merged_to_use = mauna_loa_merged[mauna_loa_merged["time"].isin(mauna_loa_use_years)]
if mauna_loa_law_dome_merged_to_use.shape[0] != mauna_loa_use_years.size:
    raise AssertionError

mauna_loa_law_dome_merged_to_use = mauna_loa_law_dome_merged_to_use.rename({"time": "year"}, axis="columns")
mauna_loa_law_dome_merged_to_use["year"] = mauna_loa_law_dome_merged_to_use["year"].astype(int)
mauna_loa_law_dome_merged_to_use

# %%
n_transition_years = 100

# A better written harmonisation function wouldn't need this.
tmp = pd.concat([mauna_loa_law_dome_merged_to_use.iloc[:1, :]] * n_transition_years)
tmp["year"] = np.arange(mauna_loa_start - n_transition_years, mauna_loa_start)
tmp["value"] = 0.0
harmonise_helper = pd.concat([tmp, mauna_loa_law_dome_merged_to_use])

mauna_loa_harmonised = (
    local.harmonisation.get_harmonised_timeseries(
        ints=harmonise_helper.set_index(["year", "unit", "gas"])["value"].unstack("year"),
        harm_units=conc_unit,
        harm_value=float(
            # Assume that Mauna Loa spline is roughly global-mean
            global_annual_mean_obs_network.pint.to(conc_unit).sel(year=join_year).data.m
        ),
        harm_year=join_year,
        n_transition_years=n_transition_years,
    )
    .stack()
    .to_frame("value")
    .reset_index()
)
mauna_loa_harmonised = mauna_loa_harmonised[
    mauna_loa_harmonised["year"].isin(mauna_loa_law_dome_merged_to_use["year"])
]
mauna_loa_harmonised

# %%
fig, ax = plt.subplots()

global_annual_mean_obs_network.plot(ax=ax, label="Obs network")
ax.plot(
    mauna_loa_merged["time"],
    mauna_loa_merged["value"],
    label="Mauna Loa Law Dome merged",
)
ax.plot(
    mauna_loa_harmonised["year"],
    mauna_loa_harmonised["value"],
    label="Mauna Loa Law Dome merged, harmonised",
    alpha=0.4,
)
ax.legend()
ax.set_xlim([1920, 2030])

# %%
ml_ld_da = xr.DataArray(
    data=mauna_loa_harmonised["value"],
    dims=["year"],
    coords=dict(year=mauna_loa_harmonised["year"]),
    attrs=dict(units=conc_unit),
).pint.quantify()
ml_ld_da

# %%
mauna_loa_law_dome_merged_years_full_field = ml_ld_da + allyears_latitudinal_gradient
mauna_loa_law_dome_merged_years_full_field

# %% [markdown]
# #### Menking et al., 2025
#
# Then we use the Menking et al., 2025 data.
# We calculate the offset by ensuring that the value in Menking et al's latitudinal bin
# matches the Menking et al., 2025 timeseries in the years in which we have Menking et al. data and don't have the observational network.

# %% [markdown]
# ##### Harmonise
#
# We also harmonise the Menking et al. data with the observational record
# to avoid jumps as we transition between the two.

# %%
join_year_menking = int(mauna_loa_law_dome_merged_years_full_field["year"].min())
join_year_menking

# %%
menking_et_al_harmonised = (
    local.harmonisation.get_harmonised_timeseries(
        ints=menking_et_al.set_index(["year", "unit", "gas", "source"])["value"].unstack("year"),
        harm_units=conc_unit,
        harm_value=float(
            mauna_loa_law_dome_merged_years_full_field.sel(lat=menking_et_al_lat, method="nearest")
            .pint.to(conc_unit)
            .sel(year=join_year_menking)
            .data.m
        ),
        harm_year=join_year_menking,
        n_transition_years=100,
    )
    .stack()
    .to_frame("value")
    .reset_index()
)
menking_et_al_harmonised

# %%
fig, ax = plt.subplots()

mauna_loa_law_dome_merged_years_full_field.sel(lat=menking_et_al_lat, method="nearest").plot(
    ax=ax, label="Mauna Loa Law Dome merged full field"
)
ax.plot(
    menking_et_al["year"],
    menking_et_al["value"],
    label="Menking et al., raw",
)
ax.plot(
    menking_et_al_harmonised["year"],
    menking_et_al_harmonised["value"],
    label="Menking et al., harmonised",
    alpha=0.4,
)
ax.legend()
ax.set_xlim([1830, 2030])
# ax.set_xlim([join_year - 20, join_year + 20])

# %%
menking_et_al_da = xr.DataArray(
    data=menking_et_al_harmonised["value"],
    dims=["year"],
    coords=dict(year=menking_et_al_harmonised["year"]),
    attrs=dict(units=conc_unit),
).pint.quantify()
menking_et_al_da

# %%
offset = menking_et_al_da - allyears_latitudinal_gradient.sel(lat=menking_et_al_lat, method="nearest")
offset

# %%
menking_et_al_years_full_field = allyears_latitudinal_gradient + offset
menking_et_al_years_full_field

# %% [markdown]
# #### Join back together

# %%
mostyears_full_field = xr.concat(
    [menking_et_al_years_full_field, mauna_loa_law_dome_merged_years_full_field, obs_network_full_field],
    "year",
)

mostyears_full_field

# %%
mostyears_full_field.plot(hue="lat")

# %% [markdown]
# #### Check our full field calculation
#
# There's a lot of steps above, if we have got this right the field will:
#
# - have an annual-average that matches:
#    - our smoothed Law Dome in the Law Dome latitude
#      (for the years of the smoothed Law Dome timeseries)
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
        mostyears_full_field.sel(lat=menking_et_al_lat, method="nearest")
        .sel(year=menking_et_al_harmonised["year"].values)
        .data.to(conc_unit)
        .m,
        menking_et_al_harmonised["value"],
    )
else:
    menking_et_al_compare_years = menking_et_al_harmonised["year"].values[
        np.isin(menking_et_al_harmonised["year"].values, out_years)  # type: ignore
    ]
    np.testing.assert_allclose(
        mostyears_full_field.sel(lat=menking_et_al_lat, method="nearest")
        .sel(year=menking_et_al_compare_years)
        .data.to(conc_unit)
        .m,
        menking_et_al_harmonised[np.isin(menking_et_al_harmonised["year"], menking_et_al_compare_years)][
            "value"
        ],
    )

# %%
tmp = mostyears_full_field.copy()
tmp.name = "mostyears_global_annual_mean"
mostyears_global_annual_mean = local.xarray_space.calculate_global_mean_from_lon_mean(tmp)
mostyears_global_annual_mean

# %% [markdown]
# #### Extending back to year 1
#
# We simply assume that global-mean concentrations are constant
# before the start of the ice core record.

# %%
back_extend_years = np.setdiff1d(
    out_years[np.where(out_years < mostyears_global_annual_mean["year"].values[-1])],
    mostyears_global_annual_mean["year"],
)
back_extend_years

# %%
if back_extend_years.size > 0:
    tmp = mostyears_global_annual_mean.sel(year=[mostyears_global_annual_mean["year"][0]])
    back_extended_global_annual_mean = (
        mostyears_global_annual_mean.pint.dequantify()
        .interp(year=back_extend_years, kwargs={"fill_value": tmp.data[0].m})
        .pint.quantify()
    )
    allyears_global_annual_mean = (
        mostyears_global_annual_mean.pint.dequantify()
        .interp(year=out_years, kwargs={"fill_value": tmp.data[0].m})
        .pint.quantify()
    )

    back_extended_full_field = (
        allyears_latitudinal_gradient.sel(year=back_extend_years) + back_extended_global_annual_mean
    )
    allyears_full_field = xr.concat([back_extended_full_field, mostyears_full_field], "year")


else:
    print("No back extension necessary")
    allyears_full_field = mostyears_full_field
    allyears_global_annual_mean = mostyears_global_annual_mean

allyears_global_annual_mean

# %%
fig, ax = plt.subplots()

global_annual_mean_obs_network.plot(ax=ax, label="Obs network", alpha=0.4)
allyears_global_annual_mean.sel(year=range(1900, 2023 + 1)).plot(ax=ax, label="output", alpha=0.4)
ax.axvline(join_year, linestyle="--", color="gray")
ax.axvline(join_year_menking, linestyle="--", color="gray")

ax.legend()

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
