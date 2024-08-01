# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # CH$_4$ - extend the latitudinal gradient principal components
#
# Extend the latitudinal gradient's principal components back in time.
# For CH$_4$, we do this by using the values from ice cores
# and a regression against emissions.

# %% [markdown]
# ## Imports

# %%
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import cast

import cf_xarray.units
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import pint_xarray
import primap2  # type: ignore
import scipy.optimize  # type: ignore
import tqdm.autonotebook as tqdman
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

# %% editable=true
step: str = "calculate_ch4_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true
# ## Load config

# %% editable=true
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_smooth_law_dome_data = get_config_for_step_id(
    config=config, step="smooth_law_dome_data", step_config_id=config_step.gas
)

config_process_neem = get_config_for_step_id(
    config=config, step="retrieve_and_process_neem_data", step_config_id="only"
)

config_retrieve_misc = get_config_for_step_id(
    config=config, step="retrieve_misc_data", step_config_id="only"
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
# This is kept constant before the observational network period.

# %%
# Quick assertion that things are as expected
exp_n_eofs = 2
if len(lat_grad_eofs_obs_network["eof"]) != exp_n_eofs:
    raise AssertionError("Rethink")

# %%
allyears_pc1 = lat_grad_eofs_obs_network["principal-components"].sel(eof=1).copy()
allyears_pc1 = allyears_pc1.pint.dequantify().interp(
    year=out_years, kwargs={"fill_value": allyears_pc1.data[0].m}
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
# #### Optimise PC0 to match Law Dome and NEEM data
#
# Over the period where we have both these timeseries.

# %% [markdown]
# A quick check below.
# We do this because we do the scaling independent of the global-mean.
# This only makes sense if the EOFs have a spatial-mean of zero,
# because then we can change the PCs however and the global-mean
# will not be affected.

# %%
np.testing.assert_allclose(
    local.xarray_space.calculate_global_mean_from_lon_mean(
        lat_grad_eofs_obs_network["eofs"]
    ).data.m,
    0.0,
    atol=1e-10,
)

# %%
law_dome_lat = get_col_assert_single_value(smooth_law_dome, "latitude")
law_dome_lat

# %%
law_dome_lat_nearest = float(
    lat_grad_eofs_obs_network.sel(lat=law_dome_lat, method="nearest")["lat"]
)
law_dome_lat_nearest

# %%
neem_lat = get_col_assert_single_value(neem_data, "latitude")
neem_lat

# %%
neem_lat_nearest = float(
    lat_grad_eofs_obs_network.sel(lat=neem_lat, method="nearest")["lat"]
)
neem_lat_nearest

# %%
conc_unit = get_col_assert_single_value(smooth_law_dome, "unit")
conc_unit

# %%
neem_unit = get_col_assert_single_value(neem_data, "unit")
if neem_unit != conc_unit:
    raise AssertionError

neem_unit


# %%
def diff_from_ice_cores(
    x: tuple[float, float], pc1: float, eofs: xr.DataArray, ice_core_data: xr.DataArray
) -> float:
    """
    Calculate the difference from the ice core data

    Parameters
    ----------
    x
        Current x-vector.
        Element zero should be the global-mean,
        element one is the value of pc0.

    pc1
        Value of PC1 to use when calculating the value at each latitude.

    eofs
        EOFs to use when calculating the value at each latitude.

    ice_core_data
        Ice core data to compare to

    Returns
    -------
        Area-weighted squared difference between the model's prediction
        and the ice core values.
    """
    global_mean, pc0 = x

    pcs = xr.DataArray([pc0, pc1], dims=["eof"], coords=dict(eof=[0, 1]))

    lat_grad = pcs @ eofs
    lat_grad.name = "lat_grad"

    lat_resolved = lat_grad + Quantity(global_mean, conc_unit)

    lat_resolved.name = "lat_resolved"
    lat_resolved = (
        lat_resolved.to_dataset()
        .cf.add_bounds("lat")
        .pint.quantify({"lat_bounds": "degrees_north"})
    )

    diff_squared = (lat_resolved - ice_core_data) ** 2
    if not str(diff_squared["lat_resolved"].data.units) == f"{conc_unit} ** 2":
        raise AssertionError(diff_squared["lat_resolved"].data.units)

    area_weighted_diff_squared = (
        local.xarray_space.calculate_area_weighted_mean_latitude_only(
            diff_squared, variables=["lat_resolved"]
        )["lat_resolved"].pint.dequantify()
    )

    return cast(float, area_weighted_diff_squared**0.5)


# %%
years_to_optimise = np.setdiff1d(neem_data["year"], obs_network_years.data)
years_to_optimise = years_to_optimise[np.in1d(years_to_optimise, out_years)]
years_to_optimise

# %%
iter_df = (
    pd.concat([neem_data, smooth_law_dome])
    .set_index("year")
    .loc[years_to_optimise]
    .set_index("source", append=True)
)
# iter_df

# %%
optimised = []
x0 = (1100, -70)
eofs = lat_grad_eofs_obs_network["eofs"]
for year, ydf in tqdman.tqdm(iter_df.groupby("year")):
    exp_n_ice_core_data_points = 2
    if ydf.shape[0] != exp_n_ice_core_data_points:
        msg = "Should have both NEEM and law dome data here..."
        raise AssertionError(msg)

    ice_core_data_year = xr.DataArray(
        data=[
            [
                ydf.loc[(year, "neem")]["value"],  # type: ignore
                ydf.loc[(year, "law_dome")]["value"],  # type: ignore
            ]
        ],
        dims=["year", "lat"],
        coords=dict(
            year=[year],
            lat=[neem_lat_nearest, law_dome_lat_nearest],
        ),
        attrs={"units": conc_unit},
    ).pint.quantify()

    diff_from_ice_cores_year = partial(
        diff_from_ice_cores,
        eofs=eofs,
        pc1=float(allyears_pc1.sel(year=year)),
        ice_core_data=ice_core_data_year,
    )

    min_res = scipy.optimize.minimize(
        diff_from_ice_cores_year,
        x0=x0,
        method="Nelder-Mead",
    )

    optimised.append([year, min_res.x[0], min_res.x[1]])

    x0 = min_res.x
    # if year > 300:
    #     break

# %% [markdown]
# Process the results.

# %%
optimised_ar = np.array(optimised)

global_annual_mean_optimised = xr.DataArray(
    name="global_annual_mean_optimised",
    data=optimised_ar[:, 1],
    dims=["year"],
    coords=dict(year=optimised_ar[:, 0]),
    attrs=dict(units=conc_unit),
).pint.quantify()

pc0_optimised = (
    xr.DataArray(
        name="pc0_optimised",
        data=optimised_ar[:, 2],
        dims=["year"],
        coords=dict(year=optimised_ar[:, 0]),
        attrs=dict(units="dimensionless"),
    )
    .assign_coords(eof=0)
    .pint.quantify()
)
pc0_optimised

# %% [markdown]
# Linearly interpolate the optimised values.

# %%
pc0_optimised_years_to_optimise = pc0_optimised.pint.dequantify().interp(
    year=years_to_optimise
)
# Assuming PC0 is constant before the start of the NEEM data.
pc0_optimised_years_to_optimise_back_to_year_one = (
    pc0_optimised_years_to_optimise.pint.dequantify().interp(
        year=np.arange(out_years[0], years_to_optimise[-1] + 1),
        kwargs={"fill_value": pc0_optimised.data[0].m},
    )
)

pc0_optimised.pint.dequantify().plot.scatter(x="year", color="tab:orange")
pc0_optimised_years_to_optimise_back_to_year_one.plot(
    linewidth=0.5, alpha=0.7, zorder=1
)

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

primap_fossil_ch4_emissions

# %%
regression_years = np.intersect1d(
    lat_grad_eofs_obs_network["year"], primap_fossil_ch4_emissions["year"]
)
regression_years

# %%
primap_regression_data = primap_fossil_ch4_emissions.sel(year=regression_years)
primap_regression_data

# %%
pc0_obs_network_regression = lat_grad_eofs_obs_network["principal-components"].sel(
    eof=0, year=regression_years
)
pc0_obs_network_regression

# %%
with axes_vertical_split() as axes:
    primap_regression_data.plot(ax=axes[0])
    pc0_obs_network_regression.plot(ax=axes[1])

# %%
x = QuantityOSCM(primap_regression_data.data.m, str(primap_regression_data.data.units))
A = np.vstack([x.m, np.ones(x.size)]).T
y = QuantityOSCM(
    pc0_obs_network_regression.data.m, str(pc0_obs_network_regression.data.units)
)

res = np.linalg.lstsq(A, y.m, rcond=None)
m, c = res[0]
m = QuantityOSCM(m, (y / x).units)
c = QuantityOSCM(c, y.units)

latitudinal_gradient_pc0_ch4_fossil_emissions_regression = (
    local.regressors.LinearRegressionResult(m=m, c=c)
)

fig, ax = plt.subplots()
ax.scatter(x.m, y.m, label="raw data")
ax.plot(x.m, (m * x + c).m, color="tab:orange", label="regression")
ax.set_ylabel("PC0")
ax.set_xlabel("PRIMAP emissions")
ax.legend()

# %%
years_to_fill_with_regression = np.setdiff1d(
    primap_fossil_ch4_emissions["year"],
    pc0_optimised_years_to_optimise_back_to_year_one["year"],
)
years_to_fill_with_regression = np.setdiff1d(
    years_to_fill_with_regression, pc0_obs_network_regression["year"]
)

years_to_fill_with_regression

# %%
pc0_emissions_extended = (
    m
    * primap_fossil_ch4_emissions.sel(
        year=years_to_fill_with_regression
    ).pint.quantify()
    + c
)
pc0_emissions_extended = pc0_emissions_extended.assign_coords(eof=0)
pc0_emissions_extended = pc0_emissions_extended.assign_coords(eof=0)
pc0_emissions_extended

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Concatenate the pieces of PC0
#
# Join:
#
# - optimised, interpolated based on Law Dome and NEEM
# - extended based on regression with emissions
# - raw values derived from the observational network

# %% editable=true slideshow={"slide_type": ""}
pc0_obs_network_raw_years = np.setdiff1d(
    lat_grad_eofs_obs_network["principal-components"].sel(
        eof=0,
    )["year"],
    pc0_obs_network_regression["year"],
)

# %% editable=true slideshow={"slide_type": ""}
allyears_pc0 = xr.concat(
    [
        pc0_optimised_years_to_optimise_back_to_year_one,
        pc0_emissions_extended,
        pc0_obs_network_regression,
        lat_grad_eofs_obs_network["principal-components"].sel(
            eof=0, year=pc0_obs_network_raw_years
        ),
    ],
    "year",
)

with axes_vertical_split() as axes:
    allyears_pc0.plot(ax=axes[0])

    pc0_optimised_years_to_optimise_back_to_year_one.sel(
        year=np.arange(
            1930, pc0_optimised_years_to_optimise_back_to_year_one["year"].max() + 1
        )
    ).plot(ax=axes[1])
    pc0_emissions_extended.plot(ax=axes[1])
    pc0_obs_network_regression.plot(ax=axes[1])

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

# %% [markdown]
# Quick check that our output matches the observational network in the years they overlap.

# %%
xr.testing.assert_allclose(
    (out["principal-components"] @ out["eofs"]).sel(year=regression_years),
    (
        lat_grad_eofs_obs_network["principal-components"]
        @ lat_grad_eofs_obs_network["eofs"]
    ).sel(year=regression_years),
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
config_step.latitudinal_gradient_pc0_ch4_fossil_emissions_regression_file.parent.mkdir(
    exist_ok=True, parents=True
)
with open(
    config_step.latitudinal_gradient_pc0_ch4_fossil_emissions_regression_file, "w"
) as fh:
    fh.write(
        local.config.converter_yaml.dumps(
            latitudinal_gradient_pc0_ch4_fossil_emissions_regression
        )
    )

latitudinal_gradient_pc0_ch4_fossil_emissions_regression
