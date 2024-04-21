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
# # CH$_4$ - extend the latitudinal gradient and global-, annual-mean
#
# Extend the latitudinal gradient and global-, annual-mean back in time.
# For CH$_4$, we do this by combining the values from ice cores etc.
# and our latitudinal gradient information.

# %% [markdown]
# ## Imports

# %%
from functools import partial

import cf_xarray.units
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import pint_xarray
import scipy.optimize
import tqdm.autonotebook as tqdman
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
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

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(
    cf_xarray.units.units
)

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

config_smooth_law_dome_data = get_config_for_step_id(
    config=config, step="smooth_law_dome_data", step_config_id="ch4"
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
# ### Load data

# %%
global_annual_mean = xr.load_dataarray(
    config_step.observational_network_global_annual_mean_file
).pint.quantify()
global_annual_mean

# %%
lat_grad_eofs = xr.load_dataset(
    config_step.observational_network_latitudinal_gradient_eofs_file
).pint.quantify()
lat_grad_eofs

# %%
smooth_law_dome = pd.read_csv(config_smooth_law_dome_data.smoothed_median_file)
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
# ### Define time axis onto which we want to interpolate

# %%
new_years = np.arange(1, lat_grad_eofs["year"].max() + 1)
new_years

# %%
current_years = lat_grad_eofs["year"]
current_years

# %% [markdown]
# ### Extend PC one
#
# This is kept constant before the observational network period.
#
# (Zero-indexing, hence this is the second PC)

# %%
# Quick assertion that things are as expected
if len(lat_grad_eofs["eof"]) != 2:
    raise AssertionError("Rethink")

# %%
new_pc1 = lat_grad_eofs["principal-components"].sel(eof=1).copy()
new_pc1 = new_pc1.pint.dequantify().interp(
    year=new_years, kwargs={"fill_value": new_pc1.data[0].m}
)
new_pc1.plot()
new_pc1

# %% [markdown]
# ## Optimise PC zero to match Law Dome and NEEM data


# %%
def get_col_assert_single_value(idf: pd.DataFrame, col: str) -> str:
    res = idf[col].unique()
    if len(res) != 1:
        raise AssertionError

    return res[0]


# %%
law_dome_lat = get_col_assert_single_value(smooth_law_dome, "latitude")
law_dome_lat

# %%
law_dome_lat_nearest = float(
    lat_grad_eofs.sel(lat=law_dome_lat, method="nearest")["lat"]
)
law_dome_lat_nearest

# %%
neem_lat = get_col_assert_single_value(neem_data, "latitude")
neem_lat

# %%
neem_lat_nearest = float(lat_grad_eofs.sel(lat=neem_lat, method="nearest")["lat"])
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
def diff_from_ice_cores(x, pc1, eofs, ice_core_data):
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
        local.xarray_space.calculate_weighted_area_mean_latitude_only(
            diff_squared, variables=["lat_resolved"]
        )["lat_resolved"].pint.dequantify()
    )

    return area_weighted_diff_squared**0.5


# %%
years_to_optimise = sorted(
    list(set(neem_data["year"]).difference(set(global_annual_mean["year"].data)))
)
display(years_to_optimise[0])
years_to_optimise[-1]

# %%
iter_df = (
    pd.concat([neem_data, smooth_law_dome])
    .set_index("year")
    .loc[years_to_optimise]
    .set_index("source", append=True)
)
# iter_df

# %%
# EOFs have spatial-mean of zero, which any scaling will preserve
np.testing.assert_allclose(
    local.xarray_space.calculate_global_mean_from_lon_mean(
        lat_grad_eofs["eofs"]
    ).data.m,
    0.0,
    atol=1e-10,
)

# %%
optimised = []
x0 = (1100, -70)
eofs = lat_grad_eofs["eofs"]
for year, ydf in tqdman.tqdm(iter_df.groupby("year")):
    if ydf.shape[0] != 2:
        msg = "Should have both NEEM and law dome data here..."
        raise AssertionError(msg)

    ice_core_data_year = xr.DataArray(
        data=[
            [
                ydf.loc[(year, "neem")]["value"],
                ydf.loc[(year, "law_dome")]["value"],
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
        pc1=float(new_pc1.sel(year=year)),
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

optimised_ar = np.array(optimised)

optimised_global_mean = xr.DataArray(
    name="optimised_global-mean",
    data=optimised_ar[:, 1],
    dims=["year"],
    coords=dict(year=optimised_ar[:, 0]),
    attrs=dict(units=conc_unit),
).pint.quantify()

optimised_pc0 = xr.DataArray(
    name="optimised_pc0",
    data=optimised_ar[:, 2][:, np.newaxis],
    dims=["year", "eof"],
    coords=dict(year=optimised_ar[:, 0], eof=[0]),
    attrs=dict(units="dimensionless"),
).pint.quantify()
optimised_pc0

# %% [markdown]
# ### Interpolate the optimised PC0

# %%
optimised_pc0

# %%
# Interpolate over the years we have
optimised_pc0_interpolated_annual = optimised_pc0.interp(year=years_to_optimise)
# Assuming PC0 is constant before the start of the NEEM data.
optimised_pc0_interpolated_annual = optimised_pc0_interpolated_annual.interp(
    year=np.arange(1, years_to_optimise[-1] + 1),
    kwargs={"fill_value": optimised_pc0.data[0]},
)

local.xarray_time.convert_year_to_time(optimised_pc0.sel(eof=0).squeeze()).plot.scatter(
    x="time", color="tab:orange"
)
local.xarray_time.convert_year_to_time(optimised_pc0_interpolated_annual).plot(
    linewidth=0.5, alpha=0.7, zorder=1
)

# %% [markdown]
# ### Extend PC0

# %%
assert False, "Use the regression against emissions to fill in 1948 to 1984"

# %% [markdown]
# ### Concatenate the pieces of PC0
#
# Join the optimised, interpolated PC0, PC0 extended based on a regression against emissions and PC0 from the observational network.

# %%
# pc0_obs_network = (
#     lat_grad_eofs["principal-components"]
#     .loc[{"eof": 0}]
# )

# %%
# optimised_pc0_interpolated_annual.plot()
# lat_grad_eofs["principal-components"].sel(eof=0).plot()

# %% [markdown]
# ### Join the PCs back together

# %%
assert False, "More steps below"

# %% [markdown]
# - concatenate PC0 and PC1 on the EOF axis
# - interpolate the PCs to monthly
# - create the monthly lat. grad (interpolated monthly PCs @ eofs)
#
# Annual-mean extension bits:
#
# - create a monthly timeseries over all latitudes by offseting the latitudinal gradient such that the value in the Law Dome box is matched to the smoothed law dome value (in the timesteps where we don't have obs network)
#   - should be a basic offset calculation
# - check that annual average of this field gives the NEEM and Law Dome values (this is the key check)
# - can then calculate global-mean values from this field (which is effectively the sum of the global-mean and the lat. gradient, as the lat. gradient has a spatial mean of zero (can check this of course))

# %%
lat_grad_eofs_updated_pc0 = lat_grad_eofs.copy(deep=True)
lat_grad_eofs_updated_pc0

# %%
pc0_keep = (
    lat_grad_eofs_updated_pc0["principal-components"]
    .loc[{"eof": 0}]
    .sel(
        year=np.arange(
            years_to_optimise[-1] + 1, lat_grad_eofs_updated_pc0["year"].max() + 1
        )
    )
)
pc0_keep

# %%
pc0_new = xr.concat([optimised_pc0_interpolated, pc0_keep], "year")
pc0_new

# %%
lat_grad_eofs_updated_pc0["principal-components"].loc[{"eof": 0}] = pc0_new
lat_grad_eofs_updated_pc0

# %% [markdown]
# ### Mean-preserving interpolation
#
# Interpolate the PC to a monthly time axis before continuing,
# so that we get a latitudinal gradient on a monthly time step.
# This is needed to avoid jumps at the boundary of years
# when we combine everything back together.

# %%
lat_grad_eofs_updated_pc0["principal-components"] = (
    lat_grad_eofs_updated_pc0["principal-components"]
    .groupby("eof", squeeze=False)
    .apply(local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly)
)
lat_grad_eofs_updated_pc0

# %%
fix, axes = plt.subplots(ncols=2, sharey=True)

lat_grad_eofs["principal-components"].plot(hue="eof", ax=axes[0])
local.xarray_time.convert_year_month_to_time(lat_grad_eofs_updated_pc0)[
    "principal-components"
].plot(hue="eof", ax=axes[1])

plt.show()

# %%
lat_gradient_updated_pc0 = (
    lat_grad_eofs_updated_pc0["principal-components"]
    @ lat_grad_eofs_updated_pc0["eofs"]
)
lat_gradient_updated_pc0

# %%
fig, axes = plt.subplots(ncols=2)
local.xarray_time.convert_year_month_to_time(
    lat_gradient_updated_pc0.sel(
        year=np.hstack([1, 1900, np.arange(1700, 1901, 100), 2020]), month=[6]
    )
).plot(y="lat", hue="time", alpha=0.7, ax=axes[0])

local.xarray_time.convert_year_month_to_time(
    lat_gradient_updated_pc0.sel(
        year=np.hstack(
            [
                np.arange(1900, 2021, 10),
                2022,
            ]
        ),
        month=[6],
    )
).plot(y="lat", hue="time", alpha=0.7, ax=axes[1])

plt.tight_layout()

# %% [markdown]
# ## Optimise global-mean to match Law Dome
#
# Now that we have a timeseries of PC zero, we now use Law Dome to optimise the global-mean over the whole timeseries.

# %% [markdown]
# Here we just add a couple of data points from EPICA to round out the Law Dome timeseries. There are probably better ways to do this (latitudes don't line up exactly, for example), but this is fine for now.

# %%
epica_data_to_add = epica_data[
    (epica_data["year"] > -12) & (epica_data["year"] < smooth_law_dome["year"].min())
]
epica_data_to_add

# %%
smooth_law_dome_plus_epica = (
    xr.DataArray(
        data=np.hstack([epica_data_to_add["value"], smooth_law_dome["value"]]),
        dims=["year"],
        coords=dict(
            year=np.hstack([epica_data_to_add["year"], smooth_law_dome["year"]])
        ),
        attrs={"units": "ppb"},
    )
    .interp(year=np.arange(1, global_annual_mean["year"].min()))
    .pint.quantify()
)
smooth_law_dome_plus_epica

# %%
global_annual_mean_extension = (
    smooth_law_dome_plus_epica
    - lat_gradient_updated_pc0.sel(
        lat=law_dome_lat_nearest, year=smooth_law_dome_plus_epica["year"]
    )
)
global_annual_mean_extension["lat"] = 0.0
global_annual_mean_extension

# %%
import cftime

# %%
fig, axes = plt.subplots(ncols=2)
tmp = global_annual_mean.copy()
tmp["year"] = [cftime.datetime(y, 1, 1) for y in tmp["year"]]
tmp.plot(ax=axes[0])
local.xarray_time.convert_year_month_to_time(global_annual_mean_extension).plot(
    ax=axes[0]
)
tmp.plot(ax=axes[1])
local.xarray_time.convert_year_month_to_time(global_annual_mean_extension).plot(
    ax=axes[1]
)
# axes[1].set_xlim([0, 2000])
plt.tight_layout()

# %%
global_annual_mean_full

# %%
global_annual_mean_full = xr.concat(
    [global_annual_mean_extension, global_annual_mean], "year"
)
global_annual_mean_full.plot()
global_annual_mean_full

# %%
# Checks:
# - match law dome in all years in which we used law dome
# - match neem in all years in which we used neem
# - match EPICA in all years in which we used EPICA
# - jumps/steps/bumps at joins between data sets that make things look weird/bad

# %%
fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
optimised_lat_values.sel(lat=neem_lat, method="nearest").plot(ax=axes[0])
optimised_global_mean.plot(ax=axes[0], color="tab:green", alpha=0.5)
neem_data.plot.scatter(x="year", y="value", ax=axes[0], color="tab:orange")

optimised_lat_values.sel(lat=law_dome_lat, method="nearest").plot(ax=axes[1])
optimised_global_mean.plot(ax=axes[1], color="tab:green", alpha=0.5)
smooth_law_dome.plot.scatter(x="year", y="value", ax=axes[1], color="tab:orange")

axes[0].set_xlim([1500, 1700])
axes[0].set_ylim([600, 800])
plt.tight_layout()

# %%
eofs.plot(hue="eof")
lat_grad_full = eofs @ new_pcs
lat_grad_full.plot(hue="lat")
lat_grad_full
new_pcs = xr.concat([optimised_pc0_interpolated, new_pc1], "eof")
new_pcs.plot(hue="eof")

# %% [markdown]
# ### Save

# %%
config_step.latitudinal_gradient_file.parent.mkdir(exist_ok=True, parents=True)
lat_gradient_updated_pc0.pint.dequantify().to_netcdf(
    config_step.latitudinal_gradient_file
)
lat_gradient_updated_pc0

# %%
config_step.global_annual_mean_file.parent.mkdir(exist_ok=True, parents=True)
global_annual_mean_full.pint.dequantify().to_netcdf(config_step.global_annual_mean_file)
global_annual_mean_full
