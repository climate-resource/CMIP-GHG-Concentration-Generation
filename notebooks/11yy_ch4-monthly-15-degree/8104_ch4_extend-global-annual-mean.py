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
# Re-write to:
#
# - extend everything in year space first
# - join with obs network stuff
# - mean-preserving interpolation to months
# - save

# %% [markdown]
# ## Imports

# %%
from functools import partial

import cf_xarray.units
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import pint_xarray
import primap2
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

# %%
QuantityOSCM = openscm_units.unit_registry.Quantity

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

config_retrieve_misc = get_config_for_step_id(
    config=config, step="retrieve_misc_data", step_config_id="only"
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

optimised_pc0 = (
    xr.DataArray(
        name="optimised_pc0",
        data=optimised_ar[:, 2],
        dims=["year"],
        coords=dict(year=optimised_ar[:, 0]),
        attrs=dict(units="dimensionless"),
    )
    .assign_coords(eof=0)
    .pint.quantify()
)
optimised_pc0

# %% [markdown]
# ### Interpolate the optimised PC0

# %%
# Interpolate over the years we have
optimised_pc0_interpolated_annual = optimised_pc0.interp(year=years_to_optimise)
# Assuming PC0 is constant before the start of the NEEM data.
optimised_pc0_interpolated_annual = optimised_pc0_interpolated_annual.interp(
    year=np.arange(1, years_to_optimise[-1] + 1),
    kwargs={"fill_value": optimised_pc0.data[0]},
)

local.xarray_time.convert_year_to_time(optimised_pc0.squeeze()).plot.scatter(
    x="time", color="tab:orange"
)
local.xarray_time.convert_year_to_time(optimised_pc0_interpolated_annual).plot(
    linewidth=0.5, alpha=0.7, zorder=1
)

# %% [markdown]
# ### Extend PC0

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
obs_network_pc0 = lat_grad_eofs["principal-components"].sel(eof=0)
obs_network_pc0

# %%
years_to_fill_with_regression = np.array(
    list(
        set(primap_fossil_ch4_emissions["year"].data)
        .difference(set(optimised_pc0_interpolated_annual["year"].data))
        .difference(set(obs_network_pc0["year"].data))
    )
)
years_to_fill_with_regression

# %%
fig, axes = plt.subplots(ncols=2)

primap_regression_data = primap_fossil_ch4_emissions.sel(year=lat_grad_eofs["year"])
primap_regression_data.plot(ax=axes[0])

obs_network_pc0.plot(ax=axes[1])

plt.tight_layout()

# %%
x = QuantityOSCM(primap_regression_data.data, str(primap_regression_data.data.units))
A = np.vstack([x.m, np.ones(x.size)]).T
y = QuantityOSCM(obs_network_pc0.data, str(obs_network_pc0.data.units))

res = np.linalg.lstsq(A, y.m, rcond=None)
m, c = res[0]
m = QuantityOSCM(m, (y / x).units)
c = QuantityOSCM(c, y.units)

fig, ax = plt.subplots()
ax.scatter(x.m, y.m, label="raw data")
ax.plot(x.m, (m * x + c).m, color="tab:orange", label="regression")
ax.set_ylabel("PC0")
ax.set_xlabel("PRIMAP emissions")
ax.legend()

# %%
pc0_emissions_extended = (
    m
    * primap_fossil_ch4_emissions.sel(
        year=years_to_fill_with_regression
    ).pint.quantify()
    + c
)
# pc0_emissions_extended.name = new_pc1.name
pc0_emissions_extended = pc0_emissions_extended.assign_coords(eof=0)
pc0_emissions_extended = pc0_emissions_extended.assign_coords(eof=0)
pc0_emissions_extended

# %% [markdown]
# ### Concatenate the pieces of PC0
#
# Join the optimised, interpolated PC0, PC0 extended based on a regression against emissions and PC0 from the observational network.

# %%
pc0_obs_network = lat_grad_eofs["principal-components"].loc[{"eof": 0}]

# %%
optimised_pc0_interpolated_annual.plot()
pc0_emissions_extended.plot()
pc0_obs_network.plot()
plt.xlim([1800, 2025])

# %%
new_pc0 = xr.concat(
    [
        optimised_pc0_interpolated_annual,
        pc0_emissions_extended,
        pc0_obs_network,
    ],
    "year",
)
new_pc0

# %% [markdown]
# ### Join the PCs back together

# %%
new_pcs = xr.concat([new_pc0, new_pc1], "eof").pint.dequantify().pint.quantify()
new_pcs

# %% [markdown]
# ### Interpolate the PCs to monthly time step

# %%
new_pcs_monthly = new_pcs.groupby("eof", squeeze=False).apply(
    local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly
)
new_pcs_monthly

# %% [markdown]
# ### Create latitudinal gradient

# %%
lat_gradient_full = new_pcs_monthly @ lat_grad_eofs["eofs"]
lat_gradient_full

# %% [markdown]
# ### Create full field over entire timeseries
#
# We do this by ensuring that the value in Law Dome's bin
# matches our smoothed Law Dome timeseries.

# %%
law_dome_da = xr.DataArray(
    data=smooth_law_dome["value"],
    dims=["year"],
    coords=dict(year=smooth_law_dome["year"]),
    attrs=dict(units=conc_unit),
).pint.quantify()

# %%
offset = law_dome_da - lat_gradient_full.sel(lat=law_dome_lat, method="nearest")
offset

# %%
full_field_law_dome = lat_gradient_full + offset
full_field_law_dome

# %% [markdown]
# Here we just add a couple of data points from EPICA to round out the Law Dome timeseries
# so we can have a timeseries that goes back to year 1.
# There are probably better ways to do this, but this is fine for now.

# %%
epica_data_to_add = epica_data[
    (epica_data["year"] > -12) & (epica_data["year"] < smooth_law_dome["year"].min())
].sort_values(by="year")
epica_data_to_add

# %%
epica_lat = get_col_assert_single_value(epica_data_to_add, "latitude")
epica_lat

# %% [markdown]
# We simply linearly interpolate the EPICA data to get a yearly timeseries over the period of interest.
# We make sure that the interpolated values match our current values at the year in which Law Dome starts.
# This isn't perfect and could be investigated further, but will do for now.

# %%
law_dome_start_year = law_dome_da["year"].min()
years_use_epica = np.arange(1, law_dome_start_year)
years_use_epica

# %%
harmonisation_value = float(
    full_field_law_dome.sel(year=law_dome_start_year)
    .sel(lat=epica_lat, method="nearest")
    .mean("month")
    .data.m
)
harmonisation_value

# %%
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
epica_data[(epica_data["year"] > -1000) & (epica_data["year"] < 200)].plot.scatter(
    x="year", y="value", ax=ax, color="tab:orange", label="EPICA raw"
)
ax.legend()
epica_da

# %%
epica_da

# %%
lat_gradient_full.sel(lat=epica_lat, method="nearest")

# %%
offset_epica = epica_da - lat_gradient_full.sel(lat=epica_lat, method="nearest")
full_field_epica = lat_gradient_full + offset_epica
full_field_epica

# %%
full_field = xr.concat([full_field_epica, full_field_law_dome], "year")
full_field

# %% [markdown]
# ### Check our full field calculation
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
#   - a global-mean timeseries (with dims (year, month))
#   - a latitudinal gradient (with dims (year, month, lat))
#     that has a spatial-mean of zero.
#     This latitudinal gradient should match the latitudinal
#     gradient we calculated earlier.

# %%
full_field_annual_mean = full_field.mean(dim="month")
full_field_annual_mean

# %% [markdown]
# Check agreement with NEEM

# %%
np.testing.assert_allclose(
    full_field_annual_mean.sel(lat=neem_lat, method="nearest")
    .sel(year=neem_data["year"].values)
    .data.to(conc_unit)
    .m,
    neem_data["value"],
)

# %% [markdown]
# Check agreement with Law Dome

# %%
np.testing.assert_allclose(
    full_field_annual_mean.sel(lat=law_dome_lat, method="nearest")
    .sel(year=smooth_law_dome["year"].values)
    .data.to(conc_unit)
    .m,
    smooth_law_dome["value"],
)

# %% [markdown]
# Check agreement with EPICA

# %%
np.testing.assert_allclose(
    full_field_annual_mean.sel(lat=epica_lat, method="nearest")
    .sel(year=epica_da["year"].values)
    .data.to(conc_unit)
    .m,
    epica_da.data,
)

# %%
tmp = full_field.copy()
tmp.name = "global_mean"
global_mean = local.xarray_space.calculate_global_mean_from_lon_mean(tmp)
global_mean

# %%
lat_grad = full_field - global_mean
lat_grad

# %% [markdown]
# Check that this latitudinal gradient is the same as the one we calculated previously
# (in the years in which they overlap).

# %%
xr.testing.assert_allclose(
    lat_grad,
    lat_gradient_full.sel(year=lat_grad["year"]),
)

# %% [markdown]
# ### Join it all back together
#
# Now we create a full global-mean and latitudinal gradient timeseries based on:
#
# - the observational network for the time it is available
# - our extensions above for the rest

# %%
global_annual_mean_obs_network = (
    local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly(
        global_annual_mean
    )
)
global_annual_mean_obs_network["year"]

# %%
lat_grad_obs_network = lat_grad_eofs["eofs"] @ (
    lat_grad_eofs["principal-components"]
    .groupby("eof", squeeze=False)
    .apply(local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly)
)
lat_grad_obs_network["year"]

# %%
use_extension_years = np.setdiff1d(
    lat_gradient_full["year"], lat_grad_obs_network["year"]
)
use_extension_years

# %%
out_lat_grad = xr.concat(
    [
        lat_grad.sel(year=use_extension_years),
        lat_grad_obs_network,
    ],
    "year",
)
if out_lat_grad.isnull().any():
    raise AssertionError

local.xarray_time.convert_year_month_to_time(out_lat_grad).plot(hue="lat")
out_lat_grad

# %%
out_global_mean = xr.concat(
    [
        global_mean.sel(year=use_extension_years),
        global_annual_mean_obs_network,
    ],
    "year",
)
if out_global_mean.isnull().any():
    raise AssertionError

local.xarray_time.convert_year_month_to_time(out_global_mean).plot()
out_global_mean

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
# ### Save

# %%
config_step.latitudinal_gradient_file.parent.mkdir(exist_ok=True, parents=True)
lat_gradient_updated_pc0.pint.dequantify().to_netcdf(
    config_step.latitudinal_gradient_file
)
lat_gradient_updated_pc0

# %%
config_step.global_mean_file.parent.mkdir(exist_ok=True, parents=True)
global_annual_mean_full.pint.dequantify().to_netcdf(config_step.global_annual_mean_file)
global_annual_mean_full
