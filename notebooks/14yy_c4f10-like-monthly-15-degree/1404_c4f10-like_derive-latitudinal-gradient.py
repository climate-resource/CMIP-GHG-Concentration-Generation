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
# # C$_4$F$_{10}$-like - Derive latitudinal gradient
#
# User the data to derive the latitudinal gradient.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import cf_xarray.units
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
import local.dependencies
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
cf_xarray.units.units.define("ppt = ppb / 1000")

pint_xarray.accessors.default_registry = pint_xarray.setup_registry(cf_xarray.units.units)

Quantity = pint.get_application_registry().Quantity  # type: ignore

# %%
pint_xarray.setup_registry(openscm_units.unit_registry)

QuantityOSCM = openscm_units.unit_registry.Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_c4f10_like_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "c4f10"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

config_historical_emissions = get_config_for_step_id(
    config=config, step="compile_historical_emissions", step_config_id="only"
)
config_droste = get_config_for_step_id(
    config=config, step="retrieve_and_process_droste_et_al_2020_data", step_config_id="only"
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load raw data

# %%
droste = pd.read_csv(config_droste.processed_data_file)
droste = droste[droste["gas"] == config_step.gas]
droste

# %%
local.dependencies.save_dependency_into_db(
    db=config.dependency_db,
    gas=config_step.gas,
    dependency_short_name=config_droste.source_info.short_name,
)

# %%
historical_emissions = pd.read_csv(config_historical_emissions.complete_historical_emissions_file)
historical_emissions = historical_emissions[
    historical_emissions["variable"] == f"Emissions|{config_step.gas}"
]
if historical_emissions.empty:
    msg = "No data found for gas, check your config"
    raise AssertionError(msg)
historical_emissions

# %%
hist_emms_short_names = local.dependencies.load_source_info_short_names(
    config_historical_emissions.source_info_short_names_file
)

for sn in hist_emms_short_names:
    local.dependencies.save_dependency_into_db(
        db=config.dependency_db,
        gas=config_step.gas,
        dependency_short_name=sn,
    )

# %% [markdown]
# ### Calculate latitudinal gradient
#
# We assume that the latitudinal gradient is just linear (incorporating a cosine weighting),
# then inform its size based on the Droste et al. data.

# %%
lat_bin_weights = np.sin(local.binning.LAT_BIN_BOUNDS * np.pi / 180)
lat_bin_weights = np.diff(lat_bin_weights)
lat_bin_weights

# %% [markdown]
# #### EOF

# %%
n_bins = len(local.binning.LAT_BIN_CENTRES)
lat_grad_eof = np.linspace(-1, 1, n_bins) / lat_bin_weights

nh_slice = slice(int(n_bins / 2), n_bins, 1)
sh_slice = slice(0, int(n_bins / 2), 1)
norm = 2 * np.sum(lat_bin_weights[nh_slice] * lat_grad_eof[nh_slice]) / np.sum(lat_bin_weights[nh_slice])
lat_grad_eof = lat_grad_eof / norm

if not np.isclose((lat_bin_weights * lat_grad_eof).sum(), 0.0):
    msg = "Spatial mean of lat. gradient is clearly wrong"
    raise AssertionError(msg)


if not np.isclose(
    np.sum(lat_bin_weights[nh_slice] * lat_grad_eof[nh_slice]) / np.sum(lat_bin_weights[nh_slice]), 0.5
):
    msg = "Norm of lat. gradient is wrong"
    raise AssertionError(msg)

if not np.isclose(
    np.sum(lat_bin_weights[sh_slice] * lat_grad_eof[sh_slice]) / np.sum(lat_bin_weights[sh_slice]), -0.5
):
    msg = "Norm of lat. gradient is wrong"
    raise AssertionError(msg)

lat_grad_eof = xr.DataArray(
    lat_grad_eof,
    dims=("lat",),
    coords={"lat": local.binning.LAT_BIN_CENTRES},
    attrs={"units": "ppt"},
).pint.quantify()
lat_grad_eof

# %%
fig, axes = plt.subplots(ncols=2)
axes[0].plot(local.binning.LAT_BIN_CENTRES, lat_grad_eof)  # type: ignore
axes[1].plot(local.binning.LAT_BIN_CENTRES / lat_bin_weights, lat_grad_eof)  # type: ignore


# %% [markdown]
# #### Principal components

# %%
cg_lat = -40.6833
tal_lat = 52.5127

# %%
droste_cg = droste[droste["lat"] == cg_lat]
droste_tal = droste[droste["lat"] == tal_lat]

# %%
lat_grad_eof["lat"][
    (local.binning.LAT_BIN_BOUNDS[:-1] < cg_lat) & (local.binning.LAT_BIN_BOUNDS[1:] > cg_lat)
].values.squeeze()

# %%
cg_lat_bin_centre = lat_grad_eof["lat"][
    (local.binning.LAT_BIN_BOUNDS[:-1] < cg_lat) & (local.binning.LAT_BIN_BOUNDS[1:] > cg_lat)
].values.squeeze()

tal_lat_bin_centre = lat_grad_eof["lat"][
    (local.binning.LAT_BIN_BOUNDS[:-1] < tal_lat) & (local.binning.LAT_BIN_BOUNDS[1:] > tal_lat)
].values.squeeze()

# %%
lat_grad_eof_cg = lat_grad_eof.sel(lat=cg_lat_bin_centre).data.m.squeeze()

lat_grad_eof_tal = lat_grad_eof.sel(lat=tal_lat_bin_centre).data.m.squeeze()

# %%
lat_grad_pc_df = (droste_tal.set_index("year")["value"] - droste_cg.set_index("year")["value"]) / (
    lat_grad_eof_tal - lat_grad_eof_cg
)
lat_grad_pc_df

# %%
lat_grad_pc = xr.DataArray(
    lat_grad_pc_df.values,
    dims=("year",),
    coords={"year": lat_grad_pc_df.index.values},
    attrs={"units": "dimensionless"},
)
lat_grad_pc

# %% [markdown]
# ### Create latitudinally-gridded values
#
# Check that this latitudinal gradient recovers the original data.

# %%
lat_grad_helper = lat_grad_pc * lat_grad_eof
lat_grad_helper

# %%
global_lat_gridded_tmp = (lat_grad_helper - lat_grad_helper.sel(lat=cg_lat_bin_centre)).pint.dequantify() + (
    droste_cg["value"].values[:, np.newaxis]
)
global_lat_gridded_tmp.attrs["units"] = str(lat_grad_helper.data.u)
global_lat_gridded_tmp

# %%
for lat_sel, ref in (
    (lat_grad_helper["lat"] == tal_lat_bin_centre, droste_tal),
    (lat_grad_helper["lat"] == cg_lat_bin_centre, droste_cg),
):
    np.testing.assert_allclose(ref["value"].values, global_lat_gridded_tmp.sel(lat=lat_sel).data.squeeze())  # type: ignore

# %% [markdown]
# ### Global-, annual-mean

# %%
global_annual_mean = local.xarray_space.calculate_global_mean_from_lon_mean(
    global_lat_gridded_tmp.pint.quantify()
)
global_annual_mean.plot.line()
global_annual_mean

# %% [markdown]
# Check that this, plus the latitudinal gradient, gets back to Droste et al.

# %%
lat_grad_annual_mean = global_lat_gridded_tmp.pint.quantify() - global_annual_mean
lat_grad_annual_mean

# %%
np.testing.assert_allclose(
    local.xarray_space.calculate_global_mean_from_lon_mean(lat_grad_annual_mean).data.m,
    0.0,
    atol=1e-10,
)

# %%
for lat_sel, ref in (
    (lat_grad_helper["lat"] == tal_lat_bin_centre, droste_tal),
    (lat_grad_helper["lat"] == cg_lat_bin_centre, droste_cg),
):
    np.testing.assert_allclose(
        ref["value"].values,  # type: ignore
        (global_annual_mean + lat_grad_annual_mean).sel(lat=lat_sel).data.m.squeeze(),
    )

# %% [markdown]
# For now, basic linear extrapolation to get to 2022.
# Ideally, we don't need this in future.

# %%
min_last_year = 2022
if global_annual_mean["year"].max() < min_last_year:
    global_annual_mean = (
        global_annual_mean.pint.dequantify()
        .interp(
            year=range(global_annual_mean["year"].min().values, min_last_year + 1),
            method="linear",
            kwargs=dict(fill_value="extrapolate"),
        )
        .pint.quantify()
    )
    global_annual_mean.plot.line()

# %% [markdown]
# Extrapolate backwards in time to get complete coverage.

# %%
if not np.isclose(global_annual_mean.isel(year=0).data.m, 0.0, atol=1e-5):
    raise ValueError

# %%
global_annual_mean = (
    global_annual_mean.pint.dequantify()
    .interp(
        year=range(1, global_annual_mean["year"].max().values + 1),
        method="linear",
        kwargs=dict(fill_value=0.0),
    )
    .pint.quantify()
)
global_annual_mean.plot.line()

# %% [markdown]
# #### Regress the latitudinal gradient against emissions

# %%
lat_grad_pc = lat_grad_pc.pint.quantify()
lat_grad_pc

# %%
historical_emissions = historical_emissions.rename({"time": "year"}, axis="columns")
historical_emissions

# %%
regression_years = np.intersect1d(lat_grad_pc["year"], historical_emissions["year"])
regression_years

# %%
unit = historical_emissions["unit"].unique()
assert len(unit) == 1
unit = unit[0]

historical_emissions_xr = xr.DataArray(
    historical_emissions["value"],
    dims=("year",),
    coords=dict(year=historical_emissions["year"]),
    attrs={"units": unit},
).pint.quantify(unit_registry=openscm_units.unit_registry)
# historical_emissions_xr = historical_emissions_xr.sel(
#     year=historical_emissions_xr["year"] <= regression_years.max()
# )


historical_emissions_xr

# %%
historical_emissions_regression_data = historical_emissions_xr.sel(year=regression_years)

historical_emissions_regression_data

# %%
lat_grad_pc_regression = lat_grad_pc.sel(year=regression_years)
lat_grad_pc_regression

# %%
fig, axes = plt.subplots(ncols=2)
historical_emissions_regression_data.plot(ax=axes[0])  # type: ignore
lat_grad_pc_regression.plot(ax=axes[1])  # type: ignore
fig.tight_layout()

# %%
# The code below fixes the y-intercept to be zero, so that the latitudinal gradient is zero
# when emissions are zero.
# This is fine for gases like SF6, whose pre-industrial concentrations are zero.
# Have to be more careful when the pre-industrial concentrations are non-zero.
x = QuantityOSCM(
    historical_emissions_regression_data.data.m,
    str(historical_emissions_regression_data.data.units),
)
A = x.m[:, np.newaxis]
y = QuantityOSCM(lat_grad_pc_regression.data.m, str(lat_grad_pc_regression.data.units))

res = np.linalg.lstsq(A, y.m, rcond=None)
m = res[0]
m = QuantityOSCM(m, (y / x).units)
c = QuantityOSCM(0.0, y.units)

latitudinal_gradient_pc0_total_emissions_regression = local.regressors.LinearRegressionResult(m=m, c=c)

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

latitudinal_gradient_pc0_total_emissions_regression

# %% [markdown]
# ### Calculate global-, annual-mean monthly

# %%
# global_annual_mean_monthly = local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly(
#     global_annual_mean,
#     algorithm=LaiKaplanInterpolator(
#         get_wall_control_points_y_from_interval_ys=get_wall_control_points_y_linear_with_flat_override_on_left,
#         min_val=global_annual_mean.min().data,
#     ),
# )
# global_annual_mean_monthly

# %%
# fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
# if isinstance(axes, matplotlib.axes.Axes):
#     raise TypeError(type(axes))

# local.xarray_time.convert_year_month_to_time(global_annual_mean_monthly, calendar="proleptic_gregorian").plot(  # type: ignore
#     ax=axes[0]
# )
# local.xarray_time.convert_year_to_time(global_annual_mean, calendar="proleptic_gregorian").plot.scatter(
#     x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[0]
# )

# local.xarray_time.convert_year_month_to_time(
#     global_annual_mean_monthly.sel(year=global_annual_mean_monthly["year"][1:10]),
#     calendar="proleptic_gregorian",
# ).plot(ax=axes[1])  # type: ignore
# local.xarray_time.convert_year_to_time(
#     global_annual_mean.sel(year=global_annual_mean_monthly["year"][1:10]),
#     calendar="proleptic_gregorian",
# ).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[1])

# local.xarray_time.convert_year_month_to_time(
#     global_annual_mean_monthly.sel(year=global_annual_mean_monthly["year"][-10:]),
#     calendar="proleptic_gregorian",
# ).plot(ax=axes[2])  # type: ignore
# local.xarray_time.convert_year_to_time(
#     global_annual_mean.sel(year=global_annual_mean_monthly["year"][-10:]),
#     calendar="proleptic_gregorian",
# ).plot.scatter(x="time", color="tab:orange", zorder=3, alpha=0.5, ax=axes[2])

# plt.tight_layout()
# plt.show()

# %% [markdown]
# ### Latitudinal-gradient extension

# %% [markdown]
# Firstly extend back to year 1.

# %%
if not np.isclose(lat_grad_pc.isel(year=0).data.m, 0.0, atol=1e-5):
    raise AssertionError

lat_grad_pc_extended = (
    lat_grad_pc.pint.dequantify()
    .interp(
        year=range(1, lat_grad_pc.year.max().data.squeeze() + 1),
        method="linear",
        kwargs=dict(fill_value=0.0),
    )
    .pint.quantify()
)
lat_grad_pc_extended

# %% [markdown]
# Extend forward using linear extrapolation
# (this can also be done with the regression against emissions,
# but it is more complicated than it needs to be).

# %%
# # The code for doing this with a regression
# years_to_fill_with_regression = np.setdiff1d(
#     range(1, min_last_year + 1),
#     lat_grad_pc_extended["year"],
# )

# years_to_fill_with_regression

# historical_emissions_xr.sel(year=years_to_fill_with_regression).plot.line()

# lat_grad_pc_extended_reg_part = (
#     m
#     * historical_emissions_xr.sel(year=years_to_fill_with_regression).pint.quantify(
#         unit_registry=openscm_units.unit_registry
#     )
#     + c
# )
# lat_grad_pc_extended_reg_part
# lat_grad_pc_extended = xr.concat([lat_grad_pc_extended, lat_grad_pc_extended_reg_part], "year")
# lat_grad_pc_extended.plot.line()
# lat_grad_pc_extended

# %%
lat_grad_pc_extended = (
    lat_grad_pc_extended.pint.dequantify()
    .interp(
        year=range(1, min_last_year + 1),
        method="linear",
        kwargs=dict(fill_value="extrapolate"),
    )
    .pint.quantify()
)
lat_grad_pc_extended.sel(year=range(min_last_year - 10, min_last_year + 1)).plot.line()
lat_grad_pc_extended

# %%
# lat_grad_pc_monthly = local.mean_preserving_interpolation.interpolate_annual_mean_to_monthly(
#     lat_grad_pc_extended,
#     algorithm=LaiKaplanInterpolator(
#         get_wall_control_points_y_from_interval_ys=get_wall_control_points_y_linear_with_flat_override_on_left,
#         min_val=lat_grad_pc_extended.min().data,
#     ),
# )
# lat_grad_pc_monthly

# %%
# pcs_annual = lat_grad_pc_extended
# pcs_monthly = lat_grad_pc_monthly

# fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
# if isinstance(axes, matplotlib.axes.Axes):
#     raise TypeError(type(axes))


# local.xarray_time.convert_year_month_to_time(pcs_monthly, calendar="proleptic_gregorian").plot(ax=axes[0])
# local.xarray_time.convert_year_to_time(pcs_annual, calendar="proleptic_gregorian").plot.scatter(
#     x="time", zorder=3, alpha=0.5, ax=axes[0]
# )

# local.xarray_time.convert_year_month_to_time(
#     pcs_monthly.sel(year=pcs_monthly["year"][1:10]), calendar="proleptic_gregorian"
# ).plot(ax=axes[1])
# local.xarray_time.convert_year_to_time(
#     pcs_annual.sel(year=pcs_monthly["year"][1:10]), calendar="proleptic_gregorian"
# ).plot.scatter(x="time", zorder=3, alpha=0.5, ax=axes[1])

# local.xarray_time.convert_year_month_to_time(
#     pcs_monthly.sel(year=pcs_monthly["year"][-10:]), calendar="proleptic_gregorian"
# ).plot(ax=axes[2])
# local.xarray_time.convert_year_to_time(
#     pcs_annual.sel(year=pcs_monthly["year"][-10:]), calendar="proleptic_gregorian"
# ).plot.scatter(x="time", zorder=3, alpha=0.5, ax=axes[2])

# plt.tight_layout()
# plt.show()

# %%
# latitudinal_gradient_monthly = lat_grad_eof @ lat_grad_pc_monthly

# # Ensure spatial mean is zero
# tmp = latitudinal_gradient_monthly
# tmp.name = "latitudinal-gradient"
# np.testing.assert_allclose(
#     local.xarray_space.calculate_global_mean_from_lon_mean(tmp).data.to("ppt").m,
#     0.0,
#     atol=1e-10,
# )

# latitudinal_gradient_monthly

# %%
# tmp = global_annual_mean_monthly + latitudinal_gradient_monthly
# if tmp.min() < 0.0:
#     msg = (
#         "When combining the global values and the latitudinal gradient, "
#         f"the minimum value is less than 0.0. {tmp.min()=}"
#     )
#     print(msg)
#     # raise AssertionError(msg)

#     atol_close = 1e-8
#     print(
#         "Trying with a forced update of the latitudinal gradient "
#         f"to be zero where the global-mean is within {atol_close} of zero"
#     )
#     latitudinal_gradient_monthly_candidate = latitudinal_gradient_monthly.copy(deep=True)
#     for yr, yr_da in tqdman.tqdm(global_annual_mean_monthly.groupby("year", squeeze=False)):
#         for month, month_da in yr_da.groupby("month", squeeze=False):
#             if np.isclose(month_da.data.m, 0.0):
#                 # print(yr)
#                 latitudinal_gradient_monthly_candidate.loc[{"year": yr, "month": month}] = 0.0
#                 continue

#             # The latitudinal gradient can't be bigger than the global-mean value,
#             # because this leads to negative values.
#             # This actually points to an issue in the overall workflow,
#             # because what we're actually seeing is a disagreement between the
#             # concentration assumptions (i.e. pre-industrial values)
#             # and the emissions assumptions (which drive the latitudinal gradient).
#             # However, fixing this is an issue for the future.
#             # We squeeze even harder, to avoid seasonality breaking things too.
#             min_grad_val = latitudinal_gradient_monthly_candidate.loc[{"year": yr, "month": month}].min()
#             if np.abs(min_grad_val) > month_da * 0.5:
#                 shrink_ratio = (0.5 * month_da / np.abs(min_grad_val)).squeeze()
#                 new_val = (
#                     shrink_ratio * latitudinal_gradient_monthly_candidate.loc[{"year": yr, "month": month}]
#                 )

#                 msg = (
#                     "TODO: fix consistency issue. "
#                     f"In {yr:04d}-{month:02d}, "
#                     f"the minimum latitudinal gradient value is: {min_grad_val.data}. "
#                     f"The global-mean value is {month_da.data}. "
#                     f"This makes no sense. For now, force overriding to {new_val.data}."
#                 )
#                 print(msg)

#                 latitudinal_gradient_monthly_candidate.loc[{"year": yr, "month": month}] = new_val

#     tmp2 = global_annual_mean_monthly + latitudinal_gradient_monthly_candidate
#     if tmp2.min() < 0.0:
#         msg = "Even after the force update, " f"the minimum value is less than 0.0. {tmp2.min()=}"
#         raise AssertionError(msg)

#     print("Updated the latitudinal gradient")
#     latitudinal_gradient_monthly = latitudinal_gradient_monthly_candidate

#     # Ensure spatial mean is still zero
#     tmp = latitudinal_gradient_monthly
#     tmp.name = "latitudinal-gradient"
#     np.testing.assert_allclose(
#         local.xarray_space.calculate_global_mean_from_lon_mean(tmp).data.to("ppb").m,
#         0.0,
#         atol=1e-10,
#     )

# latitudinal_gradient_monthly

# %%
# local.xarray_time.convert_year_month_to_time(
#     latitudinal_gradient_monthly, calendar="proleptic_gregorian"
# ).plot(hue="lat")

# %%
lat_grad_pc_extended.name = "principal-components"
lat_grad_pc_extended = lat_grad_pc_extended.assign_coords(eof=0).expand_dims(dim={"eof": [0]})
lat_grad_eof.name = "eofs"
lat_grad_eof = lat_grad_eof.assign_coords(eof=0).expand_dims(dim={"eof": [0]})
lat_grad_pieces_out = xr.merge([lat_grad_pc_extended, lat_grad_eof]).sortby("year")
lat_grad_pieces_out

# %% [markdown]
# ### Save

# %%
config_step.global_annual_mean_allyears_file.parent.mkdir(exist_ok=True, parents=True)
global_annual_mean.pint.dequantify().to_netcdf(config_step.global_annual_mean_allyears_file)
global_annual_mean

# %%
config_step.latitudinal_gradient_allyears_pcs_eofs_file.parent.mkdir(exist_ok=True, parents=True)
lat_grad_pieces_out.pint.dequantify().to_netcdf(config_step.latitudinal_gradient_allyears_pcs_eofs_file)
lat_grad_pieces_out

# %%
config_step.latitudinal_gradient_pc0_total_emissions_regression_file.parent.mkdir(exist_ok=True, parents=True)
with open(config_step.latitudinal_gradient_pc0_total_emissions_regression_file, "w") as fh:
    fh.write(local.config.converter_yaml.dumps(latitudinal_gradient_pc0_total_emissions_regression))

latitudinal_gradient_pc0_total_emissions_regression
