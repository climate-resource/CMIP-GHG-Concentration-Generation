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
# # CO$_2$ - interpolate observational network
#
# Interpolate the observational network data onto our grid.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import cftime
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import tqdm.autonotebook as tqdman
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.diagnostics
import local.raw_data_processing
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

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

# %%
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)
config_process_scaled_sat_data = get_config_for_step_id(
    config=config,
    step="process_scaled_sat_data",
    step_config_id=config_step.gas,
)

# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
# `bin_averages_sat` is empty if satellite data isn't switched on for this gas
# (see `1200a_co2_bin_satellite_data`).
bin_averages_ground = pd.read_csv(config_step.processed_bin_averages_file)
bin_averages_sat = pd.read_csv(config_process_scaled_sat_data.interim_data_path)

bin_averages = pd.concat([bin_averages_ground, bin_averages_sat])

# %% [markdown]
# ## Interpolate

# %%
MIN_POINTS_FOR_SPATIAL_INTERPOLATION = 4

# %%
# Diagnostics: the years satellite data can possibly affect (same reference
# period whether or not satellite data is actually switched on for this run -
# see `local.diagnostics.get_satellite_data_year_range` - so it can be used
# to scope stats comparably across a `nosat` run and a `SAT_{fit}` run).
satellite_period_start_year, satellite_period_end_year = local.diagnostics.get_satellite_data_year_range(
    config_process_scaled_sat_data.scaled_data_path
)

# %%
times_l = []
interpolated_dat_l = []
# Diagnostics: how much does the input coverage change, and how many months
# does that coverage change save from being dropped, when satellite data is added?
# Tracked both over the full record and restricted to the satellite-covered
# years, because satellite data structurally cannot change anything outside
# that window - averaging over the full record dilutes its effect into
# looking much smaller than it is.
n_months_total = 0
n_months_dropped_insufficient_points = 0
n_months_dropped_nan = 0
n_points_per_month_l = []
n_months_total_satellite_period = 0
n_months_dropped_insufficient_points_satellite_period = 0
n_months_dropped_nan_satellite_period = 0
n_points_per_month_satellite_period_l = []

# Diagnostics: spatial coverage. `points_per_year_month` above counts *rows*,
# which double-counts a bin that both a ground station and a satellite grid
# cell happen to land in - it measures how much data gets averaged together,
# not how much of the 15x60 degree grid is actually observed vs left for
# `local.binned_data_interpolation.interpolate` to guess via `griddata`. This
# tracks the latter: how many of the grid's spatial bins have at least one
# real (ground or satellite) value each month, both as a per-month count and
# as a running per-bin "fraction of months populated" map.
lat_bin_index = {v: i for i, v in enumerate(local.binning.LAT_BIN_CENTRES)}
lon_bin_index = {v: i for i, v in enumerate(local.binning.LON_BIN_CENTRES)}
n_lat_bins = len(local.binning.LAT_BIN_CENTRES)
n_lon_bins = len(local.binning.LON_BIN_CENTRES)
n_spatial_bins_total = n_lat_bins * n_lon_bins

n_bins_populated_l = []
n_bins_populated_satellite_period_l = []
bin_populated_count_full_record = np.zeros((n_lat_bins, n_lon_bins))
bin_populated_count_satellite_period = np.zeros((n_lat_bins, n_lon_bins))

year_month_plot = (
    (1983, 1),
    (1984, 1),
    (1984, 3),
    (2000, 4),
    (2022, 12),
    (2023, 1),
    (2023, 12),
)

for (year, month), ymdf in tqdman.tqdm(bin_averages.groupby(["year", "month"])):
    in_satellite_period = satellite_period_start_year <= year <= satellite_period_end_year

    n_months_total += 1
    n_points_per_month_l.append(ymdf.shape[0])
    if in_satellite_period:
        n_months_total_satellite_period += 1
        n_points_per_month_satellite_period_l.append(ymdf.shape[0])

    populated_bins = ymdf[["lat_bin", "lon_bin"]].drop_duplicates()
    n_bins_populated_l.append(len(populated_bins))
    if in_satellite_period:
        n_bins_populated_satellite_period_l.append(len(populated_bins))

    for _, populated_bin in populated_bins.iterrows():
        i = lat_bin_index[populated_bin["lat_bin"]]
        j = lon_bin_index[populated_bin["lon_bin"]]
        bin_populated_count_full_record[i, j] += 1
        if in_satellite_period:
            bin_populated_count_satellite_period[i, j] += 1

    if ymdf.shape[0] < MIN_POINTS_FOR_SPATIAL_INTERPOLATION:
        msg = f"Not enough data ({ymdf.shape[0]} data points) for {year=}, {month=}, not performing spatial interpolation"
        print(msg)
        n_months_dropped_insufficient_points += 1
        if in_satellite_period:
            n_months_dropped_insufficient_points_satellite_period += 1
        continue

    interpolated_ym = local.binned_data_interpolation.interpolate(ymdf)
    show_plot = False
    if np.isnan(interpolated_ym).any():
        msg = f"Nan data after interpolation for {year=}, {month=}, not including spatial interpolation in output"
        print(msg)
        show_plot = True
        n_months_dropped_nan += 1
        if in_satellite_period:
            n_months_dropped_nan_satellite_period += 1

    else:
        interpolated_dat_l.append(interpolated_ym)
        times_l.append(cftime.datetime(year, month, 15))

    if (year, month) in year_month_plot or show_plot:
        # This will break if we ever change our internal gridding logic, ok for now.
        lon_grid, lat_grid = np.meshgrid(
            local.binning.LON_BIN_CENTRES,
            local.binning.LAT_BIN_CENTRES,
        )

        plt.pcolormesh(lon_grid, lat_grid, interpolated_ym.T, shading="auto")
        plt.plot(ymdf["lon_bin"], ymdf["lat_bin"], "ok", label="input point")
        plt.legend()
        plt.colorbar()
        # plt.axis("equal")
        plt.ylim([-90, 90])
        plt.title(f"{year} {month}")
        plt.show()

# %%
out = local.binned_data_interpolation.to_xarray_dataarray(
    name=config_step.gas,
    bin_averages_df=bin_averages,
    data=interpolated_dat_l,
    times=times_l,
)
out

# %% [markdown]
# ### Diagnostics
#
# How much did satellite data change the *inputs* to spatial interpolation,
# independent of anything downstream (EOFs, extensions, etc.)?

# %%
diagnostics_suffix = local.diagnostics.get_satellite_suffix(
    config_step.include_satellite_data, config_step.satellite_fit
)

n_points_per_month = np.array(n_points_per_month_l)
n_points_per_month_satellite_period = np.array(n_points_per_month_satellite_period_l)
n_bins_populated = np.array(n_bins_populated_l)
n_bins_populated_satellite_period = np.array(n_bins_populated_satellite_period_l)
local.diagnostics.save_yaml_diagnostics(
    config_step.diagnostics_dir
    / (
        local.diagnostics.diagnostics_file_stem(config_step.gas, "interpolation", diagnostics_suffix)
        + ".yaml"
    ),
    include_satellite_data=config_step.include_satellite_data,
    satellite_fit=config_step.satellite_fit,
    n_ground_network_bin_rows=int(bin_averages_ground.shape[0]),
    n_satellite_bin_rows=int(bin_averages_sat.shape[0]),
    n_year_months_total=int(n_months_total),
    n_year_months_dropped_insufficient_points=int(n_months_dropped_insufficient_points),
    n_year_months_dropped_nan_after_interpolation=int(n_months_dropped_nan),
    n_year_months_kept=int(len(times_l)),
    points_per_year_month_mean=float(n_points_per_month.mean()),
    points_per_year_month_min=int(n_points_per_month.min()),
    points_per_year_month_max=int(n_points_per_month.max()),
    # Scoped to the years satellite data can possibly affect - see the note
    # above the counters in the "Interpolate" section for why this matters.
    satellite_period_start_year=int(satellite_period_start_year),
    satellite_period_end_year=int(satellite_period_end_year),
    n_year_months_total_satellite_period=int(n_months_total_satellite_period),
    n_year_months_dropped_insufficient_points_satellite_period=int(
        n_months_dropped_insufficient_points_satellite_period
    ),
    n_year_months_dropped_nan_satellite_period=int(n_months_dropped_nan_satellite_period),
    points_per_year_month_mean_satellite_period=float(n_points_per_month_satellite_period.mean()),
    points_per_year_month_min_satellite_period=int(n_points_per_month_satellite_period.min()),
    points_per_year_month_max_satellite_period=int(n_points_per_month_satellite_period.max()),
    # Spatial coverage: how many of the `n_spatial_bins_total` grid cells have
    # at least one real (ground or satellite) value each month, i.e. are not
    # left entirely to `griddata` to guess.
    n_spatial_bins_total=int(n_spatial_bins_total),
    bins_populated_per_year_month_mean=float(n_bins_populated.mean()),
    bins_populated_per_year_month_min=int(n_bins_populated.min()),
    bins_populated_per_year_month_max=int(n_bins_populated.max()),
    bins_populated_per_year_month_mean_satellite_period=float(n_bins_populated_satellite_period.mean()),
    bins_populated_per_year_month_min_satellite_period=int(n_bins_populated_satellite_period.min()),
    bins_populated_per_year_month_max_satellite_period=int(n_bins_populated_satellite_period.max()),
)

# %%
# Save the per-bin "fraction of months populated" maps, so you can see *where*
# (not just how much) satellite data fills in gaps left by the ground network.
spatial_coverage_ds = xr.Dataset(
    {
        "bin_fraction_months_populated_full_record": (
            ("lat", "lon"),
            bin_populated_count_full_record / n_months_total,
        ),
        "bin_fraction_months_populated_satellite_period": (
            ("lat", "lon"),
            bin_populated_count_satellite_period / n_months_total_satellite_period,
        ),
    },
    coords={"lat": local.binning.LAT_BIN_CENTRES, "lon": local.binning.LON_BIN_CENTRES},
)
local.diagnostics.save_nc_diagnostics(
    config_step.diagnostics_dir
    / (local.diagnostics.diagnostics_file_stem(config_step.gas, "interpolation", diagnostics_suffix) + ".nc"),
    spatial_coverage_ds,
)
spatial_coverage_ds

# %% [markdown]
# ### Save

# %%
config_step.observational_network_interpolated_file.parent.mkdir(exist_ok=True, parents=True)
out.to_netcdf(config_step.observational_network_interpolated_file)
out

# %%
