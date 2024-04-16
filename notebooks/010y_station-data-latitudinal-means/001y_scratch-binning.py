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
# # Scratch - binning
#
# Toy notebook for working out how the binning is going to work.

# %% [markdown]
# ## Imports

# %%
import cftime
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.raw_data_processing
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_station_data_latitudinal_means"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
# TODO: put this in config_step
gas = "co2"
# config_step = get_config_for_step_id(
#     config=config, step=step, step_config_id=step_config_id
# )

config_noaa = get_config_for_step_id(
    config=config, step="retrieve_misc_data", step_config_id="only"
)

# Introduce a notebook which writes out the 'database' for each gas.
config_process_noaa_in_situ_data = get_config_for_step_id(
    config=config, step="process_noaa_in_situ_data", step_config_id=gas
)
config_process_noaa_surface_flask_data = get_config_for_step_id(
    config=config, step="process_noaa_surface_flask_data", step_config_id=gas
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %% [markdown]
# - stations get equal weight
# - flask/in situ networks are treated as separate stations
# - station average for the month
# - then average over all stations
#   - this order is best as you have a better chance of avoiding giving different times more weight by accident
#   - properly equally weighting all times in the month would be very hard, because you'd need to interpolate to a super fine grid first (beyond us, one for future research)


# %%
def check_and_read(f):
    out = pd.read_csv(f)
    local.raw_data_processing.check_processed_data_columns_for_latitudinal_mean(out)

    return out


# %%
all_data_l = []
for f in [
    config_process_noaa_surface_flask_data.processed_monthly_data_with_loc_file,
    config_process_noaa_in_situ_data.processed_monthly_data_with_loc_file,
]:
    try:
        all_data_l.append(check_and_read(f))
    except:
        print(f"Error reading {f}")
        raise

all_data = pd.concat(all_data_l)
all_data


# %%
def get_group(value, group_bounds, group_centres):
    if group_bounds.size != group_centres.size + 1:
        msg = f"This won't work. {group_bounds.size=}, {group_centres.size=}"
        raise ValueError(msg)

    max_bound_lt = np.max(np.where(value >= group_bounds)[0])
    if max_bound_lt == group_centres.size:
        # Value equal to last bound
        return group_centres[-1]

    return group_centres[max_bound_lt]


lon_groups = np.arange(-180, 181, 60)
lon_centres = (lon_groups[:-1] + lon_groups[1:]) / 2
lon_centres

lat_groups = np.arange(-90, 91, 15)
lat_centres = (lat_groups[:-1] + lat_groups[1:]) / 2
lat_centres

all_data["lon_bin"] = all_data["longitude"].apply(
    get_group,
    group_bounds=lon_groups,
    group_centres=lon_centres,
)
all_data["lat_bin"] = all_data["latitude"].apply(
    get_group,
    group_bounds=lat_groups,
    group_centres=lat_centres,
)

all_data

# %%
variable_info_cols = ["gas", "unit"]
time_cols = [
    "year",
    "month",
]
lat_lon_bin_cols = ["lat_bin", "lon_bin"]

binning_columns = [*variable_info_cols, *time_cols, *lat_lon_bin_cols]

equally_weighted_group_cols = [
    "network",
    "station",
    "measurement_method",
    *binning_columns,
]

value_column = "value"

# %%
units = all_data["unit"].unique()
if len(units) > 1:
    msg = f"Unit conversion required, have {units=}"
    raise AssertionError(msg)

unit = units[0]


# %%
def verbose_groupby_mean(inseries, groupby):
    out = inseries.groupby(groupby).mean()

    in_index = inseries.index.names
    out_index = out.index.names

    mean_over_cols = set(in_index) - set(out_index)
    print(f"Took mean over {sorted(mean_over_cols)}")

    return out


# %%
equal_weight_monthly_averages = verbose_groupby_mean(
    (
        all_data.dropna().set_index(  # explicitly drop nan before calculating, data is long so this is fine
            [c for c in all_data.columns if c != value_column]
        )[
            value_column
        ]
    ),
    equally_weighted_group_cols,
)
if (
    equal_weight_monthly_averages.index.to_frame()[
        [*equally_weighted_group_cols, *time_cols, *lat_lon_bin_cols]
    ]
    .duplicated()
    .any()
):
    msg = "Network-station-measurement combos should only receive a weight of one at each time in each spatial bin after this point..."
    raise AssertionError(msg)

equal_weight_monthly_averages

# %%
print("Collating data from:")

for (network, measurement_method), _ in equal_weight_monthly_averages.groupby(
    ["network", "measurement_method"]
):
    print(f"- {network} {measurement_method}")

# %%
bin_averages = verbose_groupby_mean(equal_weight_monthly_averages, binning_columns)
bin_averages

# %% [markdown]
# TODO:
#
# - clean up below here, basic idea:
#   - get the data you have
#   - duplicate the grids so that you are interpolating on a globe (i.e. can go round both directions, rather than hitting boudnaries)
#   - interpolate
#   - check for empty values still (I assume we don't have a great solution for these periods where the observational networks are relatively sparse...)
#   - if all good, add to our data array

# %%
min_n_points_for_spatial_interpolation = 5
included = 0
times = []
interpolated_dat_l = []

lon_grid, lat_grid = np.meshgrid(lon_centres, lat_centres)

for (year, month), ymdf in bin_averages.groupby(["year", "month"]):
    if ymdf.shape[0] < min_n_points_for_spatial_interpolation:
        msg = f"Not enough data ({ymdf.shape[0]} data points) for {year=}, {month=}, not performing spatial interpolation"
        print(msg)
        continue

    if year < 1984:
        continue

    if year > 1990:
        break

    times.append(cftime.datetime(year, month, 15))

    from scipy.interpolate import griddata

    grid = ["lon_bin", "lat_bin"]
    points = ymdf.reset_index()[grid].values

    # Malte's trick, duplicate the grids so we can go 'round the world' with interpolation
    lat_grid_interp = np.hstack(
        [
            lat_grid,
            lat_grid,
            lat_grid,
        ]
    )
    lon_grid_interp = np.hstack(
        [
            lon_grid - 360,
            lon_grid,
            lon_grid + 360,
        ]
    )

    points_shift_back = points.copy()
    points_shift_back[:, 0] -= 360
    points_shift_forward = points.copy()
    points_shift_forward[:, 0] += 360
    points_interp = np.vstack(
        [
            points_shift_back,
            points,
            points_shift_forward,
        ]
    )
    values_interp = np.hstack(
        [
            ymdf.values,
            ymdf.values,
            ymdf.values,
        ]
    )

    res_linear_interp = griddata(
        points=points_interp,
        values=values_interp,
        xi=(lon_grid_interp, lat_grid_interp),
        method="linear",
        # fill_value=12.0
    )
    res_linear = res_linear_interp[:, lon_grid.shape[1] : lon_grid.shape[1] * 2]
    interpolated_dat_l.append(res_linear)

    if included < 2:
        included += 1
        plt.pcolormesh(
            lon_grid_interp, lat_grid_interp, res_linear_interp, shading="auto"
        )
        plt.plot(points_interp[:, 0], points_interp[:, 1], "ok", label="input point")
        plt.legend()
        plt.colorbar()
        # plt.axis("equal")
        plt.ylim([-90, 90])
        plt.show()

        plt.pcolormesh(lon_grid, lat_grid, res_linear, shading="auto")
        plt.plot(points[:, 0], points[:, 1], "ok", label="input point")
        plt.legend()
        plt.colorbar()
        # plt.axis("equal")
        plt.ylim([-90, 90])
        plt.show()

# %%
units = bin_averages.index.get_level_values("unit").unique()
if len(units) > 1:
    msg = f"Need unit conversion, {units=}"
    raise AssertionError(msg)

unit = units[0]

# %%
da = xr.DataArray(
    data=np.array(interpolated_dat_l),
    dims=["time", "lat", "lon"],
    coords=dict(
        time=times,
        lat=lat_centres,
        lon=lon_centres,
    ),
    attrs=dict(
        description="TBD",
        units=unit,
    ),
)
assert not da.isnull().any()
display(da)
assert False, "Convert time to year-month"
assert (
    False
), "Use calculate_weighted_area_mean_latitude_only from carpet concentrations"
assert False, (
    "Check with Nicolai, the mean-preserving interpolation must preserve the mean. "
    "Otherwise it's just a median regression of an arbitrary degree polynomial to the data "
    "(i.e. it's basically, get as good a fit to this data as possible, under the constraint of preserving the mean)?"
)
assert False, (
    "Calculate residuals by subtracting smoothed global-mean (year, month, lat) timeseries from binned (year, month, lat) fields. "
    "Calculate annual-average residuals by calculating mean over the month dimension of these (year, month, lat) residuals, which results in (year, lat) fields. "
    "Do SVD on this (n_years, n_lat) matrix of (annual-average) residuals from the global-mean. "
    "That gets you the latitudinal gradient (which you can now represent using one or two EOFs). "
    "Check that you haven't mucked it up by making sure that, if you take all EOFs and scores, "
    "you get back the original (n_years, n_lat) matrix of (annual-average) residuals from the global-mean."
)


# %%
# TODO: put this somewhere sensible
def split_time_to_year_month(
    inp: xr.Dataset,
    time_axis: str = "time",
) -> xr.Dataset:
    """
    Convert the time dimension to year and month without stacking

    This means there is still a single time dimension in the output, but there
    is now also accompanying year and month information

    Parameters
    ----------
    inp
        Data to convert

    Returns
    -------
        Data with year and month information for the time axis

    Raises
    ------
    NonUniqueYearMonths
        The years and months are not unique
    """
    out = inp.assign_coords(
        {
            "month": inp[time_axis].dt.month,
            "year": inp[time_axis].dt.year,
        }
    ).set_index({time_axis: ("year", "month")})

    # Could be updated when https://github.com/pydata/xarray/issues/7104 is
    # closed
    unique_vals, counts = np.unique(out[time_axis].values, return_counts=True)

    if (counts > 1).any():
        raise NonUniqueYearMonths(unique_vals, counts)

    return out


def convert_time_to_year_month(
    inp: xr.Dataset,
    time_axis: str = "time",
) -> xr.Dataset:
    """
    Convert the time dimension to year and month co-ordinates

    Parameters
    ----------
    inp
        Data to convert

    Returns
    -------
        Data with year and month co-ordinates
    """
    return split_time_to_year_month(
        inp=inp,
        time_axis=time_axis,
    ).unstack(time_axis)


# %%
# Seasonality something like this
tmp = convert_time_to_year_month(da.mean(dim="lon"))
annual_means = tmp.mean("month")
seasonality = (tmp - annual_means).mean("year")
seasonality

# %%
seasonality.sum("month")
