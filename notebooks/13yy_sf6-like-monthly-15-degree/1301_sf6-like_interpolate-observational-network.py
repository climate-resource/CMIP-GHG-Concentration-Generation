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
# # SF$_6$-like - interpolate observational network
#
# Interpolate the observational network data onto our grid.

# %% [markdown]
# ## Imports

# %%
import cftime
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

import local.binned_data_interpolation
import local.binning
import local.raw_data_processing
from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "calculate_sf6_like_monthly_fifteen_degree_pieces"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "chcl3"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Load data

# %%
bin_averages = pd.read_csv(config_step.processed_bin_averages_file)
bin_averages

# %%
all_data_with_bins = pd.read_csv(config_step.processed_all_data_with_bins_file)
all_data_with_bins

# %% [markdown]
# ## Interpolate

# %%
# TODO: interpolate in time before in space

# %%
MIN_POINTS_FOR_SPATIAL_INTERPOLATION = 4

# %%
times_l = []
interpolated_dat_l = []
year_month_plot = (
    (1983, 1),
    (1984, 1),
    (1984, 3),
    (2000, 4),
    (2018, 1),
    (2018, 2),
    (2022, 12),
    (2023, 1),
    (2023, 12),
)

for (year, month), ymdf in tqdman.tqdm(bin_averages.groupby(["year", "month"])):
    if ymdf.shape[0] < MIN_POINTS_FOR_SPATIAL_INTERPOLATION:
        msg = f"Not enough data ({ymdf.shape[0]} data points) for {year=}, {month=}, not performing spatial interpolation"
        print(msg)
        continue

    interpolated_ym = local.binned_data_interpolation.interpolate(ymdf)
    show_plot = False
    include_data = True
    if np.isnan(interpolated_ym).any():
        if config_step.allow_long_poleward_extension:
            all_data_with_bins_ym = all_data_with_bins[
                (all_data_with_bins["year"] == year)
                & (all_data_with_bins["month"] == month)
            ]

            # TODO: change this so that it just sets the value at the pole,
            # then tries to interpolate again.
            furthest_south_idx = np.where(~np.isnan(interpolated_ym.T).all(axis=1))[
                0
            ].min()
            if furthest_south_idx > 0:
                south_input = all_data_with_bins_ym[
                    all_data_with_bins_ym["latitude"]
                    == all_data_with_bins_ym["latitude"].min()
                ]
                south_pole_value = south_input["value"].mean()
                interpolated_ym[:, :furthest_south_idx] = south_pole_value

                msg = f"Fixed South Pole with poleward extension of {south_pole_value}"
                print(msg)

            furthest_north_idx = np.where(~np.isnan(interpolated_ym.T).all(axis=1))[
                0
            ].max()
            if furthest_north_idx < interpolated_ym.T.shape[0] - 1:
                north_input = all_data_with_bins_ym[
                    all_data_with_bins_ym["latitude"]
                    == all_data_with_bins_ym["latitude"].max()
                ]
                north_pole_value = north_input["value"].mean()
                interpolated_ym[:, furthest_north_idx:] = north_pole_value

                msg = f"Fixed North Pole with poleward extension of {north_pole_value}"
                print(msg)

            if np.isnan(interpolated_ym).any():
                msg = "Should be no more nan now"
                raise AssertionError(msg)

        elif config_step.allow_poleward_extension:
            if np.isnan(interpolated_ym.T[1:-1, :]).any():
                msg = f"Nan data after interpolation for {year=}, {month=}, not including spatial interpolation in output"
                print(msg)
                include_data = False

            else:
                all_data_with_bins_ym = all_data_with_bins[
                    (all_data_with_bins["year"] == year)
                    & (all_data_with_bins["month"] == month)
                ]

                north_pole_has_nan = np.isnan(interpolated_ym.T[-1, :]).any()
                south_pole_has_nan = np.isnan(interpolated_ym.T[0, :]).any()

                if north_pole_has_nan:
                    north_input = all_data_with_bins_ym[
                        all_data_with_bins_ym["latitude"]
                        == all_data_with_bins_ym["latitude"].max()
                    ]
                    north_pole_value = north_input["value"].mean()
                    interpolated_ym[:, -1] = north_pole_value

                    msg = f"Fixed North Pole with poleward extension of {north_pole_value}"
                    print(msg)

                if south_pole_has_nan:
                    south_input = all_data_with_bins_ym[
                        all_data_with_bins_ym["latitude"]
                        == all_data_with_bins_ym["latitude"].min()
                    ]
                    south_pole_value = south_input["value"].mean()
                    interpolated_ym[:, 0] = south_pole_value

                    msg = f"Fixed South Pole with poleward extension of {south_pole_value}"
                    print(msg)

                if np.isnan(interpolated_ym).any():
                    msg = "Should be no more nan now"
                    raise AssertionError(msg)

        else:
            msg = (
                f"Nan data after interpolation for {year=}, {month=} and no poleward extension allowed, "
                "not including spatial interpolation in output"
            )
            print(msg)
            include_data = False

        show_plot = True

    if include_data:
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
if not interpolated_dat_l:
    msg = "Not enough observational data to create any fields"
    raise AssertionError(msg)

# %%
out = local.binned_data_interpolation.to_xarray_dataarray(
    name=config_step.gas,
    bin_averages_df=bin_averages,
    data=interpolated_dat_l,
    times=times_l,
)
out

# %% [markdown]
# ### Save

# %%
config_step.observational_network_interpolated_file.parent.mkdir(
    exist_ok=True, parents=True
)
out.to_netcdf(config_step.observational_network_interpolated_file)
out
