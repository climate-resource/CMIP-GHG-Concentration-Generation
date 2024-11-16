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
# # NOAA - process surface flask
#
# Process data from NOAA's surface flask network to add lat-lon information to the monthly data.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.axes
import matplotlib.pyplot as plt
import openscm_units
import pandas as pd
import pint
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

import local.raw_data_processing
from local.config import load_config_from_file
from local.noaa_processing import PROCESSED_DATA_COLUMNS

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %%
step: str = "process_noaa_surface_flask_data"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "sf6"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

config_retrieve = get_config_for_step_id(config=config, step="retrieve_misc_data", step_config_id="only")
config_retrieve_noaa = get_config_for_step_id(
    config=config,
    step="retrieve_and_extract_noaa_data",
    step_config_id=f"{config_step.gas}_surface-flask",
)

# %% [markdown]
# ## Action

# %% editable=true slideshow={"slide_type": ""}
df_events = pd.read_csv(config_retrieve_noaa.interim_files["events_data"])
df_months = pd.read_csv(config_retrieve_noaa.interim_files["monthly_data"])

# %% editable=true slideshow={"slide_type": ""}
df_months["year"].max()

# %% editable=true slideshow={"slide_type": ""}
df_events["year"].max()


# %% editable=true slideshow={"slide_type": ""}
def get_site_code_grouped_dict(indf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Get dictionary grouped by site site_code

    Parameters
    ----------
    indf
        Input :obj:`pd.DataFrame`

    Returns
    -------
        Dictionary where keys are site codes and values are :obj:`pd.DataFrame`
        for that site code
    """
    return {str(site_code): scdf for site_code, scdf in indf.groupby("site_code_filename")}


df_events_sc_g = get_site_code_grouped_dict(df_events)
df_months_sc_g = get_site_code_grouped_dict(df_months)

# %%
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %% [markdown]
# ### Estimate representative location for each monthly value

# %%
stationary_site_movement_tolerance: float = 1.0
"""
Size of movement in site location which will cause a message to be printed

These are basically all fine (or we can't do anything about them) and the
plots are generally more helpful, but just in case we also print a message
for large deviations.
"""


monthly_dfs_with_loc_l = []
for site_code_filename, site_monthly_df in tqdman.tqdm(df_months_sc_g.items(), desc="Monthly sites"):
    no_lat_site_code_length = 3
    if len(site_code_filename) == no_lat_site_code_length:
        site_events_df = df_events_sc_g[site_code_filename]

    else:
        site_code_indicator = site_code_filename[:3]
        lat_indicator = site_code_filename[-3:]

        if site_code_indicator.startswith("poc"):
            # Guessing from files
            band_width = 2.5
        elif site_code_indicator.startswith("scs"):
            # Guessing from files
            band_width = 1.5
        else:
            raise NotImplementedError(site_code_indicator)

        if lat_indicator == "000":
            lat_band = [-band_width, band_width]

        elif lat_indicator.startswith("n"):
            centre = int(lat_indicator[-2:])
            lat_band = [centre - band_width, centre + band_width]

        elif lat_indicator.startswith("s"):
            centre = -int(lat_indicator[-2:])
            lat_band = [centre - band_width, centre + band_width]

        else:
            raise NotImplementedError(lat_indicator)

        site_events_df = df_events_sc_g[site_code_indicator]
        site_events_df = site_events_df[
            (site_events_df["latitude"] >= lat_band[0]) & (site_events_df["latitude"] <= lat_band[1])
        ].copy()

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    if isinstance(axes, matplotlib.axes.Axes):
        raise TypeError(type(axes))

    colours = {"surface": "tab:orange", "shipboard": "tab:blue"}
    countries.plot(color="lightgray", ax=axes[0])

    surf_or_ship_arr = site_events_df["surf_or_ship"].unique()
    assert len(surf_or_ship_arr) == 1
    surf_or_ship = str(surf_or_ship_arr[0])

    site_events_df.plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        ax=axes[0],
        color=colours[surf_or_ship],
        alpha=0.4,
        label="events",
        # s=50,
    )

    pdf = site_monthly_df.copy()
    pdf["year-month"] = pdf["year"] + pdf["month"] / 12
    pdf.plot(
        x="year-month",
        y="value",
        kind="scatter",
        ax=axes[1],
        label="monthly data",
        color="tab:green",
        alpha=0.8,
        zorder=3,
    )

    pdf = site_events_df.copy()
    pdf["year-month"] = pdf["year"] + pdf["month"] / 12
    pdf.plot(
        x="year-month",
        y="value",
        kind="scatter",
        ax=axes[1],
        label="events data",
        color="tab:pink",
        marker="x",
        alpha=0.4,
    )

    time_cols = ["year", "month"]
    spatial_cols = ["latitude", "longitude"]
    loc_calc_cols = [*time_cols, "surf_or_ship"]

    lon_max = site_events_df.groupby(loc_calc_cols)["longitude"].max()
    lon_min = site_events_df.groupby(loc_calc_cols)["longitude"].min()

    # Have longitudes going across the 180 line, need to flip before averaging
    flip_lons = (site_events_df["longitude"] >= 0).any() and (site_events_df["longitude"] < 0).any()

    lat_max = site_events_df.groupby(loc_calc_cols)["latitude"].max()
    lat_min = site_events_df.groupby(loc_calc_cols)["latitude"].min()

    if flip_lons:
        # put lons in range 0-360 rather than -180 to 180 which will cause issues for averaging
        site_events_df.loc[site_events_df["longitude"] < 0, "longitude"] += 360

        # assert we are now away from annoying boundaries
        def check_no_crossover_for_month(
            space_info_for_month: pd.Series[Any],
            danger_lon_max: int = 250,
            danger_lon_min: int = 50,
        ) -> None:
            """
            Check that there is no crossover of awkward longitudes

            This check is assumed to be applied to each individual month bin.
            """
            assert not (
                (space_info_for_month["longitude"] >= danger_lon_max).any()
                and (space_info_for_month["longitude"] < danger_lon_min).any()
            ), space_info_for_month

        site_events_df.groupby(loc_calc_cols)[spatial_cols].apply(check_no_crossover_for_month)  # type: ignore

    locs_means = site_events_df.groupby(loc_calc_cols)[spatial_cols].mean()
    locs_stds = site_events_df.groupby(loc_calc_cols)[spatial_cols].std()

    if locs_stds.max().max() > stationary_site_movement_tolerance:
        print(f"Large move in location of station {site_code_filename}:\n{locs_stds.max()}")

    if flip_lons:
        # undo the operation (longitudes that are greater than 180 are returned back to being negative)
        locs_means["longitude"][locs_means["longitude"] > 180] -= 360  # noqa: PLR2004

    lat_big_deviation_threshold = 10  # above this, probably a sign of a processing problem
    if (lat_max - lat_min >= lat_big_deviation_threshold).any():
        raise NotImplementedError()

    locs_means.plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        ax=axes[0],
        color="tab:pink",
        alpha=0.4,
        label="mean locations",
        marker="x",
        s=50,
    )

    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-90, 90])

    axes[1].legend()

    fig.suptitle(site_code_filename)
    fig.tight_layout()
    plt.show()

    site_monthly_with_loc = (
        site_monthly_df
        # All monthly data labelled as surface, which isn't true hence
        # extract from events df instead
        .drop("surf_or_ship", axis="columns")
        .set_index(time_cols)
        .join(locs_means)
    )
    assert not site_monthly_with_loc["value"].isna().any()

    # interpolate lat and lon columns (for case where monthly data is
    # based on interpolation, see 7.7 here
    # https://gml.noaa.gov/aftp/data/greenhouse_gases/co2/flask/surface/README_co2_surface-flask_ccgg.html
    # which says
    # > Monthly means are produced for each site by first averaging all
    # > valid measurement results in the event file with a unique sample
    # > date and time.  Values are then extracted at weekly intervals from
    # > a smooth curve (Thoning et al., 1989) fitted to the averaged data
    # > and these weekly values are averaged for each month to give the
    # > monthly means recorded in the files.  Flagged data are excluded from the
    # > curve fitting process.  Some sites are excluded from the monthly
    # > mean directory because sparse data or a short record does not allow a
    # > reasonable curve fit.  Also, if there are 3 or more consecutive months
    # > without data, monthly means are not calculated for these months.
    site_monthly_with_loc[spatial_cols] = site_monthly_with_loc[spatial_cols].interpolate()
    assert not site_monthly_with_loc.isna().any().any()

    monthly_dfs_with_loc_l.append(site_monthly_with_loc)

monthly_dfs_with_loc = pd.concat(monthly_dfs_with_loc_l).sort_index().reset_index()[PROCESSED_DATA_COLUMNS]

# %%
# Handy check to see if all months have at least some data
pd.MultiIndex.from_product([range(1967, 2022 + 1), range(1, 13)]).difference(  # type: ignore
    monthly_dfs_with_loc.set_index(["year", "month"]).index.drop_duplicates()
)

# %% [markdown]
# ### Examine stations for which there is only events data

# %%
only_events_stations = set(df_events_sc_g.keys()) - set(df_months_sc_g.keys())

for station in tqdman.tqdm(sorted(only_events_stations), desc="Stations with only events data"):
    header = f"Examining {station}"
    print("=" * len(header))
    print(header)
    monthly_same_start = [k for k in df_months_sc_g.keys() if k.startswith(station)]
    if monthly_same_start:
        print(f"Assuming that {station} is captured in monthly summaries: {monthly_same_start}")
        print("=" * len(header))
        print()
        continue

    site_events_df = df_events_sc_g[station]

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    if isinstance(axes, matplotlib.axes.Axes):
        raise TypeError(type(axes))

    colours = {"surface": "tab:orange", "shipboard": "tab:blue"}
    countries.plot(color="lightgray", ax=axes[0])

    surf_or_ship_arr = site_events_df["surf_or_ship"].unique()
    assert len(surf_or_ship_arr) == 1
    surf_or_ship = str(surf_or_ship_arr[0])

    site_events_df.plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        ax=axes[0],
        color=colours[surf_or_ship],
        alpha=0.4,
        label="events",
        # s=50,
    )

    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-90, 90])

    pdf = site_events_df.copy()
    pdf["year-month"] = pdf["year"] + pdf["month"] / 12
    pdf.plot(
        x="year-month",
        y="value",
        kind="scatter",
        ax=axes[1],
        label="events data",
        color="tab:pink",
        marker="x",
        alpha=0.4,
    )

    axes[1].legend()

    plt.suptitle(station)
    plt.tight_layout()
    plt.show()

    print("=" * len(header))
    print()

# %%
only_events_stations

# %% [markdown]
# Conclusion: I can't really work out why these sites don't have monthly data (they pass quality control and seem to have enough data to do a fit...).
#
# TODO: include this question when I reach out to NOAA people.

# %% [markdown]
# ### Prepare and check output

# %%
out = monthly_dfs_with_loc.copy()
out["network"] = "NOAA"
out["station"] = out["site_code"].str.lower()
out["measurement_method"] = out["source"]

out

# %%
local.raw_data_processing.check_processed_data_columns_for_spatial_binning(out)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Save out result

# %% editable=true slideshow={"slide_type": ""}
assert set(monthly_dfs_with_loc["gas"]) == {config_step.gas}
out.to_csv(config_step.processed_monthly_data_with_loc_file, index=False)
out
