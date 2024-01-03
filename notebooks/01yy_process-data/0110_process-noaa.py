# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NOAA
#
# Process data from NOAA.
#
# To-do's:
#
# - parameterise notebook so we can do same for CH4, N2O and SF6 observations

# %% [markdown]
# ## Imports

# %%
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import tqdm.autonotebook as tqdman

from local.config import load_config_from_file
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "process"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %%
config_retrieve = get_config_for_step_id(
    config=config, step="retrieve", step_config_id="only"
)

# %% [markdown]
# ## Action

# %%
available_zip_files = list(config_retrieve.noaa_network.raw_dir.rglob("*.zip"))
available_zip_files

# %%
import re
import zipfile
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pandas as pd


def get_metadata_from_file_default(filename: str) -> dict[str, str]:
    event_file_regex = r"(?P<gas>[a-z0-9]*)_(?P<site_code_filename>[a-z]{3}[a-z0-9]*)_(?P<surf_or_ship>[a-z]*)-flask_1_ccgg_(?P<reporting_id>[a-z]*).txt"

    re_match = re.search(event_file_regex, filename)

    return {
        k: re_match.group(k)
        for k in ["gas", "site_code_filename", "surf_or_ship", "reporting_id"]
    }


def filter_df_events_default(inp: pd.DataFrame) -> pd.DataFrame:
    """
    Notes
    -----
    If you use the monthly data, this filtering largely doesn't matter
    anyway because NOAA have already done it. It is also a much better
    idea to use NOAA's monthly values because their curve fitting approach
    is non-trivial to say the least (see https://gml.noaa.gov/ccgg/mbl/crvfit/crvfit.html).
    For completeness, notes on filtering below.

    The filtering keeps only those samples that have a quality flag
    that starts with two periods. This is derived from Section 7.5
    here: https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/README_co2_surface-flask_ccgg.html

    > If the first character is not a period, the sample result should be
    > rejected for scientific use due to sample collection and/or measurement
    > issue. A second column character other than a period indicates a sample
    > that is likely valid but does not meet selection for representativeness
    > such as midday sampling or background air sampling. A third column flag
    > other than a period indicates abnormal circumstances that are not thought
    > to affect the data quality.

    """
    out = inp[inp["qcflag"].str.startswith("..")]

    return out


def read_event_data(
    open_zip: zipfile.ZipFile,
    zip_info: zipfile.ZipInfo,
    get_metadata_from_file: Callable[[str], str] | None = None,
    filter_df_events: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if get_metadata_from_file is None:
        get_metadata_from_file = get_metadata_from_file_default

    if filter_df_events is None:
        filter_df_events = filter_df_events_default

    filename_metadata = get_metadata_from_file(zip_info.filename)

    file_content = open_zip.read(zip_info).decode("utf-8")
    df_events = pd.read_csv(
        StringIO(file_content),
        comment="#",
        date_format={},
        parse_dates=["datetime", "analysis_datetime"],
        delim_whitespace=True,
    )

    for k, v in filename_metadata.items():
        df_events[k] = v

    df_events_filtered = filter_df_events(df_events)

    return df_events_filtered


def read_month_data(
    open_zip: zipfile.ZipFile,
    zip_info: zipfile.ZipInfo,
    get_metadata_from_file: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    if get_metadata_from_file is None:
        get_metadata_from_file = get_metadata_from_file_default

    filename_metadata = get_metadata_from_file(zip_info.filename)

    file_content = open_zip.read(zip_info).decode("utf-8")

    # Get headers
    for line in file_content.splitlines():
        if line.startswith("# data_fields:"):
            data_fields = line.split("# data_fields: ")[1].split(" ")
            break

    df_monthly = pd.read_csv(
        StringIO(file_content),
        header=None,
        names=data_fields,
        comment="#",
        delim_whitespace=True,
        converters={"site": str},  # keep '000' as string
    )

    for k, v in filename_metadata.items():
        df_monthly[k] = v

    df_monthly = df_monthly.rename({"site": "site_code"}, axis="columns")
    return df_monthly


def event_file_identifier_default(filename: str) -> bool:
    return "event" in filename


def month_file_identifier_default(filename: str) -> bool:
    return "month" in filename


def read_noaa_zip(
    noaa_zip_file: Path,
    event_file_identifier: Callable[[str], bool] | None = None,
    month_file_identifier: Callable[[str], bool] | None = None,
):
    if event_file_identifier is None:
        event_file_identifier = event_file_identifier_default

    if month_file_identifier is None:
        month_file_identifier = month_file_identifier_default

    with ZipFile(noaa_zip_file) as zip:
        event_files = [
            item for item in zip.filelist if event_file_identifier(item.filename)
        ]
        df_events = pd.concat(
            [
                read_event_data(zip, event_file_item)
                for event_file_item in tqdman.tqdm(event_files)
            ]
        )

        month_files = [
            item for item in zip.filelist if month_file_identifier(item.filename)
        ]
        df_months = pd.concat(
            [
                read_month_data(zip, month_files_item)
                for month_files_item in tqdman.tqdm(month_files)
            ]
        )

    # Make sure we haven't ended up with any obviously bogus data
    assert not (df_events["value"] <= -999.0).any()
    assert not (df_events["longitude"] <= -999.0).any()
    assert not (df_months["value"] <= -999.0).any()

    return df_events, df_months


# %%
assert len(available_zip_files) == 1
df_events, df_months = read_noaa_zip(available_zip_files[0])
display(df_events)
display(df_months)


# %%
def get_site_code_grouped_dict(indf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {site_code: scdf for site_code, scdf in indf.groupby("site_code_filename")}


df_events_sc_g = get_site_code_grouped_dict(df_events)
df_months_sc_g = get_site_code_grouped_dict(df_months)

# %%
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir
    / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
# Work out best-estimate location of each monthly value
stationary_site_movement_tolerance = 1

monthly_dfs_with_loc = []
for site_code_filename, site_monthly_df in tqdman.tqdm(
    df_months_sc_g.items(), desc="Monthly sites"
):
    # site_events_df =
    if len(site_code_filename) == 3:
        site_events_df = df_events_sc_g[site_code_filename]

    else:
        site_code_indicator = site_code_filename[:3]
        if site_code_indicator.startswith("poc"):
            # Guessing from files
            band_width = 2.5
        elif site_code_indicator.startswith("scs"):
            # Guessing from files
            band_width = 1.5
        else:
            raise NotImplementedError(site_code_indicator)

        lat_indicator = site_code_filename[-3:]

        site_events_df = df_events_sc_g[site_code_indicator]

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

        site_events_df = site_events_df[
            (site_events_df["latitude"] >= lat_band[0])
            & (site_events_df["latitude"] <= lat_band[1])
        ]

    spatial_cols = ["latitude", "longitude"]
    locs = site_events_df[spatial_cols].drop_duplicates()

    locs_means = site_events_df.groupby(["year", "month"])[spatial_cols].mean()
    locs_stds = site_events_df.groupby(["year", "month"])[spatial_cols].std()

    if locs_stds.max().max() > stationary_site_movement_tolerance:
        print(
            f"Surprisingly large move in location of station {site_code_filename}:\n{locs_stds.max()}"
        )

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

    colours = {"surface": "tab:orange", "shipboard": "tab:blue"}
    countries.plot(color="lightgray", ax=axes[0])

    surf_or_ship = site_events_df["surf_or_ship"].unique()
    assert len(surf_or_ship) == 1
    surf_or_ship = surf_or_ship[0]

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

    locs_means.plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        ax=axes[0],
        color=colours[surf_or_ship],
        alpha=0.4,
        label="mean locations",
        marker="x",
        s=50,
    )

    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-90, 90])
    time_cols = ["year", "month"]

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

    axes[1].legend()

    plt.suptitle(site_code_filename)
    plt.tight_layout()
    plt.show()

    site_monthly_with_loc = site_monthly_df.set_index(time_cols).join(locs_means)
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
    site_monthly_with_loc[spatial_cols] = site_monthly_with_loc[
        spatial_cols
    ].interpolate()
    assert not site_monthly_with_loc.isna().any().any()

    monthly_dfs_with_loc.append(site_monthly_with_loc)

monthly_dfs_with_loc = pd.concat(monthly_dfs_with_loc).sort_index()

# %%
# Handy check to see if all months have at least some data
monthly_dfs_with_loc.index.drop_duplicates().difference(
    pd.MultiIndex.from_product([range(1968, 2022 + 1), range(1, 13)])
)

# %%
assert set(monthly_dfs_with_loc["gas"]) == {config_step.expected_gas}
monthly_dfs_with_loc.to_csv(config_step.processed_monthly_data_with_loc_file)
monthly_dfs_with_loc
