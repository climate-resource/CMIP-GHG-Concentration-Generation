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
df_events.columns.tolist()

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
# Work out best-estimate location of each monthly value
stationary_site_movement_tolerance = 0.5


for site_code_filename, site_monthly_df in tqdman.tqdm(
    df_months_sc_g.items(), desc="Monthly sites"
):
    # site_events_df =
    if len(site_code_filename) == 3:
        site_events_df = df_events_sc_g[site_code_filename]

        spatial_cols = ["latitude", "longitude"]
        locs = site_events_df[spatial_cols].drop_duplicates()
        assert (
            False
        ), "Group by year-month here to handle fact that some sites move a bit"
        site_events_df.groupby(["year", "month"])[spatial_cols].mean()
        site_events_df.groupby(["year", "month"])[spatial_cols].std()

        means = {}
        for col in spatial_cols:
            mean_col = locs[col].mean()
            std_col = locs[col].std()
            if std_col >= stationary_site_movement_tolerance:
                raise ValueError(
                    f"Surprisingly large move in {col} of stationary station: {locs}"
                )

            means[col] = mean_col
    else:
        pass
        # raise NotImplementedError(site_code_filename)
    # break

# %%
site_events_df.groupby(["year", "month"])[spatial_cols].std().max()

# %%
site_code_filename

# %%
locs

# %%
site_code_filename

# %%

# %%
sorted(df_months_sc_g.keys())

# %%
df_events_sc_g["poc"]

# %%
df_months_sc_g["poc000"]

# %%
sorted(df_months_sc_g.keys())

# %%
set(df_events_sc_g.keys()) - set(df_months_sc_g.keys())

# %%
set(df_months_sc_g.keys()) - set(df_events_sc_g.keys())

# %%
df_events

# %%
df_months

# %%
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir
    / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
assert False

# %% [markdown]
# To-do's:
# - read monthly data
# - make plot of events and monthly against time on left-hand side and location of samples on right-hand side
#     - this will also check that all monthly data has event data sitting underneath it
# - save data
# - parameterise notebook so we can do same for CH4, N2O and SF6 observations

# %%
coords = df_events[["latitude", "longitude", "surf_or_ship"]].drop_duplicates()
fig, ax = plt.subplots()
countries.plot(color="lightgray", ax=ax)

for kind, colour, alpha in (
    ("surface", "tab:orange", 0.7),
    ("shipboard", "tab:blue", 0.1),
):
    coords[coords["surf_or_ship"] == kind].plot(
        x="longitude",
        y="latitude",
        kind="scatter",
        ax=ax,
        color=colour,
        alpha=alpha,
        label=kind,
    )

ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax.set_title("Sampling sites")
