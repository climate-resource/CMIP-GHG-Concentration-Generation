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


def get_metadata_from_event_file_default(filename: str) -> dict[str, str]:
    event_file_regex = r"(?P<gas>[a-z0-9]*)_(?P<site_code_filename>[a-z]{3})_(?P<surf_or_ship>[a-z]*)-flask_1_ccgg_event.txt"

    re_match = re.search(event_file_regex, filename)

    return {k: re_match.group(k) for k in ["gas", "site_code_filename", "surf_or_ship"]}


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
    get_metadata_from_event_file: Callable[[str], str] | None = None,
    filter_df_events: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if get_metadata_from_event_file is None:
        get_metadata_from_event_file = get_metadata_from_event_file_default

    if filter_df_events is None:
        filter_df_events = filter_df_events_default

    filename_metadata = get_metadata_from_event_file(zip_info.filename)

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


def event_file_identifier_default(filename: str) -> bool:
    return "event" in filename


def read_noaa_zip(
    noaa_zip_file: Path, event_file_identifier: Callable[[], bool] | None = None
):
    if event_file_identifier is None:
        event_file_identifier = event_file_identifier_default

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

    # Make sure we haven't ended up with any obviously bogus data
    assert not (df_events["value"] <= -999.0).any()
    assert not (df_events["longitude"] <= -999.0).any()
    return df_events


# %%
for zf in tqdman.tqdm(available_zip_files, desc="ZIP files"):
    df_events = read_noaa_zip(zf)
    display(df_events)

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
