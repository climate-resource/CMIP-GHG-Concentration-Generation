"""
NOAA processing tools and functions
"""
import re
import zipfile
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pandas as pd
import tqdm.autonotebook as tqdman


def get_metadata_from_file_default(filename: str) -> dict[str, str]:
    event_file_regex = r"(?P<gas>[a-z0-9]*)_(?P<site_code_filename>[a-z]{3}[a-z0-9]*)_(?P<surf_or_ship>[a-z]*)-flask_1_ccgg_(?P<reporting_id>[a-z]*).txt"

    re_match = re.search(event_file_regex, filename)

    return {
        k: re_match.group(k)
        for k in ["gas", "site_code_filename", "surf_or_ship", "reporting_id"]
    }


def filter_df_events_default(inp: pd.DataFrame) -> pd.DataFrame:
    """
    Filter events data (default implementation)

    Parameters
    ----------
    inp
        Input :obj:`pd.DataFrame`

    Returns
    -------
        Filtered :obj:`pd.DataFrame`

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

    with zipfile.ZipFile(noaa_zip_file) as zip:
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
    bogus_flag = -999.0
    if (df_events["value"] <= bogus_flag).any():
        raise ValueError("Obviously wrong values in events data")

    if (df_events["longitude"] <= bogus_flag).any():
        raise ValueError("Obviously wrong longitude in events data")

    if (df_months["value"] <= bogus_flag).any():
        raise ValueError("Obviously wrong values in monthly data")

    return df_events, df_months
