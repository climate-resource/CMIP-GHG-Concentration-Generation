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

PROCESSED_DATA_COLUMNS: list[str] = [
    "gas",
    "reporting_id",
    "year",
    "month",
    "latitude",
    "longitude",
    "value",
    "unit",
    "site_code_filename",
    "site_code",
    "surf_or_ship",
    "source",
]
"""Columns in the processed data output"""

UNIT_MAP: dict[str, str] = {
    "micromol mol-1": "ppm",
    "nanomol mol-1": "ppb",
    "picomol mol-1": "ppt",
}
"""Mapping from NOAA units to convention we use"""


def get_metadata_from_filename_default(filename: str) -> dict[str, str]:
    """
    Get metadata from filename - default implementation

    Parameters
    ----------
    filename
        Filename from which to retrieve metadata

    Returns
    -------
        Metadata extracted from the filename
    """
    event_file_regex = (
        r"(?P<gas>[a-z0-9]*)"
        r"_(?P<site_code_filename>[a-z]{3}[a-z0-9]*)"
        r"_(?P<surf_or_ship>[a-z]*)"
        r"-(?P<source>[a-z0-9-]*)"
        r"_1_ccgg"
        r"_(?P<reporting_id>[a-zA-Z]*)"
        ".txt"
    )

    re_match = re.search(event_file_regex, filename)
    if not re_match:
        raise ValueError(  # noqa: TRY003
            f"Failed to extract metadata from filename: {filename}"
        )

    return {
        k: re_match.group(k)
        for k in ["gas", "site_code_filename", "surf_or_ship", "source", "reporting_id"]
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


def read_data_incl_datetime(
    open_zip: zipfile.ZipFile,
    zip_info: zipfile.ZipInfo,
    get_metadata_from_filename: Callable[[str], dict[str, str]] | None = None,
    filter_df_events: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    datetime_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read data that includes date time information

    Parameters
    ----------
    open_zip
        Open zip archive from which to read the data

    zip_info
        Zip info about the file in the zip archive from which we want
        to read the data

    get_metadata_from_filename
        Function to use to retrieve metadata from the file name.
        If not supplied, :func:`get_metadata_from_filename_default`
        is used.

    filter_df_events
        Function to use to filter the events from the read data.
        If not supplied, :func:`filter_df_events_default` is used.

    datetime_columns
        Columns to read as date times

    Returns
    -------
        Read data
    """
    if get_metadata_from_filename is None:
        get_metadata_from_filename = get_metadata_from_filename_default

    if filter_df_events is None:
        filter_df_events = filter_df_events_default

    if datetime_columns is None:
        datetime_columns = ["datetime", "analysis_datetime"]

    filename_metadata = get_metadata_from_filename(zip_info.filename)

    file_content = open_zip.read(zip_info).decode("utf-8")

    for line in file_content.splitlines():
        if line.startswith("# value:units : "):
            units = line.split("# value:units : ")[1]
            break
    else:
        raise ValueError(  # noqa: TRY003
            f"Units not found. File contents:\n{file_content}"
        )

    try:
        units = UNIT_MAP[units]
    except KeyError:
        print(f"Could not map {units}")

    df_events = pd.read_csv(
        StringIO(file_content),
        comment="#",
        date_format={},
        parse_dates=datetime_columns,
        delim_whitespace=True,
    )
    df_events["unit"] = units

    for k, v in filename_metadata.items():
        df_events[k] = v

    df_events_filtered = filter_df_events(df_events)

    return df_events_filtered


def read_flask_monthly_data(
    open_zip: zipfile.ZipFile,
    zip_info: zipfile.ZipInfo,
    get_metadata_from_filename: Callable[[str], dict[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Read monthly flask data

    This doesn't have any location information so needs to be treated a
    bit differently

    Parameters
    ----------
    open_zip
        Open zip archive from which to read the data

    zip_info
        Zip info about the file in the zip archive from which we want
        to read the data

    get_metadata_from_filename
        Function to use to retrieve metadata from the file name.
        If not supplied, :func:`get_metadata_from_filename_default`
        is used.

    Returns
    -------
        Monthly flask data
    """
    if get_metadata_from_filename is None:
        get_metadata_from_filename = get_metadata_from_filename_default

    filename_metadata = get_metadata_from_filename(zip_info.filename)

    file_content = open_zip.read(zip_info).decode("utf-8")

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


def is_event_file_default(filename: str) -> bool:
    """
    Identify whether file contains event data

    Parameters
    ----------
    filename
        Name of the file

    Returns
    -------
        ``True`` if the file contains event data
    """
    return "event" in filename


def is_monthly_file_default(filename: str) -> bool:
    """
    Identify whether file contains monthly data (default implementation)

    Parameters
    ----------
    filename
        Filename to check

    Returns
    -------
        ``True`` if file contains monthly data
    """
    return "month" in filename


def read_noaa_flask_zip(
    noaa_zip_file: Path,
    gas: str,
    is_event_file: Callable[[str], bool] | None = None,
    is_monthly_file: Callable[[str], bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read flask data from NOAA zip archive

    Parameters
    ----------
    noaa_zip_file
        Path to NOAA zip archive

    gas
        Gas we expect the file to contain

    is_event_file
        Function which identifies whether a file contains
        event data. If not supplied, :func:`is_event_file_default`
        is used.

    is_monthly_file
        Function which identifies whether a file contains
        monthly data. If not supplied. :func:`is_monthly_file_default`
        is used.

    Returns
    -------
        Events data and monthly data as two separate :obj:`pd.DataFrame`
    """
    ASSUMED_MONTHLY_UNITS = {
        "co2": "ppm",
        "ch4": "ppb",
        "n2o": "ppb",
        "sf6": "ppt",
    }
    """Units aren't provided in monthly files so we have to asssume them instead"""

    if is_event_file is None:
        is_event_file = is_event_file_default

    if is_monthly_file is None:
        is_monthly_file = is_monthly_file_default

    with zipfile.ZipFile(noaa_zip_file) as zip:
        event_files = [item for item in zip.filelist if is_event_file(item.filename)]
        df_events = pd.concat(
            [
                read_data_incl_datetime(zip, event_file_item)
                for event_file_item in tqdman.tqdm(event_files)
            ]
        )

        month_files = [item for item in zip.filelist if is_monthly_file(item.filename)]
        df_months = pd.concat(
            [
                read_flask_monthly_data(zip, month_files_item)
                for month_files_item in tqdman.tqdm(month_files)
            ]
        )
        df_months["unit"] = ASSUMED_MONTHLY_UNITS[gas]

    # Make sure we have the expected gas
    if not (df_events["gas"] == gas).all():
        raise AssertionError("Assumed the wrong gas")  # noqa: TRY003

    if not (df_months["gas"] == gas).all():
        raise AssertionError("Assumed the wrong gas")  # noqa: TRY003

    # Make sure we haven't ended up with any obviously bogus data
    bogus_flag = -999.0
    if (df_events["value"] <= bogus_flag).any():
        raise ValueError("Obviously wrong values in events data")  # noqa: TRY003

    if (df_events["longitude"] <= bogus_flag).any():
        raise ValueError("Obviously wrong longitude in events data")  # noqa: TRY003

    if (df_months["value"] <= bogus_flag).any():
        raise ValueError("Obviously wrong values in monthly data")  # noqa: TRY003

    return df_events, df_months


def is_monthly_file_in_situ(filename: str) -> bool:
    """
    Identify whether an in-situ file contains monthly data

    Parameters
    ----------
    filename
        Filename to check


    Returns
    -------
        ``True`` if the file contains monthly data
    """
    return "MonthlyData" in filename


def read_noaa_in_situ_zip(
    noaa_zip_file: Path,
    is_monthly_file: Callable[[str], bool] | None = None,
) -> pd.DataFrame:
    """
    Read in-situ data from a NOAA zip archive

    Parameters
    ----------
    noaa_zip_file
        Zip archive from which to read data

    is_monthly_file
        Function to use to identify which files contain monthly data

    Returns
    -------
        Read data
    """
    if is_monthly_file is None:
        is_monthly_file = is_monthly_file_in_situ

    with zipfile.ZipFile(noaa_zip_file) as zip:
        month_files = [item for item in zip.filelist if is_monthly_file(item.filename)]
        df_months = pd.concat(
            [
                read_data_incl_datetime(
                    zip, month_files_item, datetime_columns=["datetime"]
                )
                for month_files_item in tqdman.tqdm(month_files)
            ]
        )

    # Make sure we haven't ended up with any obviously bogus data
    bogus_flag = -999.0

    if (df_months["longitude"] <= bogus_flag).any():
        raise ValueError("Obviously wrong longitude in events data")  # noqa: TRY003

    if (df_months["value"] <= bogus_flag).any():
        raise ValueError("Obviously wrong values in monthly data")  # noqa: TRY003

    return df_months
