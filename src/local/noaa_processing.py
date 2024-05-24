"""
NOAA processing tools and functions
"""

import re
import zipfile
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import cast

import pandas as pd
import tqdm.autonotebook as tqdman

from local.regexp_helpers import re_search_and_retrieve_group

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


HATS_GAS_NAME_MAPPING: dict[str, str] = {
    "c2f6": "PFC-116_C",
    "cfc11": "F11",
    "cfc12": "F12",
    "cfc113": "F113",
    "cfc114": "F114",
    "ccl4": "CCl4",
    "ch2cl2": "CH2Cl2",
    "ch3br": "CH3BR",
    "ch3ccl3": "CH3CCl3",
    "ch3cl": "CH3Cl",
    "halon1211": "HAL1211",
    "halon1301": "H-1301_C",
    "halon2402": "Hal2402",
    "hfc125": "HFC-125_C",
    "hfc143a": "HFC-143a_C",
    "hfc152a": "hf152a",
    "hfc227ea": "F227_",
    "hfc236fa": "HFC-236fa_C",
    "hfc32": "HFC-32_C",
    "hfc365mfc": "F365_",
    "nf3": "NF3_C",
    "so2f2": "SO2F2_C",
}
"""Mapping from HATS names for gases to our names"""

HATS_GAS_NAME_MAPPING_REVERSE = {v: k for k, v in HATS_GAS_NAME_MAPPING.items()}

HATS_M2_PR1_FILE_MAPPING: dict[str, str] = {
    "c2f6": "PFC-116",
    "halon1301": "H-1301",
    "hfc125": "HFC-125",
    "hfc143a": "HFC-143a",
    "hfc236fa": "HFC-236fa",
    "hfc32": "HFC-32",
}
HATS_M2_PR1_FILE_MAPPING_REVERSE = {v: k for k, v in HATS_M2_PR1_FILE_MAPPING.items()}

HATS_ASSUMED_LOCATION: dict[str, dict[str, float]] = {
    "alt": {"latitude": 82.5, "longitude": -62.3},
    "sum": {"latitude": 72.6, "longitude": -38.4},
    "brw": {"latitude": 71.3, "longitude": -156.6},
    "mhd": {"latitude": 53.0, "longitude": -10.0},
    "thd": {"latitude": 41.0, "longitude": -124.0},
    "nwr": {"latitude": 40.052, "longitude": -105.585},
    "kum": {"latitude": 19.5, "longitude": -154.8},
    "mlo": {"latitude": 19.5, "longitude": -155.6},
    # Guessing that this is Mauna Loa's emergency site...
    "mlo_pfp": {"latitude": 19.823, "longitude": -155.469},
    "smo": {"latitude": -14.3, "longitude": -170.6},
    "cgo": {"latitude": -40.7, "longitude": 144.8},
    "psa": {"latitude": -64.6, "longitude": -64.0},
    "spo": {"latitude": -90.0, "longitude": 0.0},
    "hfm": {"latitude": 42.5, "longitude": -72.2},
    "lef": {"latitude": 45.6, "longitude": -90.27},
    "amy": {"latitude": 36.5389, "longitude": 126.3295},
}


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
        sep=r"\s+",
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
        sep=r"\s+",
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


def read_noaa_hats(  # noqa: PLR0913
    infile: Path,
    gas: str,
    source: str,
    sep: str = r"\s+",
    unit_assumed: str = "ppt",
    time_col_assumed: str = "yyyymmdd",
) -> pd.DataFrame:
    """
    Read NOAA HATS data file

    Parameters
    ----------
    infile
        File to read

    gas
        Gas we assume is in the file

    source
        Source of the file

    sep
        Separator to assume when reading the file

    unit_assumed
        Assumed unit of the data

    time_col_assumed
        Assumed time column of the data

    Returns
    -------
        Read data
    """
    with open(infile) as fh:
        file_content = fh.read()

    gas_file = infile.stem.split("_")[0]

    if gas_file in HATS_GAS_NAME_MAPPING_REVERSE:
        gas_file_mapped = HATS_GAS_NAME_MAPPING_REVERSE[gas_file]

    elif gas_file in ("HFC-227ea", "HFC-365mfc"):
        gas_file_mapped = gas_file.lower().replace("-", "")

    else:
        gas_file_mapped = gas_file

    gas_file_mapped = gas_file_mapped.lower()

    if gas != gas_file_mapped:
        msg = f"{gas=}, {gas_file=}, {gas_file_mapped=}"
        raise AssertionError(msg)

    res = pd.read_csv(StringIO(file_content), skiprows=1, sep=sep)
    res["year"] = res[time_col_assumed].astype(str).apply(lambda x: x[:4]).astype(int)
    res["month"] = res[time_col_assumed].astype(str).apply(lambda x: x[4:6]).astype(int)

    if gas in ("ch3br", "ch3ccl3", "hcfc141b", "hcfc142b", "hcfc22"):
        res["value"] = res[gas.upper()]
    elif gas in ("ch3cl", "halon1211", "halon2402", "hfc152a", "hfc227ea", "hfc365mfc"):
        res["value"] = res[HATS_GAS_NAME_MAPPING[gas]]
    else:
        res["value"] = res[gas]

    res["unit"] = unit_assumed
    res["gas"] = gas
    res["source"] = "hats"
    res["latitude"] = res["site"].apply(lambda x: HATS_ASSUMED_LOCATION[x]["latitude"])
    res["longitude"] = res["site"].apply(
        lambda x: HATS_ASSUMED_LOCATION[x]["longitude"]
    )
    res = res.rename({"site": "site_code"}, axis="columns")
    res = res[
        [
            "year",
            "month",
            "value",
            "site_code",
            "latitude",
            "longitude",
            "gas",
            "source",
            "unit",
        ]
    ]

    # Take average where there is more than one observation in a month.
    # Bit annoying that NOAA doesn't have a monthly file, oh well.
    all_except_value = list(set(res.columns) - {"value"})
    res = res.groupby(all_except_value)["value"].mean().to_frame("value").reset_index()

    return res


def read_noaa_hats_combined(  # noqa: PLR0912, PLR0915
    infile: Path, gas: str, source: str, sep: str = r"\s+"
) -> pd.DataFrame:
    """
    Read NOAA HATS data file

    Parameters
    ----------
    infile
        File to read

    gas
        Gas we assume is in the file

    source
        Source of the file

    sep
        Separator to assume when reading the file

    Returns
    -------
        Read data
    """
    with open(infile) as fh:
        file_content = fh.read()

    gas_file = infile.stem.split("_")[2]

    if gas_file in HATS_GAS_NAME_MAPPING_REVERSE:
        gas_file_mapped = HATS_GAS_NAME_MAPPING_REVERSE[gas_file]

    else:
        gas_file_mapped = gas_file

    gas_file_mapped = gas_file_mapped.lower()

    if gas != gas_file_mapped:
        msg = f"{gas=}, {gas_file=}, {gas_file_mapped=}"
        raise AssertionError(msg)

    try:
        unit = re_search_and_retrieve_group(
            r"Global monthly data are provided in .*, (?P<unit>\S*)",
            file_content,
            "unit",
        )
    except ValueError:
        print(f"Missing units for {infile=}")
        raise

    if unit.endswith("."):
        unit = unit[:-1]

    tmp = pd.read_csv(StringIO(file_content), comment="#", sep=sep)

    year_col = tmp.columns[0]
    if not year_col.endswith("YYYY"):
        raise AssertionError()

    month_col = tmp.columns[1]
    if not month_col.endswith("MM"):
        raise AssertionError()

    tmp = tmp.set_index([year_col, month_col])
    tmp.index.names = ["year", "month"]

    if gas in ("ccl4",):
        gas_end = HATS_GAS_NAME_MAPPING[gas]
    else:
        gas_end = gas.upper()

    res_l = []
    for c in tmp:
        c = cast(str, c)
        if (
            c.endswith("sd")
            or not c.endswith(gas_end)
            or any(v in c for v in ("NH", "SH", "Global"))
        ):
            continue

        station = c.split("_")[1]
        station_dat = (
            tmp[c].dropna().to_frame("value")
        )  # .rename({c: "value"}, axis="columns")
        station_dat["site_code"] = station

        for line in file_content.splitlines():
            if line.startswith(f"#  {station}"):
                lat_lon_info = line.split("(")[1].split(")")[0]

                if lat_lon_info == "90S":
                    lat = -90.0
                    lon = 0.0

                else:
                    lat_s = lat_lon_info.split(",")[0]
                    if lat_s.endswith("N"):
                        lat = float(lat_s[:-1])
                    elif lat_s.endswith("S"):
                        lat = -float(lat_s[:-1])
                    else:
                        raise AssertionError(lat_s)

                    lon_s = lat_lon_info.split(",")[1]
                    if lon_s.endswith("W"):
                        lon = -float(lon_s[:-1])
                    elif lon_s.endswith("E"):
                        lon = float(lon_s[:-1])
                    else:
                        raise AssertionError(lon_s)

        station_dat["latitude"] = lat
        station_dat["longitude"] = lon

        res_l.append(station_dat)

    res = pd.concat(res_l)
    res["gas"] = gas
    res["source"] = source
    res["unit"] = unit

    return res.reset_index()


def read_noaa_hats_m2_and_pr1(  # noqa: PLR0913
    infile: Path,
    gas: str,
    source: str,
    sep: str = r"\s+",
    comment: str = "#",
    time_col_assumed: str = "yyyymmdd",
) -> pd.DataFrame:
    """
    Read NOAA HATS data file that contains M2 and PR1 data

    Parameters
    ----------
    infile
        File to read

    gas
        Gas we assume is in the file

    source
        Source of the file

    sep
        Separator to assume when reading the file

    comment
        Indicator of comments in the file

    Returns
    -------
        Read data
    """
    with open(infile) as fh:
        file_content = fh.read()

    gas_file = infile.stem.split("_")[0]

    if gas_file in HATS_M2_PR1_FILE_MAPPING_REVERSE:
        gas_file_mapped = HATS_M2_PR1_FILE_MAPPING_REVERSE[gas_file]

    else:
        gas_file_mapped = gas_file

    gas_file_mapped = gas_file_mapped.lower()

    if gas != gas_file_mapped:
        msg = f"{gas=}, {gas_file=}, {gas_file_mapped=}"
        raise AssertionError(msg)

    try:
        unit = re_search_and_retrieve_group(
            r"Units: .*\((?P<unit>\S*)\)",
            file_content,
            "unit",
        )
    except ValueError:
        print(f"Missing units for {infile=}")
        raise

    res = pd.read_csv(StringIO(file_content), comment=comment, sep=sep)
    res["year"] = res[time_col_assumed].astype(str).apply(lambda x: x[:4]).astype(int)
    res["month"] = res[time_col_assumed].astype(str).apply(lambda x: x[4:6]).astype(int)

    res["value"] = res[HATS_GAS_NAME_MAPPING[gas]]

    res["unit"] = unit
    res["gas"] = gas
    res["source"] = "hats"
    res["latitude"] = res["site"].apply(
        lambda x: HATS_ASSUMED_LOCATION[x.lower()]["latitude"]
    )
    res["longitude"] = res["site"].apply(
        lambda x: HATS_ASSUMED_LOCATION[x.lower()]["longitude"]
    )
    res = res.rename({"site": "site_code"}, axis="columns")
    res = res[
        [
            "year",
            "month",
            "value",
            "site_code",
            "latitude",
            "longitude",
            "gas",
            "source",
            "unit",
        ]
    ]

    # Take average where there is more than one observation in a month.
    # Bit annoying that NOAA doesn't have a monthly file, oh well.
    all_except_value = list(set(res.columns) - {"value"})
    res = res.groupby(all_except_value)["value"].mean().to_frame("value").reset_index()

    return res
