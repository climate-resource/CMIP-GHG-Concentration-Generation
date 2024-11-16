"""
Tools for binning data
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

LON_BIN_BOUNDS = np.arange(-180, 181, 60)
LON_BIN_CENTRES = (LON_BIN_BOUNDS[:-1] + LON_BIN_BOUNDS[1:]) / 2

LAT_BIN_BOUNDS = np.arange(-90, 91, 15)
LAT_BIN_CENTRES = (LAT_BIN_BOUNDS[:-1] + LAT_BIN_BOUNDS[1:]) / 2


VARIABLE_INFO_COLS = ["gas", "unit"]
"""Columns that hold information about the variable we're binning"""

TIME_COLS = [
    "year",
    "month",
]
"""Columns that hold information about the time axis"""

LAT_LON_BIN_COLS = ["lat_bin", "lon_bin"]
"""Columns that hold information about the latitudide and longitude bins"""

BINNING_COLUMNS = [*VARIABLE_INFO_COLS, *TIME_COLS, *LAT_LON_BIN_COLS]
"""Columns to use when binning the data"""

EQUALLY_WEIGHTED_GROUP_COLS = [
    "network",
    "station",
    "measurement_method",
    *BINNING_COLUMNS,
]
"""Columns that specify a group which should receive an equal weight"""

VALUE_COLUMN = "value"
"""Column that specifies the value of the data"""


def get_spatial_bin(
    value: float,
    bin_bounds: npt.NDArray[np.float64],
    bin_centres: npt.NDArray[np.float64],
) -> np.float64:
    """
    Get spatial bin for a given value

    Parameters
    ----------
    value
        Value to bin

    bin_bounds
        Bin bounds

    bin_centres
        Bin centres

    Returns
    -------
        The bin in which this value belongs
    """
    if bin_bounds.size != bin_centres.size + 1:
        msg = f"This won't work. {bin_bounds.size=}, {bin_centres.size=}"
        raise ValueError(msg)

    max_bound_lt = np.max(np.where(value >= bin_bounds)[0])
    if max_bound_lt == bin_centres.size:
        # Value equal to last bound
        return cast(np.float64, bin_centres[-1])

    return cast(np.float64, bin_centres[max_bound_lt])


def add_lat_lon_bin_columns(
    indf: pd.DataFrame,
    lon_bin_col: str = "lon_bin",
    lon_col: str = "longitude",
    lat_bin_col: str = "lat_bin",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """
    Add latitudide and longitude bin columns

    Parameters
    ----------
    indf
        Dataframe to which to add the bin columns

    lon_bin_col
        Name of the column in which to store the longitude bin information

    lon_col
        Name of the column which contains longitude information in ``indf``

    lat_bin_col
        Name of the column in which to store the latitudide bin information

    lat_col
        Name of the column which contains latitudide information in ``indf``

    Returns
    -------
        Dataframe with binning information added.
    """
    out = indf.copy()

    out[lon_bin_col] = out[lon_col].apply(
        get_spatial_bin,  # type: ignore
        bin_bounds=LON_BIN_BOUNDS,
        bin_centres=LON_BIN_CENTRES,
    )
    out[lat_bin_col] = out[lat_col].apply(
        get_spatial_bin,  # type: ignore
        bin_bounds=LAT_BIN_BOUNDS,
        bin_centres=LAT_BIN_CENTRES,
    )

    return out


def get_network_summary(source_df: pd.DataFrame, max_show_all_stations: int = 6) -> str:
    """
    Get a summary of the source :obj:`pd.DataFrame`'s networks and measurement methods

    Parameters
    ----------
    source_df
        :obj:`pd.DataFrame` to summarise

    max_show_all_stations
        Up to this number of stations, all station names are printed.
        If there are more stations than this, we just print the first few
        and the last few.

    Returns
    -------
        Summary of ``source_df``'s networks and measurement methods
    """
    out = ["Collating data from:"]

    for (network, measurement_method), nmmdf in source_df.groupby(["network", "measurement_method"]):
        stations = sorted(nmmdf["station"].unique())
        if len(stations) < max_show_all_stations:
            station_list = ", ".join(stations)
        else:
            station_list = f"{', '.join(stations[:4])} ... {', '.join(stations[-4:])}"

        out.append(f"- {network} {measurement_method} ({len(stations)} stations: {station_list})")

    return "\n".join(out)


def verbose_groupby_mean(inseries: pd.Series[float], groupby: list[str]) -> pd.Series[float]:
    """
    Verbose version of groupby-mean.

    This also prints information about which columns the mean was taken over.

    Parameters
    ----------
    inseries
        Input :obj:`pd.Series` on which to operate

    groupby
        Columns to groupby.

    Returns
    -------
        ``inseries.groupby(groupby).mean()``
    """
    out = inseries.groupby(groupby).mean()

    in_index = inseries.index.names
    out_index = out.index.names

    mean_over_cols = set(in_index) - set(out_index)
    print(f"Took mean over {sorted(mean_over_cols)}")

    return out


def calculate_bin_averages(station_monthly_averages: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average value in each bin

    Parameters
    ----------
    station_monthly_averages
        :obj:`pd.DataFrame` containing monthly averages for each station
        (in each bin).

    Returns
    -------
        Average bin values.
    """
    units = station_monthly_averages["unit"].unique()
    if len(units) > 1:
        msg = f"Unit conversion required, have {units=}"
        raise AssertionError(msg)

    keep_cols = [*EQUALLY_WEIGHTED_GROUP_COLS, VALUE_COLUMN]
    ignore_columns = [c for c in station_monthly_averages.columns if c not in keep_cols]
    if ignore_columns:
        print(f"Will ignore columns: {ignore_columns}")

    station_monthly_averages = station_monthly_averages[keep_cols]

    # explicitly drop nan in columns we care about before calculating,
    # data is long and we have removed columns we don't care about so this is fine.
    station_monthly_averages = station_monthly_averages.dropna()

    # Reset the index so we can see that this mean is actually doing something
    station_monthly_averages = station_monthly_averages.reset_index()
    all_cols_except_value = [c for c in station_monthly_averages.columns if c != VALUE_COLUMN]
    equal_weight_monthly_averages = verbose_groupby_mean(
        (station_monthly_averages.set_index(all_cols_except_value)[VALUE_COLUMN]),
        EQUALLY_WEIGHTED_GROUP_COLS,
    )
    if (
        equal_weight_monthly_averages.index.to_frame()[
            [*EQUALLY_WEIGHTED_GROUP_COLS, *TIME_COLS, *LAT_LON_BIN_COLS]
        ]
        .duplicated()
        .any()
    ):
        msg = (
            "Network-station-measurement combos "
            "should only receive a weight of one at each time "
            "in each spatial bin after this point..."
        )
        raise AssertionError(msg)

    bin_averages = verbose_groupby_mean(equal_weight_monthly_averages, BINNING_COLUMNS)

    return bin_averages.reset_index()
