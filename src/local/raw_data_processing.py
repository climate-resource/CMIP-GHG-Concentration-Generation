"""
Raw data processing
"""

from __future__ import annotations

import pandas as pd

LATITUDINAL_MEAN_CALCULATION_COLUMNS: tuple[str, ...] = (
    "gas",
    "year",
    "month",
    "latitude",
    "longitude",
    "value",
    "unit",
    "network",
    "station",
    "measurement_method",
)
"""Columns that are required for calculating latitudinal-means"""


def check_processed_data_columns_for_latitudinal_mean(indf: pd.DataFrame) -> None:
    missing_latitudinal_mean_cols = set(LATITUDINAL_MEAN_CALCULATION_COLUMNS) - set(
        indf.columns
    )
    if missing_latitudinal_mean_cols:
        msg = f"{missing_latitudinal_mean_cols=}"
        raise AssertionError(msg)
