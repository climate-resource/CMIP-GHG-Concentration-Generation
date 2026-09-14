"""
Metrics for comparing a pipeline's monthly output against held-out ground observations

Used by the train/test split evaluation notebooks - see
``notebooks/evaluation/README.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr


def match_monthly(
    pipeline_da: xr.DataArray, test_monthly: pd.DataFrame, value_col: str = "value"
) -> pd.DataFrame:
    """
    Match a pipeline's monthly, ``time``-indexed output against held-out test data by (year, month)

    Parameters
    ----------
    pipeline_da
        Pipeline output, a 1D :obj:`xr.DataArray` with a ``time`` dimension
        of ``cftime.datetime`` values.

    test_monthly
        Held-out test data, aggregated to one row per (year, month) - must
        have ``year``, ``month`` and ``value_col`` columns.

    value_col
        Name of the column in ``test_monthly`` holding the test value.

    Returns
    -------
        One row per (year, month) present in both inputs, with
        ``pipeline_value``, ``test_value`` and ``residual``
        (``pipeline_value - test_value``) columns, alongside ``year``/``month``.
    """
    pipeline_df = pipeline_da.to_dataframe(name="pipeline_value").reset_index()
    pipeline_df["year"] = [t.year for t in pipeline_df["time"]]
    pipeline_df["month"] = [t.month for t in pipeline_df["time"]]

    matched = pipeline_df[["year", "month", "pipeline_value"]].merge(
        test_monthly[["year", "month", value_col]].rename(columns={value_col: "test_value"}),
        on=["year", "month"],
        how="inner",
    )
    matched["residual"] = matched["pipeline_value"] - matched["test_value"]

    return matched


def match_pointwise(
    pipeline_da: xr.DataArray,
    test_binned: pd.DataFrame,
    lat_col: str = "lat_bin",
    lon_col: str | None = "lon_bin",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Match each held-out test bin against the pipeline's own value at that exact location

    Unlike :func:`match_monthly` (which compares two differently-sampled
    "global means" - the pipeline's true global mean vs. a mean over
    whichever bins happened to have spare data to hold out), this compares
    like with like: for each (year, month, lat_bin[, lon_bin]) the test
    data actually has a value for, it looks up the pipeline's value at that
    *same* location and month. This avoids conflating real pipeline error
    with an artifact of the held-out sample's spatial coverage being
    unrepresentative of the globe (e.g. for a gas with strong spatial
    gradients like CH4, a test sample skewed towards one hemisphere/latitude
    band would otherwise show a spurious "bias" that's really just sampling
    geometry, not pipeline error).

    Parameters
    ----------
    pipeline_da
        Pipeline output. Must have a ``time`` dimension
        (``cftime.datetime`` values) and a ``lat`` dimension; if ``lon_col``
        is not ``None``, must also have a ``lon`` dimension (e.g. this
        won't work for a zonal-mean-only product - pass ``lon_col=None``
        for those, which matches by latitude band only).

    test_binned
        Held-out test data, aggregated to one row per (year, month,
        lat_bin[, lon_bin]) - e.g. ``local.binning.calculate_bin_averages``'s
        output. Must have ``year``, ``month``, ``lat_col`` (and
        ``lon_col``, if given) and ``value_col`` columns.

    lat_col
        Column in ``test_binned`` holding the latitude bin to match on.
        Must use the same grid as ``pipeline_da``'s ``lat`` coordinate
        (e.g. ``local.binning.LAT_BIN_CENTRES``).

    lon_col
        Column in ``test_binned`` holding the longitude bin to match on
        (same grid as ``pipeline_da``'s ``lon`` coordinate, e.g.
        ``local.binning.LON_BIN_CENTRES``), or ``None`` to match by
        latitude band only.

    value_col
        Name of the column in ``test_binned`` holding the test value.

    Returns
    -------
        One row per (year, month, lat_bin[, lon_bin]) present in both
        inputs, with ``pipeline_value``, ``test_value`` and ``residual``
        (``pipeline_value - test_value``) columns.
    """
    pipeline_df = pipeline_da.to_dataframe(name="pipeline_value").reset_index()
    pipeline_df["year"] = [t.year for t in pipeline_df["time"]]
    pipeline_df["month"] = [t.month for t in pipeline_df["time"]]
    pipeline_df = pipeline_df.rename(columns={"lat": lat_col, **({"lon": lon_col} if lon_col else {})})

    merge_on = ["year", "month", lat_col, *([lon_col] if lon_col else [])]

    matched = test_binned[[*merge_on, value_col]].merge(
        pipeline_df[[*merge_on, "pipeline_value"]],
        on=merge_on,
        how="inner",
    )
    matched = matched.rename(columns={value_col: "test_value"})
    matched["residual"] = matched["pipeline_value"] - matched["test_value"]

    return matched


def summarise_agreement(matched: pd.DataFrame) -> dict[str, float]:
    """
    Summarise a matched pipeline/test dataframe (see :func:`match_monthly`) into agreement metrics

    Parameters
    ----------
    matched
        Output of :func:`match_monthly`.

    Returns
    -------
        ``n_matched`` (number of matched months), ``bias`` (mean residual,
        pipeline minus test), ``mae`` (mean absolute residual), ``rmse``,
        ``r_squared`` (fraction of the test data's variance the pipeline
        explains - ``nan`` if the test data has zero variance) and
        ``correlation`` (Pearson correlation between pipeline and test
        values).
    """
    residual = matched["residual"]
    pipeline_value = matched["pipeline_value"]
    test_value = matched["test_value"]

    ss_residual = (residual**2).sum()
    ss_total = ((test_value - test_value.mean()) ** 2).sum()

    return {
        "n_matched": len(matched),
        "bias": residual.mean(),
        "mae": residual.abs().mean(),
        "rmse": np.sqrt((residual**2).mean()),
        "r_squared": 1 - ss_residual / ss_total if ss_total > 0 else np.nan,
        "correlation": pipeline_value.corr(test_value),
    }
