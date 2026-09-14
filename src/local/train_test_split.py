"""
Tools for splitting ground-based observational data into train/test sets
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import ConvexHull, QhullError


def _hull_relevant_bins(lon_lat: npt.NDArray[np.float64]) -> set[tuple[float, float]]:
    """
    Identify which of a month's distinct ``(lon_bin, lat_bin)`` points are hull-relevant

    A point is "hull-relevant" if it's a vertex of the convex hull that
    ``local.binned_data_interpolation.interpolate`` actually builds for that
    month (including its "round the world" longitude wrap) - i.e. removing
    it could shrink the region ``griddata`` can fill in and introduce new
    NaNs. Points *not* returned are safely interior: by the definition of a
    convex hull, removing a non-vertex point can never change the hull, so
    it can never affect which grid cells `griddata` can produce a value for.

    Mirrors ``interpolate``'s "round the world" trick exactly (tripling
    every point at -360/0/+360 in longitude before triangulating) so a point
    is treated as hull-relevant if *any* of its three wrapped copies is a
    hull vertex - ``interpolate`` triangulates all three copies together as
    one point set, so the same holds there.

    Parameters
    ----------
    lon_lat
        ``(n, 2)`` array of distinct ``(lon_bin, lat_bin)`` points observed
        in one ``(year, month)``.

    Returns
    -------
        Set of ``(lon_bin, lat_bin)`` tuples that are hull-relevant. If
        there are too few distinct points for a well-defined 2D hull (fewer
        than 3, or degenerate/collinear), every point is returned - nothing
        can safely be assumed interior.
    """
    min_points_for_hull = 3
    n = lon_lat.shape[0]
    if n < min_points_for_hull:
        return {tuple(row) for row in lon_lat}

    lon = lon_lat[:, 0]
    lat = lon_lat[:, 1]
    wrapped = np.vstack(
        [
            np.column_stack([lon - 360, lat]),
            np.column_stack([lon, lat]),
            np.column_stack([lon + 360, lat]),
        ]
    )
    try:
        hull = ConvexHull(wrapped)
    except QhullError:
        # Degenerate (e.g. all points collinear) - can't safely assume anything is interior.
        return {tuple(row) for row in lon_lat}

    hull_vertex_indices = set(hull.vertices.tolist())
    return {tuple(lon_lat[i]) for i in range(n) if {i, i + n, i + 2 * n} & hull_vertex_indices}


def stratified_test_split(
    df: pd.DataFrame,
    eligible: npt.NDArray[np.bool_] | pd.Series,
    test_fraction: float,
    seed: int | None,
    group_cols: tuple[str, ...] = ("year", "month", "lat_bin", "lon_bin"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split rows into train/test sets, holding out ``test_fraction`` of all eligible rows

    Only rows where ``eligible`` is ``True`` can ever be selected for the
    test set (e.g. NOAA surface-flask rows, leaving continuously-operating
    in-situ/AGAGE/GAGE/ALE "backbone" stations untouched) - this is part of
    a workaround for the fact that a pure random sample can easily empty out
    an already-thinly-covered spatial bin, which cascades into
    ``scipy.interpolate.griddata`` failing to fill the grid for that month
    (see ``local.binned_data_interpolation.interpolate``, called from
    ``1101_ch4_.../1201_co2_..._interpolate-observational-network.py``),
    that month being dropped from ``observational_network_interpolated_file``
    entirely, and - if enough months drop out of the same year - into
    non-uniformly-spaced years, which breaks later steps that assume
    uniform year spacing (e.g. the seasonality decomposition's
    mean-preserving interpolation).

    For each group (by default one (year, month)'s contribution to one
    spatial bin - i.e. one input point to ``interpolate``'s ``griddata``
    call for that month), a per-group safety cap (``max_test``) is computed
    - the most that group could ever lose without risking the above. A group
    can be safely emptied entirely (``max_test = n_eligible``) if either:

    - it has any *non-eligible* ("protected") rows, which alone guarantee
      the group stays populated; or
    - it is not "hull-relevant" this month - see ``_hull_relevant_bins``:
      by the definition of a convex hull, removing a point that isn't a
      hull vertex can never shrink the hull, so it can never introduce a
      new NaN, regardless of protection.

    Otherwise (100% eligible rows *and* hull-relevant this month),
    ``max_test = n_eligible - 1``: at least 1 row is always kept in
    training, so the split can never shrink the interpolatable region for a
    group that had data.

    ``test_fraction`` then controls the *overall* holdout rate, not a
    per-group one: all eligible rows are shuffled together into one random
    order (governed by ``seed``), then taken in that order into the test set
    one at a time, skipping (leaving in training) any row whose group has
    already reached its safety cap, until ``round(test_fraction *
    n_eligible_total)`` rows have been taken (or every group is at capacity,
    if that happens first). This is what makes the achieved rate track the
    requested ``test_fraction`` directly - unlike capping each group
    individually at ``floor(test_fraction * n_eligible_in_group)``, which
    rounds to 0 for the vast majority of groups (most are only 1-4 rows) and
    ends up holding out a much smaller fraction than requested overall.

    Parameters
    ----------
    df
        Data to split - must already have whatever columns ``group_cols``
        names (e.g. call ``local.binning.add_lat_lon_bin_columns`` first for
        the default ``group_cols``). Must have a default ``RangeIndex``
        (0..len(df)-1), e.g. via ``df.reset_index(drop=True)``.

    eligible
        Boolean mask (same length and order as ``df``) marking which rows
        are allowed to be selected for the test set.

    test_fraction
        Target fraction of *all* eligible rows to hold out, overall. Capped
        at the total safety capacity (``sum`` of every group's
        ``max_test``) if that's smaller - in practice this is only a
        concern for very large ``test_fraction`` values, since the capacity
        is usually close to 100% of eligible rows.

    seed
        Seed for the random number generator, for reproducibility.

    group_cols
        Columns to stratify by.

    Returns
    -------
        ``(train, test)`` dataframes, row-disjoint and together covering all
        of ``df``.
    """
    if not df.index.equals(pd.RangeIndex(len(df))):
        msg = "`df` must have a default RangeIndex - call `df.reset_index(drop=True)` first"
        raise AssertionError(msg)

    eligible_arr = np.asarray(eligible, dtype=bool)
    if eligible_arr.shape[0] != len(df):
        msg = f"`eligible` must be the same length as `df` ({len(df)}), got {eligible_arr.shape[0]}"
        raise AssertionError(msg)

    rng = np.random.default_rng(seed=seed)

    # Hull-relevance is only meaningful for the (year, month, lat_bin,
    # lon_bin) stratification this module was built for - fall back to the
    # old, protection-only behaviour for any other `group_cols`.
    use_hull_relaxation = set(group_cols) == {"year", "month", "lat_bin", "lon_bin"}
    hull_relevant_by_month: dict[tuple[Any, Any], set[tuple[float, float]]] = {}
    if use_hull_relaxation:
        for (year, month), month_df in df.groupby(["year", "month"]):
            distinct_bins = month_df[["lon_bin", "lat_bin"]].drop_duplicates().to_numpy()
            hull_relevant_by_month[(year, month)] = _hull_relevant_bins(distinct_bins)

    # First pass: compute each group's safety cap (`max_test`) and record
    # which group every eligible row belongs to - no selection happens yet.
    max_test_by_group: dict[Any, int] = {}
    group_of_row: dict[int, Any] = {}
    for group_key, group in df.groupby(list(group_cols), sort=True):
        group_index = group.index.to_numpy()
        eligible_index = group_index[eligible_arr[group_index]]
        n_eligible = eligible_index.size
        if n_eligible == 0:
            continue

        has_protected_rows = n_eligible < group_index.size

        is_hull_relevant = True
        if use_hull_relaxation:
            group_values = dict(zip(group_cols, group_key, strict=True))
            month_key = (group_values["year"], group_values["month"])
            bin_point = (group_values["lon_bin"], group_values["lat_bin"])
            is_hull_relevant = bin_point in hull_relevant_by_month[month_key]

        safe_to_fully_empty = has_protected_rows or not is_hull_relevant
        max_test_by_group[group_key] = n_eligible if safe_to_fully_empty else max(0, n_eligible - 1)
        for idx in eligible_index:
            group_of_row[int(idx)] = group_key

    # Second pass: pick a random order over *all* eligible rows (globally,
    # not per group), then walk that order taking rows into the test set
    # one at a time - skipping any row whose group is already at its safety
    # cap - until the overall target count is reached.
    eligible_index_all = np.fromiter(group_of_row.keys(), dtype=np.int64)
    n_eligible_total = eligible_index_all.size
    total_capacity = sum(max_test_by_group.values())
    n_target = min(int(round(n_eligible_total * test_fraction)), total_capacity)

    remaining_capacity = dict(max_test_by_group)
    test_positions_l: list[int] = []
    for idx in rng.permutation(eligible_index_all):
        if len(test_positions_l) >= n_target:
            break
        g = group_of_row[int(idx)]
        if remaining_capacity[g] > 0:
            test_positions_l.append(int(idx))
            remaining_capacity[g] -= 1

    test_positions = np.array(test_positions_l, dtype=int)
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_positions] = True

    return df[~test_mask], df[test_mask]
