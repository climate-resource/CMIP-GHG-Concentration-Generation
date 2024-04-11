# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Law dome - smoothing
#
# Smooth the Law Dome data.
#
# Follows Section 2.1.6 of [Meinshausen et al., 2017](https://gmd.copernicus.org/articles/10/2057/2017/), with some tweaks.

# %% [markdown]
# ## Imports

# %%
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pint
import tqdm.autonotebook as tqdman
from attrs import asdict
from openscm_units import unit_registry
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file
from local.point_selection import PointSelector
from local.regressors import WeightedQuantileRegressor

# %%
ur = unit_registry
pint.set_application_registry(ur)  # type: ignore
Q = ur.Quantity

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "smooth_law_dome_data"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "co2"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_process_law_dome = get_config_for_step_id(
    config=config, step="retrieve_and_process_law_dome_data", step_config_id="only"
)

# %% [markdown]
# ## Action

# %%
full_df = pd.read_csv(config_process_law_dome.processed_data_with_loc_file)
full_df

# %% [markdown]
# ### Just get the data for the gas of interest

# %%
gas_df = full_df[full_df["gas"] == config_step.gas]
assert gas_df["gas"].unique() == [config_step.gas]
gas_df

# %%
gas_unit = gas_df["unit"].unique()
if len(gas_unit) > 1:
    msg = f"More than one unit {gas_unit=}"
    raise ValueError(msg)
gas_unit = gas_unit[0]

x_raw = Q(gas_df["time"].values, "yr")
y_raw = Q(gas_df["value"].values, gas_unit)

# %%
plt.scatter(x_raw.m, y_raw.m)
plt.xlabel(str(x_raw.units))
plt.ylabel(str(y_raw.units))

# %% [markdown]
# ## Demonstrate how the noise adder works

# %%
noise_adder = config_step.noise_adder
noise_adder

# %%
print(
    "Random time axis error is "
    f"{(noise_adder.x_relative_random_error * Q(2000, 'yr')).to('year')} "
    "per 2000 years"
)


# %%
int(config_step.gas)


# %%
def string_to_seed(inp: str) -> int:
    """
    Convert a string to an integer that can be used as a random seed

    Parameters
    ----------
    inp
        Input to convert

    Returns
    -------
        Integer that can be used as a random seed
    """
    res = []
    for v in inp:
        try:
            res.append(int(v))
        except ValueError:
            try:
                res.append(string.ascii_lowercase.index(v))
            except ValueError:
                res.append(string.ascii_uppercase.index(v))

    return sum(res)


# %%
x_plus_noise, y_plus_noise = noise_adder.add_noise(
    x=x_raw,
    y=y_raw,
    seed=config.base_seed + string_to_seed(config_step.gas),
)

# %%
fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(8, 6))

x_units = [x_raw.units, x_plus_noise.units]
assert len(set(x_units)) == 1, set(x_units)
x_units = x_units[0]

y_units = [y_raw.units, y_plus_noise.units]
assert len(set(y_units)) == 1, set(y_units)
y_units = y_units[0]

x_raw_m = x_raw.m
y_raw_m = y_raw.m
x_plus_noise_m = x_plus_noise.m
y_plus_noise_m = y_plus_noise.m

axes[0][0].scatter(x_raw_m, x_plus_noise_m)
axes[0][0].set_xlabel(f"x raw ({x_units})")
axes[0][0].set_ylabel(f"x plus noise ({x_units})")

axes[0][1].scatter(x_raw_m, x_plus_noise_m - x_raw_m)
axes[0][1].set_xlabel(f"x raw ({x_units})")
axes[0][1].set_ylabel(f"x plus noise - x raw ({x_units})")

axes[0][2].hist(x_plus_noise_m - x_raw_m)
axes[0][2].set_xlabel(f"x plus noise - x raw ({x_units})")
axes[0][2].set_ylabel("count")

axes[1][0].scatter(y_raw_m, y_plus_noise_m)
axes[1][0].set_xlabel(f"y raw ({y_units})")
axes[1][0].set_ylabel(f"y plus noise ({y_units})")

axes[1][1].scatter(y_raw_m, y_plus_noise_m - y_raw_m)
axes[1][1].set_xlabel(f"y raw ({y_units})")
axes[1][1].set_ylabel(f"y plus noise - y raw ({y_units})")

axes[1][2].hist(y_plus_noise_m - y_raw_m)
axes[1][2].set_xlabel(f"y plus noise - y raw ({y_units})")
axes[1][2].set_ylabel("count")

fig.suptitle(config_step.gas)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2)

for i, (xlim, ylim) in enumerate(((None, None), ((1500, 1750), (260, 300)))):
    axes[i].scatter(x_raw_m, y_raw_m)
    axes[i].scatter(x_plus_noise_m, y_plus_noise_m, alpha=0.3, zorder=3)

    if xlim is not None:
        axes[i].set_xlim(xlim)

    if ylim is not None:
        axes[i].set_ylim(ylim)

# %% [markdown]
# ## Demonstrate how the point selector works

# %%
point_selector = PointSelector(
    pool=(x_plus_noise, y_plus_noise), **asdict(config_step.point_selector_settings)
)

config_step.point_selector_settings

# %%
plt_yrs_width = (
    (x_raw.m.min(), 500),
    (240, 500),
    (250, 500),
    (500, 500),
    (750, 900),
    (1500, 500),
    (1600, 500),
    (1750, 500),
    (1900, 50),
    (1950, 50),
    (1990, 50),
    (x_raw.m.max(), 50),
)

for yr, xlim_width in plt_yrs_width:
    fig, ax = plt.subplots()

    target_year = Q(yr, "yr")
    selected_points = point_selector.get_points(target_year)
    selected_points_x = selected_points[0]
    selected_points_y = selected_points[1]

    print(f"{target_year=}")
    print(f"{len(selected_points_x)=}")
    print(f"{(selected_points_x >= target_year).sum()=}")
    print(f"{(selected_points_x < target_year).sum()=}")

    ax.scatter(
        point_selector.pool[0].m,
        point_selector.pool[1].m,
        alpha=0.9,
        s=60,
        marker="o",
        zorder=2,
        label="pool",
    )
    ax.scatter(
        selected_points_x.m,
        selected_points_y.m,
        alpha=0.9,
        s=60,
        marker="x",
        zorder=3,
        label="selected",
    )

    ax.axvline(target_year.m)
    ax.axvspan(
        target_year.m - point_selector.window_width.m,
        target_year.m + point_selector.window_width.m,
        color="tab:gray",
        label="Within window",
    )
    ax.set_xlim([target_year.m - xlim_width, target_year.m + xlim_width])
    ax.set_ylim([0.9 * selected_points[1].min().m, 1.1 * selected_points[1].max().m])
    ax.legend()

    plt.show()

# %% [markdown]
# ## Demonstrate how the quantile regressor works

# %% [markdown]
# Use same quantile weighting model and weighting function for all gases, hence hard-coded below.

# %%
weighted_quantile_regressor = WeightedQuantileRegressor(quantile=0.5, model_order=3)
weighted_quantile_regressor


# %%
def get_weights(
    x: pint.UnitRegistry.Quantity,
    target_year: pint.UnitRegistry.Quantity,
    window_width: pint.UnitRegistry.Quantity,
    weight_min: float = 1e-4,
) -> pint.UnitRegistry.Quantity:
    """
    Get the weights to use when performing the regression

    Parameters
    ----------
    x
        x-values

    target_year
        Year which we are targeting with the regression

    window_width
        Window width to use when calculating weights

    weight_min
        Minimum weight to return

    Returns
    -------
        Weights to use for ``x`` when performing a regression for ``target_year``.
    """
    z = x - target_year
    # Linear weights don't make sense to me, which is what was used previously
    # weights = np.max(
    #     [
    #         np.repeat(0.0001, len(z)),
    #         # 1 - np.abs(z.m) / np.max(np.abs(z.m))
    #         (1 - np.abs(z) / np.max(np.abs(z))).to("dimensionless").m,
    #     ],
    #     axis=0,
    # )
    out: pint.UnitRegistry.Quantity = np.max(
        [
            np.repeat(weight_min, len(z)),
            np.exp(-np.abs(z) / window_width).to("dimensionless").m,
        ],
        axis=0,
    )

    return out


# %%
for yr, xlim_width in plt_yrs_width:
    fig, ax = plt.subplots()

    target_year = Q(yr, "yr")
    selected_points = point_selector.get_points(target_year)
    selected_points_x = selected_points[0]
    selected_points_y = selected_points[1]

    regression_result = weighted_quantile_regressor.fit(
        x=selected_points_x,
        y=selected_points_y,
        weights=get_weights(
            x=selected_points_x,
            target_year=target_year,
            window_width=point_selector.window_width,
        ),
    )

    print(f"{target_year=}")
    print(f"{len(selected_points_x)=}")
    print(f"{(selected_points_x >= target_year).sum()=}")
    print(f"{(selected_points_x < target_year).sum()=}")

    ax.scatter(
        point_selector.pool[0].m,
        point_selector.pool[1].m,
        alpha=0.4,
        s=60,
        marker="o",
        zorder=2,
        label="pool",
    )
    ax.scatter(
        selected_points_x.m,
        selected_points_y.m,
        alpha=0.4,
        s=60,
        marker="x",
        zorder=3,
        label="selected",
    )
    ax.scatter(
        target_year.m,
        regression_result.predict(target_year).to(selected_points_y.units).m,  # type: ignore
        marker="+",
        linewidth=3,
        s=300,
        alpha=1.0,
        zorder=8.2,
        label="fitted",
    )
    sorted_selected_fine = Q(
        np.linspace(selected_points_x.m.min(), selected_points_x.m.max(), 300),
        selected_points_x.units,
    )
    ax.plot(
        sorted_selected_fine.m,
        regression_result.predict(sorted_selected_fine).to(selected_points_y.units).m,  # type: ignore
        alpha=0.9,
        zorder=4.0,
        color="tab:cyan",
        label="regression",
    )

    ax.axvline(target_year.m, color="tab:gray", linestyle="--", alpha=0.3)
    ax.set_xlim([target_year.m - xlim_width, target_year.m + xlim_width])
    ax.set_ylim([0.9 * selected_points[1].min().m, 1.1 * selected_points[1].max().m])
    ax.legend()

    plt.show()

# %% [markdown]
# ## Smooth

# %%
years_to_calculate = Q(
    np.arange(
        np.floor(x_raw.to("yr").m.min()),
        np.ceil(x_raw.to("yr").m.max()) + 1,
    ),
    "yr",
)

years_to_calculate

# %%
smoothed_all_samples_l = []
for i in tqdman.tqdm(range(config_step.n_draws)):
    x_plus_noise, y_plus_noise = noise_adder.add_noise(
        x=x_raw,
        y=y_raw,
        seed=i + config.base_seed + string_to_seed(config_step.gas),
    )

    point_selector = PointSelector(
        pool=(x_plus_noise, y_plus_noise), **asdict(config_step.point_selector_settings)
    )

    smoothed = []
    quantile_regression_success = True
    for ii, target_year in enumerate(years_to_calculate):
        selected_points = point_selector.get_points(target_year)
        selected_points_x = selected_points[0]
        selected_points_y = selected_points[1]

        weights = get_weights(
            selected_points_x, target_year, window_width=point_selector.window_width
        )

        regression_result = weighted_quantile_regressor.fit(
            x=selected_points_x,
            y=selected_points_y,
            weights=get_weights(
                x=selected_points_x,
                target_year=target_year,
                window_width=point_selector.window_width,
            ),
        )

        if not regression_result.success:
            print("Quantile regression failed, re-drawing")
            quantile_regression_success = False
            break

        smoothed.append(regression_result.predict(target_year))  # type: ignore

    if not quantile_regression_success:
        continue

    smoothed_all_samples_l.append(np.hstack(smoothed))

smoothed_all_samples: pint.UnitRegistry.Quantity = np.vstack(smoothed_all_samples_l)  # type: ignore

smoothed_all_samples_median = np.median(smoothed_all_samples, axis=0)
smoothed_all_samples_median

# %% [markdown]
# Plot

# %%
fig, axes = plt.subplots(nrows=2)

axes[0].scatter(
    x_raw.m,
    y_raw.m,
    alpha=0.4,
    s=20,
    zorder=2,
)
axes[0].plot(
    years_to_calculate.m,
    smoothed_all_samples.m.T,
    color="gray",
    alpha=0.5,
    linewidth=1.0,
)
axes[0].plot(
    years_to_calculate.m,
    smoothed_all_samples_median.m,
    color="tab:green",
    alpha=0.4,
    linewidth=3,
)

axes[1].scatter(
    x_raw.m,
    y_raw.m,
    alpha=0.6,
    s=20,
    zorder=2,
)

axes[1].plot(
    years_to_calculate.m,
    smoothed_all_samples_median.m,
    color="tab:green",
    alpha=0.8,
    linewidth=3,
)

axes[1].set_xlabel("year")
axes[1].set_ylabel(smoothed_all_samples_median.units)


# %% [markdown]
# ## Write output


# %%
def get_column_ensure_only_one(idf: pd.DataFrame, col: str) -> float | str:
    """
    Get a column's value, ensuring that there is only one value

    Parameters
    ----------
    idf
        Data frame from which to retrieve the value

    col
        Column for which to get the value

    Returns
    -------
        Retrieved value

    Raises
    ------
    AssertionError
        There is more than one value in ``idf[col]``
    """
    vals = idf[col].unique()
    if len(vals) > 1:
        raise AssertionError(vals)

    return vals[0]  # type: ignore


# %%
smoothed_median_df = pd.DataFrame(
    smoothed_all_samples_median.m,
    columns=pd.Index(["median"], name="draw"),
    index=pd.Index(years_to_calculate.m, name="year"),
)
smoothed_median_df["unit"] = str(smoothed_all_samples.units)
smoothed_median_df["gas"] = get_column_ensure_only_one(gas_df, "gas")
smoothed_median_df["latitude"] = get_column_ensure_only_one(gas_df, "latitude")
smoothed_median_df["longitude"] = get_column_ensure_only_one(gas_df, "longitude")
smoothed_median_df = (
    smoothed_median_df.set_index(["unit", "gas", "latitude", "longitude"], append=True)
    .melt(ignore_index=False)
    .reset_index()
)

smoothed_median_df

# %%
smoothed_draws_df = pd.DataFrame(
    smoothed_all_samples.m.T,
    columns=pd.Index(range(smoothed_all_samples.shape[0]), name="draw"),
    index=pd.Index(years_to_calculate.m, name="year"),
)
smoothed_draws_df["unit"] = str(smoothed_all_samples.units)
smoothed_draws_df["gas"] = get_column_ensure_only_one(gas_df, "gas")
smoothed_draws_df["latitude"] = get_column_ensure_only_one(gas_df, "latitude")
smoothed_draws_df["longitude"] = get_column_ensure_only_one(gas_df, "longitude")
smoothed_draws_df = (
    smoothed_draws_df.set_index(["latitude", "longitude", "gas", "unit"], append=True)
    .melt(ignore_index=False)
    .reset_index()
)

smoothed_draws_df

# %% [markdown]
# Check we didn't muck up our processing somehow.

# %%
pdt.assert_series_equal(
    smoothed_draws_df.groupby(["year", "unit", "gas", "latitude", "longitude"])[
        "value"
    ].median(),
    smoothed_median_df.set_index(["year", "unit", "gas", "latitude", "longitude"])[
        "value"
    ],
)

# %%
config_step.smoothed_draws_file.parent.mkdir(exist_ok=True, parents=True)
smoothed_draws_df.to_csv(config_step.smoothed_draws_file, index=False)
config_step.smoothed_draws_file

# %%
config_step.smoothed_median_file.parent.mkdir(exist_ok=True, parents=True)
smoothed_median_df.to_csv(config_step.smoothed_median_file, index=False)
config_step.smoothed_median_file
