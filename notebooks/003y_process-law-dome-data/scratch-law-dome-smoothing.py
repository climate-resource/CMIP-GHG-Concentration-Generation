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
# When translating:
#
# - split all the code out into its own module
# - put the serialised point selector, regressor etc. into the config
#     - one config per gas
# - fix up the unit handling
# - save out all the ensemble members and median
#
# Add notes from here https://docs.google.com/document/d/12r3B__DQGgwbfcI_BH6ZZjgEtOTVPN7h6c5NFhTLhJM/edit

# %%
import matplotlib.pyplot as plt
import pandas as pd
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
step: str = "retrieve_and_process_law_dome_data"
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)
config_process = get_config_for_step_id(
    config=config, step="retrieve_and_process_law_dome_data", step_config_id="only"
)

# %%
full_df = pd.read_csv(config_process.processed_data_with_loc_file)
full_df

# %%
import numpy as np
import pint
import scipy.optimize
from attrs import define
from openscm_units import unit_registry

ur = unit_registry
Q = ur.Quantity


# %%
@define
class NoiseAdder:
    time_now: pint.UnitRegistry.Quantity

    percentage_time_error: pint.UnitRegistry.Quantity

    y_random_error: pint.UnitRegistry.Quantity

    def add_noise(
        self, x, y
    ) -> tuple[pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity]:
        # Uniform noise seems weird to me, but ok,
        # will copy Nicolai for now.
        # Different choice to Nicolai here, time error scales with age,
        # with zero being now rather than minimum value in input array.
        x_plus_noise = x + (
            self.time_now - x
        ) * self.percentage_time_error * np.random.uniform(
            low=-0.5, high=0.5, size=y.shape
        )
        y_plus_noise = y + self.y_random_error * np.random.uniform(
            low=-0.5, high=0.5, size=y.shape
        )

        return (x_plus_noise, y_plus_noise)


# %%
# Super unclear what the right choice for time_now is.
time_now = Q(2024, "yr")
noise_adders = {
    "co2": NoiseAdder(
        time_now=time_now,
        percentage_time_error=Q(60 / 2000, "yr / yr"),
        y_random_error=Q(2, "ppm"),
    ),
    "ch4": NoiseAdder(
        time_now=time_now,
        percentage_time_error=Q(50 / 2000, "yr / yr"),
        y_random_error=Q(3, "ppb"),
    ),
    "n2o": NoiseAdder(
        time_now=time_now,
        percentage_time_error=Q(90 / 2000, "yr / yr"),
        y_random_error=Q(3, "ppb"),
    ),
}

point_selector_settings = {
    "co2": dict(
        window_width=Q(120, "yr"),
        minimum_data_points_either_side=7,
        maximum_data_points_either_side=25,
    ),
    "ch4": dict(
        window_width=Q(100, "yr"),
        minimum_data_points_either_side=4,
        maximum_data_points_either_side=10,
    ),
    "n2o": dict(
        window_width=Q(300, "yr"),
        minimum_data_points_either_side=7,
        maximum_data_points_either_side=15,
    ),
}


# %%
def plot_noise_addition(x_raw, y_raw, x_plus_noise, y_plus_noise, axes):
    assert axes.shape == (2, 3)

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

    return axes


# %%
@define
class PointSelector:
    window_width: pint.UnitRegistry.Quantity

    minimum_data_points_either_side: int
    """
    Minimum number of data points to pick either side of the target point.

    Obviously, if there are no data points to pick on one side of the target,
    less than this will be returned.
    """

    maximum_data_points_either_side: int
    """
    Maximum number of data points to pick either side of the target point.
    """

    x_pool: pint.UnitRegistry.Quantity
    """
    Pool of x-points to choose from
    """

    y_pool: pint.UnitRegistry.Quantity
    """
    Pool of y-points to choose from
    """

    # Nicolai's code picks minimum/maximum number either side of obs.
    # This is different to what is written in the paper.
    def get_points(
        self, target_year: pint.UnitRegistry.Quantity
    ) -> tuple[pint.UnitRegistry.Quantity, pint.UnitRegistry.Quantity]:
        # TODO: sort on entry rather than in here
        # which is a surprising side effect.
        x_pool_sorted_idx = np.argsort(self.x_pool)
        self.x_pool = self.x_pool[x_pool_sorted_idx]
        self.y_pool = self.y_pool[x_pool_sorted_idx]

        selected_x = []
        selected_y = []
        for forward_looking in [True, False]:
            if forward_looking:
                pool = self.x_pool[np.where(self.x_pool >= target_year)]
            else:
                pool = self.x_pool[np.where(self.x_pool < target_year)]

            if not pool.size:
                # Nothing in this direction, move on
                continue

            pool_abs_distance_from_target = np.abs(pool - target_year)
            within_window = pool[pool_abs_distance_from_target <= self.window_width]

            if within_window.shape[0] >= self.minimum_data_points_either_side:
                select_n_max = self.maximum_data_points_either_side
                select_from = within_window
                select_from_distance_from_target = np.abs(within_window - target_year)

            else:
                select_n_max = self.minimum_data_points_either_side
                select_from = pool
                select_from_distance_from_target = pool_abs_distance_from_target

            selected = Q(
                np.take_along_axis(
                    select_from.m,
                    np.argsort(select_from_distance_from_target)[:select_n_max],
                    axis=0,
                ),
                select_from.units,
            )

            selected_x.append(selected)
            selected_y.append(
                self.y_pool[
                    np.searchsorted(
                        self.x_pool.to(selected.units).m, selected.m, side="left"
                    )
                ]
            )

        return (
            np.hstack(selected_x),
            np.hstack(selected_y),
        )


# %%
@define
class WeightedQuantileRegression:
    quantile: float = 0.5
    """
    Quantile to fit
    """
    # TODO: add validation that this is between 0 and 1

    weight_min: float = 0.0001
    """
    Minimum weight to apply to each point

    Avoids weights of zero being given,
    which then makes things weird.
    """

    lambda_reg: float = 1e-3
    """
    Lambda to use for regularisation
    """

    model_order: int = 3
    """
    Order of the model to fit.

    1 means a linear model.
    """

    def fit(self, x, y, weights):
        if x.shape != y.shape:
            raise AssertionError

        if len(x.shape) > 1:
            raise AssertionError

        N = len(x)
        beta_len = self.model_order + 1

        c = self.lambda_reg * np.ones(2 * (beta_len + N))
        # # Prefer models with smaller higher-order terms
        # for i in range(1, beta_len + 1):
        #     c[i] = self.lambda_reg ** (1 / (i + 1))
        #     c[beta_len + i] = self.lambda_reg ** (1 / (i + 1))

        c[2 * beta_len : 2 * beta_len + N] = weights * self.quantile
        c[2 * beta_len + N :] = weights * (1 - self.quantile)

        A = np.zeros((N, 2 * (beta_len + N)))
        A[
            (
                np.arange(N),
                np.arange(2 * beta_len, 2 * beta_len + N),
            )
        ] = 1
        A[
            (
                np.arange(N),
                np.arange(2 * beta_len + N, 2 * beta_len + 2 * N),
            )
        ] = -1

        for i in range(beta_len):
            A[:, i] = (x**i).m
            A[:, i + beta_len] = -(x**i).m

        # Tracking the units through properly will take a bit more thinking
        b = y.m

        for maxiter in [1e5, 1e6, 1e7, 1e8, 1e9]:
            res = scipy.optimize.linprog(
                c,
                b_eq=b,
                A_eq=A,
                method="highs",
                bounds=(0, None),
                options=dict(maxiter=int(maxiter)),
            )
            if res.success:
                break

        else:
            print(f"didn't converge for {target_x=}")
            return None

        beta = res.x[:beta_len] - res.x[beta_len : 2 * beta_len]

        def fit(x):
            return Q(beta @ np.array([x**i for i in range(beta_len)]), y.units)

        return fit


# %%
def get_plt_index_width(years_to_calculate):
    return (
        (0, 500),
        (10, 500),
        (210, 500),
        (500, 900),
        (1500, 500),
        (1500, 500),
        (len(years_to_calculate) - 100, 50),
        (len(years_to_calculate) - 50, 50),
        (len(years_to_calculate) - 1, 50),
    )


# %%
def get_weights(x, target_year, window_width):
    z = x - target_year
    # Linear weights don't make sense to me
    # weights = np.max(
    #     [
    #         np.repeat(0.0001, len(z)),
    #         # 1 - np.abs(z.m) / np.max(np.abs(z.m))
    #         (1 - np.abs(z) / np.max(np.abs(z))).to("dimensionless").m,
    #     ],
    #     axis=0,
    # )
    return np.max(
        [
            np.repeat(0.0001, len(z)),
            np.exp(-np.abs(z) / window_width),
            # 1 - np.abs(z.m) / np.max(np.abs(z.m))
            # (1 - np.abs(z) / np.max(np.abs(z))).to("dimensionless").m,
        ],
        axis=0,
    )


# %%
def plot_regression_fit(
    point_selector,
    selected_points_x,
    selected_points_y,
    quantile_regression_fit,
    target_year,
    ax,
    xlim_width,
):
    ax.scatter(
        point_selector.x_pool.m, point_selector.y_pool.m, alpha=0.6, s=100, zorder=2
    )
    print(f"select points size={selected_points[0].shape}")

    ax.scatter(
        selected_points_x.m,
        selected_points_y.m,
        alpha=0.9,
        s=60,
        marker="x",
        zorder=3,
    )
    ax.scatter(
        target_year,
        quantile_regression_fitted(target_year.m),
        marker="o",
        s=120,
        alpha=0.4,
        zorder=2.2,
    )
    sorted_selected_fine = np.linspace(
        selected_points_x.m.min(), selected_points_x.m.max(), 300
    )
    ax.plot(
        sorted_selected_fine,
        quantile_regression_fitted(sorted_selected_fine),
        alpha=0.9,
        zorder=4.0,
        color="tab:cyan",
    )

    ax.axvline(target_year.m)
    ax.axvline(target_year.m - point_selector.window_width.m, color="k")
    ax.axvline(target_year.m + point_selector.window_width.m, color="k")
    ax.set_xlim([target_year.m - xlim_width, target_year.m + xlim_width])
    ax.set_ylim([0.9 * selected_points[1].min().m, 1.1 * selected_points[1].max().m])

    return ax


# %%
def plot_ensemble_fit(
    axes,
    point_selector,
    years_to_calculate,
    smoothed_all_samples,
    smoothed_all_samples_median,
):
    axes[0].scatter(
        point_selector.x_pool.m,
        point_selector.y_pool.m,
        alpha=0.6,
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
        point_selector.x_pool.m,
        point_selector.y_pool.m,
        alpha=0.6,
        s=20,
        zorder=2,
    )

    axes[1].plot(
        years_to_calculate.m,
        smoothed_all_samples_median.m,
        color="tab:green",
        alpha=0.4,
        linewidth=3,
    )

    axes[1].set_xlabel("year")
    axes[1].set_ylabel(smoothed_all_samples_median.units)


# %%
n_draws = 250
# n_draws = 30
# n_draws = 3

for gas, gdf in full_df.sort_values(by="time").groupby("gas"):
    # if gas != "n2o":
    #     continue

    gas_unit = gdf["unit"].unique()
    if len(gas_unit) > 1:
        raise ValueError(f"More than one unit found for {gas=}, {gas_unit=}")
    gas_unit = gas_unit[0]

    x_raw = Q(gdf["time"].values, "yr")
    y_raw = Q(gdf["value"].values, gas_unit)

    noise_adder = noise_adders[gas]
    print(noise_adder)
    print(f"{noise_adder.percentage_time_error * Q(2000, 'yr')=}")

    years_to_calculate = Q(
        np.arange(
            np.floor(x_raw.to("yr").m.min()),
            np.ceil(x_raw.to("yr").m.max()) + 1,
        ),
        "yr",
    )

    plt_index_width = get_plt_index_width(years_to_calculate)
    plt_indexes = [v[0] for v in plt_index_width]
    xlim_widths = {v[0]: v[1] for v in plt_index_width}

    regressor = WeightedQuantileRegression(quantile=0.5, model_order=3)
    print(regressor)
    # regressor = WeightedQuantileRegression(quantile=0.5, model_order=1)

    smoothed_all_samples = []
    for i in tqdman.tqdm(range(n_draws)):
        x_plus_noise, y_plus_noise = noise_adder.add_noise(
            x=x_raw,
            y=y_raw,
        )
        x_plus_noise_sorted_idx = np.argsort(x_plus_noise)
        x_plus_noise_sorted = x_plus_noise[x_plus_noise_sorted_idx]
        y_plus_noise_sorted = y_plus_noise[x_plus_noise_sorted_idx]

        point_selector = PointSelector(
            **point_selector_settings[gas],
            x_pool=x_plus_noise_sorted,
            y_pool=y_plus_noise_sorted,
        )

        if i < 1:
            print(f"{point_selector.window_width=}")
            print(f"{point_selector.minimum_data_points_either_side=}")
            print(f"{point_selector.maximum_data_points_either_side=}")

            fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(8, 6))
            plot_noise_addition(
                x_raw=x_raw,
                y_raw=y_raw,
                x_plus_noise=x_plus_noise,
                y_plus_noise=y_plus_noise,
                axes=axes,
            )
            fig.suptitle(gas)
            plt.tight_layout()
            plt.show()

        smoothed = []

        quantile_regression_success = True
        for ii, target_year in enumerate(years_to_calculate):
            selected_points = point_selector.get_points(target_year)
            selected_points_x = selected_points[0]
            selected_points_y = selected_points[1]

            weights = get_weights(
                selected_points_x, target_year, window_width=point_selector.window_width
            )

            quantile_regression_fitted = regressor.fit(
                selected_points_x, selected_points_y, weights=weights
            )
            if quantile_regression_fitted is None:
                print("Quantile regression failed, re-drawing")
                quantile_regression_success = False
                break

            if ii in plt_indexes and i < 1:
                fig, ax = plt.subplots()
                plot_regression_fit(
                    point_selector=point_selector,
                    selected_points_x=selected_points_x,
                    selected_points_y=selected_points_y,
                    quantile_regression_fit=quantile_regression_fitted,
                    target_year=target_year,
                    ax=ax,
                    xlim_width=xlim_widths[ii],
                )
                plt.show()

            smoothed.append(quantile_regression_fitted(target_year.m))

        if not quantile_regression_success:
            continue

        smoothed_all_samples.append(np.hstack(smoothed))

    smoothed_all_samples = np.vstack(smoothed_all_samples)

    smoothed_all_samples_median = np.median(smoothed_all_samples, axis=0)

    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
    plot_ensemble_fit(
        axes=axes,
        point_selector=point_selector,
        years_to_calculate=years_to_calculate,
        smoothed_all_samples=smoothed_all_samples,
        smoothed_all_samples_median=smoothed_all_samples_median,
    )
    fig.suptitle(gas)
    plt.show()
    # break
