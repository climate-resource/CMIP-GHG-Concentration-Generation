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

# %%
# TODO: move notebook to its own step

# %%
import matplotlib.pyplot as plt
import pandas as pd
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
        # Different to choice to Nicolai here, time error scales with age,
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
# Super unclear what the right choice is here.
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
for gas, gdf in full_df.sort_values(by="time").groupby("gas"):
    gas_unit = gdf["unit"].unique()
    if len(gas_unit) > 1:
        raise ValueError(f"More than one unit found for {gas=}, {gas_unit=}")
    gas_unit = gas_unit[0]

    x_raw = Q(gdf["time"].values, "yr")
    y_raw = Q(gdf["value"].values, gas_unit)

    noise_adder = noise_adders[gas]

    x_plus_noise, y_plus_noise = noise_adder.add_noise(
        x=x_raw,
        y=y_raw,
    )
    x_plus_noise_sorted_idx = np.argsort(x_plus_noise)
    x_plus_noise_sorted = x_plus_noise[x_plus_noise_sorted_idx]
    y_plus_noise_sorted = y_plus_noise[x_plus_noise_sorted_idx]

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
        # TODO: sort on entry
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
point_selector = PointSelector(
    **point_selector_settings[gas],
    x_pool=x_plus_noise_sorted,
    y_pool=y_plus_noise_sorted,
)
point_selector.minimum_data_points_either_side


# %%
def cubic_polynomial(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


# %%
for y, xlim_width in (
    (years_to_calculate[0], 500),
    (years_to_calculate[210], 500),
    (years_to_calculate[500], 900),
    (years_to_calculate[1500], 500),
    (years_to_calculate[-1], 25),
):
    fig, ax = plt.subplots()
    ax.scatter(
        point_selector.x_pool.m, point_selector.y_pool.m, alpha=0.6, s=100, zorder=2
    )
    selected_points = point_selector.get_points(y)
    popt, pcov = scipy.optimize.curve_fit(
        cubic_polynomial, selected_points[0].m, selected_points[1].m
    )
    fitted = cubic_polynomial(selected_points[0].m, *popt)
    print(f"select points size={selected_points[0].shape}")
    ax.scatter(
        selected_points[0].m,
        selected_points[1].m,
        alpha=0.9,
        s=60,
        marker="x",
        zorder=3,
    )
    ax.scatter(selected_points[0].m, fitted, alpha=0.8, s=60, marker="+", zorder=3)
    ax.plot(
        years_to_calculate,
        cubic_polynomial(years_to_calculate.m, *popt),
        color="orange",
        alpha=0.6,
        zorder=2,
    )
    ax.axvline(y.m)
    ax.axvline(y.m - point_selector.window_width.m, color="k")
    ax.axvline(y.m + point_selector.window_width.m, color="k")
    ax.set_xlim([y.m - xlim_width, y.m + xlim_width])
    plt.show()
    # break

# %%
