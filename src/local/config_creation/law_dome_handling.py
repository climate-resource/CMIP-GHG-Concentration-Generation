"""
Law Dome data handling config creation
"""

from __future__ import annotations

import pint

from local.config.retrieve_and_process_law_dome import RetrieveProcessLawDomeConfig
from local.config.smooth_law_dome_data import (
    PointSelectorSettings,
    SmoothLawDomeDataConfig,
)
from local.noise_addition import NoiseAdderPercentageXNoise

RETRIEVE_AND_PROCESS_LAW_DOME_STEPS = [
    RetrieveProcessLawDomeConfig(
        step_config_id="only",
        doi="https://doi.org/10.25919/5bfe29ff807fb",
        raw_dir="data/raw/law_dome",
        processed_data_with_loc_file="data/interim/law_dome/law_dome_with_location.csv",
        files_md5_sum={
            "data/raw/law_dome/data/Law_Dome_GHG_2000years.xlsx": "f7dd24e36565b2e213b20f90c88c990e"
        },
    )
]


def create_smooth_law_dome_data_config(
    gases: tuple[str, ...], n_draws: int
) -> list[SmoothLawDomeDataConfig]:
    """
    Create configuration for smoothing Law Dome data

    Parameters
    ----------
    gases
        Gases for which to create the config

    n_draws
        Number of draws to use when smoothing the Law Dome data

    Returns
    -------
        Configuration for smoothing Law Dome data
    """
    Q = pint.get_application_registry().Quantity

    res = []

    interim_dir = "data/interim/law_dome"

    for gas in gases:
        if gas == "ch4":
            res.append(
                SmoothLawDomeDataConfig(
                    step_config_id=gas,
                    gas=gas,
                    n_draws=n_draws,
                    smoothed_draws_file=f"{interim_dir}/law-dome_{gas}_smoothed_all-draws.csv",
                    smoothed_median_file=f"{interim_dir}/law-dome_{gas}_smoothed_median.csv",
                    noise_adder=NoiseAdderPercentageXNoise(
                        x_ref=Q(2024, "yr"),
                        x_relative_random_error=Q(50, "yr") / Q(2000, "yr"),
                        y_random_error=Q(3, "ppb"),
                    ),
                    point_selector_settings=PointSelectorSettings(
                        window_width=Q(100, "yr"),
                        minimum_data_points_either_side=4,
                        maximum_data_points_either_side=10,
                    ),
                )
            )

        elif gas == "co2":
            res.append(
                SmoothLawDomeDataConfig(
                    step_config_id=gas,
                    gas=gas,
                    n_draws=n_draws,
                    smoothed_draws_file=f"{interim_dir}/law-dome_{gas}_smoothed_all-draws.csv",
                    smoothed_median_file=f"{interim_dir}/law-dome_{gas}_smoothed_median.csv",
                    noise_adder=NoiseAdderPercentageXNoise(
                        x_ref=Q(2024, "yr"),
                        x_relative_random_error=Q(60, "yr") / Q(2000, "yr"),
                        y_random_error=Q(2, "ppm"),
                    ),
                    point_selector_settings=PointSelectorSettings(
                        window_width=Q(120, "yr"),
                        minimum_data_points_either_side=7,
                        maximum_data_points_either_side=25,
                    ),
                )
            )
        else:
            raise NotImplementedError(gas)

    return res
