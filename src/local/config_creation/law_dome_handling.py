"""
Law Dome data handling config creation
"""

from __future__ import annotations

from pathlib import Path

import pint

from local.config.retrieve_and_process_law_dome import RetrieveProcessLawDomeConfig
from local.config.smooth_law_dome_data import (
    PointSelectorSettings,
    SmoothLawDomeDataConfig,
)
from local.dependencies import SourceInfo
from local.noise_addition import NoiseAdderPercentageXNoise

SOURCE_INFO_SHORT_NAME = "Law Dome ice core"
SOURCE_INFO = SourceInfo(
    short_name=SOURCE_INFO_SHORT_NAME,
    licence="CC BY 4.0",
    reference=(
        "Rubino, Mauro; Etheridge, David; ... Van Ommen, Tas; & Smith, Andrew (2019): "
        "Law Dome Ice Core 2000-Year CO2, CH4, N2O and d13C-CO2. v3. CSIRO. "
        "Data Collection. https://doi.org/10.25919/5bfe29ff807fb"
    ),
    doi="https://doi.org/10.25919/5bfe29ff807fb",
    url="https://doi.org/10.25919/5bfe29ff807fb",
    resource_type="dataset",
)

RETRIEVE_AND_PROCESS_LAW_DOME_STEPS = [
    RetrieveProcessLawDomeConfig(
        step_config_id="only",
        doi="https://doi.org/10.25919/5bfe29ff807fb",
        raw_dir=Path("data/raw/law_dome"),
        processed_data_with_loc_file=Path("data/interim/law_dome/law_dome_with_location.csv"),
        files_md5_sum={
            Path("data/raw/law_dome/data/Law_Dome_GHG_2000years.xlsx"): "f7dd24e36565b2e213b20f90c88c990e"
        },
        source_info=SOURCE_INFO,
    )
]


def create_smooth_law_dome_data_config(gases: tuple[str, ...], n_draws: int) -> list[SmoothLawDomeDataConfig]:
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
    Q = pint.get_application_registry().Quantity  # type: ignore

    res = []

    interim_dir = Path("data/interim/law_dome")

    for gas in gases:
        if gas == "ch4":
            res.append(
                SmoothLawDomeDataConfig(
                    step_config_id=gas,
                    gas=gas,
                    n_draws=n_draws,
                    source_info_short_name=SOURCE_INFO_SHORT_NAME,
                    smoothed_draws_file=interim_dir / f"law-dome_{gas}_smoothed_all-draws.csv",
                    smoothed_median_file=interim_dir / f"law-dome_{gas}_smoothed_median.csv",
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
                    source_info_short_name=SOURCE_INFO_SHORT_NAME,
                    smoothed_draws_file=interim_dir / f"law-dome_{gas}_smoothed_all-draws.csv",
                    smoothed_median_file=interim_dir / f"law-dome_{gas}_smoothed_median.csv",
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

        elif gas == "n2o":
            res.append(
                SmoothLawDomeDataConfig(
                    step_config_id=gas,
                    gas=gas,
                    n_draws=n_draws,
                    source_info_short_name=SOURCE_INFO_SHORT_NAME,
                    smoothed_draws_file=interim_dir / f"law-dome_{gas}_smoothed_all-draws.csv",
                    smoothed_median_file=interim_dir / f"law-dome_{gas}_smoothed_median.csv",
                    noise_adder=NoiseAdderPercentageXNoise(
                        x_ref=Q(2024, "yr"),
                        x_relative_random_error=Q(90, "yr") / Q(2000, "yr"),
                        y_random_error=Q(3, "ppb"),
                    ),
                    point_selector_settings=PointSelectorSettings(
                        window_width=Q(300, "yr"),
                        minimum_data_points_either_side=7,
                        maximum_data_points_either_side=15,
                    ),
                )
            )

        else:
            raise NotImplementedError(gas)

    return res
