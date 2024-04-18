"""
Base configuration classes
"""

from __future__ import annotations

from pathlib import Path

from attrs import field, frozen
from pydoit_nb.attrs_helpers import make_attrs_validator_compatible_single_input
from pydoit_nb.config_helpers import (
    assert_path_exists,
    assert_path_is_absolute,
    assert_path_is_subdirectory_of_root_dir_output,
    assert_step_config_ids_are_unique,
)

from .calculate_n2o_monthly_15_degree import CalculateN2OMonthly15DegreeConfig
from .grid import GridConfig
from .gridded_data_processing import GriddedDataProcessingConfig
from .plot_input_data_overviews import PlotInputDataOverviewsConfig
from .process_noaa_in_situ_data import ProcessNOAAInSituDataConfig
from .process_noaa_surface_flask_data import ProcessNOAASurfaceFlaskDataConfig
from .quick_crunch import QuickCrunchConfig
from .retrieve_and_extract_agage import RetrieveExtractAGAGEDataConfig
from .retrieve_and_extract_ale import RetrieveExtractALEDataConfig
from .retrieve_and_extract_gage import RetrieveExtractGAGEDataConfig
from .retrieve_and_extract_noaa import RetrieveExtractNOAADataConfig
from .retrieve_and_process_epica_data import RetrieveProcessEPICAConfig
from .retrieve_and_process_law_dome import RetrieveProcessLawDomeConfig
from .retrieve_and_process_neem_data import RetrieveProcessNEEMConfig
from .retrieve_and_process_scripps_data import RetrieveProcessScrippsConfig
from .retrieve_misc_data import RetrieveMiscDataConfig
from .smooth_law_dome_data import SmoothLawDomeDataConfig
from .write_input4mips import WriteInput4MIPsConfig


@frozen
class Config:
    """
    Configuration class used across all notebooks

    This is the key communication class between our notebooks. It must be used
    for passing all parameters into the notebooks via papermill.
    """

    name: str
    """Name of the configuration"""

    version: str
    """Version ID for this configuration"""
    # TODO: add validation that this matches semantic versioning

    base_seed: int
    """
    Base value to use for setting random seeds

    This value is not used directly to avoid accidental correlations between draws.
    Instead, it is a base to which offsets can be added to ensure reproducibility
    while avoiding spurious correlations.
    """

    ci: bool
    """
    Is this configuration for a CI run?

    We use this to help us create a short-cut path that can reasonably be run
    as part of our CI workflow.
    """

    retrieve_misc_data: list[RetrieveMiscDataConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use with the retrieve step"""

    retrieve_and_extract_noaa_data: list[RetrieveExtractNOAADataConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for retrieving and extracting NOAA data"""

    process_noaa_surface_flask_data: list[ProcessNOAASurfaceFlaskDataConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for processing NOAA surface flask data"""

    process_noaa_in_situ_data: list[ProcessNOAAInSituDataConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for processing NOAA in-situ data"""

    retrieve_and_extract_agage_data: list[RetrieveExtractAGAGEDataConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for retrieving and extracting AGAGE data"""

    retrieve_and_extract_gage_data: list[RetrieveExtractGAGEDataConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for retrieving and extracting GAGE data"""

    retrieve_and_extract_ale_data: list[RetrieveExtractALEDataConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for retrieving and extracting ALE data"""

    retrieve_and_process_law_dome_data: list[RetrieveProcessLawDomeConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for retrieving and processing Law Dome data"""

    retrieve_and_process_scripps_data: list[RetrieveProcessScrippsConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for retrieving and processing Scripps data"""

    retrieve_and_process_epica_data: list[RetrieveProcessEPICAConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for retrieving and processing EPICA data"""

    retrieve_and_process_neem_data: list[RetrieveProcessNEEMConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for retrieving and processing NEEM data"""

    plot_input_data_overviews: list[PlotInputDataOverviewsConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for the plotting step"""

    smooth_law_dome_data: list[SmoothLawDomeDataConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for the smoothing of Law Dome data step"""

    calculate_n2o_monthly_15_degree: list[CalculateN2OMonthly15DegreeConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use for calculating the 15 degree, monthly data for N2O"""

    grid: list[GridConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use with the grid step"""

    gridded_data_processing: list[GriddedDataProcessingConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use with the gridded data processing step"""

    write_input4mips: list[WriteInput4MIPsConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use with the write input4MIPs step"""

    quick_crunch: list[QuickCrunchConfig] = field(
        validator=[
            make_attrs_validator_compatible_single_input(
                assert_step_config_ids_are_unique
            )
        ]
    )
    """Configurations to use with the quick crunch step"""


@frozen
class ConfigBundle:
    """
    Configuration bundle

    Has all key components in one place
    """

    run_id: str
    """ID for the run"""

    config_hydrated: Config
    """Hydrated config"""

    config_hydrated_path: Path
    """Path in/from which to read/write ``config_hydrated``"""

    root_dir_output: Path = field(
        validator=[
            make_attrs_validator_compatible_single_input(assert_path_is_absolute),
            make_attrs_validator_compatible_single_input(assert_path_exists),
        ]
    )

    """Root output directory"""

    root_dir_output_run: Path = field(
        validator=[
            make_attrs_validator_compatible_single_input(assert_path_is_absolute),
            make_attrs_validator_compatible_single_input(assert_path_exists),
            assert_path_is_subdirectory_of_root_dir_output,
        ]
    )
    """Root output directory for this run"""
