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

from .calculate_c4f10_like_monthly_fifteen_degree_pieces import (
    CalculateC4F10LikeMonthlyFifteenDegreePieces,
)
from .calculate_c8f18_like_monthly_fifteen_degree_pieces import (
    CalculateC8F18LikeMonthlyFifteenDegreePieces,
)
from .calculate_ch4_monthly_15_degree import (
    CalculateCH4MonthlyFifteenDegreePieces,
)
from .calculate_co2_monthly_15_degree import CalculateCO2MonthlyFifteenDegreePieces
from .calculate_n2o_monthly_15_degree import CalculateN2OMonthlyFifteenDegreePieces
from .calculate_sf6_like_monthly_15_degree import (
    CalculateSF6LikeMonthlyFifteenDegreePieces,
)
from .compile_historical_emissions import CompileHistoricalEmissionsConfig
from .crunch_equivalent_species import EquivalentSpeciesCrunchingConfig
from .crunch_grid import GridCrunchingConfig
from .plot_input_data_overviews import PlotInputDataOverviewsConfig
from .process_noaa_hats_data import ProcessNOAAHATSDataConfig
from .process_noaa_in_situ_data import ProcessNOAAInSituDataConfig
from .process_noaa_surface_flask_data import ProcessNOAASurfaceFlaskDataConfig
from .retrieve_and_extract_agage import RetrieveExtractAGAGEDataConfig
from .retrieve_and_extract_ale import RetrieveExtractALEDataConfig
from .retrieve_and_extract_gage import RetrieveExtractGAGEDataConfig
from .retrieve_and_extract_noaa import RetrieveExtractNOAADataConfig
from .retrieve_and_process_epica_data import RetrieveProcessEPICAConfig
from .retrieve_and_process_law_dome import RetrieveProcessLawDomeConfig
from .retrieve_and_process_neem_data import RetrieveProcessNEEMConfig
from .retrieve_and_process_scripps_data import RetrieveProcessScrippsConfig
from .retrieve_and_process_velders_et_al_2022_data import RetrieveExtractVeldersEtal2022Data
from .retrieve_and_process_western_et_al_2024_data import RetrieveExtractWesternEtal2024Data
from .retrieve_and_process_wmo_2022_ozone_assessment_ch7_data import (
    RetrieveProcessWMO2022OzoneAssessmentCh7Config,
)
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

    doi: str
    """
    DOI to write into each ESGF-ready file
    """

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
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use with the retrieve step"""

    retrieve_and_extract_noaa_data: list[RetrieveExtractNOAADataConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and extracting NOAA data"""

    process_noaa_surface_flask_data: list[ProcessNOAASurfaceFlaskDataConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for processing NOAA surface flask data"""

    process_noaa_in_situ_data: list[ProcessNOAAInSituDataConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for processing NOAA in-situ data"""

    process_noaa_hats_data: list[ProcessNOAAHATSDataConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for processing NOAA HATS data"""

    retrieve_and_extract_agage_data: list[RetrieveExtractAGAGEDataConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and extracting AGAGE data"""

    retrieve_and_extract_gage_data: list[RetrieveExtractGAGEDataConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and extracting GAGE data"""

    retrieve_and_extract_ale_data: list[RetrieveExtractALEDataConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and extracting ALE data"""

    retrieve_and_process_law_dome_data: list[RetrieveProcessLawDomeConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and processing Law Dome data"""

    retrieve_and_process_scripps_data: list[RetrieveProcessScrippsConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and processing Scripps data"""

    retrieve_and_process_epica_data: list[RetrieveProcessEPICAConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and processing EPICA data"""

    retrieve_and_process_neem_data: list[RetrieveProcessNEEMConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and processing NEEM data"""

    retrieve_and_process_wmo_2022_ozone_assessment_ch7_data: list[
        RetrieveProcessWMO2022OzoneAssessmentCh7Config
    ] = field(validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)])
    """Configurations to use for retrieving and processing WMO 2022 ozone assessment ch. 7 data"""

    retrieve_and_process_western_et_al_2024_data: list[RetrieveExtractWesternEtal2024Data] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and processing Western et al. (2024) data"""

    retrieve_and_process_velders_et_al_2022_data: list[RetrieveExtractVeldersEtal2022Data] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for retrieving and processing Velders et al. (2022) data"""

    plot_input_data_overviews: list[PlotInputDataOverviewsConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for the plotting step"""

    compile_historical_emissions: list[CompileHistoricalEmissionsConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for the compilation of historical emissions data"""

    smooth_law_dome_data: list[SmoothLawDomeDataConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for the smoothing of Law Dome data step"""

    calculate_co2_monthly_fifteen_degree_pieces: list[CalculateCO2MonthlyFifteenDegreePieces] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for calculating the 15 degree, monthly data for CO2"""

    calculate_ch4_monthly_fifteen_degree_pieces: list[CalculateCH4MonthlyFifteenDegreePieces] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for calculating the 15 degree, monthly data for CH4"""

    calculate_n2o_monthly_fifteen_degree_pieces: list[CalculateN2OMonthlyFifteenDegreePieces] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use for calculating the 15 degree, monthly data for N2O"""

    calculate_sf6_like_monthly_fifteen_degree_pieces: list[CalculateSF6LikeMonthlyFifteenDegreePieces] = (
        field(validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)])
    )
    """Configurations to use for calculating the 15 degree, monthly data for gases we handle like SF6"""

    calculate_c4f10_like_monthly_fifteen_degree_pieces: list[CalculateC4F10LikeMonthlyFifteenDegreePieces] = (
        field(validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)])
    )
    """Configurations to use for calculating the 15 degree, monthly data for gases we handle like C4F10"""

    calculate_c8f18_like_monthly_fifteen_degree_pieces: list[CalculateC8F18LikeMonthlyFifteenDegreePieces] = (
        field(validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)])
    )
    """Configurations to use for calculating the 15 degree, monthly data for gases we handle like C8F18"""

    crunch_grids: list[GridCrunchingConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use with the grid crunching step"""

    crunch_equivalent_species: list[EquivalentSpeciesCrunchingConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use with the equivalent species crunching step"""

    write_input4mips: list[WriteInput4MIPsConfig] = field(
        validator=[make_attrs_validator_compatible_single_input(assert_step_config_ids_are_unique)]
    )
    """Configurations to use with the write input4MIPs step"""


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
