"""
Write our configuration files
"""

# Have to the pint registry before doing other imports, hence funny order
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import openscm_units
import pint
from pydoit_nb.config_handling import insert_path_prefix

pint.set_application_registry(openscm_units.unit_registry)


import local
from local.config import Config, converter_yaml
from local.config.plot_input_data_overviews import PlotInputDataOverviewsConfig
from local.config_creation.agage_handling import create_agage_handling_config
from local.config_creation.ale_handling import RETRIEVE_AND_EXTRACT_ALE_STEPS
from local.config_creation.compile_historical_emissions import (
    COMPILE_HISTORICAL_EMISSIONS_STEPS,
)
from local.config_creation.crunch_equivalent_species import (
    create_crunch_equivalent_species_config,
)
from local.config_creation.crunch_grids import create_crunch_grids_config
from local.config_creation.epica_handling import RETRIEVE_AND_PROCESS_EPICA_STEPS
from local.config_creation.gage_handling import RETRIEVE_AND_EXTRACT_GAGE_STEPS
from local.config_creation.law_dome_handling import (
    RETRIEVE_AND_PROCESS_LAW_DOME_STEPS,
    create_smooth_law_dome_data_config,
)
from local.config_creation.monthly_fifteen_degree_pieces import (
    create_monthly_fifteen_degree_pieces_configs,
)
from local.config_creation.neem_handling import RETRIEVE_AND_PROCESS_NEEM_STEPS
from local.config_creation.noaa_handling import create_noaa_handling_config
from local.config_creation.retrieve_misc_data import RETRIEVE_MISC_DATA_STEPS
from local.config_creation.write_input4mips import create_write_input4mips_config


def create_dev_config() -> Config:
    """
    Create our (relative) dev config
    """
    equivalent_species_to_write = ("cfc11eq", "cfc12eq", "hfc134aeq")

    gases_to_write = (
        "co2",
        "ch4",
        "n2o",
        "c2f6",
        "c3f8",
        "c4f10",
        "c5f12",
        "c6f14",
        "c7f16",
        "c8f18",
        "cc4f8",
        "ccl4",
        "cf4",
        "cfc11",
        "cfc113",
        "cfc114",
        "cfc115",
        "cfc12",
        "ch2cl2",
        "ch3br",
        "ch3ccl3",
        "ch3cl",
        "chcl3",
        "halon1211",
        "halon1301",
        "halon2402",
        "hcfc141b",
        "hcfc142b",
        "hcfc22",
        "hfc125",
        "hfc134a",
        "hfc143a",
        "hfc152a",
        "hfc227ea",
        "hfc23",
        "hfc236fa",
        "hfc245fa",
        "hfc32",
        "hfc365mfc",
        "hfc4310mee",
        "nf3",
        "sf6",
        "so2f2",
    )
    # TODO: there shouldn't be this many gasses in here
    gases_long_poleward_extension = (
        "c2f6",
        "c3f8",
        "cc4f8",
        "cf4",
        "cfc114",
        "cfc115",
        "ch3ccl3",
        "chcl3",
        "halon1301",
        "hcfc141b",
        "hfc125",
        "hfc143a",
        "hfc152a",
        "hfc23",
        "hfc236fa",
        "hfc245fa",
        "hfc32",
        "hfc4310mee",
        "nf3",
        "so2f2",
    )
    # TODO: add this to CI too
    gases_drop_obs_data_years_before_inclusive = {
        "cf4": 2007,
        "c2f6": 2007,
        "cfc115": 2007,
        "ch2cl2": 2013,
        "halon2402": 2009,
        "so2f2": 2007,
    }
    gases_drop_obs_data_years_after_inclusive = {}

    start_year = 1
    end_year = 2022

    noaa_handling_steps = create_noaa_handling_config(
        data_sources=(
            ("co2", "in-situ"),
            ("co2", "surface-flask"),
            ("ch4", "in-situ"),
            ("ch4", "surface-flask"),
            ("n2o", "hats"),
            # # Don't use N2O surface flask to avoid double counting
            # ("n2o", "surface-flask"),
            ("c2f6", "hats"),
            ("ccl4", "hats"),
            ("cf4", "hats"),
            ("cfc11", "hats"),
            ("cfc113", "hats"),
            ("cfc12", "hats"),
            ("ch2cl2", "hats"),
            ("ch3br", "hats"),
            ("ch3ccl3", "hats"),
            ("ch3cl", "hats"),
            ("halon1211", "hats"),
            ("halon1301", "hats"),
            ("halon2402", "hats"),
            ("hcfc141b", "hats"),
            ("hcfc142b", "hats"),
            ("hcfc22", "hats"),
            ("hfc125", "hats"),
            ("hfc134a", "hats"),
            ("hfc143a", "hats"),
            ("hfc152a", "hats"),
            ("hfc227ea", "hats"),
            ("hfc236fa", "hats"),
            ("hfc32", "hats"),
            ("hfc365mfc", "hats"),
            ("nf3", "hats"),
            ("sf6", "hats"),
            # # Don't use SF6 surface flask to avoid double counting
            # ("sf6", "surface-flask"),
            ("so2f2", "hats"),
        )
    )

    retrieve_and_extract_agage_data = create_agage_handling_config(
        data_sources=(
            ("ch4", "gc-md", "monthly"),
            ("n2o", "gc-md", "monthly"),
            ("sf6", "gc-md", "monthly"),
            ("sf6", "gc-ms-medusa", "monthly"),
            ("c2f6", "gc-ms-medusa", "monthly"),
            ("c3f8", "gc-ms-medusa", "monthly"),
            ("c3f8", "gc-ms", "monthly"),
            ("cc4f8", "gc-ms-medusa", "monthly"),
            ("cc4f8", "gc-ms", "monthly"),
            ("ccl4", "gc-md", "monthly"),
            ("ccl4", "gc-ms-medusa", "monthly"),
            ("ccl4", "gc-ms", "monthly"),
            ("cf4", "gc-ms-medusa", "monthly"),
            ("cfc11", "gc-md", "monthly"),
            ("cfc11", "gc-ms-medusa", "monthly"),
            ("cfc11", "gc-ms", "monthly"),
            ("cfc113", "gc-md", "monthly"),
            ("cfc113", "gc-ms-medusa", "monthly"),
            ("cfc114", "gc-ms", "monthly"),
            ("cfc114", "gc-ms-medusa", "monthly"),
            ("cfc115", "gc-ms", "monthly"),
            ("cfc115", "gc-ms-medusa", "monthly"),
            ("cfc12", "gc-md", "monthly"),
            ("cfc12", "gc-ms-medusa", "monthly"),
            ("cfc12", "gc-ms", "monthly"),
            ("ch2cl2", "gc-ms-medusa", "monthly"),
            ("ch2cl2", "gc-ms", "monthly"),
            ("ch3br", "gc-ms-medusa", "monthly"),
            ("ch3br", "gc-ms", "monthly"),
            ("ch3ccl3", "gc-md", "monthly"),
            ("ch3ccl3", "gc-ms-medusa", "monthly"),
            ("ch3ccl3", "gc-ms", "monthly"),
            ("ch3cl", "gc-ms-medusa", "monthly"),
            ("ch3cl", "gc-ms", "monthly"),
            ("chcl3", "gc-md", "monthly"),
            ("chcl3", "gc-ms-medusa", "monthly"),
            ("chcl3", "gc-ms", "monthly"),
            ("halon1211", "gc-ms-medusa", "monthly"),
            ("halon1211", "gc-ms", "monthly"),
            ("halon1301", "gc-ms-medusa", "monthly"),
            ("halon1301", "gc-ms", "monthly"),
            ("halon2402", "gc-ms-medusa", "monthly"),
            ("hcfc141b", "gc-ms", "monthly"),
            ("hcfc141b", "gc-ms-medusa", "monthly"),
            ("hcfc142b", "gc-ms", "monthly"),
            ("hcfc142b", "gc-ms-medusa", "monthly"),
            ("hcfc22", "gc-ms", "monthly"),
            ("hcfc22", "gc-ms-medusa", "monthly"),
            ("hfc125", "gc-ms-medusa", "monthly"),
            ("hfc125", "gc-ms", "monthly"),
            ("hfc134a", "gc-ms-medusa", "monthly"),
            ("hfc134a", "gc-ms", "monthly"),
            ("hfc143a", "gc-ms-medusa", "monthly"),
            ("hfc143a", "gc-ms", "monthly"),
            ("hfc152a", "gc-ms-medusa", "monthly"),
            ("hfc152a", "gc-ms", "monthly"),
            ("hfc227ea", "gc-ms-medusa", "monthly"),
            ("hfc227ea", "gc-ms", "monthly"),
            ("hfc23", "gc-ms-medusa", "monthly"),
            ("hfc236fa", "gc-ms-medusa", "monthly"),
            ("hfc236fa", "gc-ms", "monthly"),
            ("hfc245fa", "gc-ms-medusa", "monthly"),
            ("hfc245fa", "gc-ms", "monthly"),
            ("hfc32", "gc-ms-medusa", "monthly"),
            ("hfc32", "gc-ms", "monthly"),
            ("hfc365mfc", "gc-ms-medusa", "monthly"),
            ("hfc365mfc", "gc-ms", "monthly"),
            ("hfc4310mee", "gc-ms-medusa", "monthly"),
            ("nf3", "gc-ms-medusa", "monthly"),
            ("so2f2", "gc-ms-medusa", "monthly"),
            ("so2f2", "gc-ms", "monthly"),
        )
    )

    smooth_law_dome_data = create_smooth_law_dome_data_config(
        gases=("co2", "ch4", "n2o"), n_draws=250
    )

    monthly_fifteen_degree_pieces_configs = create_monthly_fifteen_degree_pieces_configs(
        gases=gases_to_write,
        gases_long_poleward_extension=gases_long_poleward_extension,
        gases_drop_obs_data_years_before_inclusive=gases_drop_obs_data_years_before_inclusive,
        gases_drop_obs_data_years_after_inclusive=gases_drop_obs_data_years_after_inclusive,
    )

    return Config(
        name="dev",
        version=f"{local.__version__}-dev",
        doi="ci-hence-no-valid-doi",
        base_seed=20240428,
        ci=False,
        retrieve_misc_data=RETRIEVE_MISC_DATA_STEPS,
        **noaa_handling_steps,
        retrieve_and_extract_agage_data=retrieve_and_extract_agage_data,
        retrieve_and_extract_gage_data=RETRIEVE_AND_EXTRACT_GAGE_STEPS,
        retrieve_and_extract_ale_data=RETRIEVE_AND_EXTRACT_ALE_STEPS,
        retrieve_and_process_law_dome_data=RETRIEVE_AND_PROCESS_LAW_DOME_STEPS,
        retrieve_and_process_scripps_data=[],
        retrieve_and_process_epica_data=RETRIEVE_AND_PROCESS_EPICA_STEPS,
        retrieve_and_process_neem_data=RETRIEVE_AND_PROCESS_NEEM_STEPS,
        plot_input_data_overviews=[PlotInputDataOverviewsConfig(step_config_id="only")],
        compile_historical_emissions=COMPILE_HISTORICAL_EMISSIONS_STEPS,
        smooth_law_dome_data=smooth_law_dome_data,
        **monthly_fifteen_degree_pieces_configs,
        crunch_grids=create_crunch_grids_config(gases=gases_to_write),
        crunch_equivalent_species=create_crunch_equivalent_species_config(
            gases=equivalent_species_to_write
        ),
        write_input4mips=create_write_input4mips_config(
            # TODO: test the equivalent species in CI too somehow
            gases=(*gases_to_write, *equivalent_species_to_write),
            start_year=start_year,
            end_year=end_year,
            input4mips_cvs_source_id="CR-CMIP-testing",
            input4mips_cvs_cv_source="https://raw.githubusercontent.com/znichollscr/input4MIPs_CVs/refs/heads/cr-cmip-testing/CVs/",
        ),
    )


def create_ci_config() -> Config:
    """
    Create our (relative) CI config
    """
    gases_to_write = ("ch4", "cfc114", "hfc152a")

    gases_long_poleward_extension = (
        "cfc114",
        "hfc152a",
    )

    start_year = 1750
    end_year = 2022

    noaa_handling_steps = create_noaa_handling_config(
        data_sources=(
            ("ch4", "in-situ"),
            ("ch4", "surface-flask"),
            ("hfc152a", "hats"),
        )
    )

    retrieve_and_extract_agage_data = create_agage_handling_config(
        data_sources=(
            ("ch4", "gc-md", "monthly"),
            ("cfc114", "gc-ms", "monthly"),
            ("cfc114", "gc-ms-medusa", "monthly"),
            ("hfc152a", "gc-ms-medusa", "monthly"),
            ("hfc152a", "gc-ms", "monthly"),
        )
    )

    smooth_law_dome_data = create_smooth_law_dome_data_config(
        gases=("ch4",), n_draws=10
    )

    monthly_fifteen_degree_pieces_configs = (
        create_monthly_fifteen_degree_pieces_configs(
            gases=gases_to_write,
            gases_long_poleward_extension=gases_long_poleward_extension,
        )
    )

    return Config(
        name="CI",
        version=f"{local.__version__}-ci",
        doi="ci-hence-no-valid-doi",
        base_seed=20241108,
        ci=True,
        retrieve_misc_data=RETRIEVE_MISC_DATA_STEPS,
        **noaa_handling_steps,
        retrieve_and_extract_agage_data=retrieve_and_extract_agage_data,
        retrieve_and_extract_gage_data=RETRIEVE_AND_EXTRACT_GAGE_STEPS,
        retrieve_and_extract_ale_data=RETRIEVE_AND_EXTRACT_ALE_STEPS,
        retrieve_and_process_law_dome_data=RETRIEVE_AND_PROCESS_LAW_DOME_STEPS,
        retrieve_and_process_scripps_data=[],
        retrieve_and_process_epica_data=RETRIEVE_AND_PROCESS_EPICA_STEPS,
        retrieve_and_process_neem_data=RETRIEVE_AND_PROCESS_NEEM_STEPS,
        plot_input_data_overviews=[PlotInputDataOverviewsConfig(step_config_id="only")],
        compile_historical_emissions=COMPILE_HISTORICAL_EMISSIONS_STEPS,
        smooth_law_dome_data=smooth_law_dome_data,
        **monthly_fifteen_degree_pieces_configs,
        crunch_grids=create_crunch_grids_config(gases=gases_to_write),
        crunch_equivalent_species=[],
        write_input4mips=create_write_input4mips_config(
            gases=gases_to_write,
            start_year=start_year,
            end_year=end_year,
            input4mips_cvs_source_id="CR-CMIP-testing",
            input4mips_cvs_cv_source="https://raw.githubusercontent.com/znichollscr/input4MIPs_CVs/refs/heads/cr-cmip-testing/CVs/",
        ),
    )


def create_ci_nightly_config() -> Config:
    """
    Create our (relative) nightly CI config
    """
    gases_to_write = (
        "co2",
        "ch4",
        "n2o",
        "sf6",
        "cfc11",
        "cfc12",
        "hfc134a",
        "hfc152a",
    )

    gases_long_poleward_extension = ("hfc152a",)
    # Add cfc12eq next, which will require adding all its equivalents
    # - cfc11
    # - cfc113
    # - cfc114
    # - cfc115
    # - cfc12
    # - ccl4
    # - ch2cl2
    # - ch3br
    # - ch3ccl3
    # - ch3cl
    # - chcl3
    # - halon1211
    # - halon1301
    # - halon2402
    # - hcfc141b
    # - hcfc142b
    # - hcfc22
    start_year = 1750
    end_year = 2022

    noaa_handling_steps = create_noaa_handling_config(
        data_sources=(
            ("co2", "in-situ"),
            ("co2", "surface-flask"),
            ("ch4", "in-situ"),
            ("ch4", "surface-flask"),
            ("n2o", "hats"),
            # # Don't use N2O surface flask to avoid double counting
            # ("n2o", "surface-flask"),
            ("sf6", "hats"),
            # # Don't use SF6 surface flask to avoid double counting
            # ("sf6", "surface-flask"),
            ("cfc11", "hats"),
            ("cfc12", "hats"),
            ("hfc134a", "hats"),
            ("hfc152a", "hats"),
        )
    )

    retrieve_and_extract_agage_data = create_agage_handling_config(
        data_sources=(
            ("ch4", "gc-md", "monthly"),
            ("n2o", "gc-md", "monthly"),
            ("sf6", "gc-md", "monthly"),
            ("sf6", "gc-ms-medusa", "monthly"),
            ("cfc11", "gc-md", "monthly"),
            ("cfc11", "gc-ms-medusa", "monthly"),
            ("cfc11", "gc-ms", "monthly"),
            ("cfc12", "gc-md", "monthly"),
            ("cfc12", "gc-ms-medusa", "monthly"),
            ("cfc12", "gc-ms", "monthly"),
            ("hfc134a", "gc-ms-medusa", "monthly"),
            ("hfc134a", "gc-ms", "monthly"),
            ("hfc152a", "gc-ms-medusa", "monthly"),
            ("hfc152a", "gc-ms", "monthly"),
        )
    )

    smooth_law_dome_data = create_smooth_law_dome_data_config(
        gases=("co2", "ch4", "n2o"), n_draws=10
    )

    monthly_fifteen_degree_pieces_configs = (
        create_monthly_fifteen_degree_pieces_configs(
            gases=gases_to_write,
            gases_long_poleward_extension=gases_long_poleward_extension,
        )
    )

    return Config(
        name="CI nightly",
        version=f"{local.__version__}-ci-nightly",
        doi="ci-hence-no-valid-doi",
        base_seed=20241109,
        ci=True,
        retrieve_misc_data=RETRIEVE_MISC_DATA_STEPS,
        **noaa_handling_steps,
        retrieve_and_extract_agage_data=retrieve_and_extract_agage_data,
        retrieve_and_extract_gage_data=RETRIEVE_AND_EXTRACT_GAGE_STEPS,
        retrieve_and_extract_ale_data=RETRIEVE_AND_EXTRACT_ALE_STEPS,
        retrieve_and_process_law_dome_data=RETRIEVE_AND_PROCESS_LAW_DOME_STEPS,
        retrieve_and_process_scripps_data=[],
        retrieve_and_process_epica_data=RETRIEVE_AND_PROCESS_EPICA_STEPS,
        retrieve_and_process_neem_data=RETRIEVE_AND_PROCESS_NEEM_STEPS,
        plot_input_data_overviews=[PlotInputDataOverviewsConfig(step_config_id="only")],
        compile_historical_emissions=COMPILE_HISTORICAL_EMISSIONS_STEPS,
        smooth_law_dome_data=smooth_law_dome_data,
        **monthly_fifteen_degree_pieces_configs,
        crunch_grids=create_crunch_grids_config(gases=gases_to_write),
        # TODO: test this in CI
        crunch_equivalent_species=[],
        write_input4mips=create_write_input4mips_config(
            gases=gases_to_write,
            start_year=start_year,
            end_year=end_year,
            input4mips_cvs_source_id="CR-CMIP-testing",
            input4mips_cvs_cv_source="https://raw.githubusercontent.com/znichollscr/input4MIPs_CVs/refs/heads/cr-cmip-testing/CVs/",
        ),
    )


def write_config_files(
    run_id: str,
    config_rel: Config,
    file_rel: Path,
    file_absolute: Path,
    root_dir_output: Path,
) -> None:
    """
    Write config files for a specific target
    """
    with open(file_rel, "w") as fh:
        fh.write(converter_yaml.dumps(config_rel))

    print(f"Updated {file_rel}")

    config_absolute = insert_path_prefix(
        config=config_rel,
        prefix=root_dir_output / run_id,
    )

    with open(file_absolute, "w") as fh:
        fh.write(converter_yaml.dumps(config_absolute))

    print(f"Updated {file_absolute}")


if __name__ == "__main__":
    ROOT_DIR_OUTPUT: Path = Path(__file__).parent.parent.absolute() / "output-bundles"

    ### Dev config
    write_config_files(
        run_id="dev-test-run",
        config_rel=create_dev_config(),
        file_rel=Path("dev-config.yaml"),
        file_absolute=Path("dev-config-absolute.yaml"),
        root_dir_output=ROOT_DIR_OUTPUT,
    )

    ### Config CI
    write_config_files(
        run_id="ci",
        config_rel=create_ci_config(),
        file_rel=Path("ci-config.yaml"),
        file_absolute=Path("ci-config-absolute.yaml"),
        root_dir_output=ROOT_DIR_OUTPUT,
    )

    ### Config nightly CI
    write_config_files(
        run_id="ci-nightly",
        config_rel=create_ci_nightly_config(),
        file_rel=Path("ci-nightly-config.yaml"),
        file_absolute=Path("ci-nightly-config-absolute.yaml"),
        root_dir_output=ROOT_DIR_OUTPUT,
    )
