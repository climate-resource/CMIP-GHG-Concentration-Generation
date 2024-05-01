"""
Write our configuration files
"""

from __future__ import annotations

from pathlib import Path

import openscm_units
import pint
from pydoit_nb.config_handling import insert_path_prefix

import local
from local.config import Config, converter_yaml
from local.config_creation.agage_handling import create_agage_handling_config
from local.config_creation.ale_handling import RETRIEVE_AND_EXTRACT_ALE_STEPS
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

pint.set_application_registry(openscm_units.unit_registry)


def create_ci_config() -> Config:
    """
    Create our (relative) CI config
    """
    gases_to_write = ("ch4",)

    noaa_handling_steps = create_noaa_handling_config(
        data_sources=(
            ("ch4", "in-situ"),
            ("ch4", "surface-flask"),
        )
    )

    retrieve_and_extract_agage_data = create_agage_handling_config(
        data_sources=(("ch4", "gc-md", "monthly"),)
    )

    smooth_law_dome_data = create_smooth_law_dome_data_config(
        gases=("ch4",), n_draws=10
    )

    monthly_fifteen_degree_pieces_configs = (
        create_monthly_fifteen_degree_pieces_configs(gases=gases_to_write)
    )

    return Config(
        name="CI",
        version=f"{local.__version__}-ci",
        base_seed=20240427,
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
        plot_input_data_overviews=[],
        smooth_law_dome_data=smooth_law_dome_data,
        **monthly_fifteen_degree_pieces_configs,
        calculate_n2o_monthly_15_degree=[],  # Will move into config creation function in future
        crunch_grids=create_crunch_grids_config(gases=gases_to_write),
        write_input4mips=create_write_input4mips_config(gases=gases_to_write),
    )


if __name__ == "__main__":
    ROOT_DIR_OUTPUT: Path = Path(__file__).parent.parent.absolute() / "output-bundles"
    CI_RUN_ID: str = "CI"

    ### Config CI
    CI_FILE: Path = Path("ci-config.yaml")
    ci_config = create_ci_config()
    with open(CI_FILE, "w") as fh:
        fh.write(converter_yaml.dumps(ci_config))

    print(f"Updated {CI_FILE}")

    ### Config CI absolute
    CI_ABSOLUTE_FILE: Path = Path("ci-config-absolute.yaml")

    ci_config_absolute = insert_path_prefix(
        config=ci_config,
        prefix=ROOT_DIR_OUTPUT / CI_RUN_ID,
    )
    with open(CI_ABSOLUTE_FILE, "w") as fh:
        fh.write(converter_yaml.dumps(ci_config_absolute))

    print(f"Updated {CI_ABSOLUTE_FILE}")
