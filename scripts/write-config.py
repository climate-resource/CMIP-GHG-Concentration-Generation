"""
Write our configuration files
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_handling import insert_path_prefix

import local
from local.config import Config, converter_yaml


def create_ci_config() -> Config:
    """
    Create our (relative) CI config
    """
    return Config(
        name="CI",
        version=f"{local.__version__}-ci",
        base_seed=20240427,
        ci=True,
        retrieve_misc_data=[],
        retrieve_and_extract_noaa_data=[],
        process_noaa_surface_flask_data=[],
        process_noaa_in_situ_data=[],
        retrieve_and_extract_agage_data=[],
        retrieve_and_extract_gage_data=[],
        retrieve_and_extract_ale_data=[],
        retrieve_and_process_law_dome_data=[],
        retrieve_and_process_scripps_data=[],
        retrieve_and_process_epica_data=[],
        retrieve_and_process_neem_data=[],
        plot_input_data_overviews=[],
        smooth_law_dome_data=[],
        calculate_ch4_monthly_fifteen_degree_pieces=[],
        calculate_n2o_monthly_15_degree=[],
        crunch_grids=[],
        grid=[],
        gridded_data_processing=[],
        write_input4mips=[],
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
