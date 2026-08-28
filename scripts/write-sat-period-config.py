"""
Write config for the "satellite period only" experiment

Standalone and deliberately not wired into ``write-config.py`` - this is a
one-off check (do the observational-network EOFs/PCs look different if we
restrict the ground network to the same years the satellite product covers,
2003 onwards, rather than using the full multi-decade record?), not a
permanent part of the pipeline. Keeping it a separate script/config file
means it can be deleted cleanly later without touching the main dev config.

CO2 only for now. CH4 was tried too, but its global-annual-mean extension
harmonises directly against the Law Dome ice core (smoothed CH4 record ends
1997), so restricting the observational network to 2003 onwards opens a
6-year gap (1998-2002) that step can't bridge (`KeyError: 2003`). CO2 uses
the continuous, still-ongoing Mauna Loa/Scripps record for that handoff
instead, so it isn't affected. Revisit CH4 once that's resolved.

Ice-core/firn data used to extend the record back in time is untouched -
only the ground-based observational network (NOAA) is restricted to 2003
onwards, via ``year_drop_observational_data_before_and_including=2002``.

Writes ``sat-period-config.yaml``/``sat-period-config-absolute.yaml``
(distinct from ``dev-config.yaml``/``dev-config-absolute.yaml``) so this
never collides with the main dev config, and uses ``run_id
"dev-test-run-sat-period"`` so all of its output lives in a completely
separate ``output-bundles/dev-test-run-sat-period/`` tree.

Usage
-----
    # No satellite data, ground network restricted to 2003 onwards
    pixi run python scripts/write-sat-period-config.py

    # With satellite data (a specific fit), ground network restricted to 2003 onwards
    SAT_GAS=True SAT_FIT=LINEAR_STD_WEIGHT_FIT pixi run python scripts/write-sat-period-config.py

Before running ``doit`` against the written config, copy over
``output-bundles/dev-test-run/data/{raw,interim}`` into
``output-bundles/dev-test-run-sat-period/data/`` and reuse the same
``DOIT_DB_FILE`` as the main dev run, so none of the shared upstream steps
(raw downloads, NOAA/AGAGE/ice-core processing) need to redo any work -
see the project discussion for why this works (doit task identity doesn't
include ``run_id``, and ``config_changed`` compares against doit's own DB).
"""

# ruff: noqa: E402
from __future__ import annotations

import os
from pathlib import Path

import openscm_units
import pint
from pydoit_nb.config_handling import insert_path_prefix

pint.set_application_registry(openscm_units.unit_registry)


import local
from local.config import Config, converter_yaml
from local.config_creation.compile_historical_emissions import (
    COMPILE_HISTORICAL_EMISSIONS_STEPS,
)
from local.config_creation.crunch_grids import create_crunch_grids_config
from local.config_creation.epica_handling import RETRIEVE_AND_PROCESS_EPICA_STEPS
from local.config_creation.law_dome_handling import (
    RETRIEVE_AND_PROCESS_LAW_DOME_STEPS,
    create_smooth_law_dome_data_config,
)
from local.config_creation.menking_et_al_2025_handling import (
    RETRIEVE_AND_PROCESS_MENKING_ET_AL_2025_DATA_STEPS,
)
from local.config_creation.monthly_fifteen_degree_pieces import (
    create_monthly_fifteen_degree_pieces_configs,
)
from local.config_creation.neem_handling import RETRIEVE_AND_PROCESS_NEEM_STEPS
from local.config_creation.noaa_handling import create_noaa_handling_config
from local.config_creation.retrieve_misc_data import RETRIEVE_MISC_DATA_STEPS
from local.config_creation.scaled_sat_handling import create_scaled_sat_handling_config
from local.config_creation.scripps_handling import RETRIEVE_AND_PROCESS_SCRIPPS_DATA
from local.config_creation.write_input4mips import create_write_input4mips_config

RUN_ID = "dev-test-run-sat-period"
GASES_TO_WRITE = ("co2",)
# Ground network restricted to 2003 onwards - satellite coverage starts
# 2003-01, confirmed to have data in every month of 2003 for both gases.
YEAR_DROP_OBSERVATIONAL_DATA_BEFORE_AND_INCLUDING = 2002


def get_satellite_gases_and_fit_from_env() -> tuple[tuple[str, ...], str]:
    """Same env-var contract as ``write-config.py`` - ``SAT_GAS``/``SAT_FIT``"""
    sat_fit = os.environ.get("SAT_FIT", "LINEAR_FIT")

    if os.environ.get("SAT_GAS", "False").lower() != "true":
        return (), sat_fit

    return GASES_TO_WRITE, sat_fit


def create_sat_period_config() -> Config:
    """
    Create the (relative) "satellite period only" config

    Adapted from ``write-config.py``'s ``create_dev_config``, trimmed to
    just CO2/CH4 and with the ground network restricted to 2003 onwards.
    """
    start_year = 1
    end_year = 2022

    gases_with_satellite_data, satellite_fit = get_satellite_gases_and_fit_from_env()

    scaled_sat_handling_steps = create_scaled_sat_handling_config(
        data_sources=tuple((gas, "scaled-sat", satellite_fit) for gas in GASES_TO_WRITE)
    )

    noaa_handling_steps = create_noaa_handling_config(
        data_sources=(
            ("co2", "in-situ"),
            ("co2", "surface-flask"),
        )
    )

    # AGAGE/GAGE/ALE don't cover CO2 and aren't referenced by CO2's own
    # notebook steps (unlike CH4, which needs GAGE/ALE present even when
    # unused - see `calculate_ch4_monthly_fifteen_degree_pieces.py`).
    smooth_law_dome_data = create_smooth_law_dome_data_config(gases=("co2",), n_draws=250)

    monthly_fifteen_degree_pieces_configs = create_monthly_fifteen_degree_pieces_configs(
        gases=GASES_TO_WRITE,
        gases_drop_obs_data_years_before_inclusive={
            gas: YEAR_DROP_OBSERVATIONAL_DATA_BEFORE_AND_INCLUDING for gas in GASES_TO_WRITE
        },
        gases_with_satellite_data=gases_with_satellite_data,
        satellite_fit=satellite_fit,
    )

    return Config(
        name="dev-sat-period",
        version=f"{local.__version__}-dev-sat-period",
        doi="dev-run-hence-no-valid-doi",
        base_seed=20240428,
        ci=False,
        dependency_db=Path("data/processed/dependencies.db"),
        retrieve_misc_data=RETRIEVE_MISC_DATA_STEPS,
        **noaa_handling_steps,
        process_noaa_hats_data=[],
        **scaled_sat_handling_steps,
        retrieve_and_extract_agage_data=[],
        retrieve_and_extract_gage_data=[],
        retrieve_and_extract_ale_data=[],
        retrieve_and_process_law_dome_data=RETRIEVE_AND_PROCESS_LAW_DOME_STEPS,
        retrieve_and_process_scripps_data=RETRIEVE_AND_PROCESS_SCRIPPS_DATA,
        retrieve_and_process_epica_data=RETRIEVE_AND_PROCESS_EPICA_STEPS,
        retrieve_and_process_neem_data=RETRIEVE_AND_PROCESS_NEEM_STEPS,
        retrieve_and_process_wmo_2022_ozone_assessment_ch7_data=[],
        retrieve_and_process_western_et_al_2024_data=[],
        retrieve_and_process_velders_et_al_2022_data=[],
        retrieve_and_process_droste_et_al_2020_data=[],
        retrieve_and_process_adam_et_al_2024_data=[],
        retrieve_and_process_ghosh_et_al_2023_data=[],
        retrieve_and_process_menking_et_al_2025_data=RETRIEVE_AND_PROCESS_MENKING_ET_AL_2025_DATA_STEPS,
        retrieve_and_process_trudinger_et_al_2016_data=[],
        # Its overview notebook has a hardcoded list of gas/network combos
        # (n2o hats, cfc11 hats, ...) unrelated to our restricted gas set -
        # skip it, it's just a diagnostic plot, not on the critical path.
        plot_input_data_overviews=[],
        compile_historical_emissions=COMPILE_HISTORICAL_EMISSIONS_STEPS,
        smooth_law_dome_data=smooth_law_dome_data,
        smooth_ghosh_et_al_2023_data=[],
        **monthly_fifteen_degree_pieces_configs,
        crunch_grids=create_crunch_grids_config(gases=GASES_TO_WRITE),
        crunch_equivalent_species=[],
        write_input4mips=create_write_input4mips_config(
            gases=GASES_TO_WRITE,
            start_year=start_year,
            end_year=end_year,
            input4mips_cvs_source_id="CR-CMIP-testing",
            input4mips_cvs_cv_source="https://raw.githubusercontent.com/znichollscr/input4MIPs_CVs/refs/heads/cr-cmip-testing/CVs/",
            gases_with_satellite_data=gases_with_satellite_data,
            satellite_fit=satellite_fit,
        ),
    )


if __name__ == "__main__":
    ROOT_DIR_OUTPUT: Path = Path(__file__).parent.parent.absolute() / "output-bundles"

    config_rel = create_sat_period_config()
    file_rel = Path("sat-period-config.yaml")
    file_absolute = Path("sat-period-config-absolute.yaml")

    with open(file_rel, "w") as fh:
        fh.write(converter_yaml.dumps(config_rel))
    print(f"Updated {file_rel}")

    config_absolute = insert_path_prefix(config=config_rel, prefix=ROOT_DIR_OUTPUT / RUN_ID)
    with open(file_absolute, "w") as fh:
        fh.write(converter_yaml.dumps(config_absolute))
    print(f"Updated {file_absolute}")
