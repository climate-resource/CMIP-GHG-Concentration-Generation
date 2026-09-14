r"""
Write config for the ground-based observational-network train/test split evaluation

Standalone and deliberately not wired into ``write-config.py`` - this
produces a config for a pipeline run in which, for CO2 and CH4,
``TEST_SPLIT_FRACTION`` of all NOAA surface-flask rows (one row = one
station-month observation) are randomly held out before binning (see
``local.train_test_split.stratified_test_split``, called from
``1100_ch4_bin-observational-network.py``/``1200_co2_bin-observational-network.py``),
so the pipeline only ever trains on the remaining rows. NOAA in-situ (and
for CH4 also AGAGE/GAGE/ALE) rows are never held out. The held-out rows are
written to ``held_out_test_data_file`` (see
``src/local/config/calculate_{ch4,co2}_monthly_15_degree.py``) for
out-of-sample evaluation - see ``notebooks/evaluation/``.

**The achieved fraction is (up to rounding) exactly ``TEST_SPLIT_FRACTION``**,
of *all* eligible rows globally - it is not applied per spatial bin. Each
bin-month still has a per-group safety cap (it can never be fully emptied
unless a protected row or hull-interior status - see
``local.train_test_split._hull_relevant_bins`` - makes that provably safe),
but the target fraction itself is achieved by randomly selecting from the
whole eligible pool at once, skipping any row whose group has already hit
its cap, rather than applying the fraction independently within each group
(which would round to 0 for the vast majority of bin-months, since most
only have 1-4 contributing stations in any given month). See
``notebooks/evaluation/README.md`` for the full story.

The split is deterministic (fixed ``TEST_SPLIT_SEED``) and independent of
satellite data (the split happens before satellite data is combined with
the ground network - see ``1101_ch4_.../1201_co2_..._interpolate-observational-network.py``),
so the held-out set is identical no matter how ``SAT_GAS``/``SAT_FIT``
(below) are set. That's what makes it possible to use the same held-out
set as a fixed benchmark across a no-satellite baseline run and several
satellite-fit runs, and compare how much each fit changes the agreement
with the (always-excluded) held-out data.

Same ``SAT_GAS``/``SAT_FIT`` env var contract as ``write-config.py``:

    # No satellite data (baseline)
    pixi run python scripts/write-eval-split-config.py

    # Satellite data, a specific fit, for both CO2 and CH4
    SAT_GAS=True SAT_FIT=LINEAR_STD_WEIGHT_FIT pixi run python scripts/write-eval-split-config.py

All variants share the same ``run_id`` ("dev-test-run-eval-split") and the
same ``eval-split-config.yaml``/``eval-split-config-absolute.yaml`` file
names - re-running this script and then ``doit`` overwrites the previous
variant's *intermediate* per-gas files (they aren't satellite-fit-specific
paths), but ``write_input4mips``'s final output filenames do carry a
``SAT_<fit>`` suffix (or none, for the no-satellite baseline), so those
accumulate side by side under the same ``output-bundles/dev-test-run-eval-split/``
tree - see ``local.diagnostics_reporting.discover_gridded_files``, which is
built to find exactly this "one baseline + several per-fit variants in one
directory" layout.

Writes ``eval-split-config.yaml``/``eval-split-config-absolute.yaml``
(distinct from ``dev-config.yaml``/``dev-config-absolute.yaml``) so this
never collides with the main dev config, and uses ``run_id
"dev-test-run-eval-split"`` so all of its output lives in a completely
separate ``output-bundles/dev-test-run-eval-split/`` tree.

Runs the full pipeline through ``crunch_grids`` and ``write_input4mips``
for CO2 and CH4 (but not other gases, and not ``crunch_equivalent_species``)
so the evaluation notebooks can compare against the actual final
gridded/input4MIPs product, not an intermediate piece.

Usage
-----
    pixi run python scripts/write-eval-split-config.py

Before running ``doit`` against the written config, copy over
``output-bundles/dev-test-run/data/{raw,interim}`` into
``output-bundles/dev-test-run-eval-split/data/`` and reuse the same
``DOIT_DB_FILE`` as the main dev run, so none of the shared upstream steps
(raw downloads, NOAA/AGAGE processing) need to redo any work - see
``scripts/write-sat-period-config.py`` for the same trick and why it works
(doit task identity doesn't include ``run_id``, and ``config_changed``
compares against doit's own DB). Then:

    DOIT_CONFIGURATION_FILE=eval-split-config-absolute.yaml \\
    DOIT_RUN_ID=dev-test-run-eval-split \\
    pixi run doit run --verbosity=2
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
from local.config_creation.agage_handling import create_agage_handling_config
from local.config_creation.ale_handling import RETRIEVE_AND_EXTRACT_ALE_STEPS
from local.config_creation.compile_historical_emissions import (
    COMPILE_HISTORICAL_EMISSIONS_STEPS,
)
from local.config_creation.crunch_grids import create_crunch_grids_config
from local.config_creation.epica_handling import RETRIEVE_AND_PROCESS_EPICA_STEPS
from local.config_creation.gage_handling import RETRIEVE_AND_EXTRACT_GAGE_STEPS
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

RUN_ID = "dev-test-run-eval-split"
GASES_TO_WRITE = ("co2", "ch4")
TEST_SPLIT_FRACTION = 0.1
TEST_SPLIT_SEED = 42
START_YEAR = 1
END_YEAR = 2022


def get_satellite_gases_and_fit_from_env() -> tuple[tuple[str, ...], str]:
    """Use the same env-var contract as ``write-config.py`` - ``SAT_GAS``/``SAT_FIT``"""
    sat_fit = os.environ.get("SAT_FIT", "LINEAR_FIT")

    if os.environ.get("SAT_GAS", "False").lower() != "true":
        return (), sat_fit

    return GASES_TO_WRITE, sat_fit


def create_eval_split_config() -> Config:
    """
    Create the (relative) train/test split evaluation config

    Adapted from ``write-config.py``'s ``create_dev_config``, trimmed to
    CO2/CH4 with the observational network's rows randomly split 80/20 for
    train/test. Satellite data is controlled by ``SAT_GAS``/``SAT_FIT``,
    same as ``write-config.py``.
    """
    gases_with_satellite_data, satellite_fit = get_satellite_gases_and_fit_from_env()

    scaled_sat_handling_steps = create_scaled_sat_handling_config(
        data_sources=tuple((gas, "scaled-sat", satellite_fit) for gas in GASES_TO_WRITE)
    )

    noaa_handling_steps = create_noaa_handling_config(
        data_sources=(
            ("co2", "in-situ"),
            ("co2", "surface-flask"),
            ("ch4", "in-situ"),
            ("ch4", "surface-flask"),
        )
    )

    retrieve_and_extract_agage_data = create_agage_handling_config(
        data_sources=(("ch4", "gc-md", "monthly"),)
    )

    smooth_law_dome_data = create_smooth_law_dome_data_config(gases=GASES_TO_WRITE, n_draws=250)

    monthly_fifteen_degree_pieces_configs = create_monthly_fifteen_degree_pieces_configs(
        gases=GASES_TO_WRITE,
        gases_with_satellite_data=gases_with_satellite_data,
        satellite_fit=satellite_fit,
        gases_test_split_fraction={gas: TEST_SPLIT_FRACTION for gas in GASES_TO_WRITE},
        gases_test_split_seed={gas: TEST_SPLIT_SEED for gas in GASES_TO_WRITE},
    )

    return Config(
        name="dev-eval-split",
        version=f"{local.__version__}-dev-eval-split",
        doi="dev-run-hence-no-valid-doi",
        base_seed=20240428,
        ci=False,
        dependency_db=Path("data/processed/dependencies.db"),
        retrieve_misc_data=RETRIEVE_MISC_DATA_STEPS,
        **noaa_handling_steps,
        process_noaa_hats_data=[],
        **scaled_sat_handling_steps,
        retrieve_and_extract_agage_data=retrieve_and_extract_agage_data,
        retrieve_and_extract_gage_data=RETRIEVE_AND_EXTRACT_GAGE_STEPS,
        retrieve_and_extract_ale_data=RETRIEVE_AND_EXTRACT_ALE_STEPS,
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
        plot_input_data_overviews=[],
        compile_historical_emissions=COMPILE_HISTORICAL_EMISSIONS_STEPS,
        smooth_law_dome_data=smooth_law_dome_data,
        smooth_ghosh_et_al_2023_data=[],
        **monthly_fifteen_degree_pieces_configs,
        crunch_grids=create_crunch_grids_config(gases=GASES_TO_WRITE),
        crunch_equivalent_species=[],
        write_input4mips=create_write_input4mips_config(
            gases=GASES_TO_WRITE,
            start_year=START_YEAR,
            end_year=END_YEAR,
            input4mips_cvs_source_id="CR-CMIP-testing",
            input4mips_cvs_cv_source="https://raw.githubusercontent.com/znichollscr/input4MIPs_CVs/refs/heads/cr-cmip-testing/CVs/",
            gases_with_satellite_data=gases_with_satellite_data,
            satellite_fit=satellite_fit,
        ),
    )


if __name__ == "__main__":
    ROOT_DIR_OUTPUT: Path = Path(__file__).parent.parent.absolute() / "output-bundles"

    config_rel = create_eval_split_config()
    file_rel = Path("eval-split-config.yaml")
    file_absolute = Path("eval-split-config-absolute.yaml")

    with open(file_rel, "w") as fh:
        fh.write(converter_yaml.dumps(config_rel))
    print(f"Updated {file_rel}")

    config_absolute = insert_path_prefix(config=config_rel, prefix=ROOT_DIR_OUTPUT / RUN_ID)
    with open(file_absolute, "w") as fh:
        fh.write(converter_yaml.dumps(config_absolute))
    print(f"Updated {file_absolute}")
