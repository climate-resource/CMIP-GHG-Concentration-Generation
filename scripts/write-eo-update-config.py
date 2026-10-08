r"""
Write config for the EO-update run (CO2 and CH4 with scaled satellite data)

Standalone and deliberately not wired into ``write-config.py``, so the
baseline dev run is never affected. This reuses ``write-config.py``'s
``create_dev_config`` (so the science settings stay identical to the dev
config) with satellite data switched on for CO2 and CH4, using

- CO2: ``LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT``
- CH4: ``NONLINEAR_LAT_STD_WEIGHT_FIT``

both weighted by the inverse of the retrieval uncertainty, and then swaps in
a ``write_input4mips`` config whose metadata you control from the
"Metadata" section below.

Writes ``eo-update-config.yaml``/``eo-update-config-absolute.yaml`` and uses
``run_id "eo-update"``, so all output lives in
``output-bundles/eo-update/``.

Usage
-----
    pixi run python scripts/write-eo-update-config.py

Before running ``doit`` against the written config, copy over
``output-bundles/dev-test-run/data/{raw,interim}`` into
``output-bundles/eo-update/data/`` and use a copy of the dev doit DB
(see ``scripts/write-sat-period-config.py`` for the same trick), then

    DOIT_CONFIGURATION_FILE=eo-update-config-absolute.yaml \\
    DOIT_RUN_ID=eo-update \\
    DOIT_DB_BACKEND=json DOIT_DB_FILE=doit-db-eo-update.json \\
    pixi run doit --verbosity=2 -n 4 \\
    "${PWD}/output-bundles/eo-update/data/processed/esgf-ready/co2_input4MIPs_esgf-ready.complete"
"""

# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import json
import os
import urllib.request
from pathlib import Path

import openscm_units
import pint
from attrs import evolve
from pydoit_nb.config_handling import insert_path_prefix

pint.set_application_registry(openscm_units.unit_registry)

from local.config import Config, converter_yaml
from local.config_creation.write_input4mips import create_write_input4mips_config

REPO_ROOT = Path(__file__).parent.parent.absolute()

RUN_ID = "eo-update"
GASES_TO_WRITE = ("co2", "ch4")
START_YEAR = 1
END_YEAR = 2022

SATELLITE_FIT = {
    "co2": "LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT",
    "ch4": "NONLINEAR_LAT_STD_WEIGHT_FIT",
}

# %% Metadata

CV_BASE_URL = "https://raw.githubusercontent.com/znichollscr/input4MIPs_CVs/refs/heads/cr-cmip-testing/CVs/"
"""
Remote CVs used for every CV file except source_id and activity_id

The CV folder is generated from this (see ``write_cvs``) every time this script runs.
"""

CV_DIR = REPO_ROOT / "output-bundles" / RUN_ID / "cvs"
"""Where the generated CV folder is written"""

INPUT4MIPS_CVS_SOURCE_ID = "CR-CMIP-EO-update"
"""Name of the source ID entry, written into the ``source_id`` attribute and file paths"""

SOURCE_ID_ENTRY = {
    "contact": "anna.lanteri@climate-resource.com;"
    "zebedee.nicholls@climate-resource.com;malte.meinshausen@climate-resource.com",
    "further_info_url": "http://www.tbd.invalid",
    "institution_id": "CR",
    "license_id": "CC BY 4.0",
    "mip_era": "CMIP6Plus",
    "source_version": "EO-update",
}
"""Contents of the source ID entry (contact, further_info_url etc.)"""

ACTIVITY_ID = "EO-update-input4MIPs"
"""Activity ID, written into the ``activity_id`` attribute and file paths"""

ACTIVITY_ID_ENTRY = {
    "URL": "None",
    "long_name": "Updated input forcing datasets for Model Intercomparison Projects"
    " using Earth Observation data",
}
"""Contents of the activity ID entry"""

DOI = "TBD, once published on Zenodo"
"""DOI to write into the ``doi`` attribute"""

COMMENT = (
    "Data compiled by Climate Resource, based on science by many others "
    "(see 'references*' attributes). "
    "For funding information, see the 'funding*' attributes."
)
"""``comment`` attribute. If blank, the notebook's default comment is used"""

SATELLITE_DATA = {
    "co2": {
        "variable": "xco2",
        "reference": "https://cds.climate.copernicus.eu/datasets/satellite-carbon-dioxide?tab=download",
    },
    "ch4": {
        "variable": "xch4",
        "reference": "https://cds.climate.copernicus.eu/datasets/satellite-methane?tab=overview",
    },
}
"""Satellite product details per gas (variable prefix and Copernicus CDS dataset reference)"""


def get_satellite_metadata(gas: str) -> dict[str, str]:
    """Get the attributes describing the satellite data, fit and weighting used for a gas"""
    variable = SATELLITE_DATA[gas]["variable"]

    return {
        "satellite_data_source": (
            f"Copernicus Climate Change Service (C3S) Level 3 column-averaged {variable.upper()} "
            "(OBS4MIPs merged product v4.6, 2003-01 to 2023-12), scaled to the ground-based network"
        ),
        "satellite_data_reference": SATELLITE_DATA[gas]["reference"],
        "satellite_fit": SATELLITE_FIT[gas],
        "satellite_weighting": (
            "Satellite bins are combined with the ground-based data using inverse-variance weighting "
            f"(inverse variance of {variable}_scaled_stderr_rchi2); each ground-based station has weight one"
        ),
    }


def write_cvs() -> Path:
    """
    Write the CV folder for this run

    All files are downloaded from ``CV_BASE_URL`` except the source ID and
    activity ID files, which are built from the entries defined above.
    """
    CV_DIR.mkdir(parents=True, exist_ok=True)

    cv_files = json.loads(
        urllib.request.urlopen(
            "https://api.github.com/repos/znichollscr/input4MIPs_CVs/contents/CVs?ref=cr-cmip-testing"
        ).read()
    )
    for name in sorted(v["name"] for v in cv_files):
        if name in ("input4MIPs_source_id.json", "input4MIPs_activity_id.json"):
            continue
        (CV_DIR / name).write_bytes(urllib.request.urlopen(f"{CV_BASE_URL}{name}").read())

    (CV_DIR / "input4MIPs_source_id.json").write_text(
        json.dumps({INPUT4MIPS_CVS_SOURCE_ID: SOURCE_ID_ENTRY}, indent=2) + "\n"
    )
    (CV_DIR / "input4MIPs_activity_id.json").write_text(
        json.dumps({ACTIVITY_ID: ACTIVITY_ID_ENTRY}, indent=2) + "\n"
    )

    return CV_DIR


def load_write_config_module():
    """Load ``write-config.py`` (hyphen in the name means we can't just import it)"""
    spec = importlib.util.spec_from_file_location("write_config", REPO_ROOT / "scripts" / "write-config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def create_eo_update_config() -> Config:
    """Create the (relative) EO-update config"""
    write_config = load_write_config_module()

    # Switch satellite data on, weighted, for both gases.
    # Fits are left to ``DEFAULT_SATELLITE_FIT_BY_GAS``, checked below.
    os.environ["SAT_GAS"] = "True"
    os.environ["SAT_WEIGHT"] = "True"
    os.environ["GAS"] = "all"
    os.environ.pop("SAT_FIT", None)

    assert (
        write_config.DEFAULT_SATELLITE_FIT_BY_GAS == SATELLITE_FIT
    ), write_config.DEFAULT_SATELLITE_FIT_BY_GAS

    dev_config = write_config.create_dev_config()

    write_input4mips = create_write_input4mips_config(
        gases=GASES_TO_WRITE,
        start_year=START_YEAR,
        end_year=END_YEAR,
        input4mips_cvs_source_id=INPUT4MIPS_CVS_SOURCE_ID,
        input4mips_cvs_cv_source=str(write_cvs()),
        comment=COMMENT or None,
        activity_id=ACTIVITY_ID,
        extra_metadata_by_gas={gas: get_satellite_metadata(gas) for gas in GASES_TO_WRITE},
    )

    return evolve(
        dev_config,
        name="eo-update",
        version=f"{dev_config.version}-eo-update",
        doi=DOI or "eo-update-hence-no-valid-doi",
        write_input4mips=write_input4mips,
    )


if __name__ == "__main__":
    ROOT_DIR_OUTPUT: Path = REPO_ROOT / "output-bundles"

    config_rel = create_eo_update_config()
    file_rel = Path("eo-update-config.yaml")
    file_absolute = Path("eo-update-config-absolute.yaml")

    with open(file_rel, "w") as fh:
        fh.write(converter_yaml.dumps(config_rel))
    print(f"Updated {file_rel}")

    config_absolute = insert_path_prefix(config=config_rel, prefix=ROOT_DIR_OUTPUT / RUN_ID)
    with open(file_absolute, "w") as fh:
        fh.write(converter_yaml.dumps(config_absolute))
    print(f"Updated {file_absolute}")
