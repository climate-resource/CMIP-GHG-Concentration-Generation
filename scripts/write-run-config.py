"""
Write a run configuration, based on our dev configuration
"""

# Have to the pint registry before doing other imports, hence funny order
# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import openscm_units
import pint
from attrs import evolve

pint.set_application_registry(openscm_units.unit_registry)

from local.config import Config, converter_yaml

if __name__ == "__main__":
    RUN_NAME = "20240729"
    VERSION = "0.2.0"
    SEED = 20240729

    ROOT_DIR_OUTPUT: Path = Path(__file__).parent.parent.absolute() / "output-bundles"

    DEV_FILE: Path = Path("dev-config.yaml")
    with open(DEV_FILE) as fh:
        dev_config = converter_yaml.loads(fh, Config)

    run_config = evolve(
        dev_config,
        name=RUN_NAME,
        version=VERSION,
        base_seed=SEED,
    )
    RUN_FILE: Path = Path(f"{RUN_NAME}-config.yaml")
    with open(RUN_FILE, "w") as fh:
        fh.write("# Generated with scripts/write-run-config.py\n")
        fh.write(converter_yaml.dumps(run_config))

    print(f"Wrote {RUN_FILE}")
