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

from local.zenodo import get_zenodo_doi

pint.set_application_registry(openscm_units.unit_registry)


from local.config import Config, converter_yaml

if __name__ == "__main__":
    RUN_NAME = "v0.5.0"
    VERSION = "0.5.0"
    SEED = 20250201

    ROOT_DIR_OUTPUT: Path = Path(__file__).parent.parent.absolute() / "output-bundles"

    known_zenodo_doi = None
    known_zenodo_doi = "https://doi.org/10.5281/zenodo.14892947"
    if known_zenodo_doi is None:
        # TODO: push this up into openscm-zenodo
        # so other users don't have my experience.
        zenodo_doi = get_zenodo_doi(any_deposition_id="14892938")

        msg = f"Go to Zenodo and press 'reserve DOI'. Expected DOI: https://doi.org/{zenodo_doi}"
        raise ValueError(msg)

    else:
        zenodo_doi = known_zenodo_doi

    DEV_FILE: Path = Path("dev-config.yaml")
    with open(DEV_FILE) as fh:
        dev_config = converter_yaml.loads(fh, Config)

    write_input4mips_old = dev_config.write_input4mips
    write_input4mips_new = [
        evolve(
            v,
            input4mips_cvs_source_id=f"CR-CMIP-{VERSION.replace('.', '-')}",
            input4mips_cvs_cv_source="gh:cr-cmip-0-5-0",
        )
        for v in write_input4mips_old
    ]
    run_config = evolve(
        dev_config,
        name=RUN_NAME,
        version=VERSION,
        doi=zenodo_doi,
        base_seed=SEED,
        write_input4mips=write_input4mips_new,
    )
    RUN_FILE: Path = Path(f"{RUN_NAME}-config.yaml")
    with open(RUN_FILE, "w") as fh:
        fh.write("# Generated with scripts/write-run-config.py\n")
        fh.write(converter_yaml.dumps(run_config))

    print(f"Wrote {RUN_FILE}")
