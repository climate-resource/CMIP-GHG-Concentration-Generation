"""
Create dev config with absolute paths

This should be run before working with the notebooks in dev mode to make sure
that you don't end up with files everywhere.
"""
from __future__ import annotations

from pathlib import Path

from attrs import evolve

from local.config import Config, converter_yaml
from local.pydoit_nb.config_handling import insert_path_prefix

DEV_FILE: Path = Path("dev-config.yaml")

DEV_ABSOLUTE_FILE: Path = Path("dev-config-absolute.yaml")
DEV_RUN_ID: str = "dev-test-run"

CI_FILE: Path = Path("ci-config.yaml")
CI_ABSOLUTE_FILE: Path = Path("ci-config-absolute.yaml")
CI_RUN_ID: str = "CI"

ROOT_DIR_OUTPUT: Path = Path(__file__).parent.parent.absolute() / "output-bundles"

with open(DEV_FILE) as fh:
    config_relative = converter_yaml.loads(fh.read(), Config)

config = insert_path_prefix(
    config=config_relative,
    prefix=ROOT_DIR_OUTPUT / DEV_RUN_ID,
)

with open(DEV_ABSOLUTE_FILE, "w") as fh:
    fh.write(converter_yaml.dumps(config))

print(f"Updated {DEV_ABSOLUTE_FILE}")

ci_config = evolve(config_relative, ci=True, name=CI_RUN_ID)
with open(CI_FILE, "w") as fh:
    fh.write(converter_yaml.dumps(ci_config))

print(f"Updated {CI_FILE}")

ci_config_absolute = insert_path_prefix(
    config=evolve(config_relative, ci=True, name=CI_RUN_ID),
    prefix=ROOT_DIR_OUTPUT / CI_RUN_ID,
)
with open(CI_ABSOLUTE_FILE, "w") as fh:
    fh.write(converter_yaml.dumps(ci_config_absolute))

print(f"Updated {CI_ABSOLUTE_FILE}")
