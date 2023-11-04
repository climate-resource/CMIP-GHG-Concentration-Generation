"""
Create dev config with absolute paths

This should be run before working with the notebooks in dev mode to make sure
that you don't end up with files everywhere.
"""
from __future__ import annotations

from pathlib import Path

from local.config import Config, converter_yaml
from local.pydoit_nb.config_handling import insert_path_prefix

DEV_FILE: Path = Path("dev-config.yaml")
DEV_ABSOLUTE_FILE: Path = Path("dev-config-absolute.yaml")
ROOT_OUTPUT_DIR: Path = Path(__file__).parent.parent.absolute() / "output-bundles"
RUN_ID: str = "dev-run"

with open(DEV_FILE) as fh:
    config_relative = converter_yaml.loads(fh.read(), Config)

config = insert_path_prefix(
    config=config_relative,
    prefix=ROOT_OUTPUT_DIR / RUN_ID,
)

with open(DEV_ABSOLUTE_FILE, "w") as fh:
    fh.write(converter_yaml.dumps(config))

print(f"Updated {DEV_ABSOLUTE_FILE}")
