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
from local.pydoit_nb.config_tools import URLSource

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
retrieve_config = config.retrieve_and_extract_noaa_data
process_noaa_surface_flask_data = config.process_noaa_surface_flask_data
process_noaa_in_situ_data = config.process_noaa_in_situ_data

for gas, source, download_hash in [
    (
        "ch4",
        "surface-flask",
        "e541578315328857f01eb7432b5949e39beabab2017c09e46727ac49ec728087",
    ),
    (
        "ch4",
        "in-situ",
        "c8ad74288d860c63b6a027df4d7bf6742e772fc4e3f99a4052607a382d7fefb2",
    ),
    (
        "n2o",
        "surface-flask",
        "6b7e09c37b7fa456ab170a4c7b825b3d4b9f6eafb0ff61a2a46554b0e63e84b1",
    ),
    (
        "sf6",
        "surface-flask",
        "376c78456bba6844cca78ecd812b896eb2f10cc6b8a9bf6cad7a52dc39e31e9a",
    ),
]:
    for c in retrieve_config:
        if c.step_config_id == f"co2_{source}":
            base_retrieve = c
            break
    else:
        raise ValueError("Didn't find base retrieve")  # noqa: TRY003

    interim_files_updated = {
        k: Path(str(v).replace("co2", gas))
        for k, v in base_retrieve.interim_files.items()
    }
    updated_retrieve = evolve(
        base_retrieve,
        step_config_id=f"{gas}_{source}",
        gas=gas,
        download_urls=[
            URLSource(
                url=base_retrieve.download_urls[0].url.replace("co2", gas),
                known_hash=download_hash,
            )
        ],
        download_complete_file=Path(
            str(base_retrieve.download_complete_file).replace("co2", gas)
        ),
        interim_files=interim_files_updated,
    )
    retrieve_config.append(updated_retrieve)

    if "surface-flask" in source:
        for c in process_noaa_surface_flask_data:
            if c.step_config_id == "co2":
                base_process_noaa_surface = c
                break
        else:
            raise ValueError(  # noqa: TRY003
                "Didn't find base process_noaa_surface_flask_data"
            )

        interim_files_updated = {
            k: Path(str(v).replace("co2", gas))
            for k, v in base_retrieve.interim_files.items()
        }
        updated_process_noaa_surface = evolve(
            base_process_noaa_surface,
            step_config_id=gas,
            gas=gas,
            processed_monthly_data_with_loc_file=Path(
                str(
                    base_process_noaa_surface.processed_monthly_data_with_loc_file
                ).replace("co2", gas)
            ),
        )
        process_noaa_surface_flask_data.append(updated_process_noaa_surface)
    elif "in-situ" in source:
        for c in process_noaa_in_situ_data:
            if c.step_config_id == "co2":
                base_process_noaa_surface = c
                break
        else:
            raise ValueError(  # noqa: TRY003
                "Didn't find base process_noaa_in_situ_data"
            )

        interim_files_updated = {
            k: Path(str(v).replace("co2", gas))
            for k, v in base_retrieve.interim_files.items()
        }
        updated_process_noaa_surface = evolve(
            base_process_noaa_surface,
            step_config_id=gas,
            gas=gas,
            processed_monthly_data_with_loc_file=Path(
                str(
                    base_process_noaa_surface.processed_monthly_data_with_loc_file
                ).replace("co2", gas)
            ),
        )
        process_noaa_in_situ_data.append(updated_process_noaa_surface)

config_full = evolve(
    config,
    retrieve_and_extract_noaa_data=retrieve_config,
    process_noaa_surface_flask_data=process_noaa_surface_flask_data,
    process_noaa_in_situ_data=process_noaa_in_situ_data,
)

with open(DEV_ABSOLUTE_FILE, "w") as fh:
    fh.write(converter_yaml.dumps(config_full))

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
