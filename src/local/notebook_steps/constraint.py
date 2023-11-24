"""
Constraint notebook steps
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import asdict

from ..config import get_config_for_step_id
from ..pydoit_nb.notebook import ConfiguredNotebook, UnconfiguredNotebook
from ..pydoit_nb.notebook_step import UnconfiguredNotebookBasedStep

if TYPE_CHECKING:
    from ..config.base import Config, ConfigBundle


def configure_notebooks(
    unconfigured_notebooks: Iterable[UnconfiguredNotebook],
    config_bundle: ConfigBundle,
    step_name: str,
    step_config_id: str,
) -> Iterable[ConfiguredNotebook]:
    """
    Configure notebooks

    Parameters
    ----------
    unconfigured_notebooks
        Unconfigured notebooks

    config_bundle
        Configuration bundle from which to take configuration values

    step_name
        Name of the step

    step_config_id
        Step config ID to use when configuring the notebook

    Returns
    -------
        Configured notebooks
    """
    uc_nbs_dict = {nb.notebook_path: nb for nb in unconfigured_notebooks}

    config = config_bundle.config_hydrated

    config_step = get_config_for_step_id(
        config=config, step=step_name, step_config_id=step_config_id
    )

    config_preparation = get_config_for_step_id(
        config=config, step="preparation", step_config_id="only"
    )

    configured_notebooks = [
        ConfiguredNotebook(
            **asdict(uc_nbs_dict[Path("2xx_constraint") / "210_draw-samples"]),
            configuration=(config_step.constraint_gradient,),
            dependencies=(config_preparation.seed_file,),
            targets=(config_step.draw_file,),
            config_file=config_bundle.config_hydrated_path,
            step_config_id=step_config_id,
        )
    ]

    return configured_notebooks


step: UnconfiguredNotebookBasedStep[Config] = UnconfiguredNotebookBasedStep(
    step_name="constraint",
    unconfigured_notebooks=[
        UnconfiguredNotebook(
            notebook_path=Path("2xx_constraint") / "210_draw-samples",
            raw_notebook_ext=".py",
            summary="constraint - draw samples",
            doc="Draw samples with constraint",
        )
    ],
    # I can't make mypy behave with the below. I think the type hints are
    # correct, but removing leads to an error I just can't figure out (I think
    # it's to do with how the generic is compared but I don't actually know).
    configure_notebooks=configure_notebooks,  # type: ignore
)
