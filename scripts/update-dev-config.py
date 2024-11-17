"""
Update development config

Goes quickly out of date, use with care (more intended as a helper script
than for continuous use)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from local.config import Config, converter_yaml
from local.config.constraint import ConstraintConfig
from local.config.covariance import CovarianceConfig
from local.config.figures import FiguresConfig
from local.config.preparation import PreparationConfig

DEV_FILE: Path = Path("dev-config.yaml")

with open(DEV_FILE) as fh:
    config_relative = converter_yaml.loads(fh.read(), Config)

config = Config(
    name="dev",
    preparation=[
        PreparationConfig(
            step_config_id="only",
            seed=2847539,
            seed_file=Path("data") / "interim" / "000_seed.txt",
        )
    ],
    covariance=[
        CovarianceConfig(
            step_config_id=step_config_id,
            covariance=cov,
            draw_file=Path("data") / "interim" / f"110_{step_config_id}_draws.csv",
        )
        for step_config_id, cov in [
            ["cov", np.array([[0.25, 0.5], [0.5, 0.55]])],
            ["no-cov", np.array([[0.25, 0], [0, 0.55]])],
        ]
    ],
    constraint=[
        ConstraintConfig(
            step_config_id="only",
            constraint_gradient=1.2,
            draw_file=Path("data") / "interim" / "210_constraint_draws.csv",
        )
    ],
    figures=[
        FiguresConfig(
            step_config_id="only",
            misc_figures_dir=Path("figures") / "misc",
            draw_comparison_table=Path("data") / "processed" / "509_draw-table.csv",
            draw_comparison_figure=Path("figures") / "510_draw-comparison.pdf",
        )
    ],
)

with open(DEV_FILE, "w") as fh:
    fh.write(converter_yaml.dumps(config))
