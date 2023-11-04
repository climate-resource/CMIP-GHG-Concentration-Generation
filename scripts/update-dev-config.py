"""
Update development config
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from local.config import Config, converter_yaml
from local.config.covariance import CovarianceConfig
from local.config.preparation import PreparationConfig

DEV_FILE: Path = Path("config-dev.yaml")


config = Config(
    name="dev",
    preparation=[
        PreparationConfig(
            branch_config_id="only",
            seed=2847539,
            seed_file=Path("seed.txt"),
        )
    ],
    covariance=[
        CovarianceConfig(
            branch_config_id="cov",
            covariance=np.array([[0.25, 0.5], [0.5, 0.55]]),
        ),
        CovarianceConfig(
            branch_config_id="no-cov",
            covariance=np.array([[0.25, 0], [0, 0.55]]),
        ),
    ],
)

with open(DEV_FILE, "w") as fh:
    fh.write(converter_yaml.dumps(config))
