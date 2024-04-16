"""
Configuration handling
"""

from __future__ import annotations

from functools import partial
from typing import TypeVar

import pydoit_nb.serialization

from .base import Config

T = TypeVar("T")

converter_yaml = pydoit_nb.serialization.converter_yaml


load_config_from_file = partial(
    pydoit_nb.serialization.load_config_from_file,
    target=Config,
    converter=converter_yaml,
)
