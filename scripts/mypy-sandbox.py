"""
A useful sandbox for figuring out type issues without stuff being everywhere

Not currently used, but may be a helpful example for others. If it is very
confusing, can also be deleted.
"""

# ruff: noqa
from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from attrs import define


class ConfigLike(Protocol):
    @property
    def name(self) -> str: ...


T = TypeVar("T", bound=ConfigLike, contravariant=True)


class SupportsProcessing(Protocol[T]):
    def __call__(self, config: T) -> None: ...


@define
class Processor(Generic[T]):
    process: SupportsProcessing[T]

    def run_process(self, config: T) -> None:
        print(f"Value: {config.name}")
        self.process(config)


@define
class Config:
    name: str
    height: int


def process_func(config: Config) -> None:
    print(f"{config.height=}")


proc_concrete = Processor(process=process_func)
if TYPE_CHECKING:
    reveal_type(proc_concrete)

config = Config(name="bill", height=12)

proc_concrete.run_process(config)
