from typing import Any

from numpy.typing import NDArray

class Cube:
    attributes: dict[str, str]
    data: NDArray[Any]

    def name(self) -> str: ...

class CubeList:
    def __getitem__(self, key: int) -> Cube: ...
    def __len__(self) -> int: ...
