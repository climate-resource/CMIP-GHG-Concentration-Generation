from collections.abc import Callable
from pathlib import Path

class Pooch: ...

class Unzip:
    def __init__(
        self, members: list[str] | None = None, extract_dir: str | None = None
    ): ...
    def __call__(self, fname: str, action: str, pooch: Pooch) -> list[Path]: ...

def retrieve(
    url: str,
    known_hash: str,
    fname: str | None = None,
    path: Path | None = None,
    processor: None | Callable[[str, str, Pooch], Path | list[Path]] = None,
    progressbar: bool = False,
) -> Path | list[Path]: ...
