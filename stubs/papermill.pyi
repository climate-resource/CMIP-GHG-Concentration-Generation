from pathlib import Path
from typing import Any

import nbformat

def execute_notebook(
    input_notebook: Path,
    output_notebook: Path,
    parameters: dict[str, Any],
) -> nbformat.NotebookNode: ...
