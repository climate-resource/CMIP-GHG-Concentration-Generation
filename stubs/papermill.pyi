import os
from typing import Any

import nbformat

def execute_notebook(
    input_notebook: os.PathLike,
    output_notebook: os.PathLike,
    parameters: dict[str, Any],
) -> nbformat.NotebookNode: ...
