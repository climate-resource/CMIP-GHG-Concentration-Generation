"""
pyproject.toml
Local code to support running the notebooks
"""
import importlib.metadata

__version__ = importlib.metadata.version("local")


def get_key_info() -> str:
    return "\n".join(
        [
            "CMIP greenhouse gas concentration generation",
            f"Version: {__version__} (tranlsation: prototype)",
            "See the main repository for license, docs etc.",
        ]
    )
