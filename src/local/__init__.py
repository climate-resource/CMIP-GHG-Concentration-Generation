"""
Local code to support running the notebooks
"""
import importlib.metadata

__version__ = importlib.metadata.version("local")


def get_key_info() -> str:
    """
    Get key information about the project

    Returns
    -------
        Key information
    """
    return "\n".join(
        [
            "CMIP greenhouse gas concentration generation",
            f"Version: {__version__} (translation: prototype)",
            "See the main repository for license, docs etc.",
        ]
    )
