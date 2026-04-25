"""Data format detection and registry for Ludwig.

Provides automatic format detection from file extensions and a registry of format-specific preprocessor classes.

Extracted from the monolithic preprocessing.py for better modularity.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Maps file extensions to Ludwig format names
EXTENSION_TO_FORMAT = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".feather": "feather",
    ".fwf": "fwf",
    ".html": "html",
    ".orc": "orc",
    ".sas7bdat": "sas",
    ".sav": "spss",
    ".dta": "stata",
    ".pickle": "pickle",
    ".pkl": "pickle",
    ".hdf5": "hdf5",
    ".h5": "hdf5",
}


def detect_format(path: str) -> str | None:
    """Detect data format from file extension.

    Args:
        path: Path to the data file.

    Returns:
        Format string (e.g., "csv", "parquet") or None if unrecognized.
    """
    if not isinstance(path, str):
        return None

    _, ext = os.path.splitext(path.lower())
    return EXTENSION_TO_FORMAT.get(ext)


def detect_format_from_dataset(dataset) -> str:
    """Detect format from a dataset argument (path, dict, or DataFrame).

    Args:
        dataset: Input dataset (str path, dict, pd.DataFrame, etc.)

    Returns:
        Format string.
    """
    import pandas as pd

    if isinstance(dataset, pd.DataFrame):
        return "df"
    elif isinstance(dataset, dict):
        return "dict"
    elif isinstance(dataset, str):
        detected = detect_format(dataset)
        if detected:
            return detected
        # Could be a directory or unknown format
        if os.path.isdir(dataset):
            return "auto"
        return "auto"
    else:
        return "auto"
