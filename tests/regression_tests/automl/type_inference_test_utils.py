from pathlib import Path

REGISTRY = {"adult_census_income", "mnist"}


def get_dataset_golden_types_path(dataset_name: str) -> str:
    return str(Path(__file__).resolve().parent / "golden" / f"{dataset_name}.types.json")
