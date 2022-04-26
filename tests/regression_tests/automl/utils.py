from pathlib import Path

from ludwig.datasets import dataset_registry
from ludwig.datasets.base_dataset import BaseDataset

REGISTRY = {"adult_census_income", "mnist"}


def get_dataset_golden_types_path(dataset_name: str) -> str:
    return str(Path(__file__).resolve().parent / "golden" / f"{dataset_name}.types.json")


def get_dataset_object(dataset_name: str) -> BaseDataset:
    return dataset_registry[dataset_name]()
