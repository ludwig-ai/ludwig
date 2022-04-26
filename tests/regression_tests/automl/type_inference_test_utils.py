# from os import path
from pathlib import Path

from ludwig.datasets import adult_census_income, mnist

REGISTRY = {adult_census_income, mnist}


def get_dataset_golden_types_path(dataset_module) -> str:
    return str(Path(__file__).resolve().parent / "golden" / f"{dataset_module.__name__.split('.')[-1]}.types.json")


for d in REGISTRY:
    print(get_dataset_golden_types_path(d))
