from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):
    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def from_path(cls, path: str) -> "VectorIndex":
        pass

    @abstractmethod
    def from_embeddings(cls, embeddings: np.ndarray) -> "VectorIndex":
        pass
