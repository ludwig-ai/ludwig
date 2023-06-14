import faiss
import numpy as np

from ludwig.vector_index.base import VectorIndex


class FaissIndex(VectorIndex):
    def __init__(self, index: faiss.Index):
        self.index = index

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        top_k = self.index.search(query.reshape(1, -1), k)
        return top_k[1].tolist()[0]

    def save(self, path: str):
        faiss.write_index(self.index, path)

    @classmethod
    def from_path(cls, path: str) -> "VectorIndex":
        index = faiss.read_index(path)
        return cls(index)

    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray) -> "VectorIndex":
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return cls(index)
