import hashlib
import json
from typing import List

import faiss
import pandas as pd
import ray
import numpy as np
from sentence_transformers import SentenceTransformer


SENTENCE_TRANSFORMER_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"


def df_checksum(df: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()


# TODO: support Ray backend
# TODO: support loading index from disk
class RetrievalModel:
    def create_dataset_index(self, df: pd.DataFrame):
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> List[int]:
        raise NotImplementedError

    def ping(self) -> bool:
        return True


class RandomRetrieval(RetrievalModel):
    def __init__(self, **kwargs):
        self.index = None

    def create_dataset_index(self, df: pd.DataFrame):
        self.index = range(len(df))

    def search(self, query: str, k: int = 10) -> List[int]:
        return np.random.choice(self.index, k, replace=False).tolist()


# @ray.remote
class IndexRetrieval(RetrievalModel):
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index = None
        self.checksum = None

    def create_dataset_index(self, df: pd.DataFrame):
        new_checksum = df_checksum(df)
        if self.checksum == new_checksum:
            # Reuse existing index
            # TODO(travis): could use an LRU cache to support multiple datasets concurrently
            return

        self.checksum = new_checksum
        rows = df.to_dict(orient="records")
        row_strs = [json.dumps(r) for r in rows]
        embeddings = self.model.encode(row_strs)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, k: int = 10) -> List[int]:
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, k)
        indices = top_k[1].tolist()[0]
        return indices

    def save_index(self, path: str):
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        self.index = faiss.read_index(path)

    def ping(self) -> bool:
        return True


def get_retrieval_model(type: str, **kwargs) -> RetrievalModel:
    if type == "random":
        return RandomRetrieval(**kwargs)
    elif type == "index":
        return IndexRetrieval(**kwargs)
    else:
        raise ValueError(f"Unsupported retrieval model type: {type}")
