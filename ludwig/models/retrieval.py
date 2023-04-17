import hashlib
import json
from typing import Any, Dict, List, Optional, Union

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
    def create_dataset_index(self, df: pd.DataFrame, columns_to_index: Optional[List[str]] = None):
        raise NotImplementedError

    def search(self, query: str, k: int = 10, return_data: bool = False) -> Union[List[int], List[Dict[str, Any]]]:
        raise NotImplementedError

    def save_index(self, name: str):
        raise NotImplementedError
    
    def load_index(self, name: str):
        raise NotImplementedError
    
    def ping(self) -> bool:
        return True

class RandomRetrieval(RetrievalModel):
    def __init__(self, **kwargs):
        self.index = None
        self.index_data = None

    def create_dataset_index(self, df: pd.DataFrame, columns_to_index: Optional[List[str]] = None):
        self.index = np.array(range(len(df)))
        self.index_data = df

    def search(self, query: str, k: int = 10, return_data: bool = False) -> Union[List[int], List[Dict[str, Any]]]:
        indices = np.random.choice(self.index, k, replace=False).tolist()

        if return_data:
            return self.index_data.iloc[indices].to_dict(orient="records")
        return indices
    
    def save_index(self, name: str):
        np.save(name + ".index", self.index)
        self.index_data.to_csv(name + ".csv", index=False)
        
    def load_index(self, name: str):
        self.index = np.load(name + ".index")
        self.index_data = pd.read_csv(name + ".csv")



# @ray.remote
class SemanticRetrieval(RetrievalModel):
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index = None
        self.index_data: pd.DataFrame = None
        self.checksum = None

    def create_dataset_index(self, df: pd.DataFrame, columns_to_index: Optional[List[str]] = None):
        if columns_to_index is None:
            columns_to_index = df.columns
        df_to_index = df[columns_to_index]

        new_checksum = df_checksum(df_to_index)
        if self.checksum == new_checksum:
            # Reuse existing index
            # TODO(travis): could use an LRU cache to support multiple datasets concurrently
            return

        self.checksum = new_checksum
        rows = df_to_index.to_dict(orient="records")
        row_strs = [json.dumps(r) for r in rows]
        embeddings = self.model.encode(row_strs)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        # Save the entire df so we can return the full row when searching
        self.index_data = df

    def search(self, query: str, k: int = 10, return_data: bool = False) -> Union[List[int], List[Dict[str, Any]]]:
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, k)
        indices = top_k[1].tolist()[0]
        
        if return_data:
            return self.index_data.iloc[indices].to_dict(orient="records")
        return indices

    def save_index(self, name: str):
        faiss.write_index(self.index, name + ".index")
        self.index_data.to_csv(name + ".csv", index=False)

    def load_index(self, name: str):
        self.index = faiss.read_index(name + ".index")
        self.index_data = pd.read_csv(name + ".csv")


def get_retrieval_model(type: str, **kwargs) -> RetrievalModel:
    if type == "random":
        return RandomRetrieval(**kwargs)
    elif type == "semantic":
        return SemanticRetrieval(**kwargs)
    else:
        raise ValueError(f"Unsupported retrieval model type: {type}")
