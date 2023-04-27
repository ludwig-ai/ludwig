import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None

if TYPE_CHECKING:
    from ludwig.backend.base import Backend

from ludwig.utils.batch_size_tuner import BatchSizeEvaluator
from ludwig.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)


def df_checksum(df: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()


class RetrievalModel:
    def create_dataset_index(self, df: pd.DataFrame, backend: "Backend", columns_to_index: Optional[List[str]] = None):
        """Creates an index for the dataset.

        If `columns_to_index` is None, all columns are indexed. Otherwise, only the columns in `columns_to_index` are
        used for indexing, but all columns in `df` are returned in the search results.
        """
        raise NotImplementedError

    def search(
        self, df, backend: "Backend", k: int = 10, return_data: bool = False
    ) -> Union[List[int], List[Dict[str, Any]]]:
        """Retrieve the top k results for the given query.

        If `return_data` is True, returns the data associated with the indices. Otherwise, returns the indices.
        """
        raise NotImplementedError

    def save_index(self, name: str, cache_directory: str):
        """Saves the index to the cache directory."""
        raise NotImplementedError

    def load_index(self, name: str, cache_directory: str):
        """Loads the index from the cache directory."""
        raise NotImplementedError


class RandomRetrieval(RetrievalModel):
    """Random retrieval model.

    Gets k random indices from the dataset regardless of the query.
    """

    def __init__(self, **kwargs):
        self.index = None
        self.index_data = None

    def create_dataset_index(self, df: pd.DataFrame, backend: "Backend", columns_to_index: Optional[List[str]] = None):
        self.index = np.array(range(len(df)))
        self.index_data = df

    def search(
        self, df, backend: "Backend", k: int = 10, return_data: bool = False
    ) -> Union[List[int], List[Dict[str, Any]]]:
        indices = np.random.choice(self.index, k, replace=False).tolist()

        if return_data:
            return self.index_data.iloc[indices].to_dict(orient="records")
        return indices

    def save_index(self, name: str, cache_directory: str):
        logger.info(f"Saving index to cache directory {cache_directory} with name {name}")
        index_file_path = os.path.join(cache_directory, name + ".index")
        np.save(index_file_path, self.index)

        index_data_file_path = os.path.join(cache_directory, name + "_data.csv")
        self.index_data.to_csv(index_data_file_path, index=False)

    def load_index(self, name: str, cache_directory: str):
        logger.info(f"Loading index from cache directory {cache_directory} with name {name}")
        index_file_path = os.path.join(cache_directory, name + ".index")
        self.index = np.load(index_file_path)

        index_data_file_path = os.path.join(cache_directory, name + "_data.csv")
        self.index_data = pd.read_csv(index_data_file_path)


class SemanticRetrieval(RetrievalModel):
    """Semantic retrieval model.

    Uses a sentence transformer model to encode the dataset and retrieve the top k most similar results to the query.
    """

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.index: faiss.Index = None
        self.index_data: pd.DataFrame = None
        self.checksum = None

        # best batch size computed during the encoding step
        self.best_batch_size = None

    def create_dataset_index(self, df: pd.DataFrame, backend: "Backend", columns_to_index: Optional[List[str]] = None):
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

        embeddings = self._encode(row_strs, backend)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        # Save the entire df so we can return the full row when searching
        self.index_data = df

    def _encode(self, row_strs: List[str], backend: "Backend") -> np.ndarray:
        # only do this step once
        if self.best_batch_size is None:
            self.best_batch_size = backend.tune_batch_size(
                create_semantic_retrieval_model_evaluator(self.model_name, row_strs), len(row_strs)
            )

        transform_fn = create_semantic_retrieval_model_fn(self.model_name, self.best_batch_size)
        df = backend.df_engine.from_pandas(pd.DataFrame({"data": row_strs}))
        df = backend.batch_transform(df, self.best_batch_size, transform_fn)
        df = backend.df_engine.compute(df)
        embeddings = np.stack(df["data"].values).astype(np.float32)
        return embeddings

    def search(
        self, df: pd.DataFrame, backend: "Backend", k: int = 10, return_data: bool = False
    ) -> Union[List[int], List[Dict[str, Any]]]:
        rows = df.to_dict(orient="records")
        row_strs = [json.dumps(r) for r in rows]

        query_vectors = self._encode(row_strs, backend)
        results = []
        # TODO(geoffrey): figure out why self.index.search segfaults with larger batch sizes
        for query_vector in tqdm(query_vectors, total=query_vectors.shape[0]):
            top_k = self.index.search(query_vector.reshape(1, -1), k)
            indices = top_k[1].tolist()[0]

            if return_data:
                result = self.index_data.iloc[indices].to_dict(orient="records")
            else:
                result = indices
            results.append(result)
        return results

    def save_index(self, name: str, cache_directory: str):
        logger.info(f"Saving index to cache directory {cache_directory} with name {name}")
        index_file_path = os.path.join(cache_directory, name + ".index")
        faiss.write_index(self.index, index_file_path)

        index_data_file_path = os.path.join(cache_directory, name + "_data.csv")
        self.index_data.to_csv(index_data_file_path, index=False)

    def load_index(self, name: str, cache_directory: str):
        logger.info(f"Loading index from cache directory {cache_directory} with name {name}")
        index_file_path = os.path.join(cache_directory, name + ".index")
        self.index = faiss.read_index(index_file_path)

        index_data_file_path = os.path.join(cache_directory, name + "_data.csv")
        self.index_data = pd.read_csv(index_data_file_path)


def create_semantic_retrieval_model_evaluator(model_name, samples):
    class _RetrievalModelEvaluator(BatchSizeEvaluator):
        def __init__(self):
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=get_torch_device())
            self.samples = samples

        def step(self, batch_size: int):
            self.model.encode(self.samples[:batch_size], batch_size=batch_size, show_progress_bar=False)

    return _RetrievalModelEvaluator


def create_semantic_retrieval_model_fn(model_name, batch_size):
    class _RetrievalModelFn:
        def __init__(self):
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=get_torch_device())
            self.batch_size = batch_size

        def __call__(self, df: pd.DataFrame) -> np.ndarray:
            row_strs = df["data"].tolist()
            result = self.model.encode(row_strs, batch_size=self.batch_size, show_progress_bar=False)
            df["data"] = result.tolist()
            return df

    return _RetrievalModelFn


def get_retrieval_model(type: str, **kwargs) -> RetrievalModel:
    if type == "random":
        return RandomRetrieval(**kwargs)
    elif type == "semantic":
        return SemanticRetrieval(**kwargs)
    else:
        raise ValueError(f"Unsupported retrieval model type: {type}")
