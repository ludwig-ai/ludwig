import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ludwig.vector_index import FAISS, get_vector_index_cls
from ludwig.vector_index.base import VectorIndex

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from ludwig.backend.base import Backend

from ludwig.utils.batch_size_tuner import BatchSizeEvaluator
from ludwig.utils.torch_utils import get_torch_device


def df_checksum(df: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()


def df_to_row_strs(df: pd.DataFrame) -> List[str]:
    rows = df.to_dict(orient="records")
    row_strs = [json.dumps(r) for r in rows]
    return row_strs


class RetrievalModel(ABC):
    @abstractmethod
    def create_dataset_index(self, df: pd.DataFrame, backend: "Backend", columns_to_index: Optional[List[str]] = None):
        """Creates an index for the dataset.

        If `columns_to_index` is None, all columns are indexed. Otherwise, only the columns in `columns_to_index` are
        used for indexing, but all columns in `df` are returned in the search results.
        """
        pass

    @abstractmethod
    def search(
        self, df, backend: "Backend", k: int = 10, return_data: bool = False
    ) -> Union[List[int], List[Dict[str, Any]]]:
        """Retrieve the top k results for the given query.

        If `return_data` is True, returns the data associated with the indices. Otherwise, returns the indices.
        """
        pass

    @abstractmethod
    def save_index(self, name: str, cache_directory: str):
        """Saves the index to the cache directory."""
        pass

    @abstractmethod
    def load_index(self, name: str, cache_directory: str):
        """Loads the index from the cache directory."""
        pass


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
        results = []
        for _ in tqdm(range(len(df))):
            indices = np.random.choice(self.index, k, replace=False)

            if return_data:
                result = self.index_data.iloc[indices].to_dict(orient="records")
            else:
                result = indices
            results.append(result)
        return results

    def save_index(self, name: str, cache_directory: str):
        index_file_path = os.path.join(cache_directory, name + ".index")
        # open file to prevent using the .npy extension
        # https://numpy.org/doc/stable/reference/generated/numpy.save.html
        with open(index_file_path, "wb") as f:
            np.save(f, self.index)

        index_data_file_path = os.path.join(cache_directory, name + "_data.csv")
        self.index_data.to_csv(index_data_file_path, index=False)

    def load_index(self, name: str, cache_directory: str):
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
        self.model = get_semantic_retrieval_model(self.model_name)
        self.index: VectorIndex = None
        self.index_data: pd.DataFrame = None

        # best batch size computed during the encoding step
        self.best_batch_size = None

    def create_dataset_index(self, df: pd.DataFrame, backend: "Backend", columns_to_index: Optional[List[str]] = None):
        if columns_to_index is None:
            columns_to_index = df.columns
        df_to_index = df[columns_to_index]
        row_strs = df_to_row_strs(df_to_index)

        embeddings = self._encode(row_strs, backend)
        self.index = get_vector_index_cls(FAISS).from_embeddings(embeddings)
        # Save the entire df so we can return the full row when searching
        self.index_data = df

    def _encode(self, row_strs: List[str], backend: "Backend") -> np.ndarray:
        # only do this step once
        if self.best_batch_size is None:
            self.best_batch_size = backend.tune_batch_size(
                create_semantic_retrieval_model_evaluator(self.model, row_strs), len(row_strs)
            )

        transform_fn = create_semantic_retrieval_model_fn(self.model, self.best_batch_size)
        df = backend.df_engine.from_pandas(pd.DataFrame({"data": row_strs}))
        df = backend.batch_transform(df, self.best_batch_size, transform_fn)
        df = backend.df_engine.compute(df)
        embeddings = np.stack(df["data"].values).astype(np.float32)
        return embeddings

    def search(
        self, df: pd.DataFrame, backend: "Backend", k: int = 10, return_data: bool = False
    ) -> Union[List[int], List[Dict[str, Any]]]:
        row_strs = df_to_row_strs(df)

        query_vectors = self._encode(row_strs, backend)
        results = []
        # TODO(geoffrey): figure out why self.index.search segfaults with larger batch sizes
        for query_vector in tqdm(query_vectors, total=query_vectors.shape[0]):
            indices = self.index.search(query_vector.reshape(1, -1), k)
            if return_data:
                result = self.index_data.iloc[indices].to_dict(orient="records")
            else:
                result = indices
            results.append(result)
        return results

    def save_index(self, name: str, cache_directory: str):
        index_file_path = os.path.join(cache_directory, name + ".index")
        self.index.save(index_file_path)

        index_data_file_path = os.path.join(cache_directory, name + "_data.csv")
        self.index_data.to_csv(index_data_file_path, index=False)

    def load_index(self, name: str, cache_directory: str):
        index_file_path = os.path.join(cache_directory, name + ".index")
        self.index = get_vector_index_cls(FAISS).from_path(index_file_path)

        index_data_file_path = os.path.join(cache_directory, name + "_data.csv")
        self.index_data = pd.read_csv(index_data_file_path)


def create_semantic_retrieval_model_evaluator(
    model: "SentenceTransformer", samples: List[str]
) -> Type[BatchSizeEvaluator]:
    class _RetrievalModelEvaluator(BatchSizeEvaluator):
        def __init__(self):
            self.model = model.to(get_torch_device())
            self.samples = samples

        def step(self, batch_size: int):
            self.model.encode(self.samples[:batch_size], batch_size=batch_size, show_progress_bar=False)

    return _RetrievalModelEvaluator


def create_semantic_retrieval_model_fn(
    model: "SentenceTransformer", batch_size: int
) -> Callable[[pd.DataFrame], np.ndarray]:
    class _RetrievalModelFn:
        def __init__(self):
            self.model = model.to(get_torch_device())
            self.batch_size = batch_size

        def __call__(self, df: pd.DataFrame) -> np.ndarray:
            row_strs = df["data"].tolist()
            result = self.model.encode(row_strs, batch_size=self.batch_size, show_progress_bar=False)
            df["data"] = result.tolist()
            return df

    return _RetrievalModelFn


def get_semantic_retrieval_model(model_name: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=get_torch_device())


def get_retrieval_model(type: str, **kwargs) -> RetrievalModel:
    if type == "random":
        return RandomRetrieval(**kwargs)
    elif type == "semantic":
        return SemanticRetrieval(**kwargs)
    else:
        raise ValueError(f"Unsupported retrieval model type: {type}")
