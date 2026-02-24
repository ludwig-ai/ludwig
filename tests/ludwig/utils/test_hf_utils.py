import os
import shutil
from typing import Type

import pytest
from transformers import AlbertModel, BertModel, BertTokenizer

from ludwig.encoders.text_encoders import ALBERTEncoder, BERTEncoder
from ludwig.utils.hf_utils import (
    load_pretrained_hf_model_from_hub,
    load_pretrained_hf_model_with_hub_fallback,
    upload_folder_to_hfhub,
)


@pytest.mark.parametrize(
    ("model", "name"),
    [
        (AlbertModel, ALBERTEncoder.DEFAULT_MODEL_NAME),
        (BertTokenizer, "bert-base-uncased"),
    ],
)
def test_load_pretrained_hf_model_from_hub(model: type, name: str, tmpdir: os.PathLike):
    """Ensure that the HF models used in ludwig download correctly."""
    cache_dir = os.path.join(tmpdir, name.replace(os.path.sep, "_") if name else str(model.__name__))
    os.makedirs(cache_dir, exist_ok=True)
    loaded_model = load_pretrained_hf_model_from_hub(model, name, cache_dir=cache_dir, force_download=True)
    assert isinstance(loaded_model, model)
    assert os.listdir(cache_dir)


def test_load_pretrained_hf_model_with_hub_fallback(tmpdir):
    """Ensure that the HF models used in ludwig download correctly with S3 or hub fallback."""
    # Don't set env var.
    _, used_fallback = load_pretrained_hf_model_with_hub_fallback(AlbertModel, ALBERTEncoder.DEFAULT_MODEL_NAME)
    assert used_fallback

    # Download the model, load it from tmpdir, and set env var.
    load_pretrained_hf_model_from_hub(AlbertModel, "albert-base-v2").save_pretrained(
        os.path.join(tmpdir, "albert-base-v2")
    )
    os.environ["LUDWIG_PRETRAINED_MODELS_DIR"] = f"file://{tmpdir}"  # Needs to be an absolute path.
    _, used_fallback = load_pretrained_hf_model_with_hub_fallback(AlbertModel, ALBERTEncoder.DEFAULT_MODEL_NAME)
    assert not used_fallback

    # Fallback is used for a model that doesn't exist in models directory.
    _, used_fallback = load_pretrained_hf_model_with_hub_fallback(BertModel, BERTEncoder.DEFAULT_MODEL_NAME)
    assert used_fallback

    # Clean up.
    del os.environ["LUDWIG_PRETRAINED_MODELS_DIR"]


@pytest.fixture
def tmp_folder_with_file(tmpdir):
    # Create a temporary folder
    tmp_folder = str(tmpdir.mkdir("tmp_folder"))

    # Create a file within the temporary folder
    file_path = os.path.join(tmp_folder, "test_file.txt")
    with open(file_path, "w") as f:
        f.write("Test content")

    yield tmp_folder

    # Clean up: Remove the temporary folder and its contents
    shutil.rmtree(tmp_folder)


def test_upload_folder_to_hfhub_folder_not_exist():
    with pytest.raises(FileNotFoundError, match=r"Folder .* does not exist."):
        upload_folder_to_hfhub("test_repo", "/nonexistent_folder")


def test_upload_folder_to_hfhub_folder_empty(tmpdir):
    empty_folder = str(tmpdir.mkdir("empty_folder"))
    with pytest.raises(ValueError, match=r"Folder .* is empty."):
        upload_folder_to_hfhub("test_repo", empty_folder)


def test_upload_folder_to_hfhub_folder_is_file(tmpdir):
    file_path = str(tmpdir.join("test_file.txt"))
    with open(file_path, "w") as f:
        f.write("Test content")
    with pytest.raises(ValueError, match=r"Folder .* is a file. Please provide a folder."):
        upload_folder_to_hfhub("test_repo", file_path)


def test_upload_folder_to_hfhub_invalid_repo_type(tmp_folder_with_file):
    with pytest.raises(ValueError, match=r"Invalid repo_type .*"):
        upload_folder_to_hfhub("test_repo", tmp_folder_with_file, repo_type="invalid_type")
