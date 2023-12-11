from __future__ import annotations

import logging
import pathlib
import shutil

import pytest

from ludwig.utils.upload_utils import HuggingFaceHub

logger = logging.getLogger(__name__)


def _build_dummy_tmp_model_repo(destination_directory: str, file_names: list[str]) -> None:
    """This utility function accepts the "destination_directory" and list of file names on input.

    It then makes directory hierarchy "my_simple_experiment_run" / "model" / "model_weights" under
    "destination_directory" and creates empty files for each file name specified in bottom-most (leaf) directory (file
    names must be leaf file names, not paths).
    """
    # Create a temporary folder designating training output directory.
    model_directory: str = pathlib.Path(destination_directory) / "my_simple_experiment_run" / "model"
    model_weights_directory: str = model_directory / "model_weights"
    model_weights_directory.mkdir(parents=True, exist_ok=True)

    # Create files within the "model_weights" subdirectory.
    file_name: str
    for file_name in file_names:
        pathlib.Path(model_weights_directory / file_name).touch()


@pytest.fixture
def output_directory_manager(tmpdir) -> str:
    """This convenience fixture creates temporary directory "training_results_output" and yields it to user test
    functions.

    When the user test functions complete their execution, this fixture resumes and cleans up the temporary directory.
    """
    # Create a temporary folder designating training output directory.
    output_directory: str = str(tmpdir.mkdir("training_results_output"))

    yield output_directory

    # Clean up: Remove the temporary output directory and its contents.
    shutil.rmtree(output_directory)


@pytest.mark.parametrize(
    "file_names,error_raised",
    [
        pytest.param(
            [
                "pytorch_model.bin",
            ],
            None,
            id="pretrained_model_weights_bin",
        ),
        pytest.param(
            [
                "adapter_model.bin",
            ],
            None,
            id="adapter_model_weights_bin_unmerged",  # backward compatibility for peft versions < 0.7.0
        ),
        pytest.param(
            [
                "adapter_model.safetensors",
            ],
            None,
            id="adapter_model_weights_safetensors_unmerged",
        ),
        pytest.param(
            [
                "adapter_model.bin",
                "adapter_model.safetensors",
            ],
            None,
            id="adapter_model_weights_bin_and_safetensors_unmerged",  # backward compatibility for peft versions < 0.7.0
        ),
        pytest.param(
            [
                "pytorch_model.bin",
                "adapter_model.safetensors",
            ],
            None,
            id="pretrained_model_weights_bin_and_adapter_model_weights_safetensors_merged",
        ),
        pytest.param(
            [],
            (
                ValueError,
                "Can't find model weights at {model_weights_path}. Trained model weights should either be saved as `pytorch_model.bin` for regular model training, or have `adapter_model.bin`or `adapter_model.safetensors` if using parameter efficient fine-tuning methods like LoRA.",  # noqa E501
            ),
            id="model_weights_missing",
        ),
        pytest.param(
            [
                "pytorch_model.safetensors",
            ],
            (
                ValueError,
                "Can't find model weights at {model_weights_path}. Trained model weights should either be saved as `pytorch_model.bin` for regular model training, or have `adapter_model.bin`or `adapter_model.safetensors` if using parameter efficient fine-tuning methods like LoRA.",  # noqa E501
            ),
            id="model_weights_unexpected_name_format_combination",
        ),
        pytest.param(
            [
                "pytorch_model.unkn",
            ],
            (
                ValueError,
                "Can't find model weights at {model_weights_path}. Trained model weights should either be saved as `pytorch_model.bin` for regular model training, or have `adapter_model.bin`or `adapter_model.safetensors` if using parameter efficient fine-tuning methods like LoRA.",  # noqa E501
            ),
            id="model_weights_unrecognized_format",
        ),
        pytest.param(
            [
                "unknown_model.safetensors",
            ],
            (
                ValueError,
                "Can't find model weights at {model_weights_path}. Trained model weights should either be saved as `pytorch_model.bin` for regular model training, or have `adapter_model.bin`or `adapter_model.safetensors` if using parameter efficient fine-tuning methods like LoRA.",  # noqa E501
            ),
            id="model_weights_unrecognized_name",
        ),
    ],
)
@pytest.mark.unit
def test_upload_to_hf_hub__validate_upload_parameters(
    output_directory_manager, file_names: list[str], error_raised: tuple[type, str] | None
):
    """Test "HuggingFaceHub._validate_upload_parameters()", which is executed in the path of upload to HuggingFace
    Hub; for example: `upload hf_hub -repo_id "hf-account/repo-name" --model_path.

    /content/results/api_experiment_run`.

    Each test case consists of: 1) Populating the temporary output directory ("training_results_output) with zero or
    more test model weights file; 2) Executing "HuggingFaceHub._validate_upload_parameters()"; and 3) Asserting on
    presence/absence of errors.
    """
    output_directory: str = output_directory_manager
    _build_dummy_tmp_model_repo(destination_directory=output_directory, file_names=file_names)

    model_path: pathlib.Path = pathlib.Path(output_directory) / "my_simple_experiment_run"
    model_weights_path: pathlib.Path = pathlib.Path(model_path / "model" / "model_weights")

    repo_id: str = "dummy_account/dummy_repo"
    model_path: str = str(model_path)
    if error_raised:
        error_class: type  # noqa [F842]  # incorrect flagging of "local variable is annotated but never used
        error_message: str  # noqa [F842]  # incorrect flagging of "local variable is annotated but never used
        error_class, error_message = error_raised
        with pytest.raises(error_class) as excinfo:
            HuggingFaceHub._validate_upload_parameters(
                repo_id=repo_id,
                model_path=model_path,
            )

        assert str(excinfo.value) == error_message.format(model_weights_path=model_weights_path)
    else:
        try:
            HuggingFaceHub._validate_upload_parameters(
                repo_id=repo_id,
                model_path=model_path,
            )
        except Exception as exc:
            assert False, f'"HuggingFaceHub._validate_upload_parameters()" raised an exception: "{exc}".'
