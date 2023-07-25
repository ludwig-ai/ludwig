import logging
import os
from typing import Optional

from huggingface_hub import HfApi, login

logger = logging.getLogger(__name__)


class HuggingFaceHub:
    def __init__(self):
        self.api = None

    def login(self):
        """Login to huggingface hub using the token stored in ~/.cache/huggingface/token and returns a HfApi client
        object that can be used to interact with HF Hub."""
        cached_token_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")

        if not os.path.exists(cached_token_path):
            login(add_to_git_credential=True)

        with open(cached_token_path) as f:
            hf_token = f.read()

        hf_api = HfApi(token=hf_token)
        assert hf_api.token == hf_token

        self.api = hf_api

    def upload_to_hfhub(
        self,
        repo_id: str,
        model_path: str,
        repo_type: Optional[str] = None,
        private: Optional[bool] = False,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
    ):
        """Create an empty repo on the HuggingFace Hub and upload trained model artifacts to that repo.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            model_path (`str`):
                The path of the saved model. This is the top level directory where
                the models weights as well as other associated training artifacts
                are saved.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            private (`bool`, *optional*, defaults to `False`):
                Whether the model repo should be private.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to:
                `f"Upload {path_in_repo} with huggingface_hub"`
            commit_description (`str` *optional*):
                The description of the generated commit
        """
        # Validate repo_id has both a namespace and a repo name
        assert "/" in repo_id, (
            "`repo_id` must be a namespace (user or an organization) and a repo name separated by a `/`."
            " For example, if your HF username is `johndoe` and you want to create a repository called `test`, the"
            " repo_id should be johndoe/test"
        )

        # Make sure the model's save path is actually a valid path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The path '{model_path}' does not exist.")

        # Make sure the model is actually trained
        trained_model_artifacts_path = os.path.join(model_path, "model", "model_weights")
        if not os.path.exists(trained_model_artifacts_path):
            raise Exception(
                f"Model artifacts not found at {trained_model_artifacts_path}. "
                f"It is possible that model at '{model_path}' hasn't been trained yet, or something went"
                "wrong during training where the model's weights were not saved."
            )

        # Make sure the model's saved artifacts either contain:
        # 1. pytorch_model.bin -> regular model training, such as ECD or for LLMs
        # 2. adapter_model.bin -> LLM fine-tuning using PEFT
        files = set(os.listdir(trained_model_artifacts_path))
        if "pytorch_model.bin" not in files and "adapter_model.bin" not in files:
            raise Exception(
                f"Can't find model weights at {trained_model_artifacts_path}. Trained model weights should "
                "either be saved as `pytorch_model.bin` for regular model training, or have `adapter_model.bin`"
                "if using parameter efficient fine-tuning methods like LoRA."
            )

        # Create empty model repo using repo_id, but it is okay if it already exists.
        self.api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type=repo_type,
            exist_ok=True,
        )

        # Upload all artifacts in model weights folder
        upload_path = self.api.upload_folder(
            folder_path=trained_model_artifacts_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
            commit_description=commit_description,
        )

        logger.info(f"Model uploaded to `{upload_path}` with repository name `{repo_id}`")
