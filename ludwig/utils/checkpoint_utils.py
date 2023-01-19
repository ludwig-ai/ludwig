"""Implements similar functionality as tf.train.Checkpoint and tf.train.CheckpointManager.

https://gist.github.com/kevinzakka/5d345421f7abefd5dbaf6a77f829e70a.
"""

import logging
import os
import re
import signal
import tempfile
from glob import glob
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer

from ludwig.api_annotations import DeveloperAPI
from ludwig.models.base import BaseModel
from ludwig.modules.lr_scheduler import LRScheduler
from ludwig.utils.fs_utils import safe_move_file

logger = logging.getLogger(__name__)
LATEST_FNAME = "latest.ckpt"


@DeveloperAPI
def mkdir(s):
    """Create a directory if it doesn't already exist."""
    if not os.path.exists(s):
        os.makedirs(s)


@DeveloperAPI
def get_files(d, pattern, sort=True):
    """Return a list of files in a given directory.

    Args:
      d (str): The path to the directory.
      pattern (str): The wildcard to filter files with.
      sort (bool): Whether to sort the returned list. Assumes filenames contain a number value to sort by (tmp-001).
    """
    files = glob(os.path.join(d, pattern))
    files = [f for f in files if os.path.isfile(f)]
    if sort:

        def filter_numeric(s):
            return re.sub("[^0-9]", "", s)

        files.sort(key=lambda x: int(filter_numeric(os.path.basename(x).split(".")[0])))
    return files


@DeveloperAPI
def get_latest_checkpoint_path(directory: str) -> str:
    latest_path = os.path.join(directory, LATEST_FNAME)
    if os.path.exists(latest_path):
        return latest_path

    # Legacy codepath for checkpoints saved by global step number
    ckpts = get_files(directory, "*.ckpt")
    if ckpts:
        return ckpts[-1]

    return None


@DeveloperAPI
class Checkpoint:
    """Save and restore model and optimizer states."""

    def __init__(
        self, model: BaseModel, optimizer: Optional[Optimizer] = None, scheduler: Optional[LRScheduler] = None
    ):
        """Constructor."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = 0

    def restore(self, save_path: str, device: Optional[torch.device] = None) -> bool:
        """Restore a state from a saved checkpoint.

        Args:
          save_path (str): The filepath to the saved checkpoint.
          device (torch.device): The device on which to
            restore the state.

        Returns:
          True if the checkpoint was sucessfully restored, False if the checkpoint file
            could not be found.
        """
        try:
            state = torch.load(save_path, map_location=device)
            try:
                self.global_step = self._get_global_step(state, save_path)
                self.model.load_state_dict(state["model_weights"])
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(state["optim_state"])
                if self.scheduler is not None and "scheduler_state" in state:
                    self.scheduler.load_state_dict(state["scheduler_state"])
                logger.info(f"Successfully loaded model weights from {save_path}.")
                return True
            except Exception as e:
                # there was an issue loading the state which means
                # either the model definition and saved weights
                # do not agree or they were not saved in the first
                # place.
                # since this is a severe issue, we raise an error
                # rather than allowing the program to proceed.
                raise e
        except FileNotFoundError as e:
            logger.error(e)
            return False

    def save(self, save_path: str, global_step: int):
        """Save a state to disk.

        Modified from brentyi/fannypack.

        Args:
          save_path (str): The name of the checkpoint to save.
          global_step (int): The iteration number which will be used
             to name the checkpoint.
        """
        state = {
            "global_step": global_step,
            "model_weights": self.model.state_dict(),
        }
        if self.optimizer is not None:
            state["optim_state"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler_state"] = self.scheduler.state_dict()

        # ignore ctrl+c while saving
        try:
            orig_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _sig, _frame: None)
        except ValueError:
            # signal throws a ValueError if we're not in the main thread
            orig_handler = None

        try:
            # atomic save
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save to a temporary directory outside of the checkpoint dir so
                # async processes do not try and copy a partially-written checkpoint.
                # See Ray Tune and MLFlow for examples of background processes that
                # are affected by this.
                tmp_path = os.path.join(tmpdir, "temp.ckpt")
                torch.save(state, tmp_path)

                safe_move_file(tmp_path, save_path)
                logger.debug(f"Saved checkpoint at {save_path}.")
        finally:
            # restore SIGINT handler
            if orig_handler is not None:
                signal.signal(signal.SIGINT, orig_handler)

    def _get_global_step(self, state: Dict[str, Any], save_path: str) -> int:
        global_step = state.get("global_step")
        if global_step is None:
            # Legacy step detection for older checkpoint format which encoded the
            # step number in the checkpoint filename.
            return int(os.path.basename(save_path).split(".")[0])
        return global_step


@DeveloperAPI
class CheckpointManager:
    """A model and optimizer checkpoint manager."""

    def __init__(self, checkpoint: Checkpoint, directory: str, device: torch.device):
        """Constructor.

        Args:
          checkpoint (Checkpoint): An instance of `Checkpoint`.
          directory (str): The directory in which checkpoints will be saved.
          device (torch.device): The computing device on which to restore
            checkpoints.
        """
        self.checkpoint = checkpoint
        self.directory = directory
        self.device = device
        self.latest_checkpoint = None

        # create checkpoint directory if it doesn't
        # already exist
        mkdir(self.directory)

    def restore_or_initialize(self) -> int:
        """Restore items in checkpoint from the latest checkpoint file.

        Returns:
          The global iteration step. This is parsed from the latest
            checkpoint file if one is found, else 0 is returned.
        """
        last_ckpt = get_latest_checkpoint_path(self.directory)
        if last_ckpt:
            status = self.checkpoint.restore(last_ckpt, self.device)
            if not status:
                logger.warning("Could not restore latest checkpoint file.")
                return 0
            self.latest_checkpoint = last_ckpt
            return self.checkpoint.global_step
        return 0

    def save(self, global_step: int):
        """Create a new checkpoint.

        Args:
           global_step (int): The iteration number which will be used
             to name the checkpoint.
        """
        save_path = os.path.join(self.directory, LATEST_FNAME)
        self.checkpoint.save(save_path, global_step)
        self.latest_checkpoint = save_path

    def close(self):
        pass

    @staticmethod
    def load_latest_checkpoint(checkpoint: Checkpoint, directory: str, device: torch.device):
        last_ckpt = get_latest_checkpoint_path(directory)
        if last_ckpt:
            checkpoint.restore(last_ckpt, device)
        else:
            logger.error(f"No checkpoints found in {directory}.")
