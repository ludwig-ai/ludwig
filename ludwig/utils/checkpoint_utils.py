"""Implements similar functionality as tf.train.Checkpoint and tf.train.CheckpointManager.

https://gist.github.com/kevinzakka/5d345421f7abefd5dbaf6a77f829e70a.
"""
import logging
import os
import os.path as osp
import queue
import re
import signal
import threading
from glob import glob

import numpy as np
import torch

CHECKPOINTS_LOCK = threading.Lock()


def mkdir(s):
    """Create a directory if it doesn't already exist."""
    if not osp.exists(s):
        os.makedirs(s)


def get_files(d, pattern, sort=True):
    """Return a list of files in a given directory.

    Args:
      d (str): The path to the directory.
      pattern (str): The wildcard to filter files with.
      sort (bool): Whether to sort the returned list. Assumes filenames contain a number value to sort by (tmp-001).
    """
    files = glob(osp.join(d, pattern))
    files = [f for f in files if osp.isfile(f)]
    if sort:

        def filter_numeric(s):
            return re.sub("[^0-9]", "", s)

        files.sort(key=lambda x: int(filter_numeric(os.path.basename(x).split(".")[0])))
    return files


def traim_checkpoints_loop(q: queue.Queue, directory: str, max_to_keep: int):
    """Trim older checkpoints until `max_to_keep` remain."""
    while True:
        should_continue = q.get()
        if should_continue is False:
            return

        with CHECKPOINTS_LOCK:
            # get a list of checkpoints in reverse
            # chronological order
            ckpts = get_files(directory, "*.ckpt")[::-1]

            # remove until `max_to_keep` remain
            num_remove = len(ckpts) - max_to_keep
            while num_remove > 0:
                ckpt_name = ckpts.pop()
                os.remove(ckpt_name)
                num_remove -= 1


class Checkpoint:
    """Save and restore model and optimizer states."""

    def __init__(self, model, optimizer=None):
        """Constructor."""
        self.model = model
        self.optimizer = optimizer

    def restore(self, save_path, device=None):
        """Restore a state from a saved checkpoint.

        Args:
          save_path (str): The filepath to the saved checkpoint.
          device (torch.device): The device on which to
            restore the state.
        """
        try:
            state = torch.load(save_path, map_location=device)
            try:
                self.model.load_state_dict(state["model_weights"])
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(state["optim_state"])
                logging.info(f"Successfully loaded model weights from {save_path}.")
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
            logging.error(e)
            return False

    def save(self, save_path):
        """Save a state to disk.

        Modified from brentyi/fannypack.
        Args:
          save_path (str): The name of the checkpoint to save.
        """
        state = {"model_weights": self.model.state_dict()}
        if self.optimizer is not None:
            state["optim_state"] = self.optimizer.state_dict()

        # ignore ctrl+c while saving
        try:
            orig_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _sig, _frame: None)
        except ValueError:
            # signal throws a ValueError if we're not in the main thread
            orig_handler = None

        # atomic save
        save_dir = osp.dirname(save_path)
        tmp_path = osp.join(save_dir, f"tmp-{np.random.randint(1e9)}.ckpt")
        torch.save(state, tmp_path)
        # replace is an atomic operation in python
        # it is POSIX compliant according to docs
        # https://docs.python.org/3/library/os.html#os.replace
        os.replace(tmp_path, save_path)
        logging.debug(f"Saved checkpoint at {save_path}.")

        # restore SIGINT handler
        if orig_handler is not None:
            signal.signal(signal.SIGINT, orig_handler)


class CheckpointManager:
    """A model and optimizer checkpoint manager."""

    def __init__(self, checkpoint, directory, device, max_to_keep=10):
        """Constructor.

        Args:
          checkpoint (Checkpoint): An instance of `Checkpoint`.
          directory (str): The directory in which checkpoints will be saved.
          device (torch.device): The computing device on which to restore
            checkpoints.
          max_to_keep (int): The maximum number of checkpoints to keep.
            Amongst all saved checkpoints, checkpoints will be deleted
            oldest first, until `max_to_keep` remain.
        """
        assert max_to_keep > 0, "max_to_keep should be a positive integer."

        self.checkpoint = checkpoint
        self.directory = directory
        self.max_to_keep = max_to_keep
        self.device = device
        self.latest_checkpoint = None

        # create checkpoint directory if it doesn't
        # already exist
        mkdir(self.directory)

        self.queue = queue.Queue()
        self.trim_thread = threading.Thread(
            target=traim_checkpoints_loop, args=(self.queue, self.directory, self.max_to_keep)
        )
        self.trim_thread.start()

    def restore_or_initialize(self):
        """Restore items in checkpoint from the latest checkpoint file.

        Returns:
          The global iteration step. This is parsed from the latest
            checkpoint file if one is found, else 0 is returned.
        """
        ckpts = get_files(self.directory, "*.ckpt")
        if ckpts:
            last_ckpt = ckpts[-1]
            status = self.checkpoint.restore(last_ckpt, self.device)
            if not status:
                logging.info("Could not restore latest checkpoint file.")
                return 0
            self.latest_checkpoint = last_ckpt
            return int(osp.basename(last_ckpt).split(".")[0])
        return 0

    def save(self, global_step):
        """Create a new checkpoint.

        Args:
          global_step (int): The iteration number which will be used
            to name the checkpoint.
        """
        save_path = osp.join(self.directory, f"{global_step:09d}.ckpt")
        self.checkpoint.save(save_path)
        self.latest_checkpoint = save_path
        self.queue.put(True)

    def close(self):
        self.queue.put(False)
        self.trim_thread.join()

    @staticmethod
    def load_latest_checkpoint(checkpoint, directory, device):
        ckpts = get_files(directory, "*.ckpt")
        if ckpts:
            last_ckpt = ckpts[-1]
            checkpoint.restore(last_ckpt, device)
        else:
            logging.error(f"No checkpoints found in {directory}.")
