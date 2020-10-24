import os
import unittest
from unittest import mock
from ludwig.datasets.mnist import Mnist


class TestMNistWorkflow(unittest.TestCase):

    def setUp(self):
        self._cache_dir = "/home/saikatkanjilal/.ludwig_cache"
        self.dataset = Mnist(cache_dir=self._cache_dir)
        self.raw_dataset_path = os.path.join(self._cache_dir, "mnist_1.0", "raw")

    def test_mnist_download(self):
        self.dataset.download()
        assert os.path.isfile(os.path.join(self.raw_dataset_path, "train-images-idx3-ubyte"))
        assert os.path.exists(os.path.join(self.raw_dataset_path , "train-labels-idx1-ubyte"))
        assert os.path.exists(os.path.join(self.raw_dataset_path, "t10k-images-idx3-ubyte"))
        assert os.path.exists(os.path.join(self.raw_dataset_path, "t10k-labels-idx1-ubyte"))

