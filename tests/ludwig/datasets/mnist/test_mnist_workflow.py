import os
import unittest
from unittest import mock
import shutil
from ludwig.datasets.mnist import Mnist

"""This serves as a placeholder integration test, will be changed into a true unit test soon
with mocking and patching, commented out for now"""


class TestMNistWorkflow(unittest.TestCase):

    def setUp(self):
        self._cache_dir = "/home/saikatkanjilal/.ludwig_cache"
        self.dataset = Mnist(cache_dir=self._cache_dir)
        self.raw_dataset_path = os.path.join(self._cache_dir, "mnist_1.0", "raw")
        self.raw_tmp_path = os.path.join(self._cache_dir, "mnist_1.0", "raw")
        self.processed_dataset_path = os.path.join(self._cache_dir, "mnist_1.0", "processed")
        self.cleanup(self.raw_tmp_path)
        self.cleanup(self.raw_dataset_path)
        self.cleanup(self.processed_dataset_path)

    def cleanup(self, delete_path):
        for root, dirs, files in os.walk(delete_path):
            for name in files:
                # make sure what you want to keep isn't in the full filename
                os.remove(os.path.join(delete_path , name))
            for name in dirs:
                shutil.rmtree(os.path.join(delete_path, name))

    def tearDown(self):
        self.cleanup(self.raw_tmp_path)
        self.cleanup(self.raw_dataset_path)
        self.cleanup(self.processed_dataset_path)

    def test_mnist_download(self):
        self.dataset.download()
        assert os.path.isfile(os.path.join(self.raw_dataset_path, "train-images-idx3-ubyte"))
        assert os.path.exists(os.path.join(self.raw_dataset_path , "train-labels-idx1-ubyte"))
        assert os.path.exists(os.path.join(self.raw_dataset_path, "t10k-images-idx3-ubyte"))
        assert os.path.exists(os.path.join(self.raw_dataset_path, "t10k-labels-idx1-ubyte"))

    def test_mnist_process(self):
        self.dataset.download()
        self.dataset.process()
        assert os.path.isfile(os.path.join(self.processed_dataset_path, "mnist_dataset_testing.csv"))
        assert os.path.exists(os.path.join(self.processed_dataset_path , "mnist_dataset_training.csv"))

    def test_mnist_load(self):
        self.dataset.download()
        self.dataset.process()
        output_df = self.dataset.load()
        key = os.path.join(self.raw_dataset_path, "training", "0", "17603.png")
        values = output_df.image_path.values
        assert key in values

