import unittest
from ludwig.datasets.ohsumed.ohsumed import OhsuMed

class TestOhsuDatasetWorkflow(unittest.TestCase):

    def setUp(self):
        self.__ohsu_med_handle = OhsuMed()

    def test_download_success(self):
        result = self.__ohsu_med_handle.download("ohsumed")
        assert(result, True)

    def test_process_success(self):
        result = self.__ohsu_med_handle.download("ohsumed")
        assert(result, True)