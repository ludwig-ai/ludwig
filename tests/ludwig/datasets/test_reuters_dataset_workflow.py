import unittest
from ludwig.datasets.reuters.reuters import Reuters


class TestReutersDatasetWorkflow(unittest.TestCase):

    def setUp(self):
        self.__reuters_handle = Reuters()

    def test_download_success(self):
        result = self.__reuters_handle.download("reuters")
        assert(result, True)

    def test_process_success(self):
        result = self.__reuters_handle.process()
        assert(len(result.items()) > 0)

    def test_load_success(self):
        transformed_data = self.__reuters_handle.load()
        first_key = "2 NEW YORK BANK DISCOUNT WINDOW BORROWINGS 64 MLN DLRS IN FEB 25 WEEK Blah blah blah 3  "
        tmp = transformed_data['class'].where(transformed_data['text'] == first_key)
        assert (tmp[16] == 'Neg-')