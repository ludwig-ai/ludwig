import unittest
from ludwig.datasets.ohsumed.ohsumed import OhsuMed


class TestOhsuDatasetWorkflow(unittest.TestCase):

    def setUp(self):
        self.__ohsu_med_handle = OhsuMed()

    def test_download_success(self):
        result = self.__ohsu_med_handle.download("ohsumed")
        assert(result, True)

    def test_process_success(self):
        result = self.__ohsu_med_handle.process()
        assert(len(result.items()) > 0)

    def test_load_success(self):
        transformed_data = self.__ohsu_med_handle.load()
        # we test a random assortment of keys
        first_key = "Laparoscopic treatment of perforated peptic ulcer. Mouret P  Francois Y  Vignal J  Barth X  Lombard-Platet R."
        second_key = "Cuff size and blood pressure  letter  comment  Gollin S."
        tmp = transformed_data['class'].where(transformed_data['text'] == first_key)
        tmp1 = transformed_data['class'].where(transformed_data['text'] == second_key)
        assert(tmp[0] == 'Neg-')
        assert(tmp1[16] == 'Neg-')