import os
from pathlib import Path
import unittest
import yaml
from ludwig.datasets.ohsumed.ohsumed import OhsuMed


class TestOhsuDatasetWorkflow(unittest.TestCase):

    def setUp(self):
        self._ohsu_med_handle = OhsuMed()
        self._initial_path = os.path.abspath(os.path.dirname(__file__))
        self._config_file_location = os.path.join(self._initial_path, "../../../ludwig/datasets/text/versions.yaml")
        with open(self._config_file_location) as config_file:
            self._config_file_contents = yaml.load(config_file, Loader=yaml.FullLoader)
        self._cur_version = self._config_file_contents["text"]["ohsumed"]

    def test_download_success(self):
        self._ohsu_med_handle.download("ohsumed")
        download_path = Path.home().joinpath('.ludwig_cache').joinpath("ohsumed_"
                                                                       + str(self._cur_version)).joinpath('raw.csv')
        result = os.path.isfile(download_path)
        assert(result, True)

    def test_process_success(self):
        self._ohsu_med_handle.process()
        processed_data_path = Path.home().joinpath('.ludwig_cache').joinpath("ohsumed_"
                                                                       + str(self._cur_version)).joinpath('processed.csv')
        result = os.path.isfile(processed_data_path)
        assert(result, True)

    def test_load_success(self):
        transformed_data = self._ohsu_med_handle.load()
        # we test a random assortment of keys
        first_key = "Laparoscopic treatment of perforated peptic ulcer. Mouret P  Francois Y  Vignal J  Barth X  Lombard-Platet R."
        second_key = "Cuff size and blood pressure  letter  comment  Gollin S."
        tmp = transformed_data['class'].where(transformed_data['text'] == first_key)
        tmp1 = transformed_data['class'].where(transformed_data['text'] == second_key)
        assert(tmp[0] == 'Neg-')
        assert(tmp1[16] == 'Neg-')