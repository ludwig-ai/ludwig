import pytest
import random
from typing import Any, Dict

import numpy as np
import torch

from ludwig.api import LudwigModel
from ludwig.backend import LOCAL
from tests.integration_tests.utils import slow

# Set random seeds for reproducibility.
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class TestEndToEndModelTraining:

    def run_model_training(self, config: Dict, dataset: str) -> Dict[str, Dict]:
        """ Setup and train Ludwig model, and return training stats. """
        model = LudwigModel(config=config, backend=LOCAL)
        train_stats, _, _ = model.train(dataset)
        return train_stats

    @slow
    @pytest.mark.parametrize(
        'max_epochs,min_test_accuracy,min_train_accuracy',
        [(10, 0.9, 1.0)]
    )
    def test_text_model(
            self,
            max_epochs: int,
            min_test_accuracy: float,
            min_train_accuracy: float
    ):
        """ Test model performance on ATIS dataset using ParallelCNN. 
        
        Params:
            max_epochs: Max number of epochs for convergence of the model.
            min_accuracy: Min test accuracy on convergence.
            min_accuracy: Min train accuracy on convergence (for overfitting).
        """
        config = {
            'input_features': [{'name': 'message', 'type': 'text'}],
            'output_features': [{'name': 'intent', 'type': 'category'}],
        }
        # TODO(shreya): Figure out solution for local testing.
        dataset = '/Users/shreyarajpal/Downloads/atis_intents_train.csv'
        training_stats = self.run_model_training(config, dataset)
        # TODO(shreya): How to extract the num of epochs from training stats? Length?
        assert training_stats['test']['intent']['accuracy'][-1] >= min_test_accuracy
        assert training_stats['training']['intent']['accuracy'][-1] >= min_train_accuracy
