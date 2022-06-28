import collections
import random
import time
import os

import numpy as np
import pandas as pd
import torch
from ludwig.api import LudwigModel
from ludwig.data.preprocessing import preprocess_for_prediction

from ludwig.constants import TRAINER, NAME, TYPE
from ludwig.features.feature_registries import input_type_registry
from ludwig.models.inference import to_inference_module_input_from_dataframe
from ludwig.utils.misc_utils import get_from_registry
from tests.integration_tests.utils import (
    RAY_BACKEND_CONFIG,
    binary_feature,
    generate_data,
    init_backend,
    text_feature,
)

SEQ_SIZE = 6


def test_text_preproc_module_space_punct_tokenizer_speed(tmpdir):
    feature_config = text_feature()
    input_features = [feature_config]
    output_features = [binary_feature()]
    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 1}}
    data_csv_path = generate_data(input_features, output_features, os.path.join(tmpdir, "data.csv"), num_examples=100)

    backend = "ray"
    with init_backend(backend):
        if backend == "ray":
            backend = RAY_BACKEND_CONFIG
            backend["processor"]["type"] = "dask"

        ludwig_model = LudwigModel(config, backend=backend)
        _, _, output_directory = ludwig_model.train(
            dataset=data_csv_path,
            output_directory=os.path.join(tmpdir, "output"),
        )

        feature = get_from_registry(feature_config[TYPE], input_type_registry)
        preprocessing_module = feature.create_preproc_module(ludwig_model.training_set_metadata[feature_config[NAME]])
        scripted_preprocessing_module = torch.jit.script(preprocessing_module)

        scripted_model = ludwig_model.to_torchscript()
        scripted_model_path = os.path.join(tmpdir, "inference_module.pt")
        torch.jit.save(scripted_model, scripted_model_path)
        scripted_model = torch.jit.load(scripted_model_path)

        df = pd.read_csv(data_csv_path)
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        print("warming up...")
        for i in range(100):
            batch_size = random.choice(batch_sizes)
            inputs_df = df.sample(n=batch_size, replace=True)
            inputs_series = ludwig_model.backend.df_engine.from_pandas(inputs_df)[feature_config[NAME]]
            inputs_dict = to_inference_module_input_from_dataframe(inputs_df, config)

            scripted_preprocessing_module(inputs_dict[feature_config[NAME]])
            scripted_preprocessing_module.forward_old(inputs_dict[feature_config[NAME]])
            scripted_model.preprocessor_forward(inputs_dict)

        print("benchmarking...")
        for batch_size in batch_sizes:
            inputs_df = df.sample(n=batch_size, replace=True)
            inputs_series = ludwig_model.backend.df_engine.from_pandas(inputs_df)[feature_config[NAME]]
            inputs_dict = to_inference_module_input_from_dataframe(inputs_df, config)

            method_to_durations = collections.defaultdict(list)

            for i in range(100):
                start_t = time.time()
                scripted_preprocessing_module.forward_old(inputs_dict[feature_config[NAME]])
                method_to_durations["old"].append(time.time() - start_t)

                start_t = time.time()
                scripted_preprocessing_module(inputs_dict[feature_config[NAME]])
                method_to_durations["new"].append(time.time() - start_t)

                start_t = time.time()
                preprocessing_module.forward_series(inputs_series, ludwig_model.backend)
                method_to_durations["series"].append(time.time() - start_t)

                start_t = time.time()
                preprocess_for_prediction(
                    ludwig_model.config,
                    dataset=inputs_df,
                    training_set_metadata=ludwig_model.training_set_metadata,
                    include_outputs=False,
                    backend=ludwig_model.backend,
                )
                method_to_durations["ludwig"].append(time.time() - start_t)

                start_t = time.time()
                scripted_model.preprocessor_forward(inputs_dict)
                method_to_durations["ts_inf"].append(time.time() - start_t)

            print()
            print(f"Batch size: {batch_size}")
            for method, durations in method_to_durations.items():
                print(f"\t{method}:\t{np.mean(durations):.8f} +/- {np.std(durations):.8f}")

        print("done.")
