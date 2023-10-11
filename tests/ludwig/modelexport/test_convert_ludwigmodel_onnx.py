from ludwig.model_export.onnx_exporter import OnnxExporter
from tests.integration_tests.utils import binary_feature, generate_data, LocalTestBackend
from tests.integration_tests.utils import number_feature
from tests.integration_tests.utils import text_feature
from tests.integration_tests.utils import image_feature
from tests.integration_tests.utils import set_feature
from tests.integration_tests.utils import bag_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import vector_feature
from tests.integration_tests.utils import audio_feature
from tests.integration_tests.utils import timeseries_feature
from tests.integration_tests.utils import date_feature
from tests.integration_tests.utils import h3_feature

import os

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
import shutil
import torch
from copy import deepcopy
import os
import pytest

@pytest.mark.parametrize("should_load_model", [True, False])
@pytest.mark.parametrize("model_type", ["ecd", "gbm"])
def test_convert_torch_to_onnx(tmpdir, should_load_model, model_type):
    csv_filename="datafile.csv"
    #######
    # Setup
    #######
    model_type = "ecd"
    dir_path = tmpdir
    data_csv_path = os.path.join(tmpdir, csv_filename)

    # Single sequence input, single category output
    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(encoder={"type": "passthrough", "vocab_size": 3}),
        category_feature(encoder={"type": "onehot", "vocab_size": 3}),
    ]
    if model_type == "ecd":
        image_dest_folder = os.path.join(tmpdir, "generated_images")
        audio_dest_folder = os.path.join(tmpdir, "generated_audio")
        input_features.extend(
            [
                category_feature(encoder={"type": "dense", "vocab_size": 3}),
                sequence_feature(encoder={"vocab_size": 3}),
                text_feature(encoder={"vocab_size": 3}),
                vector_feature(),
                image_feature(image_dest_folder),
                audio_feature(audio_dest_folder),
                timeseries_feature(),
                date_feature(),
                date_feature(),
                h3_feature(),
                set_feature(encoder={"vocab_size": 3}),
                bag_feature(encoder={"vocab_size": 3}),
            ]
        )

    output_features = [
        category_feature(decoder={"vocab_size": 3}),
    ]
    if model_type == "ecd":
        output_features.extend(
            [
                binary_feature(),
                number_feature(),
                set_feature(decoder={"vocab_size": 3}),
                vector_feature(),
                sequence_feature(decoder={"vocab_size": 3}),
                text_feature(decoder={"vocab_size": 3}),
            ]
        )

    predictions_column_name = "{}_predictions".format(output_features[0]["name"])

    # Generate test data
    data_csv_path = generate_data(input_features, output_features, data_csv_path)

    #############
    # Train model
    #############
    backend = LocalTestBackend()
    config = {
        "model_type": model_type,
        "input_features": input_features,
        "output_features": output_features,
    }
    if model_type == "ecd":
        config[TRAINER] = {"epochs": 2}
    else:
        # Disable feature filtering to avoid having no features due to small test dataset,
        # see https://stackoverflow.com/a/66405983/5222402
        config[TRAINER] = {"num_boost_round": 2, "feature_pre_filter": False}
    ludwig_model = LudwigModel(config, backend=backend)
    ludwig_model.train(
        dataset=data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    ###################
    # save Ludwig model
    ###################
    ludwigmodel_path = os.path.join(dir_path, "ludwigmodel")
    shutil.rmtree(ludwigmodel_path, ignore_errors=True)
    ludwig_model.save(ludwigmodel_path)

    ###################
    # load Ludwig model
    ###################
    if should_load_model:
        ludwig_model = LudwigModel.load(ludwigmodel_path, backend=backend)

    ##############################
    # collect weight tensors names
    ##############################
    original_predictions_df, _ = ludwig_model.predict(dataset=data_csv_path)
    original_weights = deepcopy(list(ludwig_model.model.parameters()))
    original_weights = [t.cpu() for t in original_weights]

    # Move the model to CPU for tracing
    ludwig_model.model.cpu()

    #################
    # save torchscript
    #################
    torchscript_path = os.path.join(dir_path, "torchscript")
    shutil.rmtree(torchscript_path, ignore_errors=True)
    ludwig_model.model.save_torchscript(torchscript_path)

    ###################################################
    # load Ludwig model, obtain predictions and weights
    ###################################################
    ludwig_model = LudwigModel.load(ludwigmodel_path, backend=backend)

    ludwig_model.load_state_dict(ludwigmodel_path, None)

    print("About to call torch_model.eval")
    # set the model to inference mode
    ludwig_model.eval()

    batch_size = 1
    # Input to the model
    print("About to call torch.randn")
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    #torch_out = ludwig_model(x)

    print("About to call torch.onnx.export")
    # Export the model
    onnx_exporter = OnnxExporter()
    onnx_exporter.export_classifier(ludwigmodel_path,".")
