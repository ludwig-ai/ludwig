import os

import pandas as pd

from ludwig.profiling.dataset_profile import get_column_profile_summaries
from ludwig.profiling.type_inference import get_ludwig_type_map_from_column_profile_summaries
from tests.integration_tests.utils import (
    audio_feature,
    bag_feature,
    binary_feature,
    category_feature,
    date_feature,
    generate_data,
    generate_data_as_dataframe,
    h3_feature,
    image_feature,
    number_feature,
    sequence_feature,
    set_feature,
    text_feature,
    timeseries_feature,
    vector_feature,
)


def test_get_ludwig_type_map_from_column_profile_summaries():
    df = pd.DataFrame(
        {
            "animal": ["lion", "shark", "cat", "bear", "jellyfish", "kangaroo", "jellyfish", "jellyfish", "fish"],
            "legs": [4, 0, 4, 4.0, None, 2, None, None, "fins"],
            "weight": [14.3, 11.8, 4.3, 30.1, 2.0, 120.0, 2.7, 2.2, 1.2],
            "account_type": ["human", "bot", "human", "bot", "human", "bot", "human", "bot", "human"],
            "finite_numbers_as_numbers": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "finite_numbers_as_strings": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
            "handful_of_numbers": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "handful_of_strings": ["human", "bot", "unknown", "human", "bot", "unknown", "human", "bot", "unknown"],
            "bools_as_bools": [True, False, True, False, True, False, True, False, True],
            "bools_as_strings": ["0", "1", "0", "1", "0", "1", "0", "1", "0"],
            "floats_as_strings": ["1.5", "3.7", "2.2", "3.1", "1.8", "1.4", "9.9", "8.7", "9.1"],
        }
    )
    column_profile_summaries = get_column_profile_summaries(df)

    ludwig_type_map = get_ludwig_type_map_from_column_profile_summaries(column_profile_summaries)

    assert ludwig_type_map == {
        "account_type": "binary",
        "animal": "text",
        "bools_as_bools": "binary",
        "bools_as_strings": "binary",
        "finite_numbers_as_numbers": "number",
        "finite_numbers_as_strings": "number",
        "handful_of_numbers": "category",
        "handful_of_strings": "category",
        "floats_as_strings": "number",
        "legs": "number",
        "weight": "number",
    }


def test_synthetic_data(tmpdir):
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")
    input_features = [
        binary_feature(name="binary_feature"),
        number_feature(name="number_feature"),
        sequence_feature(name="sequence_feature", encoder={"vocab_size": 3}),
        text_feature(name="text_feature", encoder={"vocab_size": 3}),
        vector_feature(name="vector_feature"),
        image_feature(image_dest_folder, name="image_feature"),
        audio_feature(audio_dest_folder, name="audio_feature"),
        timeseries_feature(name="timeseries_feature"),
        date_feature(name="date_feature"),
        h3_feature(name="h3_feature"),
        set_feature(name="set_feature", encoder={"vocab_size": 100}),
        bag_feature(name="bag_feature", encoder={"vocab_size": 100}),
    ]
    output_features = [category_feature(name="category_feature", decoder={"vocab_size": 3})]
    dataset_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=100)

    column_profile_summaries = get_column_profile_summaries(dataset_csv)
    ludwig_type_map = get_ludwig_type_map_from_column_profile_summaries(column_profile_summaries)

    assert ludwig_type_map == {
        "audio_feature": "audio",
        "bag_feature": "text",
        "binary_feature": "binary",
        "category_feature": "category",
        "date_feature": "text",
        "h3_feature": "number",
        "image_feature": "image",
        "number_feature": "number",
        "sequence_feature": "text",
        "set_feature": "text",
        "text_feature": "text",
        "timeseries_feature": "text",
        "vector_feature": "text",
    }


def test_synthetic_data_200k_examples():
    input_features = [
        binary_feature(name="binary_feature"),
        number_feature(name="number_feature"),
        text_feature(name="text_feature", encoder={"vocab_size": 100}),
    ]
    output_features = [category_feature(name="category_feature", decoder={"vocab_size": 3})]
    df = generate_data_as_dataframe(
        input_features,
        output_features,
        num_examples=200000,
    )

    column_profile_summaries = get_column_profile_summaries(df)
    ludwig_type_map = get_ludwig_type_map_from_column_profile_summaries(column_profile_summaries)

    assert ludwig_type_map == {
        "binary_feature": "binary",
        "category_feature": "category",
        "number_feature": "number",
        "text_feature": "text",
    }
