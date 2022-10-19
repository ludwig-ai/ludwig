import time
from typing import Any, Dict, Set, Union

import whylogs as why
from whylogs.core.proto import ColumnMessage
from whylogs.core.view.column_profile_view import ColumnProfileView
from whylogs.core.view.dataset_profile_view import DatasetProfileView

from ludwig.constants import AUDIO, BINARY, CATEGORY, IMAGE, NUMBER, TEXT
from ludwig.profiling import dataset_profile_pb2
from ludwig.utils import strings_utils
from ludwig.utils.audio_utils import is_audio_score

# from ludwig.profiling.why_schema import LudwigWhySchema
from ludwig.utils.data_utils import load_dataset
from ludwig.utils.image_utils import is_image_score
from ludwig.utils.types import DataFrame


def get_dataset_profile_view(dataset: Union[str, DataFrame]) -> DatasetProfileView:
    """Returns whylogs dataset profile view."""
    dataframe = load_dataset(dataset)
    results = why.log(pandas=dataframe)
    profile = results.profile()
    profile_view = profile.view()
    return profile_view


def get_num_distinct_values(column_profile_summary):
    return int(column_profile_summary["cardinality/est"])


def get_num_nonnull_values(column_profile_summary):
    return column_profile_summary["counts/n"] - column_profile_summary["counts/null"]


def get_pct_null_values(column_profile_summary):
    return column_profile_summary["counts/null"] / column_profile_summary["counts/n"]


def get_distinct_values(column_profile_summary) -> Set[str]:
    if "frequent_items/frequent_strings" not in column_profile_summary:
        return {}
    frequent_items = column_profile_summary["frequent_items/frequent_strings"]
    if not frequent_items:
        # Can be an empty list if the feature is non-string.
        return {}
    return {frequent_item.value for frequent_item in frequent_items}


def get_pct_distinct_values(column_profile_summary) -> float:
    return get_num_distinct_values(column_profile_summary) / column_profile_summary["counts/n"]


def get_distinct_values_balance(column_profile_summary):
    if "frequent_items/frequent_strings" not in column_profile_summary:
        return -1
    frequent_items = column_profile_summary["frequent_items/frequent_strings"]
    if not frequent_items:
        # Can be an empty list if the feature is non-string.
        return -1

    max_occurence = frequent_items[0].est
    min_occurence = frequent_items[-1].est
    return min_occurence / max_occurence


def are_values_images(distinct_values: Set[str], feature_name: str):
    overall_image_score = 0
    for value in distinct_values:
        is_image_score(None, value, column=feature_name)
        if overall_image_score > 3:
            return True

    if overall_image_score > 0.5 * len(distinct_values):
        return True
    return False


def are_values_audio(distinct_values: Set[str], feature_name: str):
    overall_audio_score = 0
    for value in distinct_values:
        is_audio_score(value)
        if overall_audio_score > 3:
            return True

    if overall_audio_score > 0.5 * len(distinct_values):
        return True
    return False


def get_ludwig_type_from_column_profile_summary(feature_name: str, column_profile_summary: Dict[str, Any]) -> str:
    distinct_values = get_distinct_values(column_profile_summary)

    # Check for unstructured types.
    if are_values_images(distinct_values, feature_name):
        return IMAGE
    if are_values_audio(distinct_values, feature_name):
        return AUDIO

    if column_profile_summary["types/boolean"]:
        # True booleans.
        return BINARY
    if column_profile_summary["types/fractional"]:
        # True fractionals.
        return NUMBER
    if column_profile_summary["types/integral"]:
        # True integers.
        # Use CATEGORY if percentage of distinct values is sufficiently low.
        if get_pct_distinct_values(column_profile_summary) < 0.5:
            return CATEGORY
        return NUMBER
    if column_profile_summary["types/string"]:
        # TODO: Check for DATE.
        # Check for NUMBER, CATEGORY, BINARY.
        if get_num_distinct_values(column_profile_summary) == 2:
            return BINARY
        if get_pct_distinct_values(column_profile_summary) < 0.5:
            return CATEGORY
        if strings_utils.are_all_numbers(distinct_values):
            return NUMBER
    # Fallback to TEXT.
    return TEXT


def get_ludwig_type_map_from_column_profile_summaries(column_profile_summaries: Dict[str, Any]) -> Dict[str, str]:
    ludwig_type_map = {}
    for feature_name, column_profile_summary in column_profile_summaries.items():
        ludwig_type_map[feature_name] = get_ludwig_type_from_column_profile_summary(
            feature_name, column_profile_summary
        )
    return ludwig_type_map


def get_dataset_profile_proto(df: DataFrame, profile_view: DatasetProfileView) -> dataset_profile_pb2.DatasetProfile:
    """Returns a Ludwig DatasetProfile from a whylogs DatasetProfileView."""
    profile_view_pandas = profile_view.to_pandas()

    dataset_profile = dataset_profile_pb2.DatasetProfile()
    dataset_profile.timestamp = int(time.time())
    dataset_profile.num_examples = profile_view_pandas.iloc[0]["counts/n"]
    dataset_profile.size_bytes = sum(df.memory_usage(deep=True))
    for column_name, column_profile_view in profile_view.get_columns().items():
        feature_profile = dataset_profile_pb2.FeatureProfile()
        # Ideally, this line of code would simply be:
        # feature_profile.whylogs_metrics.CopyFrom(column_profile_view.to_protobuf())
        # However, this results in a TypeError: "Parameter to CopyFrom() must be instance of same class: expected
        # ludwigwhy.ColumnMessage got ColumnMessage.""
        #
        # To bypass this error, we serialize one proto and then deserialize into the other.
        feature_profile.whylogs_metrics.ParseFromString(column_profile_view.to_protobuf().SerializeToString())

        dataset_profile.feature_profiles[column_name].CopyFrom(feature_profile)
    return dataset_profile


def get_column_profile_summaries_from_proto(
    dataset_profile_proto: dataset_profile_pb2.DatasetProfile,
) -> Dict[str, Dict[str, Any]]:
    """Returns a mapping of feature name to ColumnProfileView."""
    column_profile_views: Dict[str, ColumnProfileView] = {}
    for feature_name, feature_profile in dataset_profile_proto.feature_profiles.items():
        whylogs_metrics_proto = ColumnMessage()
        # Extra copy+deserialization to avoid TypeError.
        whylogs_metrics_proto.ParseFromString(feature_profile.whylogs_metrics.SerializeToString())
        column_profile_view: ColumnProfileView = ColumnProfileView.from_protobuf(whylogs_metrics_proto)
        column_profile_views[feature_name] = column_profile_view.to_summary_dict()
    return column_profile_views


def get_column_profile_summaries(
    pandas_df: DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Get WhyLogs column summaries directly from a pandas dataframe."""
    dataset_profile_view = get_dataset_profile_view(pandas_df)
    dataset_profile_view_proto = get_dataset_profile_proto(pandas_df, dataset_profile_view)
    return get_column_profile_summaries_from_proto(dataset_profile_view_proto)
