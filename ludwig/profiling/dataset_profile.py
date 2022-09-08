import logging
import time
from typing import Dict, Union

import whylogs as why
from whylogs.core.view.column_profile_view import ColumnProfileView
from whylogs.core.view.dataset_profile_view import DatasetProfileView
from whylogs.core.proto import ColumnMessage

from ludwig.utils.types import DataFrame
from ludwig.utils.data_utils import load_dataset
from ludwig.profiling import dataset_profile_pb2
from ludwig.profiling.why_schema import LudwigWhySchema
from ludwig.utils.automl.field_info import FieldInfo


def get_dataset_profile_view(dataset: Union[str, DataFrame]) -> DatasetProfileView:
    """Returns whylogs dataset profile view."""
    dataframe = load_dataset(dataset)
    results = why.log(pandas=dataframe, schema=LudwigWhySchema())
    profile = results.profile()
    profile_view = profile.view()
    logging.debug(f"Dataset profiled: {profile_view.to_pandas()}")
    return profile_view


def get_dataset_profile_proto(profile_view: DatasetProfileView) -> dataset_profile_pb2.DatasetProfile:
    profile_view_pandas = profile_view.to_pandas()

    dataset_profile = dataset_profile_pb2.DatasetProfile()
    dataset_profile.timestamp = int(time.time())
    dataset_profile.num_examples = profile_view_pandas.iloc[0]["counts/n"]
    # TODO: Add size bytes.
    for column_name, column_profile_view in profile_view.get_columns().items():
        feature_profile = dataset_profile_pb2.FeatureProfile()
        # TypeError: Parameter to CopyFrom() must be instance of same class: expected ludwigwhy.ColumnMessage got ColumnMessage.
        # feature_profile.whylogs_metrics.CopyFrom(column_profile_view.to_protobuf())
        feature_profile.whylogs_metrics.ParseFromString(column_profile_view.to_protobuf().SerializeToString())

        dataset_profile.feature_profiles[column_name].CopyFrom(feature_profile)
    return dataset_profile


def get_column_profile_views_from_proto(
    dataset_profile_proto: dataset_profile_pb2.DatasetProfile,
) -> Dict[str, ColumnProfileView]:
    column_profile_views: Dict[str, ColumnProfileView] = {}
    for feature_name, feature_profile in dataset_profile_proto.feature_profiles.items():
        # column_profile_views[feature_name] = ColumnProfileView.from_protobuf(feature_profile.whylogs_metrics)
        whylogs_metrics_proto = ColumnMessage()
        whylogs_metrics_proto.ParseFromString(feature_profile.whylogs_metrics.SerializeToString())
        column_profile_views[feature_name] = ColumnProfileView.from_protobuf(whylogs_metrics_proto)
    return column_profile_views


def get_dtype_from_column_profile(column_profile_summary: Dict) -> str:
    # TODO: Better way of getting this, is it automatically available?
    if column_profile_summary["types/boolean"]:
        return "bool"
    if column_profile_summary["types/fractional"]:
        return "float"
    if column_profile_summary["types/integral"]:
        return "int64"
    if column_profile_summary["types/string"]:
        return "string"
    return "object"


def column_profile_to_field_info(feature_name: str, column_profile: ColumnProfileView) -> FieldInfo:
    """Placeholder to replicate current Ludwig type inference logic."""
    column_profile_summary = column_profile.to_summary_dict()
    field_info = FieldInfo()
    field_info.name = feature_name
    field_info.key = feature_name
    if "frequent_items/frequent_strings" in column_profile_summary:
        frequent_items = column_profile_summary["frequent_items/frequent_strings"]
        if frequent_items:  # Can be an empty list if the feature is non-string.
            max_occurence = frequent_items[0].est
            min_occurence = frequent_items[-1].est
            for frequent_item in frequent_items:
                field_info.distinct_values.append(frequent_item.value)
            field_info.distinct_values_balance = min_occurence / max_occurence
    field_info.num_distinct_values = int(column_profile_summary["cardinality/est"])
    field_info.nonnull_values = column_profile_summary["counts/n"] - column_profile_summary["counts/null"]
    field_info.image_values = column_profile_summary["ludwig_metric/image_score"]
    field_info.audio_values = column_profile_summary["ludwig_metric/audio_score"]
    field_info.dtype = get_dtype_from_column_profile(column_profile_summary)
    return field_info
