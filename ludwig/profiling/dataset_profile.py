import time
from typing import Any, Dict, Union

import whylogs as why
from whylogs.core.proto import ColumnMessage
from whylogs.core.view.column_profile_view import ColumnProfileView
from whylogs.core.view.dataset_profile_view import DatasetProfileView

from ludwig.profiling import dataset_profile_pb2
from ludwig.profiling.utils import get_ludwig_type_from_column_profile_summary
from ludwig.utils.data_utils import load_dataset
from ludwig.utils.types import DataFrame


def get_dataset_profile_view(dataset: Union[str, DataFrame]) -> DatasetProfileView:
    """Returns whylogs dataset profile view."""
    dataframe = load_dataset(dataset)

    # Hard cap to 10k rows.
    dataframe = dataset.head(10000)

    results = why.log(pandas=dataframe)
    profile = results.profile()
    profile_view = profile.view()
    return profile_view


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
