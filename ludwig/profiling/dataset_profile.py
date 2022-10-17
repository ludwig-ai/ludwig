import logging
import time
from typing import Dict, Union

import whylogs as why
from whylogs.core.proto import ColumnMessage
from whylogs.core.view.column_profile_view import ColumnProfileView
from whylogs.core.view.dataset_profile_view import DatasetProfileView

from ludwig.profiling import dataset_profile_pb2
from ludwig.profiling.why_schema import LudwigWhySchema
from ludwig.utils.data_utils import load_dataset
from ludwig.utils.types import DataFrame


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


def get_column_profile_views_from_proto(
    dataset_profile_proto: dataset_profile_pb2.DatasetProfile,
) -> Dict[str, ColumnProfileView]:
    """Returns a mapping of feature name to ColumnProfileView."""
    column_profile_views: Dict[str, ColumnProfileView] = {}
    for feature_name, feature_profile in dataset_profile_proto.feature_profiles.items():
        # column_profile_views[feature_name] = ColumnProfileView.from_protobuf(feature_profile.whylogs_metrics)
        whylogs_metrics_proto = ColumnMessage()
        whylogs_metrics_proto.ParseFromString(feature_profile.whylogs_metrics.SerializeToString())
        column_profile_views[feature_name] = ColumnProfileView.from_protobuf(whylogs_metrics_proto)
    return column_profile_views
