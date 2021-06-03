"""TF record dataset."""
import contextlib
import glob
import json
import math

import os
import tensorflow as tf
from ludwig.constants import NAME, PROC_COLUMN
from ludwig.data.batcher.iterable import IterableBatcher
from ludwig.data.dataset.base import Dataset
from ludwig.data.dataset.pandas import PandasDataset
from ludwig.data.dataset.partitioned import PartitionedDataset
from ludwig.utils.data_utils import DATA_TRAIN_HDF5_FP
from ludwig.utils.fs_utils import to_url
from ludwig.utils.misc_utils import get_combined_features, get_proc_features


class TFRecordDataset(Dataset):
    def __init__(self, url, features, training_set_metadata):
        self.url = to_url(url)[7:]
        if not os.path.isdir(self.url):
            raise RuntimeError("url for TFRecordDataset must be a folder.")
        abs_path = os.path.abspath(os.path.expanduser(self.url))
        self.file_names = [os.path.join(abs_path, name)
                           for name in glob.glob(os.path.join(abs_path, "*.tfrecords.gz"))]
        self.features = [feature[PROC_COLUMN] for feature in features]
        self.training_set_metadata = training_set_metadata

        # dataset = tf.data.TFRecordDataset(self.file_names, compression_type="GZIP")
        # self.feature, self.feature_lists = self._detect_schema(dataset)
        with open(os.path.join(abs_path, "meta.json")) as in_file:
            meta = json.load(in_file)
        self.size = meta["size"]

        self.reshape_features = {
            feature[PROC_COLUMN]: list((-1, *training_set_metadata[feature[NAME]]['reshape']))
            for feature in features
            if "reshape" in training_set_metadata[feature[NAME]]
        }

    def get(self, feature_name, sample):
        t = sample[feature_name]
        reshape_dim = self.reshape_features.get(feature_name)
        if reshape_dim is not None:
            # When we read a 1D array from disk, we need to reshape it back to its
            # full dimensions.
            t = tf.reshape(t, reshape_dim)
        return t

    def __len__(self):
        return self.size

    @contextlib.contextmanager
    def initialize_batcher(self,
                           batch_size=128,
                           should_shuffle=True,
                           seed=0,
                           ignore_last=False,
                           horovod=None):
        cur_shard, shard_count = None, None
        if horovod:
            cur_shard, shard_count = horovod.rank(), horovod.size()

        dataset = tf.data.TFRecordDataset(self.file_names, compression_type="GZIP")
        if shard_count > 1:
            dataset = dataset.shard(shard_count, cur_shard)
        total_samples = self.size
        local_samples = int(total_samples / shard_count) if shard_count else total_samples

        # map parser
        features, feature_lists = self._detect_schema(dataset)
        parser = self._get_parser(features, feature_lists)
        dataset = dataset.map(parser)
        # TODO(Hao): figure out the comments below.
        # dataset = dataset.unbatch()
        if should_shuffle:
            # rows_per_piece = max([piece.get_metadata().num_rows for piece in reader.dataset.pieces])
            # buffer_size = min(rows_per_piece, local_samples)
            buffer_size = local_samples
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)

        steps_per_epoch = math.ceil(local_samples / batch_size)

        batcher = IterableBatcher(self,
                                  dataset,
                                  steps_per_epoch,
                                  ignore_last=ignore_last)
        yield batcher

    def _detect_schema(self, dataset):
        features = {}
        feature_lists = {}

        serialized = next(iter(dataset.map(lambda serialized: serialized)))
        seq_ex = tf.train.SequenceExample.FromString(serialized.numpy())

        def _get_feature_type(feature=None, type_=None):
            if type_:
                return {
                    int: tf.int64,
                    float: tf.float32,
                    str: tf.string,
                    bytes: tf.string,
                }[type_]

            if feature:
                if feature.HasField('int64_list'):
                    return tf.int64
                if feature.HasField('float_list'):
                    return tf.float32
                if feature.HasField('bytes_list'):
                    return tf.string

        if seq_ex.context.feature:
            for key, feature in seq_ex.context.feature.items():
                features[key] = tf.io.FixedLenFeature(
                    (), _get_feature_type(feature=feature))

        if seq_ex.feature_lists.feature_list:
            for key, feature_list in seq_ex.feature_lists.feature_list.items():
                feature_lists[key] = tf.io.FixedLenSequenceFeature(
                    (), _get_feature_type(feature=feature_list.feature[0]))
        return features, feature_lists

    def _get_parser(self, features, feature_lists):
        def parse_example(feats):

            def parse(binary):
                return tf.io.parse_single_example(binary, features=feats)

            return parse

        def parse_sequence(feats, feat_lists):

            def parse(binary):
                context, sequence = tf.io.parse_single_sequence_example(
                    binary,
                    context_features=feats,
                    sequence_features=feat_lists
                )
                context.update(sequence)
                return context

            return parse

        if feature_lists:
            return parse_sequence(features, feature_lists)
        else:
            return parse_example(features)

class TFRecordDatasetManager(object):
    def __init__(self, backend):
        self.backend = backend

    def create(self, dataset, config, training_set_metadata):
        """Create a TFRecordDataset.

        Args:
            dataset (str): urls to the tfrecord binary files.
            config ():
            training_set_metadata ():

        Returns:
            TFRecordDataset
        """
        features = get_combined_features(config)
        return TFRecordDataset(
            dataset,
            features,
            training_set_metadata
        )

    def create_inference_dataset(self, dataset, tag, config, training_set_metadata):
        """We don't use TFRecord for inference."""
        if self.backend.df_engine.partitioned:
            return PartitionedDataset(
                dataset,
                get_proc_features(config),
                training_set_metadata.get(DATA_TRAIN_HDF5_FP)
            )
        else:
            return PandasDataset(
                dataset,
                get_proc_features(config),
                training_set_metadata.get(DATA_TRAIN_HDF5_FP)
            )

    def save(self, cache_path, dataset, config, training_set_metadata, tag):
        dataset_tfrecord_fp = cache_path

        features = get_combined_features(config)
        for feature in features:
            name = feature[NAME]
            proc_column = feature[PROC_COLUMN]
            reshape = training_set_metadata[name].get('reshape')
            if reshape:
                dataset[proc_column] = self.backend.df_engine.map_objects(
                    dataset[proc_column],
                    lambda x: x.reshape(-1))
        self.backend.df_engine.to_tfrecord(dataset, dataset_tfrecord_fp)
        return dataset_tfrecord_fp

    def can_cache(self, input_fname, config, skip_save_processed_input):
        return self.backend.is_coordinator()

    @property
    def data_format(self):
        return 'tfrecord'
