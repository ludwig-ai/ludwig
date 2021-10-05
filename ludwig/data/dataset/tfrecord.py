"""TF record dataset."""
import contextlib
import math

import os
from distutils.version import LooseVersion

from ludwig.constants import NAME, PROC_COLUMN
from ludwig.data.batcher.iterable import IterableBatcher
from ludwig.data.dataset.base import Dataset
from ludwig.data.dataset.pandas import PandasDataset
from ludwig.utils.data_utils import DATA_TRAIN_HDF5_FP
from ludwig.utils.fs_utils import to_url
from ludwig.utils.misc_utils import get_combined_features, get_proc_features
from ludwig.utils.data_utils import load_json


# if LooseVersion(tf.__version__) >= LooseVersion('2.4'):
#     AUTOTUNE = tf.data.AUTOTUNE
# else:
#     AUTOTUNE = tf.data.experimental.AUTOTUNE


class TFRecordDataset(Dataset):
    def __init__(self, url, features, training_set_metadata):
        self.url = to_url(url)
        self.features = [feature[PROC_COLUMN] for feature in features]
        self.training_set_metadata = training_set_metadata

        meta = load_json(os.path.join(self.url, "meta.json"))
        self.size = meta["size"]
        self.compression_type = meta["compression_type"]
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
                           shuffle_buffer_size=None,
                           seed=0,
                           ignore_last=False,
                           horovod=None):
        cur_shard, shard_count = None, None
        if horovod:
            cur_shard, shard_count = horovod.rank(), horovod.size()
        total_samples = self.size
        local_samples = int(total_samples / shard_count) if shard_count else total_samples

        # Below are routine optimizations for tf.dataset
        compression_ext = '.gz' if self.compression_type else ''
        path = os.path.join(self.url, "*.tfrecords{}".format(compression_ext))

        # Now construct the tfrecrd dataset
        files = tf.data.Dataset.list_files(path)

        # interleave the tfrecord files for parallel reading
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=self.compression_type),
            num_parallel_calls=AUTOTUNE
        )
        # Fetch one element so to get the parser.
        features, feature_lists = self._detect_schema(dataset)
        parser = self._get_parser(features, feature_lists)

        # sharding
        if shard_count and shard_count > 1:
            dataset = dataset.shard(shard_count, cur_shard)

        # parallelize parser
        dataset = dataset.map(parser, num_parallel_calls=AUTOTUNE)
        # Note(Hao) batching: ideally we should put this line before the above map.
        # but the parser func is not vectorized for now.
        dataset = dataset.batch(batch_size)

        # cache
        dataset = dataset.cache()
        if should_shuffle:
            buffer_size = shuffle_buffer_size or local_samples
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(AUTOTUNE)

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
            raise ValueError('Batch inference not supported with TFRecord format at this time')
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

    def can_cache(self, skip_save_processed_input):
        return self.backend.is_coordinator()

    @property
    def data_format(self):
        return 'tfrecords'


def get_compression_ext(compression_type):
    return '.gz' if compression_type else ''


def get_part_filename(i, compression_ext):
    return f"part.{str(i).zfill(5)}.tfrecords{compression_ext}"
