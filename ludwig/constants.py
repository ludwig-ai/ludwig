#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
INPUT_FEATURES = "input_features"
OUTPUT_FEATURES = "output_features"

BINARY = "binary"
CATEGORY = "category"
INT = "int"
FLOAT = "float"
SPACE = "space"
NUMBER = "number"
SET = "set"
BAG = "bag"
TEXT = "text"
SEQUENCE = "sequence"
TIMESERIES = "timeseries"
IMAGE = "image"
AUDIO = "audio"
DATE = "date"
H3 = "h3"
VECTOR = "vector"
HEIGHT = "height"
WIDTH = "width"
INFER_IMAGE_DIMENSIONS = "infer_image_dimensions"
INFER_IMAGE_MAX_HEIGHT = "infer_image_max_height"
INFER_IMAGE_MAX_WIDTH = "infer_image_max_width"
INFER_IMAGE_SAMPLE_SIZE = "infer_image_sample_size"
NUM_CHANNELS = "num_channels"
LOSS = "loss"
ROC_AUC = "roc_auc"
EVAL_LOSS = "eval_loss"
TRAIN_MEAN_LOSS = "train_mean_loss"
SEQUENCE_SOFTMAX_CROSS_ENTROPY = "sequence_softmax_cross_entropy"
SOFTMAX_CROSS_ENTROPY = "softmax_cross_entropy"
SIGMOID_CROSS_ENTROPY = "sigmoid_cross_entropy"
BINARY_WEIGHTED_CROSS_ENTROPY = "binary_weighted_cross_entropy"
ACCURACY = "accuracy"
HITS_AT_K = "hits_at_k"
MEAN_HITS_AT_K = "mean_hits_at_k"
ERROR = "error"
ABSOLUTE_ERROR = "absolute_error"
SQUARED_ERROR = "squared_error"
MEAN_SQUARED_ERROR = "mean_squared_error"
ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
ROOT_MEAN_SQUARED_PERCENTAGE_ERROR = "root_mean_squared_percentage_error"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
R2 = "r2"
EDIT_DISTANCE = "edit_distance"
PERPLEXITY = "perplexity"
JACCARD = "jaccard"
PREDICTIONS = "predictions"
TOP_K_PREDICTIONS = "top_k_predictions"
PROBABILITY = "probability"
PROBABILITIES = "probabilities"
SPLIT_PROBABILITIES = "split_probabilities"
TOKEN_ACCURACY = "token_accuracy"
LAST_ACCURACY = "last_accuracy"
SEQUENCE_ACCURACY = "sequence_accuracy"
LAST_PROBABILITIES = "last_probabilities"
LAST_PREDICTIONS = "last_predictions"
LENGTHS = "lengths"
TIED = "tied"
COMBINED = "combined"

PREPROCESSING = "preprocessing"
FILL_WITH_CONST = "fill_with_const"
FILL_WITH_MODE = "fill_with_mode"
FILL_WITH_MEAN = "fill_with_mean"
FILL_WITH_FALSE = "fill_with_false"
BACKFILL = "backfill"
BFILL = "bfill"
PAD = "pad"
FFILL = "ffill"
DROP_ROW = "drop_row"
MISSING_VALUE_STRATEGY_OPTIONS = [
    FILL_WITH_CONST,
    FILL_WITH_MODE,
    FILL_WITH_MEAN,
    BACKFILL,
    BFILL,
    PAD,
    FFILL,
    DROP_ROW,
]

CROP_OR_PAD = "crop_or_pad"
INTERPOLATE = "interpolate"
RESIZE_METHODS = [CROP_OR_PAD, INTERPOLATE]

TRAINER = "trainer"
METRIC = "metric"
PREDICTION = "prediction"
LOGITS = "logits"
HIDDEN = "hidden"
LAST_HIDDEN = "last_hidden"
ENCODER_OUTPUT_STATE = "encoder_output_state"
PROJECTION_INPUT = "projection_input"

RANDOM = "random"
SUM = "sum"
APPEND = "append"
SEQ_SUM = "seq_sum"
AVG_EXP = "avg_exp"

TRAINING = "training"
VALIDATION = "validation"
TEST = "test"
SPLIT = "split"
FORCE_SPLIT = "force_split"
STRATIFY = "stratify"
FULL = "full"
TRAIN_SPLIT = 0
VALIDATION_SPLIT = 1
TEST_SPLIT = 2

META = "meta"

HYPEROPT = "hyperopt"
STRATEGY = "strategy"
EXECUTOR = "executor"
MINIMIZE = "minimize"
MAXIMIZE = "maximize"
SAMPLER = "sampler"
NUM_SAMPLES = "num_samples"
SEARCH_ALG = "search_alg"
SCHEDULER = "scheduler"
PARAMETERS = "parameters"

NAME = "name"
COLUMN = "column"
TYPE = "type"

RAY = "ray"

PROC_COLUMN = "proc_column"

CHECKSUM = "checksum"

HDF5 = "hdf5"
PARQUET = "parquet"

SRC = "dataset_src"

BATCH_SIZE = "batch_size"
EVAL_BATCH_SIZE = "eval_batch_size"
LEARNING_RATE = "learning_rate"
USE_BIAS = "use_bias"
BIAS = "bias"
DEFAULT_USE_BIAS = "default_use_bias"
DEFAULT_BIAS = "default_bias"
CONV_USE_BIAS = "conv_use_bias"
CONV_BIAS = "conv_bias"
AUTO = "auto"
CONFIG = "config"

COMBINER = "combiner"

ENCODER = "encoder"
DECODER = "decoder"

DEFAULTS = "defaults"

BALANCE_PERCENTAGE_TOLERANCE = 0.03
IMBALANCE_DETECTION_RATIO = 0.05

TABULAR = "tabular"
AUTOML_DEFAULT_TABULAR_MODEL = "tabnet"
AUTOML_DEFAULT_TEXT_ENCODER = "bert"
AUTOML_SMALLER_TEXT_ENCODER = "distilbert"
AUTOML_TEXT_ENCODER_MAX_TOKEN_LEN = 512
AUTOML_SMALLER_TEXT_LENGTH = 128
AUTOML_LARGE_TEXT_DATASET = 100000
AUTOML_MAX_ROWS_PER_CHECKPOINT = 350000
AUTOML_DEFAULT_IMAGE_ENCODER = "stacked_cnn"

HYPEROPT_WARNING = (
    "You are running the ludwig train command but there’s a hyperopt section present in your config. "
    "It will be ignored. If you want to run hyperopt you should use the following command: ludwig "
    "hyperopt\n\n"
)

CONTINUE_PROMPT = "Do you want to continue? "

DEFAULT_AUDIO_TENSOR_LENGTH = 70000
AUDIO_FEATURE_KEYS = [
    "type",
    "window_length_in_s",
    "window_shift_in_s",
    "num_fft_points",
    "window_type",
    "num_filter_bands",
]

INPUT_FEATURE_KEYS = [
    "name",
    "type",
    "column",
    "proc_column",
    "encoder",
    "tied",
    "preprocessing",
    "vector_size",
]

OUTPUT_FEATURE_KEYS = [
    "name",
    "type",
    "column",
    "proc_column",
    "decoder",
    "tied",
    "preprocessing",
    "loss",
    "reduce_input",
    "dependencies",
    "reduce_dependencies",
    "top_k",
    "vector_size",
]


MODEL_TYPE = "model_type"
MODEL_ECD = "ecd"
MODEL_GBM = "gbm"
DASK_MODULE_NAME = "dask.dataframe"

PREPROCESSOR = "preprocessor"
PREDICTOR = "predictor"
POSTPROCESSOR = "postprocessor"

S3 = "s3"
CACHE = "cache"
EXPERIMENT_RUN = "experiment_run"
TRAIN_TAG = "train"
EVAL_TAG = "evaluate"
