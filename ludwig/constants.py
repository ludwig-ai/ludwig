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

INPUT = "input"
OUTPUT = "output"
BINARY = "binary"
CATEGORY = "category"
CATEGORY_DISTRIBUTION = "category_distribution"
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
NUM_CLASSES = "num_classes"
NUM_CHANNELS = "num_channels"
REQUIRES_EQUAL_DIMENSIONS = "requires_equal_dimensions"
USE_PRETRAINED = "use_pretrained"
TRAINABLE = "trainable"
CLASS_WEIGHTS = "class_weights"
LOSS = "loss"
ROC_AUC = "roc_auc"
EVAL_LOSS = "eval_loss"
TRAIN_MEAN_LOSS = "train_mean_loss"
SEQUENCE_SOFTMAX_CROSS_ENTROPY = "sequence_softmax_cross_entropy"
NEXT_TOKEN_SOFTMAX_CROSS_ENTROPY = "next_token_softmax_cross_entropy"
SOFTMAX_CROSS_ENTROPY = "softmax_cross_entropy"
SIGMOID_CROSS_ENTROPY = "sigmoid_cross_entropy"
BINARY_WEIGHTED_CROSS_ENTROPY = "binary_weighted_cross_entropy"
THRESHOLD = "threshold"
VALIDATION_METRIC = "validation_metric"
ACCURACY = "accuracy"
ACCURACY_MICRO = "accuracy_micro"
HITS_AT_K = "hits_at_k"
MEAN_HITS_AT_K = "mean_hits_at_k"
ERROR = "error"
ABSOLUTE_ERROR = "absolute_error"
SQUARED_ERROR = "squared_error"
MEAN_SQUARED_ERROR = "mean_squared_error"
ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
ROOT_MEAN_SQUARED_PERCENTAGE_ERROR = "root_mean_squared_percentage_error"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error"
HUBER = "huber"
CORN = "corn"
R2 = "r2"
EDIT_DISTANCE = "edit_distance"
PERPLEXITY = "perplexity"
NEXT_TOKEN_PERPLEXITY = "next_token_perplexity"
JACCARD = "jaccard"
PRECISION = "precision"
RECALL = "recall"
SPECIFICITY = "specificity"
PREDICTIONS = "predictions"
RESPONSE = "RESPONSE"
TOP_K = "top_k"
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
FILL_WITH_TRUE = "fill_with_true"
BFILL = "bfill"
FFILL = "ffill"
DROP_ROW = "drop_row"
MISSING_VALUE_STRATEGY = "missing_value_strategy"
MISSING_VALUE_STRATEGY_OPTIONS = [
    FILL_WITH_CONST,
    FILL_WITH_MODE,
    BFILL,
    FFILL,
    DROP_ROW,
]

CROP_OR_PAD = "crop_or_pad"
INTERPOLATE = "interpolate"
RESIZE_METHODS = [CROP_OR_PAD, INTERPOLATE]

# Special symbols for text.
STOP_SYMBOL = "<EOS>"
START_SYMBOL = "<SOS>"
PADDING_SYMBOL = "<PAD>"
UNKNOWN_SYMBOL = "<UNK>"

TRAINER = "trainer"
OPTIMIZER = "optimizer"
METRIC = "metric"
PREDICTION = "prediction"
LOGITS = "logits"
HIDDEN = "hidden"
LAST_HIDDEN = "last_hidden"
ENCODER_OUTPUT = "encoder_output"
ENCODER_OUTPUT_STATE = "encoder_output_state"
PROJECTION_INPUT = "projection_input"
LEARNING_RATE_SCHEDULER = "learning_rate_scheduler"

SEMANTIC = "semantic"

RANDOM = "random"
SUM = "sum"
APPEND = "append"
SEQ_SUM = "seq_sum"
AVG_EXP = "avg_exp"

TRAIN = "train"
TRAINING = "training"
VALIDATION = "validation"
TEST = "test"
EVALUATION = "evaluation"
SPLIT = "split"
FORCE_SPLIT = "force_split"
STRATIFY = "stratify"
FULL = "full"
TRAIN_SPLIT = 0
VALIDATION_SPLIT = 1
TEST_SPLIT = 2
MIN_DATASET_SPLIT_ROWS = 3  # The minimum number of rows in a split. Splits smaller than this size are treated as empty.

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
MAX_CONCURRENT_TRIALS = "max_concurrent_trials"
CPU_RESOURCES_PER_TRIAL = "cpu_resources_per_trial"
GPU_RESOURCES_PER_TRIAL = "gpu_resources_per_trial"
GOAL = "goal"
GRID_SEARCH = "grid_search"

NAME = "name"
COLUMN = "column"
TYPE = "type"
ACTIVE = "active"

RAY = "ray"
IN_MEMORY = "in_memory"

PROC_COLUMN = "proc_column"

CHECKSUM = "checksum"

HDF5 = "hdf5"
PARQUET = "parquet"

SRC = "dataset_src"

EARLY_STOP = "early_stop"
EPOCHS = "epochs"
BATCH_SIZE = "batch_size"
EVAL_BATCH_SIZE = "eval_batch_size"
DEFAULT_BATCH_SIZE = "auto"
FALLBACK_BATCH_SIZE = 128
# The smallest batch size that is supported on Ludwig.
MINIMUM_BATCH_SIZE = 1
# 2^40. Used for `max_batch_size` config param. Not a hard constraint for `batch_size` config param.
MAX_POSSIBLE_BATCH_SIZE = 1099511627776
# min batch size. Used as a floor for batch size tuning.
MIN_POSSIBLE_BATCH_SIZE = 1
# max batch size for dataset is 20% of dataset size
MAX_BATCH_SIZE_DATASET_FRACTION = 0.2
MAX_CPU_BATCH_SIZE = 128
LEARNING_RATE = "learning_rate"
INPUT_SIZE = "input_size"
USE_BIAS = "use_bias"
BIAS = "bias"
DEFAULT_USE_BIAS = "default_use_bias"
DEFAULT_BIAS = "default_bias"
CONV_USE_BIAS = "conv_use_bias"
CONV_BIAS = "conv_bias"
AUTO = "auto"
CONFIG = "config"

CLIP = "clip"
DEPENDENCIES = "dependencies"
REDUCE_INPUT = "reduce_input"
REDUCE_DEPENDENCIES = "reduce_dependencies"

BACKEND = "backend"
COMBINER = "combiner"

ENCODER = "encoder"
DECODER = "decoder"

TRAINABLE = "trainable"

DEFAULTS = "defaults"
DEFAULT = "default"
DEFAULT_VALIDATION_METRIC = "default_validation_metric"

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

BASE_MODEL = "base_model"
MODEL_TYPE = "model_type"
MODEL_ECD = "ecd"
MODEL_GBM = "gbm"
MODEL_LLM = "llm"
DASK_MODULE_NAME = "dask.dataframe"
LUDWIG_VERSION = "ludwig_version"

PREPROCESSOR = "preprocessor"
PREDICTOR = "predictor"
POSTPROCESSOR = "postprocessor"

GENERATION = "generation"
PROMPT = "prompt"
ADAPTER = "adapter"
QUANTIZATION = "quantization"
MIN_QUANTIZATION_BITS_FOR_MERGE_AND_UNLOAD = 8
PRETRAINED_ADAPTER_WEIGHTS = "pretrained_adapter_weights"
MERGE_ADAPTER_INTO_BASE_MODEL = "merge_adapter_into_base_model"
PROGRESSBAR = "progressbar"

# CrossEntropyLoss for LLMs
IGNORE_INDEX_TOKEN_ID = -100

S3 = "s3"
CACHE = "cache"

# If `use_torch_profiler=True` in LudwigProfiler, LUDWIG_TAG is prepended to the specified experiment tag
# (LudwigProfiler(tag="...", ..)). This edited tag is passed in to `torch.profiler.record_function` so we can
# retrieve torch ops for the tagged code blocks/functions.
LUDWIG_TAG = "[ludwig]"

# Retry constants
TRIES = 5
DELAY = 1
BACKOFF = 2
JITTER = (0, 1)

# image support constants
IMAGENET1K = "imagenet1k"

AUGMENTATION = "augmentation"

LUDWIG_SCHEMA_VALIDATION_POLICY = "LUDWIG_SCHEMA_VALIDATION_POLICY"
