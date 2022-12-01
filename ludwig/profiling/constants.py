"""Constants for standard whylogs metrics.

Docs:
https://whylogs.readthedocs.io/en/latest/examples/basic/Inspecting_Profiles.html#Understanding-The-whylogs-Profile-Statistics

By default, whylogs will track the following metrics according to the column's inferred data type:

Integral:
- counts
- types
- distribution
- ints
- cardinality
- frequent_items

Fractional:
- counts
- types
- cardinality
- distribution

String:
- counts
- types
- cardinality
- frequent_items
"""

# The total number of entries in a feature.
COUNTS_N = "counts/n"
# The number of null values.
COUNTS_NULL = "counts/null"
# The number of values consisting of an integral (whole number).
TYPES_INTEGRAL = "types/integral"
# The number of values consisting of a fractional value (float).
TYPES_FRACTIONAL = "types/fractional"
# The number of values consisting of a boolean.
TYPES_BOOLEAN = "types/boolean"
# The number of values consisting of a string.
TYPES_STRING = "types/string"
# The number of values consisting of an object. If the data is not of any of the previous types, it will be assumed as
# an object.
TYPES_OBJECT = "types/object"

# Cardinality
# The estimated unique values for each feature
CARDINALITY_EST = "cardinality/est"
# Upper bound for the cardinality estimation. The actual cardinality will always be below this number.
CARDINALITY_UPPER = "cardinality/upper_1"
# Lower bound for the cardinality estimation. The actual cardinality will always be above this number.
CARDINALITY_LOWER = "cardinality/lower_1"

# The most frequent items.
FREQUENT_ITEMS = "frequent_items/frequent_strings"

# Distribution statistics are generated when a feature contains numerical data.
# The calculated mean of the feature data.
DISTRIBUTION_MEAN = "distribution/mean"
# The calculated standard deviation of the feature data.
DISTRIBUTION_STDDEV = "distribution/stddev"
# The number of rows belonging to the feature.
DISTRIBUTION_N = "distribution/n"
# The highest (max) value in the feature.
DISTRIBUTION_MAX = "distribution/max"
# The smallest (min) value in the feature.
DISTRIBUTION_MIN = "distribution/min"
# The median value of the feature data.
DISTRIBUTION_MEDIAN = "distribution/median"
# The xx-th quantile value of the data’s distribution
DISTRIBUTION_Q_XX = "distribution/q_xx"
