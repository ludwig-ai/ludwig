from ludwig.schema.metadata.parameter_metadata import ExpectedImpact, ParameterMetadata

PREPROCESSING_METADATA = {
    "force_split": ParameterMetadata(
        ui_display_name="Force Split",
        default_value_reasoning='We do not expect most datasets to have an explicit "split" column in the data. Used '
        "mostly internally by ludwig datasets.",
        example_value=None,
        related_parameters=["split_probabilities, stratify"],
        other_information=None,
        description_implications=None,
        suggested_values=None,
        suggested_values_reasoning=None,
        commonly_used=False,
        expected_impact=ExpectedImpact.HIGH,
        literature_references=None,
        internal_only=False,
    ),
    "oversample_minority": ParameterMetadata(
        ui_display_name="Oversample Minority",
        default_value_reasoning="We do not want to randomly oversample by default since this is a strategy to deal "
        "with imbalanced datasets, but can cause issues if not implemented correctly.",
        example_value=[0.5],
        related_parameters=None,
        other_information="This parameter is one of many strategies to combat issues with class imbalance, "
        "though it is not a cure all. Oversampling too much can cause overfitting which can "
        "adversely affect your model so use with caution.",
        description_implications="The higher the value you choose gets to 1, the closer you will be to having an "
        "equal imbalance ratio (i.e. 1:1 positive to negative class), however this can lead "
        "to problems of overfitting when oversampling is used too liberally. As a rule of "
        "thumb, starting oversampling with a very conservative approach and increasing in "
        "small incremements is probably the best way to improve your model without "
        "experiencing model overfitting.",
        suggested_values="Depends on imbalance ratio and dataset size",
        suggested_values_reasoning=None,
        commonly_used=False,
        expected_impact=ExpectedImpact.MEDIUM,
        literature_references=[
            "https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/"
        ],
        internal_only=False,
    ),
    "sample_ratio": ParameterMetadata(
        ui_display_name="Sample Ratio",
        default_value_reasoning="The default value is 1.0 because we do not want to shrink the dataset by default. In "
        "the rare occurences when you do want to downsample the entire dataset, "
        "this parameter is available, however it is not enabled by default, hence a default "
        "value of 1.0",
        example_value=[0.8],
        related_parameters=None,
        other_information=None,
        description_implications="Decreases the amount of data you are inputting into the model. Could be useful if "
        "you have more data than you need and you are concerned with computational costs.",
        suggested_values="Depends on data size",
        suggested_values_reasoning=None,
        commonly_used=False,
        expected_impact=ExpectedImpact.MEDIUM,
        literature_references=None,
        internal_only=False,
    ),
    "split_probabilities": ParameterMetadata(
        ui_display_name="Split Probabilities",
        default_value_reasoning="Most of the dataset should be used for training, with some portion heldout for "
        "validation and testing.",
        example_value=None,
        related_parameters=["force_split, stratify"],
        other_information="Split data into train, validation, and test.\n\nBy default, Ludwig looks for a column "
        "named split (case-sensitive) which is expected to consist of 3 possible values that "
        "correspond to different datasets:\n\n0: train\n1: validation\n2: test\nIf the data does "
        "not contain the split column, then data is randomly split based on splitting percentages, "
        "defined by split_probabilities.\n\nIf force_split is true, the the split column in the "
        "dataset is ignored and the dataset is randomly split based on splitting percentages, "
        "defined by split_probabilities.",
        description_implications="In machine learning, data splitting is typically done to avoid overfitting. That is "
        "an instance where a machine learning model fits its training data too well and "
        "fails to reliably fit additional data.\n\nThe training set is the portion of data "
        "used to train the model. The model should observe and learn from the training set, "
        "optimizing any of its parameters.\n\nThe dev set is a data set of examples used to "
        "change learning process parameters. It is also called the cross-validation or model "
        "validation set. This set of data has the goal of ranking the model's accuracy and "
        "can help with model selection.\n\nThe testing set is the portion of data that is "
        "tested in the final model and is compared against the previous sets of data. The "
        "testing set acts as an evaluation of the final mode and algorithm.",
        suggested_values=(0.8, 0.1, 0.1),
        suggested_values_reasoning="For larger datasets, it can be beneficial to use more data for training, "
        "since the test and validation sets are still plenty big for getting a good sense "
        "of model generalization.",
        commonly_used=False,
        expected_impact=ExpectedImpact.HIGH,
        literature_references=[
            "https://www.techtarget.com/searchenterpriseai/definition/data-splitting#:~:text=Data%20splitting%20is"
            "%20when%20data,creating%20models%20based%20on%20data. "
        ],
        internal_only=False,
    ),
    "stratify": ParameterMetadata(
        ui_display_name="Stratify",
        default_value_reasoning="The default is set to None since we do not want to stratify unless specifically told "
        "to do so. There are a variety of reasons for this, but one example is that our data "
        "set may not even have a categorical feature to stratify on.",
        example_value=["Category_Feature_A"],
        related_parameters=["force_split, split_probabilities"],
        other_information=None,
        description_implications="Depends on dataset",
        suggested_values="Depends on dataset",
        suggested_values_reasoning=None,
        commonly_used=False,
        expected_impact=ExpectedImpact.HIGH,
        literature_references=[
            "https://medium.com/analytics-vidhya/stratified-sampling-in-machine-learning-f5112b5b9cfe"
        ],
        internal_only=False,
    ),
    "undersample_majority": ParameterMetadata(
        ui_display_name="Undersample Majority",
        default_value_reasoning="We do not want to randomly undersample by default since this is a strategy to deal "
        "with imbalanced datasets, but can cause issues if not implemented correctly.",
        example_value=[0.5],
        related_parameters=None,
        other_information="This parameter is one of many strategies to combat issues with class imbalance, "
        "though it is not a cure all. Undersampling too much can cause loss of data which can "
        "adversely affect your model so use with caution.",
        description_implications="The higher the value you choose gets to 1, the closer you will be to having an "
        "equal imbalance ratio (i.e. 1:1 positive to negative class), however this can lead "
        "to problems of data loss when undersampling is used too liberally. As a rule of "
        "thumb, starting undersampling with a very conservative approach and increasing in "
        "small incremements is probably the best way to improve your model without "
        "experiencing catastrophic data loss effects.",
        suggested_values="Depends on imbalance ratio and dataset size",
        suggested_values_reasoning=None,
        commonly_used=False,
        expected_impact=ExpectedImpact.MEDIUM,
        literature_references=[
            "https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/"
        ],
        internal_only=False,
    ),
}
