from ludwig.schema.metadata.parameter_metadata import ExpectedImpact, ParameterMetadata

FEATURE_METADATA = {
    "audio": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "type": ParameterMetadata(
                ui_display_name="Type",
                default_value_reasoning="The default type fbank is set based on values that we have tested and "
                "determined "
                "to be a good starting point for audio feature preprocessing. This is not to "
                "say "
                "that it is the best way to process every audio feature, it is just a good "
                "starting place that performs well in general.",
                example_value=["stft"],
                related_parameters=["audio_file_length_limit_in_s", "norm", "padding_value", "in_memory"],
                other_information="Audio feature preprocessing depends heavily on the type of audio data you are "
                "dealing "
                "with. The type of audio preprocessing you will want to use will be dictated by the "
                "audio data you are dealing with.",
                description_implications="The different type of audio you select hear will determine how your audio "
                "feature"
                " is preprocessed and transformed into trainable data for the model.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=[
                    "https://medium.com/analytics-vidhya/simplifying-audio-data-fft-stft-mfcc-for-machine-learning-and"
                    "-deep-learning-443a2f962e0e "
                ],
                internal_only=False,
            ),
            "window_length_in_s": ParameterMetadata(
                ui_display_name="Window Length in Seconds",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=["window_shift_in_s", "type", "num_filter_bands"],
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=[
                    "https://medium.com/analytics-vidhya/simplifying-audio-data-fft-stft-mfcc-for-machine-learning-and"
                    "-deep-learning-443a2f962e0e "
                ],
                internal_only=False,
            ),
            "window_shift_in_s": ParameterMetadata(
                ui_display_name="Window Shift in Seconds",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=["window_length_in_s", "type", "num_filter_bands"],
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=[
                    "https://medium.com/analytics-vidhya/simplifying-audio-data-fft-stft-mfcc-for-machine-learning-and"
                    "-deep-learning-443a2f962e0e "
                ],
                internal_only=False,
            ),
            "num_fft_points": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "num_filter_bands": ParameterMetadata(
                ui_display_name="Type",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=["window_length_in_s", "type", "window_shift_in_s"],
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=[
                    "https://medium.com/analytics-vidhya/simplifying-audio-data-fft-stft-mfcc-for-machine-learning-and"
                    "-deep-learning-443a2f962e0e "
                ],
                internal_only=False,
            ),
            "audio_file_length_limit_in_s": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "in_memory": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "norm": ParameterMetadata(
                ui_display_name="Normalization Type",
                default_value_reasoning="While batch normalization and layer normalization usually lead to "
                "improvements, "
                "it can be useful to start with fewer bells and whistles.",
                example_value=["batch"],
                related_parameters=["norm_params"],
                other_information=None,
                description_implications="Normalization helps stabilize the learning process and can have a "
                "regularizing "
                "effect that can help with generalization. It's often suggested that with "
                "normalization, you can use a higher learning rate.",
                suggested_values='"batch" or "layer"',
                suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from '
                "the "
                "changing distributions of the inputs to layers deep in the network when "
                "weights are updated. For example, batch normalization standardizes the "
                "inputs "
                "to a layer for each mini-batch. Try out different normalizations to see if "
                "that helps with training stability",
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=[
                    "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
                ],
                internal_only=False,
            ),
            "padding_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "window_type": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "bag": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "lowercase": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "most_common": ParameterMetadata(
                ui_display_name="Most common (vocabulary size)",
                default_value_reasoning="If there are more than 10000 unique categories in the data, it is likely that "
                "they will follow a long-tailed distribution and the least common ones may not "
                "provide a lot of information",
                example_value=[10000],
                related_parameters=["vocab_file, pretrained_embeddings"],
                other_information="Specifying a vocab_file overrides this parameter",
                description_implications="A smaller number will reduce the vocabulary, making the embedding matrix "
                "smaller and reduce the memory footprint, but will also collapse more tokens "
                "into the rare one, so the model may perform worse when rare tokens appear in "
                "the data",
                suggested_values="A value that covers at least 95% of the tokens in the data",
                suggested_values_reasoning="Depending on the data distribution and how important rare tokens are, 90%, "
                "95% or 99% of the number of tokens will leave out only very rare tokens "
                "that "
                "should not influence performance substantially",
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "tokenizer": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "binary": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fallback_true_label": ParameterMetadata(
                ui_display_name="Fallback True Label",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications="Modeling performance should not be affected, but the semantics of some "
                "binary "
                'metrics may change like for "false positives", "false negatives", '
                "etc. if the "
                "true label is pinned to the other value.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.MEDIUM,
                literature_references=None,
                internal_only=False,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "category": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "lowercase": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "most_common": ParameterMetadata(
                ui_display_name="Most common (vocabulary size)",
                default_value_reasoning="If there are more than 10000 unique categories in the data, it is likely that "
                "they will follow a long-tailed distribution and the least common ones may not "
                "provide a lot of information",
                example_value=[10000],
                related_parameters=["vocab_file, pretrained_embeddings"],
                other_information="Specifying a vocab_file overrides this parameter",
                description_implications="A smaller number will reduce the vocabulary, making the embedding matrix "
                "smaller and reduce the memory footprint, but will also collapse more tokens "
                "into the rare one, so the model may perform worse when rare tokens appear in "
                "the data",
                suggested_values="A value that covers at least 95% of the tokens in the data",
                suggested_values_reasoning="Depending on the data distribution and how important rare tokens are, 90%, "
                "95% or 99% of the number of tokens will leave out only very rare tokens "
                "that "
                "should not influence performance substantially",
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "date": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "datetime_format": ParameterMetadata(
                ui_display_name="Datetime format",
                default_value_reasoning="Ludwig will try to infer the date format automatically, but a specific format "
                "can be provided. The date string spec is the same as the one described in "
                "python's datetime.",
                example_value=["%d %b %Y"],
                related_parameters=None,
                other_information=None,
                description_implications="If Ludwig has trouble parsing dates, it could be useful to specify an "
                "explicit "
                "format that Ludwig should parse date feature values as. This could also "
                "serve "
                "as a form of normalization, for example, if not all datetimes have the same "
                "granularity (some have days, some have times), then the common format (i.e. "
                "%d "
                "%m %Y) serves as a truncator.",
                suggested_values=None,
                suggested_values_reasoning="Have Ludwig figure out the date format automatically.",
                commonly_used=False,
                expected_impact=ExpectedImpact.LOW,
                literature_references=None,
                internal_only=False,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "h3": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "image": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "height": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "in_memory": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "infer_image_dimensions": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "infer_image_max_height": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "infer_image_max_width": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "infer_image_num_channels": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "infer_image_sample_size": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "num_channels": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "num_processes": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "resize_method": ParameterMetadata(
                ui_display_name="Resize Method",
                default_value_reasoning="Interpolation may stretch or squish the image, but it does not remove "
                "content or "
                "change the statistical distribution of image values so it is more appropriate "
                "for most tasks.",
                example_value=None,
                related_parameters=["height, width"],
                other_information=None,
                description_implications="interpolation will not change the content of the image, but it will change "
                "the "
                "aspect ratio.\n\ncrop_or_pad will preserve the aspect ratio of the image, "
                "but may remove some content (in the case of cropping).",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.LOW,
                literature_references=None,
                internal_only=False,
            ),
            "scaling": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "width": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "number": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "normalization": ParameterMetadata(
                ui_display_name="Normalization",
                default_value_reasoning="It could be valuable to observe how the model trains without normalization, "
                "and see how the performance changes after.",
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications="The goal of normalization is to transform features to be on a similar scale. "
                "Normalization can be a form of feature smoothing that improves the "
                "performance "
                "and training stability of the model. Normalizations may result in different "
                "effects on the semantics of your number features. The best normalization "
                "technique is one that empirically works well, so try new ideas if you think "
                "they'll work well on your feature distribution.",
                suggested_values="z-score",
                suggested_values_reasoning="Z-score is a variation of scaling that represents the number of standard "
                "deviations away from the mean. You would use z-score to ensure your "
                "feature "
                "distributions have mean = 0 and std = 1. It’s useful when there are a few "
                "outliers, but not so extreme that you need clipping.",
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=[
                    "https://developers.google.com/machine-learning/data-prep/transform/normalization"
                ],
                internal_only=False,
            ),
        }
    },
    "sequence": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "lowercase": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "max_sequence_length": ParameterMetadata(
                ui_display_name="Maximum Sequence Length",
                default_value_reasoning="The default value is 256. Every sequence will be truncated to this length.",
                example_value=None,
                related_parameters=["vocab_size, embedding_size"],
                other_information=None,
                description_implications="A larger sequence length keeps more information from the data, "
                "but also makes "
                "it more computationally expensive (more memory and longer training time). A "
                "smaller sequence length keeps less information from the data, "
                "but also makes it "
                "less computationally expensive (less memory and shorter training time).",
                suggested_values="Use the lowest value that covers most of your input data. Only increase the value if "
                "crucial parts of the input data are truncated.",
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "most_common": ParameterMetadata(
                ui_display_name="Most common (vocabulary size)",
                default_value_reasoning="If there are more than 10000 unique categories in the data, it is likely that "
                "they will follow a long-tailed distribution and the least common ones may not "
                "provide a lot of information",
                example_value=[10000],
                related_parameters=["vocab_file, pretrained_embeddings"],
                other_information="Specifying a vocab_file overrides this parameter",
                description_implications="A smaller number will reduce the vocabulary, making the embedding matrix "
                "smaller and reduce the memory footprint, but will also collapse more tokens "
                "into the rare one, so the model may perform worse when rare tokens appear in "
                "the data",
                suggested_values="A value that covers at least 95% of the tokens in the data",
                suggested_values_reasoning="Depending on the data distribution and how important rare tokens are, 90%, "
                "95% or 99% of the number of tokens will leave out only very rare tokens "
                "that "
                "should not influence performance substantially",
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "padding": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "padding_symbol": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "tokenizer": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "unknown_symbol": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "vocab_file": ParameterMetadata(
                ui_display_name="Vocab File",
                default_value_reasoning="The vocabulary can be parsed automatically from the incoming input features.",
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications="It can be useful to specify your own vocabulary list if the vocabulary is "
                "very "
                "large, there's no out of the box tokenizer that fits your data, or if there "
                "are "
                "several uncommon or infrequently occurring tokens that we want to guarantee "
                "to "
                "be a part of the vocabulary, rather than treated as an unknown.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.MEDIUM,
                literature_references=None,
                internal_only=False,
            ),
            "ngram_size": ParameterMetadata(
                ui_display_name="n-gram size",
                default_value_reasoning="Size of the n-gram when using the `ngram` tokenizer.",
                example_value=3,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "set": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "lowercase": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "most_common": ParameterMetadata(
                ui_display_name="Most common (vocabulary size)",
                default_value_reasoning="If there are more than 10000 unique categories in the data, it is likely that "
                "they will follow a long-tailed distribution and the least common ones may not "
                "provide a lot of information",
                example_value=[10000],
                related_parameters=["vocab_file, pretrained_embeddings"],
                other_information="Specifying a vocab_file overrides this parameter",
                description_implications="A smaller number will reduce the vocabulary, making the embedding matrix "
                "smaller and reduce the memory footprint, but will also collapse more tokens "
                "into the rare one, so the model may perform worse when rare tokens appear in "
                "the data",
                suggested_values="A value that covers at least 95% of the tokens in the data",
                suggested_values_reasoning="Depending on the data distribution and how important rare tokens are, 90%, "
                "95% or 99% of the number of tokens will leave out only very rare tokens "
                "that "
                "should not influence performance substantially",
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "tokenizer": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "text": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name="DOCSTRING ONLY",
                default_value_reasoning=None,
                example_value=["Depends on dtype"],
                related_parameters=["missing_value_strategy, fill_value"],
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "lowercase": ParameterMetadata(
                ui_display_name="Convert to lowercase",
                default_value_reasoning="Reading the text in lowercase enables the model to treat capitalized and "
                "lowercase words as the same, effectively increasing the number of data points "
                "per word.",
                example_value=[True],
                related_parameters=["vocab_size"],
                other_information=None,
                description_implications="If you set lowercase to False, then capitalized words are seen as completely "
                "separate entities than lowercase words.",
                suggested_values="TRUE",
                suggested_values_reasoning="If there is a strong reason to treat capitalized words and lowercased "
                "words "
                "differently, then set this to False. Otherwise, it is preferable to bucket "
                "the words and make the model case-insensitive.",
                commonly_used=False,
                expected_impact=ExpectedImpact.LOW,
                literature_references=None,
                internal_only=False,
            ),
            "max_sequence_length": ParameterMetadata(
                ui_display_name="Maximum Sequence Length",
                default_value_reasoning="The default value is 256. Every sequence will be truncated to this length.",
                example_value=None,
                related_parameters=["vocab_size, embedding_size"],
                other_information=None,
                description_implications="A larger sequence length keeps more information from the data, "
                "but also makes "
                "it more computationally expensive (more memory and longer training time). A "
                "smaller sequence length keeps less information from the data, "
                "but also makes it "
                "less computationally expensive (less memory and shorter training time).",
                suggested_values="Use the lowest value that covers most of your input data. Only increase the value if "
                "crucial parts of the input data are truncated.",
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "most_common": ParameterMetadata(
                ui_display_name="Most common (vocabulary size)",
                default_value_reasoning="If there are more than 10000 unique categories in the data, it is likely that "
                "they will follow a long-tailed distribution and the least common ones may not "
                "provide a lot of information",
                example_value=[10000],
                related_parameters=["vocab_file, pretrained_embeddings"],
                other_information="Specifying a vocab_file overrides this parameter",
                description_implications="A smaller number will reduce the vocabulary, making the embedding matrix "
                "smaller and reduce the memory footprint, but will also collapse more tokens "
                "into the rare one, so the model may perform worse when rare tokens appear in "
                "the data",
                suggested_values="A value that covers at least 95% of the tokens in the data",
                suggested_values_reasoning="Depending on the data distribution and how important rare tokens are, 90%, "
                "95% or 99% of the number of tokens will leave out only very rare tokens "
                "that "
                "should not influence performance substantially",
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "padding": ParameterMetadata(
                ui_display_name="Padding",
                default_value_reasoning="We usually want to add padding to the end of a text sequence to fill in any "
                "remaining space as opposed to the beggining so we set the default to right.",
                example_value=None,
                related_parameters=["padding_symbol,\nmax_sequence_length"],
                other_information=None,
                description_implications="If you pad to the left, the encoded vector will have leading padding tokens "
                "as "
                "opposed to trailing padding tokens. This could matter based on the type of "
                "text "
                "input you are expecting.",
                suggested_values="'right'",
                suggested_values_reasoning="right padding is the usual way to add padding to a text sequence",
                commonly_used=False,
                expected_impact=ExpectedImpact.LOW,
                literature_references=None,
                internal_only=False,
            ),
            "padding_symbol": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "pretrained_model_name_or_path": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "tokenizer": ParameterMetadata(
                ui_display_name="Tokenizer",
                default_value_reasoning='The default tokenizer is `space_punct`, an abbreviation of "Space '
                'punctuation". This tokenizer creates sub-words by dividing the text on '
                "whitespace and punctuation characters. For example: The text `'hello "
                "world!isn't "
                "this great?'` would be transformed to `['hello', 'world', '!', 'isn', \"'\", "
                "'t', 'this', 'great', '?']`. This is the default value because it is a fast "
                "tokenizer that works reasonably well.",
                example_value=["space_punct"],
                related_parameters=["vocab_file, pretrained_model_name_or_path"],
                other_information=None,
                description_implications="Choosing a tokenizer can be difficult. The primary thing to check is that "
                "the "
                "tokenizer you have selected is compatible with the language(s) in your text "
                "data. This means either selecting a tokenizer that is language-specific ("
                "i.e. "
                "`french_tokenize` if working with French text) or general enough that its "
                "tokenizations are language-agnostic (i.e. `space_punct`).",
                suggested_values="sentencepiece",
                suggested_values_reasoning="SentencePiece is a tokenizer developed by Google which utilizes Byte-Pair "
                "Encoding (BPE), which strikes a good balance between character-level and "
                "word-level tokenization (more info on BPE here: "
                "https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern"
                "-nlp-eb36c7df4f10 ). This tokenizer is language-agnostic and more "
                "sophisticated than the default.",
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=["https://huggingface.co/course/chapter2/4?fw=pt"],
                internal_only=False,
            ),
            "unknown_symbol": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "vocab_file": ParameterMetadata(
                ui_display_name="Vocab File",
                default_value_reasoning="The vocabulary can be parsed automatically from the incoming input features.",
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications="It can be useful to specify your own vocabulary list if the vocabulary is "
                "very "
                "large, there's no out of the box tokenizer that fits your data, or if there "
                "are "
                "several uncommon or infrequently occurring tokens that we want to guarantee "
                "to "
                "be a part of the vocabulary, rather than treated as an unknown.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.MEDIUM,
                literature_references=None,
                internal_only=False,
            ),
            "ngram_size": ParameterMetadata(
                ui_display_name="n-gram size",
                default_value_reasoning="Size of the n-gram when using the `ngram` tokenizer.",
                example_value=3,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "timeseries": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "padding": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "padding_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "timeseries_length_limit": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
            "tokenizer": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
    "vector": {
        "preprocessing": {
            "computed_fill_value": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=True,
            ),
            "fill_value": ParameterMetadata(
                ui_display_name="Fill Value",
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "missing_value_strategy": ParameterMetadata(
                ui_display_name="Missing Value Strategy",
                default_value_reasoning="The default `fill_with_const` replaces missing values with the value "
                "specified "
                "by `fill_value`.",
                example_value=None,
                related_parameters=["fill_value"],
                other_information=None,
                description_implications="Determines how missing values will be handled in the dataset. Not all "
                "strategies are valid for all datatypes. For example, `fill_with_mean` is "
                "applicable to continuous numerical data. Note that choosing to drop rows "
                "with "
                "missing values could result in losing information, especially if there is a "
                "high proportion of missing values in the dataset.",
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.HIGH,
                literature_references=None,
                internal_only=False,
            ),
            "vector_size": ParameterMetadata(
                ui_display_name=None,
                default_value_reasoning=None,
                example_value=None,
                related_parameters=None,
                other_information=None,
                description_implications=None,
                suggested_values=None,
                suggested_values_reasoning=None,
                commonly_used=False,
                expected_impact=ExpectedImpact.UNKNOWN,
                literature_references=None,
                internal_only=False,
            ),
        }
    },
}
