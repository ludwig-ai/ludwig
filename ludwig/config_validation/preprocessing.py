def check_global_max_sequence_length_fits_prompt_template(metadata, global_preprocessing_parameters):
    """Checks that the prompt template fits within the global max sequence length."""

    if (
        "global_max_sequence_length" in global_preprocessing_parameters
        and global_preprocessing_parameters["global_max_sequence_length"] is not None
    ):
        for feature_name, feature_metadata in metadata.items():
            if (
                "prompt_template_num_tokens" in feature_metadata
                and feature_metadata["prompt_template_num_tokens"]
                > global_preprocessing_parameters["global_max_sequence_length"]
            ):
                raise ValueError(
                    f'The prompt contains ({feature_metadata["prompt_template_num_tokens"]}) tokens, which is more '
                    f"than the the global_max_sequence_length "
                    f'({global_preprocessing_parameters["global_max_sequence_length"]}), which will remove all unique '
                    "information. Shorten the prompt, or increase the global max sequence length to > "
                    f'({feature_metadata["prompt_template_num_tokens"]}) to include the full prompt.'
                )
