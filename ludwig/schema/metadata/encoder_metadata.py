from ludwig.schema.metadata.parameter_metadata import ExpectedImpact, ParameterMetadata

ENCODER_METADATA = {
    "ALBERTEncoder": {
        "attention_probs_dropout_prob": ParameterMetadata(
            ui_display_name="attention_probs_dropout_prob",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["hidden_dropout_prob, classifier_dropout_prob"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "bos_token_id": ParameterMetadata(
            ui_display_name="Beginning-of-Sentence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "classifier_dropout_prob": ParameterMetadata(
            ui_display_name="classifier_dropout_prob",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["hidden_dropout_prob, attention_probs_dropout_prob"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "eos_token_id": ParameterMetadata(
            ui_display_name="End-of-Sentence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "hidden_act": ParameterMetadata(
            ui_display_name="Hidden Layer Activation",
            default_value_reasoning="Taken from huggingface.",
            example_value=["relu"],
            related_parameters=None,
            other_information=None,
            description_implications="Changing this activation function will only affect the feed-forward layers of "
            "the transformer.",
            suggested_values="gelu",
            suggested_values_reasoning="Taken from huggingface defaults.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "[Hugging face docs for ALBERT config]("
                "https://huggingface.co/docs/transformers/model_doc/albert#transformers.AlbertConfig.hidden_act)\n\r"
                "\n[Relevant StackOverflow discussion]("
                "https://ai.stackexchange.com/questions/30341/why-does-a-transformer-not-use-an-activation-function"
                "-following-the-multi-head-a)"
            ],
            internal_only=False,
        ),
        "hidden_dropout_prob": ParameterMetadata(
            ui_display_name="hidden_dropout_prob",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["attention_probs_dropout_prob,\nclassifier_dropout_prob"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "hidden_size": ParameterMetadata(
            ui_display_name="Hidden Size",
            default_value_reasoning="Huggingface default.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the hidden size makes the model larger and slower to train, "
            "increases the model's capacity to capture more complexity. It also increases "
            "the chance of overfitting.",
            suggested_values="10 - 2048",
            suggested_values_reasoning="Increasing the hidden size makes sense if the model is underfitting. It's "
            "useful to train both smaller and larger models to see how model capacity "
            "affects performance. This should only be explored after the architecture of "
            "the model has been settled.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "inner_group_num": ParameterMetadata(
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
        "intermediate_size": ParameterMetadata(
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
        "layer_norm_eps": ParameterMetadata(
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
        "max_position_embeddings": ParameterMetadata(
            ui_display_name="Max Position Embeddings",
            default_value_reasoning="Taken from huggingface.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words or positions, which can have a large "
            "vocbulary size. Ideally, after an embedding is trained, it captures some of the "
            "semantics of the input by placing semantically similar inputs close together in "
            "the embedding space.\n\nIncreasing the embedding size may cause the model to "
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values=512,
            suggested_values_reasoning="Out of the box value based on published literature. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_attention_heads": ParameterMetadata(
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
        "num_hidden_groups": ParameterMetadata(
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
        "num_hidden_layers": ParameterMetadata(
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
        "pad_token_id": ParameterMetadata(
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
        "position_embedding_type": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="These arguments typically don't need to be specified.",
            example_value=None,
            related_parameters=["pretrained_model_name_or_path"],
            other_information=None,
            description_implications=None,
            suggested_values="Default",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pretrained_model_name_or_path": ParameterMetadata(
            ui_display_name="Pretrained model",
            default_value_reasoning="The default model is the canonical model for this model architecture, "
            "and is therefore a good starting point for most use cases.",
            example_value=None,
            related_parameters=["use_pretrained, trainable, pretrained_kwargs"],
            other_information=None,
            description_implications="There are two factors to consider when choosing a pre-trained model: (1) size, "
            "and (2) task similarity. \n\nThe larger the model, the more subtle its "
            "comprehension of inputs can become. However, larger models are also more "
            "compute and memory-intensive to train.\n\nModels pretrained on highly-related "
            "source tasks are more likely to be successful on the target task. Consider "
            "searching the HuggingFace model repository for models trained on similar tasks.",
            suggested_values="albert-large-v2, albert-base-chinese",
            suggested_values_reasoning="If you would like better performance and are not compute/memory-constrained, "
            "increasing model capacity can potentially provide a richer representation "
            "than the default. The suggested value upsizes the model while maintaining the "
            "same model architecture.\n\nLanguage models trained on general corpora "
            "typically generalize well. Consider deviating from the default only if the "
            "text in the dataset originates from another domain (e.g. languages other than "
            "English).",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://arxiv.org/abs/1909.11942"],
            internal_only=False,
        ),
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "type_vocab_size": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "AutoTransformerEncoder": {
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "trainable": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "BERTEncoder": {
        "attention_probs_dropout_prob": ParameterMetadata(
            ui_display_name="attention_probs_dropout_prob",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["hidden_dropout_prob, classifier_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "classifier_dropout": ParameterMetadata(
            ui_display_name="classifier_dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["hidden_dropout_prob, attention_probs_dropout_prob"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "gradient_checkpointing": ParameterMetadata(
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
        "hidden_act": ParameterMetadata(
            ui_display_name="Hidden Layer Activation",
            default_value_reasoning="Taken from huggingface.",
            example_value=["relu"],
            related_parameters=None,
            other_information=None,
            description_implications="Changing this activation function will only affect the feed-forward layers of "
            "the transformer.",
            suggested_values="gelu",
            suggested_values_reasoning="Taken from huggingface defaults.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "[Huggingface docs for BERT config]("
                "https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig.hidden_act)\n\r\n["
                "Relevant StackOverflow discussion]("
                "https://ai.stackexchange.com/questions/30341/why-does-a-transformer-not-use-an-activation-function"
                "-following-the-multi-head-a)"
            ],
            internal_only=False,
        ),
        "hidden_dropout_prob": ParameterMetadata(
            ui_display_name="hidden_dropout_prob",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["attention_probs_dropout_prob, classifier_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "hidden_size": ParameterMetadata(
            ui_display_name="Hidden Size",
            default_value_reasoning="Huggingface default.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the hidden size makes the model larger and slower to train, "
            "increases the model's capacity to capture more complexity. It also increases "
            "the chance of overfitting.",
            suggested_values="10 - 2048",
            suggested_values_reasoning="Increasing the hidden size makes sense if the model is underfitting. It's "
            "useful to train both smaller and larger models to see how model capacity "
            "affects performance. This should only be explored after the architecture of "
            "the model has been settled.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "intermediate_size": ParameterMetadata(
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
        "layer_norm_eps": ParameterMetadata(
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
        "max_position_embeddings": ParameterMetadata(
            ui_display_name="Max Position Embeddings",
            default_value_reasoning="Taken from huggingface.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words or positions, which can have a large "
            "vocbulary size. Ideally, after an embedding is trained, it captures some of the "
            "semantics of the input by placing semantically similar inputs close together in "
            "the embedding space.\n\nIncreasing the embedding size may cause the model to "
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values=512,
            suggested_values_reasoning="Out of the box value based on published literature. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_attention_heads": ParameterMetadata(
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
        "num_hidden_layers": ParameterMetadata(
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
        "pad_token_id": ParameterMetadata(
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
        "position_embedding_type": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "type_vocab_size": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "BagEmbedWeightedEncoder": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
            ui_display_name="(under Embeddings header) Trainable?",
            default_value_reasoning="If trained from scratch, embedding vectors are typically learned alongside the "
            "rest of the model.",
            example_value=None,
            related_parameters=["embedding_size, representation, pretrained_embeddings"],
            other_information=None,
            description_implications="Typically this value is only set to False if pre-trained embeddings are "
            "uploaded. Even then, it is reasonable to leave it as True in order to fine-tune"
            " the embeddings.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "force_embedding_size": ParameterMetadata(
            ui_display_name="Force Embedding Size",
            default_value_reasoning="It is not often the case that the user has a strict need for using an embedding "
            "size that should be larger than the vocabulary size.",
            example_value=None,
            related_parameters=["embedding_size"],
            other_information=None,
            description_implications="Should only be True if the user has a strict need for using an embedding size "
            "that should be larger than the vocabulary size. For example, there may be size "
            "requirements across multiple features imposed by downstream modules like the "
            "ComparatorCombiner.",
            suggested_values=[False],
            suggested_values_reasoning="True for advanced usage only.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "pretrained_embeddings": ParameterMetadata(
            ui_display_name="Pretrained embeddings path",
            default_value_reasoning="Embeddings are commonly trained from scratch, or incorporated as part of a "
            "pre-trained model package.",
            example_value=["~/Downloads/glove.6B.100d.txt"],
            related_parameters=["embedding_size, embeddings_trainable"],
            other_information=None,
            description_implications="If pretrained embeddings are specified, then the model may have a head start in "
            "its representation of various input entities.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "representation": ParameterMetadata(
            ui_display_name="Representation approach",
            default_value_reasoning="Trainable, randomly initialized embedding vectors often lead to more subtle "
            "representations of input entities than one-hot vectors.",
            example_value=None,
            related_parameters=["embedding_size, embeddings_trainable, pretrained_embeddings"],
            other_information="",
            description_implications="If set to sparse, the representations for input entities are fixed as one-hot "
            "vectors. This leads to less flexible representations for input entities, "
            "but could lead to faster training since there are less learnable parameters.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "use_bias": ParameterMetadata(
            ui_display_name="Use Bias",
            default_value_reasoning="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to use bias "
            "terms.\n\nBatch Normalization, however, adds a trainable shift parameter which "
            "is added to the activation. When Batch Normalization is used in a layer, "
            "bias terms are redundant and may be removed.",
            example_value=[True],
            related_parameters=["bias_initializer, fc_layers"],
            other_information="If fc_layers is not specified, or use_bias is not specified for individual layers, "
            "the value of use_bias will be used as the default for all layers.",
            description_implications="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to leave this "
            "parameter set to True.",
            suggested_values=[True],
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from published [literature]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "CTRLEncoder": {
        "attn_pdrop": ParameterMetadata(
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
        "dff": ParameterMetadata(
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
        "embd_pdrop": ParameterMetadata(
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
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "layer_norm_epsilon": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "n_ctx": ParameterMetadata(
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
        "n_embd": ParameterMetadata(
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
        "n_head": ParameterMetadata(
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
        "n_layer": ParameterMetadata(
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
        "n_positions": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "resid_pdrop": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "CamemBERTEncoder": {
        "attention_probs_dropout_prob": ParameterMetadata(
            ui_display_name="attention_probs_dropout_prob",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["classifier_dropout, hidden_dropout_prob"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "classifier_dropout": ParameterMetadata(
            ui_display_name="classifier_dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["attention_probs_dropout_prob, hidden_dropout_prob"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "gradient_checkpointing": ParameterMetadata(
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
        "hidden_act": ParameterMetadata(
            ui_display_name="Hidden Layer Activation",
            default_value_reasoning="Taken from huggingface.",
            example_value=["relu"],
            related_parameters=None,
            other_information=None,
            description_implications="Changing this activation function will only affect the feed-forward layers of "
            "the transformer.",
            suggested_values="gelu",
            suggested_values_reasoning="Taken from huggingface defaults.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "[Relevant StackOverflow discussion]("
                "https://ai.stackexchange.com/questions/30341/why-does-a-transformer-not-use-an-activation-function"
                "-following-the-multi-head-a)"
            ],
            internal_only=False,
        ),
        "hidden_dropout_prob": ParameterMetadata(
            ui_display_name="hidden_dropout_prob",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["attention_probs_dropout_prob, \nclassifier_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "hidden_size": ParameterMetadata(
            ui_display_name="Hidden Size",
            default_value_reasoning="Huggingface default.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the hidden size makes the model larger and slower to train, "
            "increases the model's capacity to capture more complexity. It also increases "
            "the chance of overfitting.",
            suggested_values="10 - 2048",
            suggested_values_reasoning="Increasing the hidden size makes sense if the model is underfitting. It's "
            "useful to train both smaller and larger models to see how model capacity "
            "affects performance. This should only be explored after the architecture of "
            "the model has been settled.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "intermediate_size": ParameterMetadata(
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
        "layer_norm_eps": ParameterMetadata(
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
        "max_position_embeddings": ParameterMetadata(
            ui_display_name="Max Position Embeddings",
            default_value_reasoning="Taken from huggingface.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words or positions, which can have a large "
            "vocbulary size. Ideally, after an embedding is trained, it captures some of the "
            "semantics of the input by placing semantically similar inputs close together in "
            "the embedding space.\n\nIncreasing the embedding size may cause the model to "
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values=512,
            suggested_values_reasoning="Out of the box value based on published literature. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_attention_heads": ParameterMetadata(
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
        "num_hidden_layers": ParameterMetadata(
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
        "pad_token_id": ParameterMetadata(
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
        "position_embedding_type": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "type_vocab_size": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "CategoricalEmbedEncoder": {
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_initializer": ParameterMetadata(
            ui_display_name="Embedding Initialization",
            default_value_reasoning="According to https://arxiv.org/abs/1711.09160, choice of embedding "
            "initialization is not important as long as the variance is kept reasonably low.",
            example_value=["kaiming"],
            related_parameters=None,
            other_information=None,
            description_implications="According to https://arxiv.org/abs/1711.09160, choice of embedding "
            "initialization is not important as long as the variance is kept reasonably low.",
            suggested_values="kaiming",
            suggested_values_reasoning="https://discuss.huggingface.co/t/state-of-the-art-technique-for-initializing"
            "-embedding-matrix/326",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://arxiv.org/abs/1711.09160"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
        "pretrained_embeddings": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
    },
    "CategoricalSparseEncoder": {
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_initializer": ParameterMetadata(
            ui_display_name="Embedding Initialization",
            default_value_reasoning="According to https://arxiv.org/abs/1711.09160, choice of embedding "
            "initialization is not important as long as the variance is kept reasonably low.",
            example_value=["kaiming"],
            related_parameters=None,
            other_information=None,
            description_implications="According to https://arxiv.org/abs/1711.09160, choice of embedding "
            "initialization is not important as long as the variance is kept reasonably low.",
            suggested_values="kaiming",
            suggested_values_reasoning="https://discuss.huggingface.co/t/state-of-the-art-technique-for-initializing"
            "-embedding-matrix/327",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://arxiv.org/abs/1711.09161"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
        "pretrained_embeddings": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
    },
    "DateEmbed": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "use_bias": ParameterMetadata(
            ui_display_name="Use Bias",
            default_value_reasoning="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to use bias "
            "terms.\n\nBatch Normalization, however, adds a trainable shift parameter which "
            "is added to the activation. When Batch Normalization is used in a layer, "
            "bias terms are redundant and may be removed.",
            example_value=[True],
            related_parameters=["bias_initializer, fc_layers"],
            other_information="If fc_layers is not specified, or use_bias is not specified for individual layers, "
            "the value of use_bias will be used as the default for all layers.",
            description_implications="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to leave this "
            "parameter set to True.",
            suggested_values=[True],
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "DateWave": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "use_bias": ParameterMetadata(
            ui_display_name="Use Bias",
            default_value_reasoning="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to use bias "
            "terms.\n\nBatch Normalization, however, adds a trainable shift parameter which "
            "is added to the activation. When Batch Normalization is used in a layer, "
            "bias terms are redundant and may be removed.",
            example_value=[True],
            related_parameters=["bias_initializer, fc_layers"],
            other_information="If fc_layers is not specified, or use_bias is not specified for individual layers, "
            "the value of use_bias will be used as the default for all layers.",
            description_implications="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to leave this "
            "parameter set to True.",
            suggested_values=[True],
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "DenseEncoder": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "input_size": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=["No"],
            other_information="Internal Only",
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "layers": ParameterMetadata(
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
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_layers": ParameterMetadata(
            ui_display_name="Number of Layers",
            default_value_reasoning="The ideal number of layers depends on the data. For many data types, "
            "one layer is sufficient.",
            example_value=[1],
            related_parameters=["layers"],
            other_information="If you have multiple input features, varying the number of layers in the combiner or "
            "output feature decoder will have more impact.",
            description_implications="Increasing the number of layers may improve model performance by allowing the "
            "model to synthesize learned features derived from the original input. If the "
            "input is simple, ex. a category with a few options, increasing the number of "
            "layers has no benefit. For more complex inputs, additional layers add more "
            "'processing power' to extract useful information from the input.\n\nHowever, "
            "more layers will increase training time and may reduce accuracy due to "
            "overfitting.",
            suggested_values="1-3",
            suggested_values_reasoning="Increasing the number of layers may improve encoder performance.  However, "
            "more layers will increase training time and may cause overfitting.  Small "
            "numbers of layers usually work best.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "use_bias": ParameterMetadata(
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
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "DistilBERTEncoder": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="This is the default activation function used in the Distillbert huggingface "
            "implementation",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "attention_dropout": ParameterMetadata(
            ui_display_name="attention_dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["dropout, qa_dropout, seq_classif_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "dim": ParameterMetadata(
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
        "dropout": ParameterMetadata(
            ui_display_name="dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["attention_dropout,\nqa_dropout,\nseq_classif_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "hidden_dim": ParameterMetadata(
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
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "max_position_embeddings": ParameterMetadata(
            ui_display_name="Max Position Embeddings",
            default_value_reasoning="Taken from huggingface.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words or positions, which can have a large "
            "vocbulary size. Ideally, after an embedding is trained, it captures some of the "
            "semantics of the input by placing semantically similar inputs close together in "
            "the embedding space.\n\nIncreasing the embedding size may cause the model to "
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values=512,
            suggested_values_reasoning="Out of the box value based on published literature. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "n_heads": ParameterMetadata(
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
        "n_layers": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "qa_dropout": ParameterMetadata(
            ui_display_name="qa_dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["dropout, attention_dropout, seq_classif_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "seq_classif_dropout": ParameterMetadata(
            ui_display_name="seq_classif_dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["dropout,\nattention_dropout,\nqa_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "sinusoidal_pos_embds": ParameterMetadata(
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
        "trainable": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "ELECTRAEncoder": {
        "attention_probs_dropout_prob": ParameterMetadata(
            ui_display_name="attention_probs_dropout_prob",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["hidden_dropout_prob, classifier_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "classifier_dropout": ParameterMetadata(
            ui_display_name="classifier_dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["hidden_dropout_prob, attention_probs_dropout_prob"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "hidden_act": ParameterMetadata(
            ui_display_name="Hidden Layer Activation",
            default_value_reasoning="Taken from huggingface.",
            example_value=["relu"],
            related_parameters=None,
            other_information=None,
            description_implications="Changing this activation function will only affect the feed-forward layers of "
            "the transformer.",
            suggested_values="gelu",
            suggested_values_reasoning="Taken from huggingface defaults.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "[Huggingface docs for ELECTRA config]("
                "https://huggingface.co/docs/transformers/model_doc/electra#transformers.ElectraConfig.hidden_act)\n"
                "\n[Relevant StackOverflow discussion]("
                "https://ai.stackexchange.com/questions/30341/why-does-a-transformer-not-use-an-activation-function"
                "-following-the-multi-head-a)"
            ],
            internal_only=False,
        ),
        "hidden_dropout_prob": ParameterMetadata(
            ui_display_name="hidden_dropout_prob",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["attention_probs_dropout_prob,\nclassifier_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "hidden_size": ParameterMetadata(
            ui_display_name="Hidden Size",
            default_value_reasoning="Huggingface default.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the hidden size makes the model larger and slower to train, "
            "increases the model's capacity to capture more complexity. It also increases "
            "the chance of overfitting.",
            suggested_values="10 - 2048",
            suggested_values_reasoning="Increasing the hidden size makes sense if the model is underfitting. It's "
            "useful to train both smaller and larger models to see how model capacity "
            "affects performance. This should only be explored after the architecture of "
            "the model has been settled.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "intermediate_size": ParameterMetadata(
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
        "layer_norm_eps": ParameterMetadata(
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
        "max_position_embeddings": ParameterMetadata(
            ui_display_name="Max Position Embeddings",
            default_value_reasoning="Taken from huggingface.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words or positions, which can have a large "
            "vocbulary size. Ideally, after an embedding is trained, it captures some of the "
            "semantics of the input by placing semantically similar inputs close together in "
            "the embedding space.\n\nIncreasing the embedding size may cause the model to "
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values=512,
            suggested_values_reasoning="Out of the box value based on published literature. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_attention_heads": ParameterMetadata(
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
        "num_hidden_layers": ParameterMetadata(
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
        "position_embedding_type": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "type_vocab_size": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "FlauBERTEncoder": {
        "asm": ParameterMetadata(
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
        "attention_dropout": ParameterMetadata(
            ui_display_name="attention_dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "bos_index": ParameterMetadata(
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
        "causal": ParameterMetadata(
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
        "dropout": ParameterMetadata(
            ui_display_name="dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["attention_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "emb_dim": ParameterMetadata(
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
        "embed_init_std": ParameterMetadata(
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
        "eos_index": ParameterMetadata(
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
        "gelu_activation": ParameterMetadata(
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
        "init_std": ParameterMetadata(
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
        "is_encoder": ParameterMetadata(
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
        "lang_id": ParameterMetadata(
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
        "layer_norm_eps": ParameterMetadata(
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
        "layerdrop": ParameterMetadata(
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
        "mask_index": ParameterMetadata(
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
        "mask_token_id": ParameterMetadata(
            ui_display_name="Mask Token ID",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "max_position_embeddings": ParameterMetadata(
            ui_display_name="Max Position Embeddings",
            default_value_reasoning="Taken from huggingface.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words or positions, which can have a large "
            "vocbulary size. Ideally, after an embedding is trained, it captures some of the "
            "semantics of the input by placing semantically similar inputs close together in "
            "the embedding space.\n\nIncreasing the embedding size may cause the model to "
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values=512,
            suggested_values_reasoning="Out of the box value based on published literature. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "n_head": ParameterMetadata(
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
        "n_langs": ParameterMetadata(
            ui_display_name="Number of Languages",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "n_layer": ParameterMetadata(
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
        "pad_index": ParameterMetadata(
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
        "pre_norm": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "sinusoidal_embeddings": ParameterMetadata(
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
        "trainable": ParameterMetadata(
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
        "unk_index": ParameterMetadata(
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
        "use_lang_emb": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "GPT2Encoder": {
        "activation_function": ParameterMetadata(
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
        "attn_pdrop": ParameterMetadata(
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
        "embd_pdrop": ParameterMetadata(
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
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "layer_norm_epsilon": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "n_ctx": ParameterMetadata(
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
        "n_embd": ParameterMetadata(
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
        "n_head": ParameterMetadata(
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
        "n_inner": ParameterMetadata(
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
        "n_layer": ParameterMetadata(
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
        "n_positions": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "resid_pdrop": ParameterMetadata(
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
        "scale_attn_weights": ParameterMetadata(
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
        "trainable": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "GPTEncoder": {
        "afn": ParameterMetadata(
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
        "attn_pdrop": ParameterMetadata(
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
        "embd_pdrop": ParameterMetadata(
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
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "layer_norm_epsilon": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "n_ctx": ParameterMetadata(
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
        "n_embd": ParameterMetadata(
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
        "n_head": ParameterMetadata(
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
        "n_layer": ParameterMetadata(
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
        "n_positions": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "resid_pdrop": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "H3Embed": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "reduce_output": ParameterMetadata(
            ui_display_name="Sequence Reducer",
            default_value_reasoning="Sums the tensors along the sequence dimension.",
            example_value=None,
            related_parameters=["max_sequence_length"],
            other_information=None,
            description_implications='"last", "sum", "mean", and "max" are the fastest and most memory-efficient '
            "operations– they result in tensors that are the same-size as a single item in "
            "the input sequence. However, these are simple aggregation operations, "
            'therefore some information may be lost. \n\n"concat" concatenates each tensor '
            'together, creating a `(sequence length)*(tensor size)`-element tensor. "concat" '
            "preserves this information, but can be very memory-intensive and should only be "
            'applied if the sequence length and/or tensor size is small. \n\n"attention" '
            "takes a weighted sum of the items in the sequence, where the weights for each "
            "item in the sequence are determined by the model on-the-fly based on the "
            "features of the item itself. This is both slower and and more memory-intensive "
            'than the other operations; however, it can also provide a richer "global" '
            "representation of the sequence.",
            suggested_values='"attention". This and the default covers 95% of use cases.',
            suggested_values_reasoning="If you would like better performance and are not compute/memory-constrained, "
            "attention-based reduction can potentially provide a richer global "
            "representation than the default.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "use_bias": ParameterMetadata(
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
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "H3RNN": {
        "activation": ParameterMetadata(
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
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "bidirectional": ParameterMetadata(
            ui_display_name="Bidirectional",
            default_value_reasoning="For short sequences, it is reasonable to use a vanilla RNN.",
            example_value=None,
            related_parameters=["cell_type, activation, recurrent_activation, use_bias"],
            other_information=None,
            description_implications="Setting bidirectional to True may increase the compute and memory requirements "
            "of the model, but may also increase model performance on long sequences.",
            suggested_values=[True],
            suggested_values_reasoning="RNNs can sometimes suffer from catastrophic forgetting (source: "
            "https://en.wikipedia.org/wiki/Catastrophic_interference ) on long sequences. "
            "Allowing the RNN to read from both the beginning and end of the sequence can "
            "improve its representation at each timestep.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://devopedia.org/bidirectional-rnn#:~:text=RNN%20has%20the%20limitation%20that,"
                "forward%20and%20reverse%20time%20order."
            ],
            internal_only=False,
        ),
        "cell_type": ParameterMetadata(
            ui_display_name="Cell Type",
            default_value_reasoning="The LSTM cell has proven to be the most performant of the three cells.",
            example_value=None,
            related_parameters=["bidirectional\nactivation\nrecurrent_activation\nuse_bias"],
            other_information=None,
            description_implications="There are two reasons to consider other cell types: (1) compute costs and (2) "
            "catastrophic forgetting (source: "
            "https://en.wikipedia.org/wiki/Catastrophic_interference ). RNNs have marginally "
            "less compute costs, but are prone to catastrophic forgetting.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["recurrent_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "hidden_size": ParameterMetadata(
            ui_display_name="Hidden Size",
            default_value_reasoning="H3 values numbers, so a small RNN dimensionality is likely sufficient.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the hidden size makes the model larger and slower to train, "
            "increases the model's capacity to capture more complexity. It also increases "
            "the chance of overfitting.",
            suggested_values="10 - 2048",
            suggested_values_reasoning="Increasing the hidden size makes sense if the model is underfitting. It's "
            "useful to train both smaller and larger models to see how model capacity "
            "affects performance. This should only be explored after the architecture of "
            "the model has been settled.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "num_layers": ParameterMetadata(
            ui_display_name="Number of Recurrent Layers",
            default_value_reasoning="The ideal number of layers depends on the data. For many data types, "
            "one layer is sufficient.",
            example_value=[1],
            related_parameters=["layers"],
            other_information="If you have multiple input features, varying the number of layers in the combiner or "
            "output feature decoder will have more impact.",
            description_implications="Increasing the number of layers may improve model performance for longer "
            "sequences or more complex tasks.",
            suggested_values="1-3",
            suggested_values_reasoning="Increasing the number of layers may improve encoder performance.  However, "
            "more layers will increase training time and may cause overfitting.  Small "
            "numbers of layers usually work best.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "recurrent_activation": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="sigmoid' is commonly used",
            example_value=None,
            related_parameters=None,
            other_information="I don't think that this parameter is used anywhere in the code base. It's being passed "
            "down but not used in the actual RNN forwarding functions.",
            description_implications=None,
            suggested_values="sigmoid, ReLu, tanh",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "recurrent_dropout": ParameterMetadata(
            ui_display_name="Recurrent Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "recurrent_initializer": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "unit_forget_bias": ParameterMetadata(
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
        "use_bias": ParameterMetadata(
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
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "H3WeightedSum": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "should_softmax": ParameterMetadata(
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
        "use_bias": ParameterMetadata(
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
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "LongformerEncoder": {
        "attention_window": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_tokens": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "sep_token_id": ParameterMetadata(
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
        "trainable": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "MLPMixerEncoder": {
        "avg_pool": ParameterMetadata(
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
        "channel_dim": ParameterMetadata(
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
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embed_size": ParameterMetadata(
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
        "num_layers": ParameterMetadata(
            ui_display_name="Number of Layers",
            default_value_reasoning="The ideal number of layers depends on the size and complexity of the input "
            "images. The default value is used in the paper and tested on several image "
            "datasets.",
            example_value=[8],
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the number of layers may improve model performance for larger images "
            "or more complex image tasks.",
            suggested_values="4 - 32",
            suggested_values_reasoning="Values from 8 - 32 are tested in the paper. It is possible that fewer layers "
            "will be sufficient for some tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["MLP-Mixer: An all-MLP Architecture for Vision\nhttps://arxiv.org/abs/2105.01601"],
            internal_only=False,
        ),
        "patch_size": ParameterMetadata(
            ui_display_name="Patch Size",
            default_value_reasoning="Taken from MLP-Mixer paper.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The implications of the image patch size for this layer depend on other "
            "factors, such as the true resolution of the incoming image dataset. If the "
            "patch size is kept consistent but a higher resolution image is used as input, "
            "then the resulting chunked sequence of tokens will be longer than it would have "
            "been if the input resolution was lower. \n\nThe original MLP-Mixer paper also "
            "notes that there is a tradeoff with respect to the projection units learned by "
            "a model. In their findings, a 32x32 patch size model learned very structured "
            "low frequency projection units, while the equivalent 16x16 model learned high "
            "frequencies and showed no clear structure.",
            suggested_values=(16, 32),
            suggested_values_reasoning="16 and 32 are the values used in the original MLP Mixer paper",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=["[MLP Mixer paper](https://arxiv.org/pdf/2105.01601.pdf)"],
            internal_only=False,
        ),
        "token_size": ParameterMetadata(
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
    },
    "MT5Encoder": {
        "d_ff": ParameterMetadata(
            ui_display_name="Dimensionality of Feed-Forward Layer",
            default_value_reasoning="Default value matches the pre-trained encoder.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="If using a pre-trained encoder, this parameter will be automatically derived "
            "from the pre-trained model.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "d_kv": ParameterMetadata(
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
        "d_model": ParameterMetadata(
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
        "decoder_start_token_id": ParameterMetadata(
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
        "dropout_rate": ParameterMetadata(
            ui_display_name="dropout_rate",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "eos_token_id": ParameterMetadata(
            ui_display_name="End-of-Sentence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "feed_forward_proj": ParameterMetadata(
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
        "initializer_factor": ParameterMetadata(
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
        "is_encoder_decoder": ParameterMetadata(
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
        "layer_norm_epsilon": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_decoder_layers": ParameterMetadata(
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
        "num_heads": ParameterMetadata(
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
        "num_layers": ParameterMetadata(
            ui_display_name="Number of Transformer Layers",
            default_value_reasoning="The default value matches the number of layers in the default pretrained encoder.",
            example_value=[8],
            related_parameters=["pretrained_model_or_path"],
            other_information=None,
            description_implications="The ideal number of transformer layers depends on the length and complexity of "
            "input sequences, as well as the task.\n\nIf using a pre-trained encoder, "
            "this parameter will be automatically derived from the pre-trained model.",
            suggested_values="1 - 12",
            suggested_values_reasoning="Increasing the number of layers may improve encoder performance.  However, "
            "more layers will increase training time and may cause overfitting.  Small "
            "numbers of layers usually work best.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pad_token_id": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "relative_attention_num_buckets": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "tie_word_embeddings": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Keeping the word embeddings separate ensures maximum modeling flexibility.",
            example_value=[True],
            related_parameters=None,
            other_information=None,
            description_implications="The main tradeoff between True and False values is in compute costs and model "
            "flexibility. If set to False, the model will require more memory, but may be "
            "more flexible. If set to True, the opposite is true.",
            suggested_values=[True],
            suggested_values_reasoning="If set to True, then the word embeddings will be shared between the encoder "
            "and decoder. There are two main reasons to set this value to True: (1) saving "
            "compute resources. Word embedding tables can be very large and using a single "
            "table between the encoder and decoder can cut one's memory usage in half. (2) "
            "If the domain of the generated text is highly similar to the input text. For "
            "example, if training a Question and Answering (QA) text model, where both the "
            "questions and answers are in the same language, the word embeddings used by "
            "the encoder are likely usable by the decoder and vice-versa. On the other "
            "hand, if training a translation model between two languages, "
            "the word embeddings are not likely to be shareable by both model components.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "tokenizer_class": ParameterMetadata(
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
        "trainable": ParameterMetadata(
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
        "use_cache": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "ParallelCNN": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "conv_layers": ParameterMetadata(
            ui_display_name="Convolutional Layers",
            default_value_reasoning=None,
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=["num_conv_layers"],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a large amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "filter_size": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_conv_layers": ParameterMetadata(
            ui_display_name="Number of Convolutional Layers",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=["conv_layers"],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a large amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_filters": ParameterMetadata(
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
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "pool_function": ParameterMetadata(
            ui_display_name="Pooling function",
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
        "pool_size": ParameterMetadata(
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
        "pretrained_embeddings": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "representation": ParameterMetadata(
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
        "should_embed": ParameterMetadata(
            ui_display_name="Not displayed",
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
        "use_bias": ParameterMetadata(
            ui_display_name="Use Bias",
            default_value_reasoning="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to use bias "
            "terms.\n\nBatch Normalization, however, adds a trainable shift parameter which "
            "is added to the activation. When Batch Normalization is used in a layer, "
            "bias terms are redundant and may be removed.",
            example_value=[True],
            related_parameters=["bias_initializer, fc_layers"],
            other_information="If fc_layers is not specified, or use_bias is not specified for individual layers, "
            "the value of use_bias will be used as the default for all layers.",
            description_implications="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to leave this "
            "parameter set to True.",
            suggested_values=[True],
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "PassthroughEncoder": {
        "input_size": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=["No"],
            other_information="Internal Only",
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        )
    },
    "ResNetEncoder": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "batch_norm_epsilon": ParameterMetadata(
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
        "batch_norm_momentum": ParameterMetadata(
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
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "conv_stride": ParameterMetadata(
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
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "first_pool_kernel_size": ParameterMetadata(
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
        "first_pool_stride": ParameterMetadata(
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
        "kernel_size": ParameterMetadata(
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
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
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
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "out_channels": ParameterMetadata(
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
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "resnet_size": ParameterMetadata(
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
        "use_bias": ParameterMetadata(
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
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
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
    },
    "RoBERTaEncoder": {
        "bos_token_id": ParameterMetadata(
            ui_display_name="Beginning-of-Sentence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "eos_token_id": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="<class 'int'>",
            example_value=["Default value used in pre-trained HF encoder."],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pad_token_id": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "SequenceEmbedEncoder": {
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pretrained_embeddings": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "representation": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "SequencePassthroughEncoder": {
        "encoding_size": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The default `reduce_output` method does not use this parameter, so by default "
            "this parameter is not set.",
            example_value=[128],
            related_parameters=["reduce_output"],
            other_information=None,
            description_implications="This parameter must be equal to the size of the input. Otherwise, an error will"
            " occur.",
            suggested_values=None,
            suggested_values_reasoning="NONE",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "reduce_output": ParameterMetadata(
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
    },
    "SetSparseEncoder": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "pretrained_embeddings": ParameterMetadata(
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
        "representation": ParameterMetadata(
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
        "use_bias": ParameterMetadata(
            ui_display_name="Use Bias",
            default_value_reasoning="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to use bias "
            "terms.\n\nBatch Normalization, however, adds a trainable shift parameter which "
            "is added to the activation. When Batch Normalization is used in a layer, "
            "bias terms are redundant and may be removed.",
            example_value=[True],
            related_parameters=["bias_initializer, fc_layers"],
            other_information="If fc_layers is not specified, or use_bias is not specified for individual layers, "
            "the value of use_bias will be used as the default for all layers.",
            description_implications="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to leave this "
            "parameter set to True.",
            suggested_values=[True],
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "Stacked2DCNN": {
        "conv_activation": ParameterMetadata(
            ui_display_name="Convolutional Activation",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "conv_bias": ParameterMetadata(
            ui_display_name="Convolutional Bias",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "conv_dropout": ParameterMetadata(
            ui_display_name="Convolutional Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["conv_dropout,\nfc_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "conv_layers": ParameterMetadata(
            ui_display_name="Convolutional Layers",
            default_value_reasoning=None,
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=["num_conv_layers"],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a large amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "conv_norm": ParameterMetadata(
            ui_display_name="Convolutional Normalization",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "conv_norm_params": ParameterMetadata(
            ui_display_name="Convolutional Normalization Parameters",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "dilation": ParameterMetadata(
            ui_display_name="Dilation",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "fc_activation": ParameterMetadata(
            ui_display_name="FC Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=["relu"],
            related_parameters=["activation, activation_function, conv_activation, recurrent_activation"],
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="relu, alternatively leakyRelu or elu",
            suggested_values_reasoning="The default value will work well in the majority of the cases",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html"],
            internal_only=False,
        ),
        "fc_bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "fc_dropout": ParameterMetadata(
            ui_display_name="FC Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["conv_dropout,\nfc_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "fc_norm": ParameterMetadata(
            ui_display_name="Fully Connected Normalization",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=["fc_norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate. See Torch's documentation on "
            "batch normalization or for layer see Torch's documentation on layer "
            "normalization.",
            suggested_values="batch",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "fc_norm_params": ParameterMetadata(
            ui_display_name="Fully Connected Normalization Parameters",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=["fc_norm"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "fc_use_bias": ParameterMetadata(
            ui_display_name="FC Use Bias",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "fc_weights_initializer": ParameterMetadata(
            ui_display_name="FC Weights Initializer",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "groups": ParameterMetadata(
            ui_display_name="Groups",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "height": ParameterMetadata(
            ui_display_name="NOT DISPLAYED",
            default_value_reasoning="Computed internally, automatically, based on image data preprocessing.",
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
        "kernel_size": ParameterMetadata(
            ui_display_name="Kernel Size",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_channels": ParameterMetadata(
            ui_display_name="NOT DISPLAYED",
            default_value_reasoning="Computed internally, automatically, based on image data preprocessing.",
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
        "num_conv_layers": ParameterMetadata(
            ui_display_name="Number of Convolutional Layers",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=["conv_layers"],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a large amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "out_channels": ParameterMetadata(
            ui_display_name="Number of Output Channels",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "padding": ParameterMetadata(
            ui_display_name="Padding",
            default_value_reasoning="When padding is set to 'valid' like in the default case, no padding is added. As "
            "a default value putting in the raw image is the goal here.",
            example_value=["'same'"],
            related_parameters=["padding_mode,\nresize method"],
            other_information=None,
            description_implications="By increasing the amount of padding, you can increase the accuracy of the image "
            "analysis for certain circumstances.",
            suggested_values="Same' padding if images are of different dimensions. \nSpecific [h, w] entries can be "
            "valuable on a per dataset basis.",
            suggested_values_reasoning="If your images already have padding, there is no need to add padding, "
            "so the default is fine. If your images come in different dimensions, "
            "then 'same' padding can help pad the images to standardized dimensions. For "
            "certain images, adding padding to the edges can help the CNN process the "
            "images better which can improve model performance. This depends on the images "
            "however.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://www.geeksforgeeks.org/cnn-introduction-to-padding/"],
            internal_only=False,
        ),
        "padding_mode": ParameterMetadata(
            ui_display_name="Padding Mode",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pool_dilation": ParameterMetadata(
            ui_display_name="Pool Dilation",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pool_function": ParameterMetadata(
            ui_display_name="Pooling function",
            default_value_reasoning='Within a given sliding window (e.g. a "patch" of a 3-channel image), the maximum '
            "value for each channel is kept. All other values in the patch are discarded. "
            "Repeat this step for every patch and you have a more compact representation of "
            "the image. \n\nIntuitively, each patch encodes the features from a particular "
            "part of an image, and it is more informative to look at the most prominent "
            "features of an image than the average of all of them.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Both average and max pooling can achieve strong performance.",
            suggested_values="Default",
            suggested_values_reasoning="No",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html\n\nhttps://machinelearningmastery"
                ".com/pooling-layers-for-convolutional-neural-networks/"
            ],
            internal_only=False,
        ),
        "pool_kernel_size": ParameterMetadata(
            ui_display_name="Pool Kernel Size",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pool_padding": ParameterMetadata(
            ui_display_name="Pool Padding",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pool_stride": ParameterMetadata(
            ui_display_name="Pool Stride",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "stride": ParameterMetadata(
            ui_display_name="Stride",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "width": ParameterMetadata(
            ui_display_name="NOT DISPLAYED",
            default_value_reasoning="Computed internally, automatically, based on image data preprocessing.",
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
    },
    "StackedCNN": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "conv_layers": ParameterMetadata(
            ui_display_name="Convolutional Layers",
            default_value_reasoning=None,
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=["num_conv_layers"],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a large amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "dilation_rate": ParameterMetadata(
            ui_display_name="Dilation Rate",
            default_value_reasoning="The standard discrete convolution is the same as a 1-dilated convolution.",
            example_value=[2],
            related_parameters=["filter_size"],
            other_information="Dilated convolution is also known as atrous convolution.",
            description_implications="Higher dilation rates increase the effective size of the convolutional filter.  "
            "Dilated convolution may improve performance if the data is very correlated "
            "locally and also contains long-term dependencies.",
            suggested_values="1-3",
            suggested_values_reasoning="The dilation rate is a factor which increases the spacing between elements of "
            "the convolutional filter",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "filter_size": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_conv_layers": ParameterMetadata(
            ui_display_name="Number of Convolutional Layers",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=["conv_layers"],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a large amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_filters": ParameterMetadata(
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
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
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
        "pool_function": ParameterMetadata(
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
        "pool_padding": ParameterMetadata(
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
        "pool_size": ParameterMetadata(
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
        "pool_strides": ParameterMetadata(
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
        "pretrained_embeddings": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "representation": ParameterMetadata(
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
        "should_embed": ParameterMetadata(
            ui_display_name="Not displayed",
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
        "strides": ParameterMetadata(
            ui_display_name="Stride",
            default_value_reasoning="In general, it makes sense to have a smaller stride that fits the input. "
            "Imagining the simple 2D image as our input, two pixels next to eachother are "
            "strongly correlated while pixels that are further apart will have a "
            "comparatively weaker correlation. Consequently, a higher stride may cause "
            "significant information loss.",
            example_value=[1],
            related_parameters=["pool_strides, default_strides, default_pool_strides, block_strides"],
            other_information=None,
            description_implications="Changing the stride of a convolutional layer is one form of downsampling ("
            "another being pooling). In the case of a large stride, significant amounts of "
            "information is thrown away as the filter convolves over its input. This should "
            "be usually avoided but may be desirable in cases in which the user has some "
            "deep knowledge of the filter or of the rest of the model architecture that "
            "makes it comfortable to allow a higher level compression in the output feature "
            "map of this layer.",
            suggested_values="1-2",
            suggested_values_reasoning="In general, points that are closer to eachother in the input feature space "
            "will be more strongly correlated to eachother, so it is a good idea to select "
            "a stride that captures these neighboring relationships.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "[d2l.ai blog post](http://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)\n"
                "\n[machinelearningmastery blogpost]("
                "https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)\n\n["
                "crossvalidated discussion](https://stats.stackexchange.com/questions/296027/choosing-filter-size"
                "-strides-etc-in-a-cnn)"
            ],
            internal_only=False,
        ),
        "use_bias": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "StackedCNNRNN": {
        "activation": ParameterMetadata(
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
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "bidirectional": ParameterMetadata(
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
        "cell_type": ParameterMetadata(
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
        "conv_activation": ParameterMetadata(
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
        "conv_dropout": ParameterMetadata(
            ui_display_name="Convolutional Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["conv_dropout,\ndropout,\nrecurrent_dropout,\nfc_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "conv_layers": ParameterMetadata(
            ui_display_name="Convolutional Layers",
            default_value_reasoning=None,
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=["num_conv_layers"],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a large amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "dilation_rate": ParameterMetadata(
            ui_display_name="Dilation Rate",
            default_value_reasoning="The standard discrete convolution is the same as a 1-dilated convolution.",
            example_value=[2],
            related_parameters=["filter_size"],
            other_information="Dilated convolution is also known as atrous convolution.",
            description_implications="Higher dilation rates increase the effective size of the convolutional filter.  "
            "Dilated convolution may improve performance if the data is very correlated "
            "locally and also contains long-term dependencies.",
            suggested_values="1-3",
            suggested_values_reasoning="The dilation rate is a factor which increases the spacing between elements of "
            "the convolutional filter",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["conv_dropout,\ndropout,\nrecurrent_dropout,\nfc_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
        "fc_activation": ParameterMetadata(
            ui_display_name="FC Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=["relu"],
            related_parameters=["activation, activation_function, conv_activation, recurrent_activation"],
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="relu, alternatively leakyRelu or elu",
            suggested_values_reasoning="The default value will work well in the majority of the cases",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html"],
            internal_only=False,
        ),
        "fc_dropout": ParameterMetadata(
            ui_display_name="FC Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["conv_dropout,\ndropout,\nrecurrent_dropout,\nfc_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "filter_size": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_conv_layers": ParameterMetadata(
            ui_display_name="Number of Convolutional Layers",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=["conv_layers"],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a large amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_filters": ParameterMetadata(
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
        "num_rec_layers": ParameterMetadata(
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
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
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
        "pool_function": ParameterMetadata(
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
        "pool_padding": ParameterMetadata(
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
        "pool_size": ParameterMetadata(
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
        "pool_strides": ParameterMetadata(
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
        "pretrained_embeddings": ParameterMetadata(
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
        "recurrent_activation": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="sigmoid' is commonly used",
            example_value=None,
            related_parameters=None,
            other_information="I don't think that this parameter is used anywhere in the code base. It's being passed "
            "down but not used in the actual RNN forwarding functions.",
            description_implications=None,
            suggested_values="sigmoid, ReLu, tanh",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "recurrent_dropout": ParameterMetadata(
            ui_display_name="Recurrent Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["conv_dropout,\ndropout,\nrecurrent_dropout,\nfc_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "recurrent_initializer": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "representation": ParameterMetadata(
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
        "should_embed": ParameterMetadata(
            ui_display_name="Not displayed",
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
        "state_size": ParameterMetadata(
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
        "strides": ParameterMetadata(
            ui_display_name="Stride",
            default_value_reasoning="In general, it makes sense to have a smaller stride that fits the input. "
            "Imagining the simple 2D image as our input, two pixels next to eachother are "
            "strongly correlated while pixels that are further apart will have a "
            "comparatively weaker correlation. Consequently, a higher stride may cause "
            "significant information loss.",
            example_value=[1],
            related_parameters=["pool_strides, default_strides, default_pool_strides, block_strides"],
            other_information=None,
            description_implications="Changing the stride of a convolutional layer is one form of downsampling ("
            "another being pooling). In the case of a large stride, significant amounts of "
            "information is thrown away as the filter convolves over its input. This should "
            "be usually avoided but may be desirable in cases in which the user has some "
            "deep knowledge of the filter or of the rest of the model architecture that "
            "makes it comfortable to allow a higher level compression in the output feature "
            "map of this layer.",
            suggested_values="1-2",
            suggested_values_reasoning="In general, points that are closer to eachother in the input feature space "
            "will be more strongly correlated to eachother, so it is a good idea to select "
            "a stride that captures these neighboring relationships.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "[d2l.ai blog post](http://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)\n"
                "\n[machinelearningmastery blogpost]("
                "https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)\n\n["
                "crossvalidated discussion](https://stats.stackexchange.com/questions/296027/choosing-filter-size"
                "-strides-etc-in-a-cnn)"
            ],
            internal_only=False,
        ),
        "unit_forget_bias": ParameterMetadata(
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
        "use_bias": ParameterMetadata(
            ui_display_name="Use Bias",
            default_value_reasoning="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to use bias "
            "terms.\n\nBatch Normalization, however, adds a trainable shift parameter which "
            "is added to the activation. When Batch Normalization is used in a layer, "
            "bias terms are redundant and may be removed.",
            example_value=[True],
            related_parameters=["bias_initializer, fc_layers"],
            other_information="If fc_layers is not specified, or use_bias is not specified for individual layers, "
            "the value of use_bias will be used as the default for all layers.",
            description_implications="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to leave this "
            "parameter set to True.",
            suggested_values=[True],
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "StackedParallelCNN": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="The default value will work well in the majority of the cases",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "filter_size": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_filters": ParameterMetadata(
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
        "num_stacked_layers": ParameterMetadata(
            ui_display_name="Number of Stacked Layers",
            default_value_reasoning=None,
            example_value=[1],
            related_parameters=["stacked_layers"],
            other_information=None,
            description_implications="While superceded by `stacked_layers`, this can directly change the depth of the "
            "current stack of parallel convolutional layers.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "pool_function": ParameterMetadata(
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
        "pool_size": ParameterMetadata(
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
        "pretrained_embeddings": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "representation": ParameterMetadata(
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
        "should_embed": ParameterMetadata(
            ui_display_name="Not displayed",
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
        "stacked_layers": ParameterMetadata(
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
        "use_bias": ParameterMetadata(
            ui_display_name="Use Bias",
            default_value_reasoning="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to use bias "
            "terms.\n\nBatch Normalization, however, adds a trainable shift parameter which "
            "is added to the activation. When Batch Normalization is used in a layer, "
            "bias terms are redundant and may be removed.",
            example_value=[True],
            related_parameters=["bias_initializer, fc_layers"],
            other_information="If fc_layers is not specified, or use_bias is not specified for individual layers, "
            "the value of use_bias will be used as the default for all layers.",
            description_implications="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to leave this "
            "parameter set to True.",
            suggested_values=[True],
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "StackedRNN": {
        "activation": ParameterMetadata(
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
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "bidirectional": ParameterMetadata(
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
        "cell_type": ParameterMetadata(
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
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["dropout,\nrecurrent_dropout,\nfc_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
        "fc_activation": ParameterMetadata(
            ui_display_name="FC Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=["relu"],
            related_parameters=["activation, activation_function, conv_activation, recurrent_activation"],
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="relu, alternatively leakyRelu or elu",
            suggested_values_reasoning="The default value will work well in the majority of the cases",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html"],
            internal_only=False,
        ),
        "fc_dropout": ParameterMetadata(
            ui_display_name="FC Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["dropout, recurrent_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_layers": ParameterMetadata(
            ui_display_name="Number of Recurrent Layers",
            default_value_reasoning="The ideal number of layers depends on the data. For many data types, "
            "one layer is sufficient.",
            example_value=[1],
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the number of layers may improve model performance for longer "
            "sequences or more complex tasks.",
            suggested_values="1-3",
            suggested_values_reasoning="Increasing the number of layers may improve encoder performance.  However, "
            "more layers will increase training time and may cause overfitting.  Small "
            "numbers of layers usually work best.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "pretrained_embeddings": ParameterMetadata(
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
        "recurrent_activation": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="sigmoid' is commonly used",
            example_value=None,
            related_parameters=None,
            other_information="I don't think that this parameter is used anywhere in the code base. It's being passed "
            "down but not used in the actual RNN forwarding functions.",
            description_implications=None,
            suggested_values="sigmoid, ReLu, tanh",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "recurrent_dropout": ParameterMetadata(
            ui_display_name="Recurrent Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["dropout,\nrecurrent_dropout,\nfc_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "recurrent_initializer": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "representation": ParameterMetadata(
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
        "should_embed": ParameterMetadata(
            ui_display_name="Not displayed",
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
        "state_size": ParameterMetadata(
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
        "unit_forget_bias": ParameterMetadata(
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
        "use_bias": ParameterMetadata(
            ui_display_name="Use Bias",
            default_value_reasoning="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to use bias "
            "terms.\n\nBatch Normalization, however, adds a trainable shift parameter which "
            "is added to the activation. When Batch Normalization is used in a layer, "
            "bias terms are redundant and may be removed.",
            example_value=[True],
            related_parameters=["bias_initializer, fc_layers"],
            other_information="If fc_layers is not specified, or use_bias is not specified for individual layers, "
            "the value of use_bias will be used as the default for all layers.",
            description_implications="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to leave this "
            "parameter set to True.",
            suggested_values=[True],
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "StackedTransformer": {
        "bias_initializer": ParameterMetadata(
            ui_display_name="Bias Initializer",
            default_value_reasoning="It is possible and common to initialize the biases to be zero, since the "
            "asymmetry breaking is provided by the small random numbers in the weights.",
            example_value=None,
            related_parameters=["weights_initializer"],
            other_information=None,
            description_implications="It's rare to see any performance gains from choosing a different bias "
            "initialization. Some practitioners like to use a small constant value such as "
            "0.01 for all biases to ensure that all ReLU units are activated in the "
            "beginning and have some effect on the gradient. However, it's still an open "
            "question as to whether this provides consistent improvement.",
            suggested_values="zeros",
            suggested_values_reasoning="It is possible and common to initialize the biases to be zero, "
            "since the asymmetry breaking is provided by the small random numbers in the "
            "weights. For ReLU non-linearities, some people like to use small constant "
            "value such as 0.01 for all biases because this ensures that all ReLU units "
            "fire in the beginning and therefore obtain and propagate some gradient. "
            "However, it is not clear if this provides a consistent improvement (in fact "
            "some results seem to indicate that this performs worse) and it is more common "
            "to simply use 0 bias initialization.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://cs231n.github.io/neural-networks-2/"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Taken from published literature (https://arxiv.org/abs/1908.07442).",
            example_value=[0.2],
            related_parameters=["fc_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "embedding_size": ParameterMetadata(
            ui_display_name="Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words, which can have a large vocbulary size. "
            "Ideally, after an embedding is trained, it captures some of the semantics of "
            "the input by placing semantically similar inputs close together in the "
            "embedding space.\n\nIn most cases, the embedding size is chosen empirically, "
            'by trial and error. From https://www.amazon.com/dp/1098115783, "one rule of '
            "thumb is to use the fourth root of the total number of unique categorical "
            "elements while another is that the embedding dimension should be approximately "
            "1.6 times the square root of the number of unique elements in the category, "
            'and no less than 600."\n\nIncreasing the embedding size may cause the model to '
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values="1.6 * sqrt(vocab_size)",
            suggested_values_reasoning="Rule of thumb suggested by a deep learning textbook. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture"
            ],
            internal_only=False,
        ),
        "embeddings_on_cpu": ParameterMetadata(
            ui_display_name="Embeddings on CPU",
            default_value_reasoning="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="By default embeddings matrices are stored on GPU memory if a GPU is used, "
            "as it allows for faster access. However, in some cases when the vocabulary size "
            "is very large, the full embedding matrix may be really big and unwieldy to have "
            "in GPU memory. This parameter forces the placement of the embedding matrix in "
            "regular memory and the CPU is used to access them. This may slow down training "
            "due to additional data transfer between CPU and GPU memory, but can lead to "
            "healthier GPU memory resource usage.",
            suggested_values=[False],
            suggested_values_reasoning="If GPU memory is not a constraint, having embeddings stored and accessed "
            "within the GPU is faster.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "embeddings_trainable": ParameterMetadata(
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
        "fc_activation": ParameterMetadata(
            ui_display_name="FC Activation",
            default_value_reasoning="The Rectified Linear Units (ReLU) function is the standard activation function "
            "used for adding non-linearity. It is simple, fast, and empirically works well ("
            "https://arxiv.org/abs/1803.08375).",
            example_value=["relu"],
            related_parameters=["activation, activation_function, conv_activation, recurrent_activation"],
            other_information=None,
            description_implications="Changing the activation functions has an impact on the computational load of "
            "the model and might require further hypterparameter tuning",
            suggested_values="relu, alternatively leakyRelu or elu",
            suggested_values_reasoning="The default value will work well in the majority of the cases",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html"],
            internal_only=False,
        ),
        "fc_dropout": ParameterMetadata(
            ui_display_name="FC Dropout",
            default_value_reasoning="Dropout can cause training to become less stable. Consider start with a "
            "dropout-free baseline, and add dropout gradually in subsequent experiments.",
            example_value=[0.2],
            related_parameters=["dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "hidden_size": ParameterMetadata(
            ui_display_name="Hidden Size",
            default_value_reasoning="Taken from literature (https://arxiv.org/abs/1706.03762)",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the hidden size makes the model larger and slower to train, "
            "increases the model's capacity to capture more complexity. It also increases "
            "the chance of overfitting.",
            suggested_values="10 - 2048",
            suggested_values_reasoning="Increasing the hidden size makes sense if the model is underfitting. It's "
            "useful to train both smaller and larger models to see how model capacity "
            "affects performance. This should only be explored after the architecture of "
            "the model has been settled.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "and the positional embedding matrix are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "norm": ParameterMetadata(
            ui_display_name="Normalization Type",
            default_value_reasoning="While batch normalization and layer normalization usually lead to improvements, "
            "it can be useful to start with fewer bells and whistles.",
            example_value=["batch"],
            related_parameters=["norm_params"],
            other_information=None,
            description_implications="Normalization helps stabilize the learning process and can have a regularizing "
            "effect that can help with generalization. It's often suggested that with "
            "normalization, you can use a higher learning rate.",
            suggested_values='"batch" or "layer"',
            suggested_values_reasoning='Normalization tries to solve "internal covariate shift" that comes from the '
            "changing distributions of the inputs to layers deep in the network when "
            "weights are updated. For example, batch normalization standardizes the inputs "
            "to a layer for each mini-batch. Try out different normalizations to see if "
            "that helps with training stability",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/"
            ],
            internal_only=False,
        ),
        "norm_params": ParameterMetadata(
            ui_display_name="Normalization Parameters",
            default_value_reasoning="The default parameters that come with Torch's implementation of these "
            "normalization types are a trusted starting point.",
            example_value=[{"num_features": 100, "momentum": 0.2, "affine": False}],
            related_parameters=["`norm`"],
            other_information=None,
            description_implications="There are a variety of ways a certain set of parameters specificed could "
            "influence performance here. Broadly speaking the different values passed in "
            "here allow for different levels of smoothness to be observed in the learning "
            "curves. Since setting this parameters depends on the type of `norm` set, "
            "see [BatchNorm2d]("
            "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) for more "
            "information on the parameters to set for batch normalization, "
            "and see [LayerNorm]("
            "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for more "
            "information on the parameters to set for layer normalization.",
            suggested_values="Depends on the type of `norm` set.",
            suggested_values_reasoning="NO",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "For BatchNorm2d: https://arxiv.org/abs/1502.03167\n\nFor LayerNorm: https://arxiv.org/abs/1607.06450"
            ],
            internal_only=False,
        ),
        "num_fc_layers": ParameterMetadata(
            ui_display_name="Number of Fully Connected Layers",
            default_value_reasoning="The encoder already has learnable parameters.Sometimes the default is 1 for "
            "modules where the FC stack is used for shape management, or the only source of "
            "learnable parameters.",
            example_value=[1],
            related_parameters=["fc_layers"],
            other_information="Not all modules that have fc_layers also have an accompanying num_fc_layers parameter. "
            "Where both are present, fc_layers takes precedent over num_fc_layers. Specifying "
            "num_fc_layers alone uses fully connected layers that are configured by the defaults in "
            "FCStack.",
            description_implications="Increasing num_fc_layers will increase the capacity of the model. The model "
            "will be slower to train, and there's a higher risk of overfitting.",
            suggested_values="0-1",
            suggested_values_reasoning="The full model likely contains many learnable parameters. Consider starting "
            "with very few, or without any additional fully connected layers and add them "
            "if you observe evidence of limited model capacity. Sometimes the default is 1 "
            "for modules where the FC stack is used for shape management, or the only "
            "source of learnable parameters.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_heads": ParameterMetadata(
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
        "num_layers": ParameterMetadata(
            ui_display_name="Number of Transformer Layers",
            default_value_reasoning="The ideal number of layers depends on the data. For many data types, "
            "one layer is sufficient.",
            example_value=[1],
            related_parameters=None,
            other_information=None,
            description_implications="The ideal number of transformer layers depends on the length and complexity of "
            "input sequences, as well as the task.\n\nFor more complex tasks, and higher "
            "number of transformer layers may be useful. However, too many layers will "
            "increase memory and slow training while providing diminishing returns of model "
            "performance.",
            suggested_values="1 - 12",
            suggested_values_reasoning="Increasing the number of layers may improve encoder performance.  However, "
            "more layers will increase training time and may cause overfitting.  Small "
            "numbers of layers usually work best.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "output_size": ParameterMetadata(
            ui_display_name="Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "pretrained_embeddings": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "representation": ParameterMetadata(
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
        "should_embed": ParameterMetadata(
            ui_display_name="Not displayed",
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
        "transformer_output_size": ParameterMetadata(
            ui_display_name="Transformer Output Size",
            default_value_reasoning="A modest value, not too small, not too large.",
            example_value=None,
            related_parameters=["num_fc_layers, fc_layers"],
            other_information="If num_fc_layers=0 and fc_layers=None, and there are no fully connected layers defined "
            "on the module, then this parameter may have no effect on the module's final output "
            "shape.",
            description_implications="If there are fully connected layers in this module, increasing the output size "
            "of each fully connected layer will increase the capacity of the model. However, "
            "the model may be slower to train, and there's a higher risk of overfitting. If "
            "it seems like the model could use even more capacity, consider increasing the "
            "number of fully connected layers, or explore other architectures.",
            suggested_values="10 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "use_bias": ParameterMetadata(
            ui_display_name="Use Bias",
            default_value_reasoning="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to use bias "
            "terms.\n\nBatch Normalization, however, adds a trainable shift parameter which "
            "is added to the activation. When Batch Normalization is used in a layer, "
            "bias terms are redundant and may be removed.",
            example_value=[True],
            related_parameters=["bias_initializer, fc_layers"],
            other_information="If fc_layers is not specified, or use_bias is not specified for individual layers, "
            "the value of use_bias will be used as the default for all layers.",
            description_implications="Bias terms may improve model accuracy, and don't have much impact in terms of "
            "memory or training speed. For most models it is reasonable to leave this "
            "parameter set to True.",
            suggested_values=[True],
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "weights_initializer": ParameterMetadata(
            ui_display_name="Layer Weights Initializer",
            default_value_reasoning="Taken from [this paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The method you choose to initialize layer weights during training can have a "
            "big impact on performance as well as the reproducibility of your final model "
            "between runs. As an example, if you were to randomly initialize weights you "
            "would risk non-reproducibility (and possibly general training performance), "
            "but sticking with constant values for initialization might significantly "
            "increase the time needed for model convergence. Generally, choosing one of the "
            "probabilistic approaches strikes a balance between the two extremes, "
            "and the literature kicked off by the landmark [*Xavier et al.* paper]("
            "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) provides a few good "
            "options. See this nice discussion from [Weights and Biases]("
            "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
            "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
            "net%20train%20better%20and%20faster.) for more information.",
            suggested_values="xavier_uniform",
            suggested_values_reasoning="Changing the weights initialization scheme is something to consider if a "
            "model is having trouble with convergence, or otherwise it is something to "
            "experiment with after other factors are considered. The default choice ("
            "`xavier_uniform`) is a suitable starting point for most tasks.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=[
                "Weights and Biases blog post: "
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural-nets#:~:text=Studies"
                "%20have%20shown%20that%20initializing,net%20train%20better%20and%20faster.\n\nXavier et al. paper: "
                "http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
            ],
            internal_only=False,
        ),
    },
    "T5Encoder": {
        "d_ff": ParameterMetadata(
            ui_display_name="Dimensionality of Feed-Forward Layer",
            default_value_reasoning="Default value matches the pre-trained encoder.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="If using a pre-trained encoder, this parameter will be automatically derived "
            "from the pre-trained model.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "d_kv": ParameterMetadata(
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
        "d_model": ParameterMetadata(
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
        "dropout_rate": ParameterMetadata(
            ui_display_name="dropout_rate",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "feed_forward_proj": ParameterMetadata(
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
        "initializer_factor": ParameterMetadata(
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
        "layer_norm_eps": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "num_decoder_layers": ParameterMetadata(
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
        "num_heads": ParameterMetadata(
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
        "num_layers": ParameterMetadata(
            ui_display_name="Number of Transformer Layers",
            default_value_reasoning="The default value matches the number of layers in the default pretrained encoder.",
            example_value=[6],
            related_parameters=["pretrained_model_or_path"],
            other_information=None,
            description_implications="The ideal number of transformer layers depends on the length and complexity of "
            "input sequences, as well as the task.\n\nIf using a pre-trained model, "
            "this parameter will be automatically derived from the pre-trained model.",
            suggested_values="1 - 12",
            suggested_values_reasoning="Increasing the number of layers may improve encoder performance.  However, "
            "more layers will increase training time and may cause overfitting.  Small "
            "numbers of layers usually work best.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "relative_attention_num_buckets": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "TransformerXLEncoder": {
        "adaptive": ParameterMetadata(
            ui_display_name="Adaptive Softmax",
            default_value_reasoning="Huggingface default.",
            example_value=None,
            related_parameters=["vocab_size"],
            other_information=None,
            description_implications="Adaptive softmax is a speedup technique for computing probability distributions "
            "over words. For text with large vocabulary, adaptive softmax improves both "
            "training speed.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "attn_type": ParameterMetadata(
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
        "clamp_len": ParameterMetadata(
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
        "cutoffs": ParameterMetadata(
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
        "d_embed": ParameterMetadata(
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
        "d_head": ParameterMetadata(
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
        "d_inner": ParameterMetadata(
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
        "d_model": ParameterMetadata(
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
        "div_val": ParameterMetadata(
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
        "dropatt": ParameterMetadata(
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
        "dropout": ParameterMetadata(
            ui_display_name="dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=None,
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "eos_token_id": ParameterMetadata(
            ui_display_name="End-of-Sequence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "init": ParameterMetadata(
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
        "init_range": ParameterMetadata(
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
        "init_std": ParameterMetadata(
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
        "layer_norm_epsilon": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "mem_len": ParameterMetadata(
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
        "n_head": ParameterMetadata(
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
        "n_layer": ParameterMetadata(
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
        "pre_lnorm": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "proj_init_std": ParameterMetadata(
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
        "proj_share_all_but_first": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "same_length": ParameterMetadata(
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
        "sample_softmax": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "untie_r": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "ViTEncoder": {
        "attention_probs_dropout_prob": ParameterMetadata(
            ui_display_name="Attention Dropout",
            default_value_reasoning="Taken from literature (https://arxiv.org/abs/2010.11929).",
            example_value=[0.2],
            related_parameters=["hidden_dropout_prob,\nattention_probs_dropout_prob"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "gradient_checkpointing": ParameterMetadata(
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
        "hidden_act": ParameterMetadata(
            ui_display_name="Hidden Layer Activation",
            default_value_reasoning="Taken from huggingface.",
            example_value=["relu"],
            related_parameters=None,
            other_information=None,
            description_implications="Changing this activation function will only affect the feed-forward layers of "
            "the transformer.",
            suggested_values="gelu",
            suggested_values_reasoning="Taken from huggingface defaults.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "[Huggingface docs for ViT config]("
                "https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTConfig.hidden_act)\n\n["
                "Relevant StackOverflow discussion]("
                "https://ai.stackexchange.com/questions/30341/why-does-a-transformer-not-use-an-activation-function"
                "-following-the-multi-head-a)"
            ],
            internal_only=False,
        ),
        "hidden_dropout_prob": ParameterMetadata(
            ui_display_name="Hidden Dropout",
            default_value_reasoning="Taken from literature (https://arxiv.org/abs/2010.11929).",
            example_value=[0.2],
            related_parameters=["hidden_dropout_prob,\nattention_probs_dropout_prob"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "hidden_size": ParameterMetadata(
            ui_display_name="Hidden Size",
            default_value_reasoning="Huggingface default.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the hidden size makes the model larger and slower to train, "
            "increases the model's capacity to capture more complexity. It also increases "
            "the chance of overfitting.",
            suggested_values="10 - 2048",
            suggested_values_reasoning="Increasing the hidden size makes sense if the model is underfitting. It's "
            "useful to train both smaller and larger models to see how model capacity "
            "affects performance. This should only be explored after the architecture of "
            "the model has been settled.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "intermediate_size": ParameterMetadata(
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
        "layer_norm_eps": ParameterMetadata(
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
        "num_attention_heads": ParameterMetadata(
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
        "num_hidden_layers": ParameterMetadata(
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
        "patch_size": ParameterMetadata(
            ui_display_name="Patch Size",
            default_value_reasoning="Taken from ViT paper.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="The implications of the image patch size for this layer depend on other "
            "factors, such as the true resolution of the incoming image dataset. If the "
            "patch size is kept consistent but a higher resolution image is used as input, "
            "then the resulting chunked sequence of tokens will be longer than it would have "
            "been if the input resolution was lower. \n\nThe ViT paper notes that decreasing "
            "the patch size in this way led to robust improvements without introducing other "
            "parameters.",
            suggested_values=(16, 32),
            suggested_values_reasoning="16 and 32 are the values used in the original ViT paper.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "[Huggingface docs](https://huggingface.co/docs/transformers/model_doc/vit)\n\n[ViT paper]("
                "https://arxiv.org/abs/2010.11929)"
            ],
            internal_only=False,
        ),
        "pretrained_model": ParameterMetadata(
            ui_display_name="Pretrained model name",
            default_value_reasoning="The default model is the canonical model for this model architecture, "
            "and is therefore a good starting point for most use cases.",
            example_value=None,
            related_parameters=["use_pretrained, trainable, pretrained_kwargs"],
            other_information=None,
            description_implications="There are two factors to consider when choosing a pre-trained model: (1) size, "
            "and (2) task similarity. \n\nThe larger the model, the more subtle its "
            "comprehension of inputs can become. However, larger models are also more "
            "compute and memory-intensive to train.\n\nModels pretrained on highly-related "
            "source tasks are more likely to be successful on the target task. Consider "
            "searching the HuggingFace model repository for models trained on similar tasks.",
            suggested_values="google/vit-large-patch16-224",
            suggested_values_reasoning="If you would like better performance and are not compute/memory-constrained, "
            "increasing model capacity can potentially provide a richer representation "
            "than the default. The suggested value upsizes the model while maintaining the "
            "same model architecture.\n\nModel trained on internet-scale datasets "
            "typically generalize well. Consider deviating from the default only if the "
            "images in the dataset originate from another domain (e.g. medical images, "
            "geospatial data).",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://arxiv.org/abs/2010.11929"],
            internal_only=False,
        ),
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
            ui_display_name="Trainable",
            default_value_reasoning="By default, model components are trainable.",
            example_value=None,
            related_parameters=["use_pretrained, pretrained_model, saved_weights_in_checkpoint"],
            other_information=None,
            description_implications="The tradeoff when using `trainable` is between speed and flexibility. If False, "
            "less weights are subject to change and the model will therefore train faster. "
            "However, the representations output by this component are fixed for each input.",
            suggested_values=[False],
            suggested_values_reasoning="Freezing the weights (i.e. `trainable = False`) is only worth trying if you "
            "are loading in pretrained weights. In that case, check to see if your model "
            "is overfitting. If so, freezing the weights (and therefore reducing model "
            "complexity) may be beneficial.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=[
                "https://www.ibm.com/cloud/learn/overfitting\n\nhttp://d2l.ai/chapter_computer-vision/fine-tuning"
                ".html"
            ],
            internal_only=False,
        ),
        "use_pretrained": ParameterMetadata(
            ui_display_name="Use pretrained model",
            default_value_reasoning="By default, the model is initialized as a pretrained model.",
            example_value=None,
            related_parameters=["trainable, pretrained_model_name, pretrained_model_name_or_path, pretrained_kwargs"],
            other_information=None,
            description_implications="Pretrained models have typically already learned features that are difficult to "
            "learn from scratch. They are particularly beneficial when training on small "
            "amounts of data.",
            suggested_values=[False],
            suggested_values_reasoning="If you have a large amount of data and/or you have data that differs from the "
            "typical distribution, then it might be worth training the model from scratch.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://machinelearningmastery.com/transfer-learning-for-deep-learning/"],
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
    },
    "XLMEncoder": {
        "asm": ParameterMetadata(
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
        "attention_dropout": ParameterMetadata(
            ui_display_name="attention_dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "bos_index": ParameterMetadata(
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
        "bos_token_id": ParameterMetadata(
            ui_display_name="Beginning-of-Sentence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "causal": ParameterMetadata(
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
        "dropout": ParameterMetadata(
            ui_display_name="dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["attention_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "emb_dim": ParameterMetadata(
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
        "embed_init_std": ParameterMetadata(
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
        "end_n_top": ParameterMetadata(
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
        "eos_index": ParameterMetadata(
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
        "gelu_activation": ParameterMetadata(
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
        "init_std": ParameterMetadata(
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
        "is_encoder": ParameterMetadata(
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
        "lang_id": ParameterMetadata(
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
        "layer_norm_eps": ParameterMetadata(
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
        "mask_index": ParameterMetadata(
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
        "mask_token_id": ParameterMetadata(
            ui_display_name="Mask Token ID",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "max_position_embeddings": ParameterMetadata(
            ui_display_name="Max Position Embeddings",
            default_value_reasoning="Taken from huggingface.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="An embedding is a relatively low-dimensional space that is used to translate "
            "high-dimensional vectors like words or positions, which can have a large "
            "vocbulary size. Ideally, after an embedding is trained, it captures some of the "
            "semantics of the input by placing semantically similar inputs close together in "
            "the embedding space.\n\nIncreasing the embedding size may cause the model to "
            "train more slowly, but the higher dimensionality can also improve overall "
            "quality.",
            suggested_values=512,
            suggested_values_reasoning="Out of the box value based on published literature. Try models with smaller "
            "or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "n_heads": ParameterMetadata(
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
        "n_langs": ParameterMetadata(
            ui_display_name="Number of Languages",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "n_layers": ParameterMetadata(
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
        "pad_index": ParameterMetadata(
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
        "pad_token_id": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "sinusoidal_embeddings": ParameterMetadata(
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
        "start_n_top": ParameterMetadata(
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
        "trainable": ParameterMetadata(
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
        "unk_index": ParameterMetadata(
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
        "use_lang_emb": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "XLMRoBERTaEncoder": {
        "add_pooling_layer": ParameterMetadata(
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
        "bos_token_id": ParameterMetadata(
            ui_display_name="Beginning-of-Sentence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "eos_token_id": ParameterMetadata(
            ui_display_name="End-of-Sentence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "pad_token_id": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "trainable": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "XLNetEncoder": {
        "attn_type": ParameterMetadata(
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
        "bi_data": ParameterMetadata(
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
        "bos_token_id": ParameterMetadata(
            ui_display_name="Beginning-of-Sentence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "clamp_len": ParameterMetadata(
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
        "d_inner": ParameterMetadata(
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
        "d_model": ParameterMetadata(
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
        "dropout": ParameterMetadata(
            ui_display_name="dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["summary_last_dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "end_n_top": ParameterMetadata(
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
        "eos_token_id": ParameterMetadata(
            ui_display_name="End-of-Sequence Token Id",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "ff_activation": ParameterMetadata(
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
        "initializer_range": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=[0.02],
            related_parameters=["weights_initializer"],
            other_information="Must be greater than 0",
            description_implications="There is an ideal value for this variable that doesn't lead to the outputs of "
            "these matrices to vanish or explode",
            suggested_values="0.01-0.05",
            suggested_values_reasoning="Large values will likely lead to very large outputs. Small values will lead "
            "to vanishing outputs.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=None,
            internal_only=False,
        ),
        "layer_norm_eps": ParameterMetadata(
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
            ui_display_name=None,
            default_value_reasoning="Sets the maximum sequence length of the expected inputs, so input/output shapes "
            "are computed accurately.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "mem_len": ParameterMetadata(
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
        "n_head": ParameterMetadata(
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
        "n_layer": ParameterMetadata(
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
        "pad_token_id": ParameterMetadata(
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
        "pretrained_kwargs": ParameterMetadata(
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
        "reduce_output": ParameterMetadata(
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
        "reuse_len": ParameterMetadata(
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
        "same_length": ParameterMetadata(
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
        "saved_weights_in_checkpoint": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning="The weights of the encoder are not necessarily saved in the checkpoint. The user "
            "has to save them first.",
            example_value=None,
            related_parameters=["skip_save_model"],
            other_information=None,
            description_implications="The memory footprint for some of these encoders can be large.",
            suggested_values=[False],
            suggested_values_reasoning="Some of these encoders are large, so it might be better to load them as "
            "needed, especially if 1. they're not used frequently 2. the user doesn't have"
            " a lot of storage.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        ),
        "start_n_top": ParameterMetadata(
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
        "summary_activation": ParameterMetadata(
            ui_display_name="Summary Activation Function",
            default_value_reasoning="Default value used in pre-trained HF encoder.",
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
        "summary_last_dropout": ParameterMetadata(
            ui_display_name="summary_last_dropout",
            default_value_reasoning="Huggingface default.",
            example_value=[0.2],
            related_parameters=["dropout"],
            other_information=None,
            description_implications="Dropout is a computationally cheap regularization method where during training, "
            "some neurons are randomly ignored or “dropped out”. Increasing dropout has the "
            "effect of making the training process more noisy and lowering overall network "
            "capacity, but it can be an effective regularization method to reduce "
            "overfitting and improve generalization.",
            suggested_values="0.05 - 0.8",
            suggested_values_reasoning="Tuning dropout is really something to be done when all of the big choices "
            "about architecture have been settled. Consider starting with 0.5 and "
            "adjusting the dropout depending on observed model performance.",
            commonly_used=False,
            expected_impact=ExpectedImpact.HIGH,
            literature_references=["https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"],
            internal_only=False,
        ),
        "summary_type": ParameterMetadata(
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
        "summary_use_proj": ParameterMetadata(
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
        "trainable": ParameterMetadata(
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
        "untie_r": ParameterMetadata(
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
        "use_mems_eval": ParameterMetadata(
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
        "use_mems_train": ParameterMetadata(
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
        "use_pretrained": ParameterMetadata(
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
        "vocab": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning="Computed and passed along internally according to preprocessing settings.",
            example_value=["a", "b", "c"],
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=True,
        ),
        "vocab_size": ParameterMetadata(
            ui_display_name="Not displayed",
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
    },
    "encoder": {
        "fc_layers": ParameterMetadata(
            ui_display_name="Fully Connected Layers",
            default_value_reasoning="By default the stack is built by using num_fc_layers, output_size, use_bias, "
            "weights_initializer, bias_initializer, norm, norm_params, activation, "
            "dropout. When a list of dictionaries is provided, the stack is built following "
            "the parameters of each dict for building each layer.",
            example_value=[{"output_size": 128, "dropout": 0.1}, {"output_size": 64, "norm": "layer"}],
            related_parameters=[
                "output_size",
                "use_bias",
                "weights_initializer",
                "bias_initializer",
                "norm",
                "norm_params",
                "activation",
                "dropout",
            ],
            other_information=None,
            description_implications="The more layers that are specified the deeper and higher capacity the model "
            "will be. This makes it possible to potentially achieve better performance when "
            "a big anough amount of data is provided, but also makes the model more "
            "computationally expensive and potentially more prone to overfitting.",
            suggested_values=None,
            suggested_values_reasoning="It is easier to define a stack of fully connected layers by just specifying "
            "num_fc_layers, output_size and the other individual parameters. It will "
            "create a stack of layers with identical properties. Use this parameter only "
            "if you need a fine grained level of control of each individual layer in the "
            "stack.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=None,
            internal_only=False,
        )
    },
}
