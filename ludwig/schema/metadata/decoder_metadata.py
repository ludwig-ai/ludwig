from ludwig.schema.metadata.parameter_metadata import ExpectedImpact, ParameterMetadata

DECODER_METADATA = {
    "Classifier": {
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
        "num_classes": ParameterMetadata(
            ui_display_name="Not Displayed",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information="Internal Only",
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
            suggested_values=True,
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
                "Weights and Biases blog post: https://wandb.ai/site/articles/the-effects-of-weight-initialization-on"
                "-neural-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
                "net%20train%20better%20and%20faster.",
                "Xavier et al. paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
            ],
            internal_only=False,
        ),
    },
    "Projector": {
        "activation": ParameterMetadata(
            ui_display_name="Activation",
            default_value_reasoning=None,
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
        "clip": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
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
            suggested_values=True,
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
                "Weights and Biases blog post: https://wandb.ai/site/articles/the-effects-of-weight-initialization-on"
                "-neural-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
                "net%20train%20better%20and%20faster.",
                "Xavier et al. paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
            ],
            internal_only=False,
        ),
    },
    "Regressor": {
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
            suggested_values=True,
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
                "Weights and Biases blog post: https://wandb.ai/site/articles/the-effects-of-weight-initialization-on"
                "-neural-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
                "net%20train%20better%20and%20faster.",
                "Xavier et al. paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
            ],
            internal_only=False,
        ),
    },
    "SequenceGeneratorDecoder": {
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
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
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
        "num_layers": ParameterMetadata(
            ui_display_name="Number of Recurrent Layers",
            default_value_reasoning="The ideal number of layers depends on the data and task. For many data types, "
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
        "reduce_input": ParameterMetadata(
            ui_display_name="Combiner Reduce Mode",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="“last”: Reduces tensor by taking the last non-zero element per sequence in the "
            "sequence dimension.\n“sum”: Reduces tensor by summing across the sequence "
            "dimension.\n“mean”: Reduces tensor by taking the mean of the sequence "
            "dimension.\n“avg”: synonym for “mean”.\n“max”: Reduces tensor by taking the "
            "maximum value of the last dimension across the sequence dimension.\n“concat”: "
            "Reduces tensor by concatenating the second and last dimension.\n“attention”: "
            "Reduces tensor by summing across the sequence dimension after applying "
            "feedforward attention.\n“none”: no reduction.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
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
            internal_only=False,
        ),
    },
    "SequenceTaggerDecoder": {
        "attention_embedding_size": ParameterMetadata(
            ui_display_name="Attention Embedding Size",
            default_value_reasoning="Not too big, not too small.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the embedding size may cause the model to train more slowly, "
            "but the higher dimensionality can also improve overall quality.",
            suggested_values="128 - 2048",
            suggested_values_reasoning="Try models with smaller or larger embedding sizes to observe relative impact.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "attention_num_heads": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
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
        "max_sequence_length": ParameterMetadata(
            ui_display_name=None,
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
        "use_attention": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
            internal_only=False,
        ),
    },
}
