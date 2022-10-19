from ludwig.schema.metadata.parameter_metadata import ExpectedImpact, ParameterMetadata

COMBINER_METADATA = {
    "type": ParameterMetadata(
        ui_display_name="Combiner Type",
        default_value_reasoning=None,
        example_value=None,
        related_parameters=None,
        other_information=None,
        description_implications=None,
        suggested_values_reasoning=None,
        commonly_used=True,
        expected_impact=ExpectedImpact.HIGH,
        literature_references=None,
        internal_only=False,
    ),
    "ComparatorCombiner": {
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
        "entity_1": ParameterMetadata(
            ui_display_name="Entity 1",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=["https://ludwig.ai/0.6/configuration/combiner/#comparator-combiner"],
            internal_only=False,
        ),
        "entity_2": ParameterMetadata(
            ui_display_name="Entity 2",
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=["https://ludwig.ai/0.6/configuration/combiner/#comparator-combiner"],
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
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
            suggested_values="15 - 1024",
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
            suggested_values="TRUE",
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
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
                "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
                "net%20train%20better%20and%20faster.",
                "Xavier et al. paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
            ],
            internal_only=False,
        ),
    },
    "ConcatCombiner": {
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
        "flatten_inputs": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
            suggested_values="16 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "residual": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
            suggested_values="TRUE",
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
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
                "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
                "net%20train%20better%20and%20faster.",
                "Xavier et al. paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
            ],
            internal_only=False,
        ),
    },
    "ProjectAggregateCombiner": {
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
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
            suggested_values="17 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "projection_size": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "residual": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
            suggested_values="TRUE",
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
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
                "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
                "net%20train%20better%20and%20faster.",
                "Xavier et al. paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
            ],
            internal_only=False,
        ),
    },
    "SequenceCombiner": {
        "encoder": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "main_sequence_feature": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
    },
    "SequenceConcatCombiner": {
        "main_sequence_feature": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
    },
    "TabNetCombiner": {
        "bn_epsilon": ParameterMetadata(
            ui_display_name="Batch Normalization Epsilon",
            default_value_reasoning="Default value found in popular ML packages like Keras and Tensorflow.",
            example_value=[1e-05],
            related_parameters=None,
            other_information=None,
            description_implications="An epsilon is added to the denominator of the batch normalization operation so "
            "that the function converges. Setting the epsilon to 0 is inadvisable.",
            suggested_values="1e-3-1e-9",
            suggested_values_reasoning="Common epsilon choices",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=[
                "[Keras example](https://keras.io/api/layers/normalization_layers/batch_normalization/)"
            ],
            internal_only=False,
        ),
        "bn_momentum": ParameterMetadata(
            ui_display_name="Batch Norm Momentum",
            default_value_reasoning=None,
            example_value=[0.05],
            related_parameters=None,
            other_information="`bn_momentum` is only used if `norm`: `batch`.  For other values of `norm` it has no "
            "effect.\n\n`bn_momentum` is different from optimizer momentum.  Batch norm moving "
            "estimate statistics are updated according to the rule:\nx_hat = (1 - momentum) * x_hat "
            "+ momentum * x_t,\nwhere x_hat is the estimated statistic and x_t is the new observed "
            "value.",
            description_implications="Higher values result in faster updates, but more sensitivity to noise in the "
            "dataset.  Lower values result in slower updates.\n\nIf momentum is set to 0, "
            "moving statistics will not be updated during training. This is likely to cause "
            "variance between train and test performance, and is not recommended.",
            suggested_values="0.01-0.2",
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=[
                "TabNet Paper: https://arxiv.org/abs/1908.07442",
                "Torch Batch Norm: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html",
            ],
            internal_only=False,
        ),
        "bn_virtual_bs": ParameterMetadata(
            ui_display_name="Ghost Normalization: Virtual batch size",
            default_value_reasoning="Paper default.",
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications="Virtual Batch Normalization is a normalization method that extends batch "
            "normalization. Regular batch normalization causes the output of a neural "
            "network for an input example  to be highly dependent on several other inputs  "
            "in the same minibatch. To avoid this problem in virtual batch normalization ("
            "VBN), each example is normalized based on the statistics collected on a "
            "reference batch of examples that are chosen once and fixed at the start of "
            "training, and on itself. The reference batch is normalized using only its own "
            "statistics. VBN is computationally expensive because it requires running "
            "forward propagation on two minibatches of data, so the authors use it only in "
            "the generator network. A higher virtual batch size could improve normalization, "
            "but it also causes training to run slower since each batch will be sampled "
            "multiple times.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=["https://paperswithcode.com/method/virtual-batch-normalization"],
            internal_only=False,
        ),
        "dropout": ParameterMetadata(
            ui_display_name="Dropout",
            default_value_reasoning="Taken from published literature (https://arxiv.org/abs/1908.07442).",
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
        "entmax_alpha": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "entmax_mode": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "num_shared_blocks": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "num_steps": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "num_total_blocks": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
            suggested_values="18 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
            literature_references=None,
            internal_only=False,
        ),
        "relaxation_factor": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "size": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "sparsity": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
    "TabTransformerCombiner": {
        "embed_input_feature_name": ParameterMetadata(
            ui_display_name="Embed Input Feature Name",
            default_value_reasoning="Though the ideal embedding size depends on the task and dataset, setting the "
            "feature embedding size equal to the hidden size and adding feature embeddings to "
            "hidden representations ('add') is a good starting point.",
            example_value=[64],
            related_parameters=["hidden_size"],
            other_information="Must be an integer, 'add', or null. If an integer, specifies the embedding size for "
            "input feature names. Input feature name embeddings will be concatenated to hidden "
            "representations. Must be less than or equal to hidden_size. If 'add', input feature "
            "names use embeddings the same size as hidden_size, and are added (element-wise) to the "
            "hidden representations. If null, input feature embeddings are not used.",
            description_implications="Input feature name embeddings have been shown to improve performance of deep "
            "learning methods on tabular data. Feature name embeddings play a similar role "
            "to positional embeddings in a language model, allowing the network to learn "
            "conditional dependencies between input features.",
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=["TabTransformer: Tabular Data Modeling Using Contextual Embeddings"],
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
            default_value_reasoning="Taken from published literature (https://arxiv.org/abs/1706.03762).",
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
            related_parameters=None,
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
        "fc_residual": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "hidden_size": ParameterMetadata(
            ui_display_name="Hidden Size",
            default_value_reasoning="Not too big, not too small.",
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
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
        "num_heads": ParameterMetadata(
            ui_display_name="Number of attention heads",
            default_value_reasoning="The middle value explored in the original TabTransformer paper. Source: "
            "https://arxiv.org/pdf/2012.06678.pdf",
            example_value=[8],
            related_parameters=None,
            other_information=None,
            description_implications="Increasing the number of attention heads can increase model performance at the "
            "cost of additional compute and memory.",
            suggested_values=16,
            suggested_values_reasoning="If your model is underperforming, increasing the number of attention heads "
            "can improve its ability to correlate items in a sequence.",
            commonly_used=False,
            expected_impact=ExpectedImpact.LOW,
            literature_references=["https://arxiv.org/pdf/2012.06678.pdf"],
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
            suggested_values="19 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
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
            suggested_values="20 - 1024",
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
            suggested_values="TRUE",
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
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
                "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
                "net%20train%20better%20and%20faster.",
                "Xavier et al. paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
            ],
            internal_only=False,
        ),
    },
    "TransformerCombiner": {
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
            default_value_reasoning="Taken from published literature (https://arxiv.org/abs/1706.03762).",
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
            related_parameters=None,
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
        "fc_residual": ParameterMetadata(
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
            other_information=None,
            description_implications=None,
            suggested_values=None,
            suggested_values_reasoning=None,
            commonly_used=False,
            expected_impact=ExpectedImpact.UNKNOWN,
            literature_references=None,
            internal_only=False,
        ),
        "hidden_size": ParameterMetadata(
            ui_display_name="Hidden Size",
            default_value_reasoning="Not too big, not too small.",
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
            ui_display_name=None,
            default_value_reasoning=None,
            example_value=None,
            related_parameters=None,
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
            suggested_values="21 - 1024",
            suggested_values_reasoning="Increasing the output size increases the capacity of the model. If this seems "
            "to have a positive effect, then it could be worth increasing the number of "
            "layers, or trying a different architecture with a larger capacity.",
            commonly_used=False,
            expected_impact=ExpectedImpact.MEDIUM,
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
            suggested_values="22 - 1024",
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
            suggested_values="TRUE",
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
                "https://wandb.ai/site/articles/the-effects-of-weight-initialization-on-neural"
                "-nets#:~:text=Studies%20have%20shown%20that%20initializing,"
                "net%20train%20better%20and%20faster.",
                "Xavier et al. paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
            ],
            internal_only=False,
        ),
    },
}
