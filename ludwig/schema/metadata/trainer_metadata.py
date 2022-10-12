# flake8: noqa
# fmt: off
from ludwig.schema.metadata.parameter_metadata import ExpectedImpact, ParameterMetadata

TRAINER_METADATA = (
    {'batch_size': ParameterMetadata(ui_display_name='Batch Size',
                                     default_value_reasoning='Not too big, not too small.',
                                     example_value=None,
                                     related_parameters=['eval_batch_size'],
                                     other_information=None,
                                     description_implications="There's conflicting advice about what batch size to "
                                                              "use. "
                                                              'Using a higher batch size will achieve the highest '
                                                              "throughput and training efficiency. However, "
                                                              "there's also "
                                                              'evidence that depending on other hyperparameters, '
                                                              'a smaller '
                                                              'batch size may produce a higher quality model.',
                                     suggested_values='auto',
                                     suggested_values_reasoning='Try at least a few different batch sizes to get a '
                                                                'sense '
                                                                'of whether batch size affects model performance',
                                     commonly_used=True,
                                     expected_impact=ExpectedImpact.HIGH,
                                     literature_references=None,
                                     internal_only=False),
      'max_batch_size': ParameterMetadata(ui_display_name='Max Batch Size',
                                                  default_value_reasoning='Not typically required.',
                                                  example_value=1024,
                                                  related_parameters=['batch_size', 'increase_batch_size_on_plateau'],
                                                  description_implications='Value used to manually limit the batch '
                                                                           'sizes explored by auto batch size tuning '
                                                                           'and batch size increasing on plateau.',
                                                  suggested_values=None,
                                                  suggested_values_reasoning=None,
                                                  commonly_used=False,
                                                  expected_impact=ExpectedImpact.MEDIUM,
                                                  literature_references=None,
                                                  internal_only=False),
     'checkpoints_per_epoch': ParameterMetadata(ui_display_name='Checkpoints per epoch',
                                                default_value_reasoning='Per-epoch behavior, which scales according '
                                                                        'to the '
                                                                        'dataset size.',
                                                example_value=None,
                                                related_parameters=['train_steps', 'steps_per_checkpoint'],
                                                other_information=None,
                                                description_implications='Epoch-based evaluation (using the default: '
                                                                         '0) is '
                                                                         'an appropriate fit for tabular datasets, '
                                                                         'which '
                                                                         'are small, fit in memory, and train '
                                                                         'quickly.\n'
                                                                         '\n'
                                                                         'However, this is a poor fit for unstructured '
                                                                         'datasets, which tend to be much larger, and '
                                                                         'train more slowly due to larger models.\n'
                                                                         '\n'
                                                                         "It's important to setup evaluation such "
                                                                         "that you "
                                                                         'do not wait several hours before getting a '
                                                                         'single evaluation result. In general, '
                                                                         'it is not '
                                                                         'necessary for models to train over the '
                                                                         'entirety '
                                                                         'of a dataset, nor evaluate over the '
                                                                         'entirety of '
                                                                         'a test set, to produce useful monitoring '
                                                                         'metrics '
                                                                         'and signals to indicate model health.\n'
                                                                         '\n'
                                                                         'It is also more engaging and more valuable '
                                                                         'to '
                                                                         'ensure a frequent pulse of evaluation '
                                                                         'metrics, '
                                                                         'even if they are partial.',
                                                suggested_values='2 - 10, for larger datasets',
                                                suggested_values_reasoning='Running evaluation too frequently can be '
                                                                           'wasteful while running evaluation not '
                                                                           'frequently enough can be prohibitively '
                                                                           'uninformative. In many large-scale '
                                                                           'training '
                                                                           'runs, evaluation is often configured to '
                                                                           'run on '
                                                                           'a sub-epoch time scale, or every few '
                                                                           'thousand '
                                                                           'steps.',
                                                commonly_used=True,
                                                expected_impact=ExpectedImpact.HIGH,
                                                literature_references=None,
                                                internal_only=False),
     'decay': ParameterMetadata(ui_display_name='Decay',
                                default_value_reasoning=None,
                                example_value=None,
                                related_parameters=['decay_rate', 'decay_steps', 'learning_rate'],
                                other_information=None,
                                description_implications='It’s almost always a good idea to use a schedule. For most '
                                                         'models, try the exponential decay schedule first.\n'
                                                         '\n'
                                                         'The exponential schedule divides the learning rate by the '
                                                         'same '
                                                         'factor (%) every epoch. This means that the learning rate '
                                                         'will '
                                                         'decrease rapidly in the first few epochs, and spend more '
                                                         'epochs '
                                                         'with a lower value, but never reach exactly zero. As a rule '
                                                         'of '
                                                         'thumb, compared to training without a schedule, you can use '
                                                         'a '
                                                         'slightly higher maximum learning rate. Since the learning '
                                                         'rate '
                                                         'changes over time, the whole training is not so sensitive '
                                                         'to the '
                                                         'value picked.',
                                suggested_values=None,
                                suggested_values_reasoning='There is no go-to schedule for all models. Changing the '
                                                           'learning rate, in general, has shown to make training less '
                                                           'sensitive to the learning rate value you pick for it. So '
                                                           'using '
                                                           'a learning rate schedule can give better training '
                                                           'performance '
                                                           'and make the model converge faster',
                                commonly_used=True,
                                expected_impact=ExpectedImpact.MEDIUM,
                                literature_references=[
                                    'https://peltarion.com/knowledge-center/documentation/modeling-view/run-a-model'
                                    '/optimization-principles-(in-deep-learning)/learning-rate-schedule '
                                ],
                                internal_only=False),
     'decay_rate': ParameterMetadata(ui_display_name='Decay Rate',
                                     default_value_reasoning='4-5% decay each step is an empirically useful decary '
                                                             'rate to '
                                                             'start with.',
                                     example_value=None,
                                     related_parameters=['decay_rate, decay_steps, learning_rate'],
                                     other_information=None,
                                     description_implications='Increasing the decay rate will lower the learning rate '
                                                              'faster. This could make the model more robust to a bad '
                                                              '(too '
                                                              'high) initial learning rate, but a decay rate that is '
                                                              'too '
                                                              'high could prohibit the model from learning anything at '
                                                              'all.',
                                     suggested_values='0.9 - 0.96',
                                     suggested_values_reasoning='Since this controls exponential decay, even a small '
                                                                'decay '
                                                                'rate will still be strongly impactful.',
                                     commonly_used=False,
                                     expected_impact=ExpectedImpact.MEDIUM,
                                     literature_references=[
                                         'https://peltarion.com/knowledge-center/documentation/modeling-view/run-a'
                                         '-model'
                                         '/optimization-principles-(in-deep-learning)/learning-rate-schedule '
                                     ],
                                     internal_only=False),
     'decay_steps': ParameterMetadata(ui_display_name='Decay Steps',
                                      default_value_reasoning='This default essentially enables the `learning_rate` to '
                                                              'decay by a factor of the `decay_rate` at 10000 training '
                                                              'steps.',
                                      example_value=[5000],
                                      related_parameters=['decay_rate', 'decay_steps', 'learning_rate'],
                                      other_information=None,
                                      description_implications='By increasing the value of decay steps, '
                                                               'you are increasing '
                                                               'the number of training steps it takes to decay the '
                                                               'learning rate by a factor of `decay_rate`. In other '
                                                               'words, '
                                                               'the bigger this parameter, the slower the learning '
                                                               'rate '
                                                               'decays.',
                                      suggested_values='10000 +/- 500 at a time',
                                      suggested_values_reasoning='The decay in the learning rate is calculated as the '
                                                                 'training step divided by the `decay_steps` plus one. '
                                                                 'Then the `decay_rate` is raised to the power of this '
                                                                 'exponent which is then multiplied to the current '
                                                                 'learning rate. All this to say that the learning '
                                                                 'rate is '
                                                                 'only decayed by a factor of the set `decay_rate` '
                                                                 'when '
                                                                 'the training step reaches the `decay_steps` and then '
                                                                 'subsequently when it reaches any multiple of '
                                                                 '`decay_steps`. You can think of `decay_steps` as a '
                                                                 'rate '
                                                                 'of decay for the `decay_rate`.',
                                      commonly_used=True,
                                      expected_impact=ExpectedImpact.MEDIUM,
                                      literature_references=None,
                                      internal_only=False),
     'early_stop': ParameterMetadata(ui_display_name='Early Stop',
                                     default_value_reasoning="Deep learning models are prone to overfitting. It's "
                                                             'generally a good policy to set up some early stopping '
                                                             "criteria as it's not useful to have a model train after "
                                                             "it's "
                                                             'maximized what it can learn. 5 consecutive rounds of '
                                                             "evaluation where there hasn't been any improvement on "
                                                             "the "
                                                             'validation set (including chance) is a reasonable '
                                                             'policy to '
                                                             'start with.',
                                     example_value=None,
                                     related_parameters=['epochs', 'train_steps'],
                                     other_information=None,
                                     description_implications='Decreasing this value is a more aggressive policy. '
                                                              'Decreasing early stopping makes model training less '
                                                              'forgiving, as the model has less runway to demonstrate '
                                                              'consecutive metric improvements before the training '
                                                              'run is '
                                                              'quit. This can be efficient for pruning bad models '
                                                              'earlier, '
                                                              'but since the training process is inherently '
                                                              'non-deterministic and noisy, sometimes improvements '
                                                              'happen '
                                                              'very gradually over a long period of time.',
                                     suggested_values='5 - 10',
                                     suggested_values_reasoning="There's potentially a lot of randomness in how models "
                                                                'train, but so many consecutive rounds of no '
                                                                'improvement '
                                                                "is usually a good indicator that there's not much "
                                                                "more to "
                                                                'learn.',
                                     commonly_used=True,
                                     expected_impact=ExpectedImpact.MEDIUM,
                                     literature_references=None,
                                     internal_only=False),
     'epochs': ParameterMetadata(ui_display_name='Epochs',
                                 default_value_reasoning='A very high training length ceiling. Models will almost '
                                                         'always '
                                                         'hit early stopping criteria before hitting a 100-epoch '
                                                         'ceiling.',
                                 example_value=None,
                                 related_parameters='eval_batch_size',
                                 other_information=None,
                                 description_implications="There's conflicting advice about what batch size to use. "
                                                          'Using a higher batch size will achieve the highest '
                                                          "throughput and training efficiency. However, there's also "
                                                          'evidence that depending on other hyperparameters, a smaller '
                                                          'batch size may produce a higher quality model.',
                                 suggested_values='auto',
                                 suggested_values_reasoning='Try at least a few different batch sizes to get a sense '
                                                            'of whether batch size affects model performance',
                                 commonly_used=True,
                                 expected_impact=ExpectedImpact.HIGH,
                                 literature_references=None,
                                 internal_only=False),
 'checkpoints_per_epoch': ParameterMetadata(ui_display_name='Checkpoints per epoch',
                                            default_value_reasoning='Per-epoch behavior, which scales according to the '
                                                                    'dataset size.',
                                            example_value=None,
                                            related_parameters='train_steps, steps_per_checkpoint',
                                            other_information=None,
                                            description_implications='Epoch-based evaluation (using the default: 0) is '
                                                                     'an appropriate fit for tabular datasets, which '
                                                                     'are small, fit in memory, and train quickly.\n'
                                                                     '\n'
                                                                     'However, this is a poor fit for unstructured '
                                                                     'datasets, which tend to be much larger, and '
                                                                     'train more slowly due to larger models.\n'
                                                                     '\n'
                                                                     "It's important to setup evaluation such that you "
                                                                     'do not wait several hours before getting a '
                                                                     'single evaluation result. In general, it is not '
                                                                     'necessary for models to train over the entirety '
                                                                     'of a dataset, nor evaluate over the entirety of '
                                                                     'a test set, to produce useful monitoring metrics '
                                                                     'and signals to indicate model health.\n'
                                                                     '\n'
                                                                     'It is also more engaging and more valuable to '
                                                                     'ensure a frequent pulse of evaluation metrics, '
                                                                     'even if they are partial.',
                                            suggested_values='2 - 10, for larger datasets',
                                            suggested_values_reasoning='Running evaluation too frequently can be '
                                                                       'wasteful while running evaluation not '
                                                                       'frequently enough can be prohibitively '
                                                                       'uninformative. In many large-scale training '
                                                                       'runs, evaluation is often configured to run on '
                                                                       'a sub-epoch time scale, or every few thousand '
                                                                       'steps.',
                                            commonly_used=True,
                                            expected_impact=ExpectedImpact.HIGH,
                                            literature_references=None,
                                            internal_only=False),
 'decay': ParameterMetadata(ui_display_name='Decay',
                            default_value_reasoning=None,
                            example_value=None,
                            related_parameters='decay_rate, decay_steps, learning_rate',
                            other_information=None,
                            description_implications='It’s almost always a good idea to use a schedule. For most '
                                                     'models, try the exponential decay schedule first.\n'
                                                     '\n'
                                                     'The exponential schedule divides the learning rate by the same '
                                                     'factor (%) every epoch. This means that the learning rate will '
                                                     'decrease rapidly in the first few epochs, and spend more epochs '
                                                     'with a lower value, but never reach exactly zero. As a rule of '
                                                     'thumb, compared to training without a schedule, you can use a '
                                                     'slightly higher maximum learning rate. Since the learning rate '
                                                     'changes over time, the whole training is not so sensitive to the '
                                                     'value picked.',
                            suggested_values=None,
                            suggested_values_reasoning='There is no go-to schedule for all models. Changing the '
                                                       'learning rate, in general, has shown to make training less '
                                                       'sensitive to the learning rate value you pick for it. So using '
                                                       'a learning rate schedule can give better training performance '
                                                       'and make the model converge faster',
                            commonly_used=True,
                            expected_impact=ExpectedImpact.MEDIUM,
                            literature_references='https://peltarion.com/knowledge-center/documentation/modeling-view/run-a-model/optimization-principles-(in-deep-learning)/learning-rate-schedule',
                            internal_only=False),
 'decay_rate': ParameterMetadata(ui_display_name='Decay Rate',
                                 default_value_reasoning='4-5% decay each step is an empirically useful decary rate to '
                                                         'start with.',
                                 example_value=None,
                                 related_parameters='decay_rate, decay_steps, learning_rate',
                                 other_information=None,
                                 description_implications='Increasing the decay rate will lower the learning rate '
                                                          'faster. This could make the model more robust to a bad (too '
                                                          'high) initial learning rate, but a decay rate that is too '
                                                          'high could prohibit the model from learning anything at '
                                                          'all.',
                                 suggested_values='0.9 - 0.96',
                                 suggested_values_reasoning='Since this controls exponential decay, even a small decay '
                                                            'rate will still be strongly impactful.',
                                 commonly_used=False,
                                 expected_impact=ExpectedImpact.MEDIUM,
                                 literature_references='https://peltarion.com/knowledge-center/documentation/modeling-view/run-a-model/optimization-principles-(in-deep-learning)/learning-rate-schedule',
                                 internal_only=False),
 'decay_steps': ParameterMetadata(ui_display_name='Decay Steps',
                                  default_value_reasoning='This default essentially enables the `learning_rate` to '
                                                          'decay by a factor of the `decay_rate` at 10000 training '
                                                          'steps.',
                                  example_value=5000,
                                  related_parameters='decay_rate, decay_steps, learning_rate',
                                  other_information=None,
                                  description_implications='By increasing the value of decay steps, you are increasing '
                                                           'the number of training steps it takes to decay the '
                                                           'learning rate by a factor of `decay_rate`. In other words, '
                                                           'the bigger this parameter, the slower the learning rate '
                                                           'decays.',
                                  suggested_values='10000 +/- 500 at a time',
                                  suggested_values_reasoning='The decay in the learning rate is calculated as the '
                                                             'training step divided by the `decay_steps` plus one. '
                                                             'Then the `decay_rate` is raised to the power of this '
                                                             'exponent which is then multiplied to the current '
                                                             'learning rate. All this to say that the learning rate is '
                                                             'only decayed by a factor of the set `decay_rate` when '
                                                             'the training step reaches the `decay_steps` and then '
                                                             'subsequently when it reaches any multiple of '
                                                             '`decay_steps`. You can think of `decay_steps` as a rate '
                                                             'of decay for the `decay_rate`.',
                                  commonly_used=True,
                                  expected_impact=ExpectedImpact.MEDIUM,
                                  literature_references=None,
                                  internal_only=False),
 'early_stop': ParameterMetadata(ui_display_name='Early Stop',
                                 default_value_reasoning="Deep learning models are prone to overfitting. It's "
                                                         'generally a good policy to set up some early stopping '
                                                         "criteria as it's not useful to have a model train after it's "
                                                         'maximized what it can learn. 5 consecutive rounds of '
                                                         "evaluation where there hasn't been any improvement on the "
                                                         'validation set (including chance) is a reasonable policy to '
                                                         'start with.',
                                 example_value=None,
                                 related_parameters='epochs, train_steps',
                                 other_information=None,
                                 description_implications='Decreasing this value is a more aggressive policy. '
                                                          'Decreasing early stopping makes model training less '
                                                          'forgiving, as the model has less runway to demonstrate '
                                                          'consecutive metric improvements before the training run is '
                                                          'quit. This can be efficient for pruning bad models earlier, '
                                                          'but since the training process is inherently '
                                                          'non-deterministic and noisy, sometimes improvements happen '
                                                          'very gradually over a long period of time.',
                                 suggested_values='5 - 10',
                                 suggested_values_reasoning="There's potentially a lot of randomness in how models "
                                                            'train, but so many consecutive rounds of no improvement '
                                                            "is usually a good indicator that there's not much more to "
                                                            'learn.',
                                 commonly_used=True,
                                 expected_impact=ExpectedImpact.MEDIUM,
                                 literature_references=None,
                                 internal_only=False),
 'epochs': ParameterMetadata(ui_display_name='Epochs',
                             default_value_reasoning='A very high training length ceiling. Models will almost always '
                                                     'hit early stopping criteria before hitting a 100-epoch ceiling.',
                             example_value=None,
                             related_parameters='train_steps',
                             other_information=None,
                             description_implications='Decreasing this will shorten the overall runway for training '
                                                      'the model.',
                             suggested_values='0 (and use train_steps), or 100',
                             suggested_values_reasoning="Usually it's sensible to leave this very high and rely on a "
                                                        'solid early stopping policy to dictate when the model should '
                                                        'stop training. Some models and hyperparameter configurations '
                                                        'require many epochs through the dataset to converge while '
                                                        'others converge before a single epoch through the data.',
                             commonly_used=False,
                             expected_impact=ExpectedImpact.HIGH,
                             literature_references=None,
                             internal_only=False),
 'eval_batch_size': ParameterMetadata(ui_display_name='Evaluation Batch Size',
                                      default_value_reasoning='Use the entire batch for model evaluation unless '
                                                              'otherwise specified',
                                      example_value=512,
                                      related_parameters='batch_size',
                                      other_information='Should only set the batch_size to a level that you can fit in '
                                                        'memory',
                                      description_implications='By increasing the `eval_batch_size` past the '
                                                               '`batch_size` parameter set value, you allow for more '
                                                               'parallelism in the batch evaluation step. For example, '
                                                               'if you have to evaluate the model on a test set of '
                                                               'size 1000, it is faster to evaluate two times with two '
                                                               'batches of size 500 as opposed to ten times with ten '
                                                               'batches of 100. Setting this parameter higher without '
                                                               'maxing out memory limits will speed up the model '
                                                               'training process overall.',
                                      suggested_values=(256, 512, 1024),
                                      suggested_values_reasoning='By observing memory consumption on training jobs, '
                                                                 'you can get a sense of how much extra memory is '
                                                                 'available for increasing this value. A good rule of '
                                                                 'thumb can be experimentally doubling the eval batch '
                                                                 'size if you do not have insight into memory usage.',
                                      commonly_used=False,
                                      expected_impact=ExpectedImpact.MEDIUM,
                                      literature_references=None,
                                      internal_only=False),
 'evaluate_training_set': ParameterMetadata(ui_display_name='Evaluate Training Set',
                                            default_value_reasoning='It could be useful to monitor evaluation metrics '
                                                                    'on the training set, as a secondary validation '
                                                                    'set.',
                                            example_value=None,
                                            related_parameters=None,
                                            other_information=None,
                                            description_implications='Running evaluation on the full training set, '
                                                                     'when your training set is large, can be a huge '
                                                                     'computational cost. Turning off training set '
                                                                     'evaluation will lead to significant gains in '
                                                                     'training throughput and efficiency. For small '
                                                                     'datasets that train and evaluate quickly, the '
                                                                     'choice is trivial.',
                                            suggested_values=False,
                                            suggested_values_reasoning='Running full-scale evaluation on the full '
                                                                       "training set doesn't usually provide any "
                                                                       'useful information over the validation '
                                                                       'dataset. Even with this set to False, '
                                                                       'continuous training loss metrics are still '
                                                                       'computed, so it will still be easy to spot '
                                                                       'signs of overfitting like when the '
                                                                       'training-validation loss curves diverge.',
                                            commonly_used=True,
                                            expected_impact=ExpectedImpact.HIGH,
                                            literature_references=None,
                                            internal_only=False),
 'gradient_clipping': ParameterMetadata(ui_display_name='Gradient Clipping',
                                        default_value_reasoning='A conservative cap on the maximum gradient size to '
                                                                'apply over a single training step.',
                                        example_value=None,
                                        related_parameters=None,
                                        other_information=None,
                                        description_implications='Gradient clipping is a technique to prevent '
                                                                 'exploding gradients in very deep networks. '
                                                                 'Increasing gradient clipping can help with model '
                                                                 'training loss curve stability, but it can also make '
                                                                 'training less efficient as weight at each training '
                                                                 'step is capped.',
                                        suggested_values=None,
                                        suggested_values_reasoning="It's usually sensible to have some conservative "
                                                                   'notion of gradient clipping to make modeling '
                                                                   'robust to a particularly bad or noisy batch of '
                                                                   'examples.',
                                        commonly_used=False,
                                        expected_impact=ExpectedImpact.MEDIUM,
                                        literature_references=None,
                                        internal_only=False),
 'increase_batch_size_eval_metric': ParameterMetadata(ui_display_name='Batch Size Increase: Evaluation Metric',
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
                                                      internal_only=False),
 'increase_batch_size_eval_split': ParameterMetadata(ui_display_name='Batch Size Increase: Evaluation Split',
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
                                                     internal_only=False),
 'increase_batch_size_on_plateau': ParameterMetadata(ui_display_name='Batch Size Increase On Plateau',
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
                                                     internal_only=False),
 'increase_batch_size_on_plateau_max': ParameterMetadata(ui_display_name='Batch Size Increase On Plateau: Cap',
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
                                                         internal_only=False),
     'increase_batch_size_on_plateau_patience': ParameterMetadata(ui_display_name='Batch Size Increase On Plateau: '
                                                                                  'Patience',
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
                                                                  internal_only=False),
     'increase_batch_size_on_plateau_rate': ParameterMetadata(ui_display_name='Batch Size Increase On Plateau: Rate',
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
                                                              internal_only=False),
 'increase_batch_size_on_plateau_rate': ParameterMetadata(ui_display_name='Batch Size Increase On Plateau: Rate',
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
                                                          internal_only=False),
 'learning_rate': ParameterMetadata(ui_display_name='Learning Rate',
                                    default_value_reasoning='Middle of the road learning rate to start with.',
                                    example_value=None,
                                    related_parameters=None,
                                    other_information=None,
                                    description_implications='The learning rate is a hyperparameter that controls how '
                                                             'much to change the model in response to the estimated '
                                                             'error each time the model weights are updated. '
                                                             'Increasing the learning rate may decrease learning curve '
                                                             'stability but also increase learning speed and '
                                                             'efficiency, leading to faster model convergence. '
                                                             'Decreasing the learning rate can help stabilize learning '
                                                             'curves at the cost of slower time to convergence.',
                                    suggested_values='0.00001 - 0.1',
                                    suggested_values_reasoning='Tabular models trained from scratch typically use '
                                                               'learning rates around 1e-3 while learning rates for '
                                                               'pre-trained models should be much smaller, typically '
                                                               'around 1e-5, which is important to mitigate '
                                                               'catastrophic forgetting. To make the model more robust '
                                                               'to any specific choice of learning rate, consider '
                                                               'turning enabling learning rate exponential decay.',
                                    commonly_used=True,
                                    expected_impact=ExpectedImpact.HIGH,
                                    literature_references=None,
                                    internal_only=False),
 'learning_rate_scaling': ParameterMetadata(ui_display_name='Learning Rate Scaling',
                                            default_value_reasoning='Traditionally the learning rate is scaled '
                                                                    'linearly with the number of workers to reflect '
                                                                    'the proportion by which the effective batch size '
                                                                    'is increased.',
                                            example_value=None,
                                            related_parameters=None,
                                            other_information=None,
                                            description_implications='Traditionally the learning rate is scaled '
                                                                     'linearly with the number of workers to reflect '
                                                                     'the proportion by which the effective batch size '
                                                                     'is increased. For very large batch sizes, a '
                                                                     'softer square-root scale can sometimes lead to '
                                                                     'better model performance. If the learning rate '
                                                                     'is hand-tuned for a given number of workers, '
                                                                     'setting this value to constant can be used to '
                                                                     'disable scale-up.',
                                            suggested_values='linear or sqrt',
                                            suggested_values_reasoning='Traditionally the learning rate is scaled '
                                                                       'linearly with the number of workers to reflect '
                                                                       'the proportion by which the effective batch '
                                                                       'size is increased. For very large batch sizes, '
                                                                       'a softer square-root scale can sometimes lead '
                                                                       'to better model performance. If the learning '
                                                                       'rate is hand-tuned for a given number of '
                                                                       'workers, setting this value to constant can be '
                                                                       'used to disable scale-up.',
                                            commonly_used=False,
                                            expected_impact=ExpectedImpact.MEDIUM,
                                            literature_references=None,
                                            internal_only=False),
 'learning_rate_warmup_epochs': ParameterMetadata(ui_display_name='Learning Rate Warmup Epochs',
                                                  default_value_reasoning='The randomness of how weights are '
                                                                          'initialized can result in strange, noisy '
                                                                          'gradient updates during the begining of '
                                                                          'your training run. Learning rate warmup has '
                                                                          'the benefit of slowly starting to tune '
                                                                          'things like attention mechanisms in your '
                                                                          'network, that may be prone to bad initial '
                                                                          'conditions.',
                                                  example_value=None,
                                                  related_parameters=None,
                                                  other_information=None,
                                                  description_implications='Learning rate warmup sets a very low '
                                                                           'learning rate for a set number of training '
                                                                           'steps (warmup steps). After your warmup '
                                                                           'steps you use your "regular" learning rate '
                                                                           'or learning rate scheduler. You can also '
                                                                           'gradually increase your learning rate over '
                                                                           'the number of warmup steps.',
                                                  suggested_values=True,
                                                  suggested_values_reasoning="You don't want to warm up for too long, "
                                                                             'as after the model is starting to hill '
                                                                             'climb, you want to use the full weight '
                                                                             'of the learning rate to descend into '
                                                                             'good loss minima.',
                                                  commonly_used=False,
                                                  expected_impact=ExpectedImpact.MEDIUM,
                                                  literature_references='https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps',
                                                  internal_only=False),
 'optimizer': ParameterMetadata(ui_display_name='Optimizer',
                                default_value_reasoning='First try Adam because it is more likely to return good '
                                                        'results without an advanced fine tuning.',
                                example_value=None,
                                related_parameters=None,
                                other_information=None,
                                description_implications='Choosing a good optimizer for your machine learning project '
                                                         'can be overwhelming. Popular deep learning libraries such as '
                                                         'PyTorch or TensorFLow offer a broad selection of different '
                                                         'optimizers — each with its own strengths and weaknesses. '
                                                         'However, picking the wrong optimizer can have a substantial '
                                                         'negative impact on the performance of your machine learning '
                                                         'model [1][2]. This makes optimizers a critical design choice '
                                                         'in the process of building, testing, and deploying your '
                                                         'machine learning model.',
                                suggested_values='adam, adamw',
                                suggested_values_reasoning='As a rule of thumb: If you have the resources to find a '
                                                           'good learning rate schedule, SGD with momentum is a solid '
                                                           'choice. If you are in need of quick results without '
                                                           'extensive hypertuning, tend towards adaptive gradient '
                                                           'methods like adam or adamw.',
                                commonly_used=True,
                                expected_impact=ExpectedImpact.HIGH,
                                literature_references='https://www.youtube.com/watch?v=mdKjMPmcWjY',
                                internal_only=False),
 'reduce_learning_rate_eval_metric': ParameterMetadata(ui_display_name='Reduce Learning Rate Eval Metric',
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
                                                       internal_only=False),
 'reduce_learning_rate_eval_split': ParameterMetadata(ui_display_name='Reduce Learning Rate Eval Split',
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
                                                      internal_only=False),
 'reduce_learning_rate_on_plateau': ParameterMetadata(ui_display_name='Reduce Learning Rate On Plateau',
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
                                                      internal_only=False),
 'reduce_learning_rate_on_plateau_patience': ParameterMetadata(ui_display_name='Reduce Learning Rate on Plateau '
                                                                               'Patience',
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
                                                               internal_only=False),
 'reduce_learning_rate_on_plateau_rate': ParameterMetadata(ui_display_name='Reduce Learning Rate On Plateau Rate',
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
                                                           internal_only=False),
 'regularization_lambda': ParameterMetadata(ui_display_name='Regularization Lambda',
                                            default_value_reasoning='How to tune the overall impact of the '
                                                                    'regularization term by multiplying its value by a '
                                                                    'scalar known as lambda (also called the '
                                                                    'regularization rate).',
                                            example_value=None,
                                            related_parameters='regularization_type',
                                            other_information=None,
                                            description_implications='When choosing a lambda value, the goal is to '
                                                                     'strike the right balance between simplicity and '
                                                                     'training-data fit:\n'
                                                                     '\n'
                                                                     'If your lambda value is too high, your model '
                                                                     'will be simple, but you run the risk of '
                                                                     "underfitting your data. Your model won't learn "
                                                                     'enough about the training data to make useful '
                                                                     'predictions.\n'
                                                                     '\n'
                                                                     'If your lambda value is too low, your model will '
                                                                     'be more complex, and you run the risk of '
                                                                     'overfitting your data. Your model will learn too '
                                                                     'much about the particularities of the training '
                                                                     "data, and won't be able to generalize to new "
                                                                     'data. The ideal value of lambda produces a model '
                                                                     'that generalizes well to new, previously unseen '
                                                                     'data. Unfortunately, that ideal value of lambda '
                                                                     "is data-dependent, so you'll need to do some "
                                                                     'tuning.',
                                            suggested_values=0.1,
                                            suggested_values_reasoning='The most common type of regularization is L2, '
                                                                       'also called simply “weight decay,” with values '
                                                                       'often on a logarithmic scale between 0 and '
                                                                       '0.1, such as 0.1, 0.001, 0.0001, etc.',
                                            commonly_used=True,
                                            expected_impact=ExpectedImpact.HIGH,
                                            literature_references='https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/lambda',
                                            internal_only=False),
 'regularization_type': ParameterMetadata(ui_display_name='Regularization Type',
                                          default_value_reasoning='L2 is a sufficiently good regularization to start '
                                                                  'with.',
                                          example_value=None,
                                          related_parameters='regularization_lambda',
                                          other_information=None,
                                          description_implications='L1 regularization penalizes the sum of absolute '
                                                                   'values of the weights, whereas L2 regularization '
                                                                   'penalizes the sum of squares of the weights. \n'
                                                                   'The L1 regularization solution is sparse. The L2 '
                                                                   'regularization solution is non-sparse.\n'
                                                                   'L2 regularization doesn’t perform feature '
                                                                   'selection, since weights are only reduced to '
                                                                   'values near 0 instead of 0. L1 regularization has '
                                                                   'built-in feature selection.\n'
                                                                   'L1 regularization is robust to outliers, L2 '
                                                                   'regularization is not.',
                                          suggested_values='L2',
                                          suggested_values_reasoning=None,
                                          commonly_used=True,
                                          expected_impact=ExpectedImpact.HIGH,
                                          literature_references='https://neptune.ai/blog/fighting-overfitting-with-l1-or-l2-regularization#:~:text=The%20differences%20between%20L1%20and,regularization%20solution%20is%20non%2Dsparse.',
                                          internal_only=False),
 'should_shuffle': ParameterMetadata(ui_display_name='Should Shuffle',
                                     default_value_reasoning="In general, it's a good idea to mix up data on each "
                                                             'batch so that the neural network gets the broadest '
                                                             'exposure to the dataset.',
                                     example_value=None,
                                     related_parameters=None,
                                     other_information=None,
                                     description_implications='Turning off mini-batch shuffling can make training '
                                                              'faster, but it may lead to worse performance overall as '
                                                              'shuffling helps mitigate overfitting.',
                                     suggested_values=True,
                                     suggested_values_reasoning='One of the most powerful things about neural networks '
                                                                'is that they can be very complex functions, allowing '
                                                                'one to learn very complex relationships between your '
                                                                'input and output data. These relationships can '
                                                                'include things you would never expect, such as the '
                                                                'order in which data is fed in per epoch. If the order '
                                                                'of data within each epoch is the same, then the model '
                                                                'may use this as a way of reducing the training error, '
                                                                'which is a sort of overfitting.',
                                     commonly_used=False,
                                     expected_impact='High',
                                     literature_references='https://stats.stackexchange.com/questions/245502/why-should-we-shuffle-data-while-training-a-neural-network#:~:text=it%20helps%20the%20training%20converge,the%20order%20of%20the%20training',
                                     internal_only=False),
 'staircase': ParameterMetadata(ui_display_name='Staircase',
                                default_value_reasoning='Extra knob for decaying the learning rate in a different way.',
                                example_value=None,
                                related_parameters=None,
                                other_information=None,
                                description_implications='An excessively aggressive decay results in optimizers never '
                                                         'reaching the minima, whereas a slow decay leads to chaotic '
                                                         'updates without significant improvement. Discrete learning '
                                                         'rate decay is another knob to help tune a balance.',
                                suggested_values=False,
                                suggested_values_reasoning='We have not found strong evidence that discretely decaying '
                                                           'the learning rate is superior to doing so continuously.',
                                commonly_used=False,
                                expected_impact=ExpectedImpact.LOW,
                                literature_references='https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler',
                                internal_only=False),
 'steps_per_checkpoint': ParameterMetadata(ui_display_name='Steps Per Checkpoint',
                                           default_value_reasoning='By default, we evaluate once per epoch, which '
                                                                   'scales according to the dataset size.',
                                           example_value=None,
                                           related_parameters='checkpoints_per_epoch',
                                           other_information=None,
                                           description_implications='Epoch-based evaluation (using the default: 0) is '
                                                                    'an appropriate fit for tabular datasets, which '
                                                                    'are small, fit in memory, and train quickly.\n'
                                                                    '\n'
                                                                    'However, this is a poor fit for unstructured '
                                                                    'datasets, which tend to be much larger, and train '
                                                                    'more slowly due to larger models.\n'
                                                                    '\n'
                                                                    "It's important to setup evaluation such that you "
                                                                    'do not wait several hours before getting a single '
                                                                    'evaluation result. In general, it is not '
                                                                    'necessary for models to train over the entirety '
                                                                    'of a dataset, nor evaluate over the entirety of a '
                                                                    'test set, to produce useful monitoring metrics '
                                                                    'and signals to indicate model health.\n'
                                                                    '\n'
                                                                    'It is also more engaging and more valuable to '
                                                                    'ensure a frequent pulse of evaluation metrics, '
                                                                    'even if they are partial.',
                                           suggested_values='O(1k) for larger datasets',
                                           suggested_values_reasoning='Running evaluation too frequently can be '
                                                                      'wasteful while running evaluation not '
                                                                      'frequently enough can be prohibitively '
                                                                      'uninformative. In many large-scale training '
                                                                      'runs, evaluation is often configured to run on '
                                                                      'a sub-epoch time scale, or every few thousand '
                                                                      'steps.',
                                           commonly_used=True,
                                           expected_impact=ExpectedImpact.HIGH,
                                           literature_references=None,
                                           internal_only=False),
 'train_steps': ParameterMetadata(ui_display_name='Train Steps',
                                  default_value_reasoning='This defaults to `epochs`, which is a very high training '
                                                          'length ceiling. Models will almost always hit early '
                                                          'stopping criteria before reaching the absolute end of the '
                                                          'training runway.',
                                  example_value=None,
                                  related_parameters='epochs',
                                  other_information=None,
                                  description_implications='Decreasing this will shorten the overall runway for '
                                                           'training the model.',
                                  suggested_values='0 (and use epochs), or 1000000, 1 for debugging',
                                  suggested_values_reasoning="Usually it's sensible to leave this very high and rely "
                                                             'on a solid early stopping policy to dictate when the '
                                                             'model should stop training. Some models and '
                                                             'hyperparameter configurations require many epochs '
                                                             'through the dataset to converge while others converge '
                                                             'before a single epoch through the data.',
                                  commonly_used=False,
                                  expected_impact=ExpectedImpact.HIGH,
                                  literature_references=None,
                                  internal_only=False),
 'validation_field': ParameterMetadata(ui_display_name='Validation Field',
                                       default_value_reasoning='Concrete evaluation metrics are usually better than '
                                                               'loss, the penalty for a bad prediction, which is only '
                                                               'a proxy for prediction correctness.',
                                       example_value=None,
                                       related_parameters='validation_field, validation_metric',
                                       other_information=None,
                                       description_implications='This parameter affects 1) what the early stopping '
                                                                'policy looks at to determine when to early stop and '
                                                                '2) hyperparameter optimization for determining the '
                                                                'best trial.',
                                       suggested_values='default behavior',
                                       suggested_values_reasoning=None,
                                       commonly_used=False,
                                       expected_impact=ExpectedImpact.HIGH,
                                       literature_references=None,
                                       internal_only=False),
 'validation_metric': ParameterMetadata(ui_display_name='Validation Metric',
                                        default_value_reasoning=None,
                                        example_value=None,
                                        related_parameters='validation_field, validation_metric',
                                        other_information=None,
                                        description_implications='This parameter affects 1) what the early stopping '
                                                                 'policy looks at to determine when to early stop and '
                                                                 '2) hyperparameter optimization for determining the '
                                                                 'best trial.',
                                        suggested_values='default behavior',
                                        suggested_values_reasoning=None,
                                        commonly_used=False,
                                        expected_impact='NONE',
                                        literature_references=None,
                                        internal_only=False)}
)
