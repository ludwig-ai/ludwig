from typing import Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils


@DeveloperAPI
@schema_utils.ludwig_dataclass
class LLMGenerationConfig(schema_utils.BaseMarshmallowConfig):
    """Parameters for LLM Generation Config.

    Should match the parameters in
    https://huggingface.co/docs/transformers/v4.28.0/en/main_classes/text_generation#transformers.GenerationConfig
    """

    # Parameters that control the length of the output

    max_length: int = schema_utils.PositiveInteger(
        default=20,
        allow_none=True,
        description="The maximum length the generated tokens can have. Corresponds to the length of the input prompt "
        "+ max_new_tokens. Its effect is overridden by max_new_tokens, if also set.",
    )

    max_new_tokens: Optional[int] = schema_utils.PositiveInteger(
        default=20,
        allow_none=True,
        description="The maximum number of new tokens to generate, ignoring the number of tokens in the input prompt.",
    )

    min_length: int = schema_utils.NonNegativeInteger(
        default=0,
        allow_none=True,
        description="The minimum length of the sequence to be generated. Corresponds to the length of the "
        "input prompt + min_new_tokens. Its effect is overridden by min_new_tokens, if also set.",
    )

    min_new_tokens: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="The minimum number of new tokens to generate, ignoring the number of tokens in the input prompt.",
    )

    early_stopping: Optional[Union[bool, str]] = schema_utils.Boolean(
        default=False,
        description="Controls the stopping condition for beam-based methods, like beam-search. It accepts the following"
        " values: True, where the generation stops as soon as there are num_beams complete candidates; False, where an "
        "heuristic is applied and the generation stops when is it very unlikely to find better candidates; `never`, "
        "where the beam search procedure only stops when there cannot be better candidates (canonical beam search "
        "algorithm)",
    )

    max_time: Optional[float] = schema_utils.FloatRange(
        default=None,
        min=None,
        max=None,
        allow_none=True,
        description="The maximum amount of time you allow the computation to run for in seconds. generation will still"
        " finish the current pass after allocated time has been passed. ",
    )

    # Parameters that control the generation strategy used

    do_sample: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether or not to use sampling ; use greedy decoding otherwise.",
    )

    num_beams: Optional[int] = schema_utils.PositiveInteger(
        default=1,
        allow_none=True,
        description="Number of beams for beam search. 1 means no beam search.",
    )

    num_beam_groups: Optional[int] = schema_utils.PositiveInteger(
        default=1,
        allow_none=True,
        description="Number of groups to divide num_beams into in order to ensure diversity among different groups of "
        "beams. 1 means no group beam search.",
    )

    penalty_alpha: Optional[float] = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="The values balance the model confidence and the degeneration penalty in contrastive "
        " search decoding.",
    )

    use_cache: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Whether or not the model should use the past last key/values attentions (if applicable to the "
        "model) to speed up decoding.",
    )

    # Parameters for manipulation of the model output logits

    temperature: Optional[float] = schema_utils.NonNegativeFloat(
        default=1.0,
        allow_none=True,
        description="The value used to module the next token probabilities.",
    )

    top_k: Optional[int] = schema_utils.PositiveInteger(
        default=50,
        allow_none=True,
        description="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )

    top_p: Optional[float] = schema_utils.FloatRange(
        default=1.0,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="If set to float < 1, only the most probable tokens with probabilities that add up to "
        "top_p or higher are kept for generation.",
    )

    typical_p: Optional[float] = schema_utils.FloatRange(
        default=1.0,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="Local typicality measures how similar the conditional probability of predicting a target token "
        "next is to the expected conditional probability of predicting a random token next, given the partial text "
        "already generated. If set to float < 1, the smallest set of the most locally typical tokens with "
        "probabilities that add up to typical_p or higher are kept for generation.",
    )

    epsilon_cutoff: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="If set to float strictly between 0 and 1, only tokens with a conditional probability greater "
        "than epsilon_cutoff will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the"
        " size of the model.",
    )

    eta_cutoff: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float "
        "strictly between 0 and 1, a token is only considered if it is greater than either eta_cutoff or "
        "sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). The latter term is intuitively the expected next"
        " token probability, scaled by sqrt(eta_cutoff). In the paper, suggested values range from 3e-4 to 2e-3, "
        "depending on the size of the model.",
    )

    diversity_penalty: Optional[float] = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="The value used to control the diversity of the generated text. The higher the value, the more "
        "diverse the text will be. The value should be between 0 and 1.0. If set to 0, no diversity is enforced."
        "This value is subtracted from a beam(s) score if it generates a token same as any beam from other group at a"
        "particular time. Note that diversity_penalty is only effective if group beam search is enabled.",
    )

    repetition_penalty: Optional[float] = schema_utils.FloatRange(
        default=1.0,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="The parameter for repetition penalty. 1.0 means no penalty.",
    )

    encoder_repetition_penalty: Optional[float] = schema_utils.FloatRange(
        default=1.0,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="An exponential penalty on sequences that are not in the original input. 1.0 means no penalty.",
    )

    length_penalty: Optional[float] = schema_utils.FloatRange(
        default=1.0,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="Exponential penalty to the length that is used with beam-based generation. It is applied as an "
        "exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is"
        " the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while "
        "length_penalty < 0.0 encourages shorter sequences.",
    )

    no_repeat_ngram_size: Optional[int] = schema_utils.NonNegativeInteger(
        default=0,
        allow_none=True,
        description="If set to int > 0, all ngrams of that size can only occur once.",
    )

    bad_words_ids: Optional[list[list[int]]] = schema_utils.List(
        default=None,
        allow_none=True,
        description="List of token ids that are not allowed to be generated. In order to get the tokens of the words "
        "that should not appear in the generated text, use tokenizer(bad_word, add_prefix_space=True).input_ids.",
    )

    force_words_ids: Optional[list[list[int]]] = schema_utils.List(
        default=None,
        allow_none=True,
        description="List of token ids that are forced to be generated by the model. In order to get the tokens of the"
        " words that should appear in the generated text, use tokenizer(force_word, add_prefix_space=True).input_ids.",
    )

    renormalize_logits: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether to renormalize the logits after temperature and top_k/top_p filtering.",
    )

    # TODO(This needs to be defined based on the Constraint class)
    # constraints:

    forced_bos_token_id: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The id of the token to force as the first generated token after the decoder_start_token_id."
        "Useful for multilingual models like mBART where the first generated token needs to be the target language"
        "token.",
    )

    forced_eos_token_id: Optional[Union[int, list[int]]] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The id of the token to force as the last generated token when max_length is reached. Optionally, "
        "use a list to set multiple end-of-sequence tokens.",
    )

    remove_invalid_values: Optional[bool] = schema_utils.Boolean(
        default=None,
        description="Whether to remove possible nan and inf outputs of the model to prevent the generation method to "
        "crash. Note that using remove_invalid_values can slow down generation.",
    )

    exponential_decay_length_penalty: Optional[tuple[int, float]] = schema_utils.FloatRange(
        default=None,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have "
        "been generated. The tuple shall consist of: (start_index, decay_factor) where start_index indicates where "
        "penalty starts and decay_factor represents the factor of exponential decay",
    )

    suppress_tokens: Optional[list[int]] = schema_utils.List(
        list_type=int,
        default=None,
        allow_none=True,
        description="A list of tokens that will be suppressed at generation. The SupressTokens logit processor will set"
        " their log probs to -inf so that they are not sampled.",
    )

    begin_suppress_tokens: Optional[list[int]] = schema_utils.List(
        list_type=int,
        default=None,
        allow_none=True,
        description="A list of tokens that will be suppressed at the beginning of the generation. The "
        "SupressBeginTokens logit processor will set their log probs to -inf so that they are not sampled.",
    )

    forced_decoder_ids: Optional[list[list[int]]] = schema_utils.List(
        default=None,
        allow_none=True,
        description="A list of forced decoder ids. The ForcedDecoderIds logit processor will set the log probs of all "
        "tokens that are not in the list to -inf so that they are not sampled.",
    )

    # Special tokens that can be used at generation time
    pad_token_id: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The id of the padding token.",
    )

    bos_token_id: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The id of the beginning of sentence token.",
    )

    eos_token_id: Optional[Union[int, list[int]]] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The id of the end of sentence token.",
    )


@DeveloperAPI
class LLMGenerationConfigField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(LLMGenerationConfig)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_marshmallow_class(LLMGenerationConfig)
