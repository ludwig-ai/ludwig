from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUDIO
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config


class PretrainedAudioEncoderConfig(BaseEncoderConfig):
    """Base config for pretrained audio encoders."""

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Download model weights from pre-trained model.",
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to load weights from the saved checkpoint.",
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Is the encoder trainable.",
    )

    reduce_output: str | None = schema_utils.StringOptions(
        ["mean", "last", "cls_pooled"],
        default="mean",
        allow_none=True,
        description=(
            "How to reduce the output tensor along the time/sequence dimension. "
            "'mean' averages over all time steps, 'last' takes the last time step, "
            "'cls_pooled' takes the first time step (CLS token position), "
            "None returns the full sequence."
        ),
    )

    def is_pretrained(self) -> bool:
        return self.use_pretrained


@DeveloperAPI
@register_encoder_config("wav2vec2", AUDIO)
class Wav2Vec2EncoderConfig(PretrainedAudioEncoderConfig):
    """Config for the Wav2Vec2 pretrained audio encoder."""

    @staticmethod
    def module_name():
        return "Wav2Vec2Encoder"

    type: str = schema_utils.ProtectedString("wav2vec2", description="Type of encoder.")

    pretrained_model_name_or_path: str = schema_utils.String(
        default="facebook/wav2vec2-base",
        allow_none=False,
        description=(
            "Name or path of the pretrained model. Can be a HuggingFace model hub identifier "
            "(e.g. 'facebook/wav2vec2-base', 'facebook/wav2vec2-large-xlsr-53') "
            "or a local path to a saved model directory."
        ),
    )


@DeveloperAPI
@register_encoder_config("whisper", AUDIO)
class WhisperEncoderConfig(PretrainedAudioEncoderConfig):
    """Config for the Whisper pretrained audio encoder."""

    @staticmethod
    def module_name():
        return "WhisperEncoder"

    type: str = schema_utils.ProtectedString("whisper", description="Type of encoder.")

    pretrained_model_name_or_path: str = schema_utils.String(
        default="openai/whisper-base",
        allow_none=False,
        description=(
            "Name or path of the pretrained model. Can be a HuggingFace model hub identifier "
            "(e.g. 'openai/whisper-base', 'openai/whisper-small', 'openai/whisper-medium', "
            "'openai/whisper-large-v3') or a local path to a saved model directory."
        ),
    )


@DeveloperAPI
@register_encoder_config("hubert", AUDIO)
class HuBERTEncoderConfig(PretrainedAudioEncoderConfig):
    """Config for the HuBERT pretrained audio encoder."""

    @staticmethod
    def module_name():
        return "HuBERTEncoder"

    type: str = schema_utils.ProtectedString("hubert", description="Type of encoder.")

    pretrained_model_name_or_path: str = schema_utils.String(
        default="facebook/hubert-base-ls960",
        allow_none=False,
        description=(
            "Name or path of the pretrained model. Can be a HuggingFace model hub identifier "
            "(e.g. 'facebook/hubert-base-ls960', 'facebook/hubert-large-ls960-ft') "
            "or a local path to a saved model directory."
        ),
    )
