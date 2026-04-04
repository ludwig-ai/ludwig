"""Pretrained audio encoders using HuggingFace transformers.

These encoders accept raw audio waveforms or preprocessed features and produce fixed-size representations using
pretrained foundation models.
"""

import logging

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUDIO, ENCODER_OUTPUT
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("wav2vec2", [AUDIO])
class Wav2Vec2Encoder(Encoder):
    """Wav2Vec2 audio encoder (Baevski et al., NeurIPS 2020).

    Self-supervised speech representation learning using contrastive learning
    over masked latent representations.

    Use when: speech recognition, audio classification, speaker identification.
    Best for: English speech tasks with the base model, multilingual with XLSR variants.
    Expects raw waveform input (16kHz sample rate).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "facebook/wav2vec2-base",
        use_pretrained: bool = True,
        trainable: bool = True,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "mean",
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        try:
            from transformers import Wav2Vec2Model
        except ImportError:
            raise RuntimeError("transformers is required for Wav2Vec2Encoder. pip install transformers")

        if use_pretrained and not saved_weights_in_checkpoint:
            self.model = Wav2Vec2Model.from_pretrained(pretrained_model_name_or_path)
        else:
            from transformers import Wav2Vec2Config

            self.model = Wav2Vec2Model(Wav2Vec2Config())

        self._output_dim = self.model.config.hidden_size
        self._reduce_output = reduce_output

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, inputs: torch.Tensor, mask=None) -> EncoderOutputDict:
        # inputs shape: [batch_size, sequence_length] (raw waveform)
        if inputs.dim() == 3:
            inputs = inputs.squeeze(1)  # Remove channel dim if present
        outputs = self.model(inputs)
        hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        if self._reduce_output == "mean":
            hidden = hidden.mean(dim=1)
        elif self._reduce_output == "last":
            hidden = hidden[:, -1, :]
        elif self._reduce_output == "cls_pooled":
            hidden = hidden[:, 0, :]
        # else: return full sequence

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls():
        from ludwig.schema.encoders.audio_encoders import Wav2Vec2EncoderConfig

        return Wav2Vec2EncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        if self._reduce_output is None:
            return torch.Size([None, self._output_dim])  # variable length
        return torch.Size([self._output_dim])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])  # raw waveform, variable length


@DeveloperAPI
@register_encoder("whisper", [AUDIO])
class WhisperEncoder(Encoder):
    """Whisper audio encoder (Radford et al., ICML 2023).

    Robust Speech Recognition via Large-Scale Weak Supervision. Uses the
    encoder portion of the Whisper model to produce audio representations
    from log-mel spectrogram input (80 mel bins).

    Use when: multilingual/noisy audio, automatic speech recognition tasks.
    Best for: robust transcription across languages, accents, and noise conditions.
    Expects log-mel spectrogram input (80 mel bins, 3000 time frames for 30s audio).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai/whisper-base",
        use_pretrained: bool = True,
        trainable: bool = True,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "mean",
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        try:
            from transformers import WhisperModel
        except ImportError:
            raise RuntimeError("transformers is required for WhisperEncoder. pip install transformers")

        if use_pretrained and not saved_weights_in_checkpoint:
            self.model = WhisperModel.from_pretrained(pretrained_model_name_or_path)
        else:
            from transformers import WhisperConfig

            self.model = WhisperModel(WhisperConfig())

        # Use only the encoder portion
        self.encoder = self.model.encoder
        self._output_dim = self.model.config.d_model
        self._reduce_output = reduce_output

        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, inputs: torch.Tensor, mask=None) -> EncoderOutputDict:
        # inputs shape: [batch_size, n_mels, seq_len] (log-mel spectrogram)
        # Whisper encoder expects [batch, n_mels, seq_len]
        if inputs.dim() == 2:
            # If [batch, seq_len], assume single mel bin - unlikely but handle gracefully
            inputs = inputs.unsqueeze(1)
        if inputs.dim() == 3 and inputs.shape[1] != 80:
            # If shape is [batch, seq_len, n_mels], transpose to [batch, n_mels, seq_len]
            if inputs.shape[2] == 80:
                inputs = inputs.transpose(1, 2)

        outputs = self.encoder(inputs)
        hidden = outputs.last_hidden_state  # [batch, seq_len, d_model]

        if self._reduce_output == "mean":
            hidden = hidden.mean(dim=1)
        elif self._reduce_output == "last":
            hidden = hidden[:, -1, :]
        elif self._reduce_output == "cls_pooled":
            hidden = hidden[:, 0, :]
        # else: return full sequence

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls():
        from ludwig.schema.encoders.audio_encoders import WhisperEncoderConfig

        return WhisperEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        if self._reduce_output is None:
            return torch.Size([None, self._output_dim])  # variable length
        return torch.Size([self._output_dim])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([80])  # 80 mel bins


@DeveloperAPI
@register_encoder("hubert", [AUDIO])
class HuBERTEncoder(Encoder):
    """HuBERT audio encoder (Hsu et al., IEEE/ACM TASLP 2021).

    Self-Supervised Speech Representation Learning by Masked Prediction of
    Hidden Units. Uses an offline clustering step to provide aligned target
    labels for a BERT-like prediction loss.

    Use when: speaker verification, emotion recognition, audio classification.
    Best for: tasks requiring robust speech representations without labeled data.
    Expects raw waveform input (16kHz sample rate).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "facebook/hubert-base-ls960",
        use_pretrained: bool = True,
        trainable: bool = True,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "mean",
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        try:
            from transformers import HubertModel
        except ImportError:
            raise RuntimeError("transformers is required for HuBERTEncoder. pip install transformers")

        if use_pretrained and not saved_weights_in_checkpoint:
            self.model = HubertModel.from_pretrained(pretrained_model_name_or_path)
        else:
            from transformers import HubertConfig

            self.model = HubertModel(HubertConfig())

        self._output_dim = self.model.config.hidden_size
        self._reduce_output = reduce_output

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, inputs: torch.Tensor, mask=None) -> EncoderOutputDict:
        # inputs shape: [batch_size, sequence_length] (raw waveform)
        if inputs.dim() == 3:
            inputs = inputs.squeeze(1)  # Remove channel dim if present
        outputs = self.model(inputs)
        hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        if self._reduce_output == "mean":
            hidden = hidden.mean(dim=1)
        elif self._reduce_output == "last":
            hidden = hidden[:, -1, :]
        elif self._reduce_output == "cls_pooled":
            hidden = hidden[:, 0, :]
        # else: return full sequence

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls():
        from ludwig.schema.encoders.audio_encoders import HuBERTEncoderConfig

        return HuBERTEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        if self._reduce_output is None:
            return torch.Size([None, self._output_dim])  # variable length
        return torch.Size([self._output_dim])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])  # raw waveform, variable length
