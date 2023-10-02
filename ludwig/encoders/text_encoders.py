#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING, TypeVar, Union

import numpy as np
import torch
from torch import nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, TEXT
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.sequence_encoders import SequenceEncoderConfig
from ludwig.schema.encoders.text_encoders import (
    ALBERTConfig,
    AutoTransformerConfig,
    BERTConfig,
    CamemBERTConfig,
    CTRLConfig,
    DebertaV2Config,
    DistilBERTConfig,
    ELECTRAConfig,
    FlauBERTConfig,
    GPT2Config,
    GPTConfig,
    LongformerConfig,
    MT5Config,
    RoBERTaConfig,
    T5Config,
    TfIdfEncoderConfig,
    TransformerXLConfig,
    XLMConfig,
    XLMRoBERTaConfig,
    XLNetConfig,
)
from ludwig.schema.llms.peft import BaseAdapterConfig
from ludwig.utils.hf_utils import load_pretrained_hf_model_with_hub_fallback
from ludwig.utils.torch_utils import FreezeModule

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ludwig.schema.encoders.text_encoders import HFEncoderConfig

logger = logging.getLogger(__name__)


def _cls_pooled_error_message(encoder: str):
    # TODO(Arnav): Remove this once we have reduce_output options set for
    # each encoder type in the schema
    raise ValueError(f"reduce_output cannot be cls_pooled for {encoder}")


class HFTextEncoder(Encoder):
    def _init_config(self, transformer, schema_keys: List[str], encoder_config: SequenceEncoderConfig):
        """Creates a config object for the encoder using the transformer model and the passed-in encoder config.

        The transformer's config is only known after it is instantiated, so we must update the
        encoder config with the values from the transformer config.

        Args:
            transformer: The transformer model.
            schema_keys: The keys in the encoder config schema. We only want to update the encoder config
                with the values from the transformer config that are in the schema.
            encoder_config: The existing encoder config containing defaults and user-specified values.
                If the values in this config differ from the transformer's config, the transformer's config
                values will override this config's values.
        Returns:
            A new encoder config object with the updated values from the transformer config.
        """
        transformer_config = transformer.config.to_dict()
        final_hf_config_params = {k: v for k, v in transformer_config.items() if k in schema_keys}
        encoder_config_dict = encoder_config.to_dict()
        encoder_config_dict.update(final_hf_config_params)
        return self.get_schema_cls().from_dict(encoder_config_dict)

    def _init_transformer_from_scratch(
        self, hf_model_cls: Type, hf_config_cls: Type, hf_config_params: Dict[str, Any], vocab_size: int
    ):
        """Initializes the transformer model from scratch. This is in contrast to loading a pre-trained model.

        Args:
            hf_model_cls: The HuggingFace model class.
            hf_config_cls: The HuggingFace config class.
            hf_config_params: The HuggingFace config parameters exposed through the Ludwig schema.
            vocab_size: The vocab size of the dataset. Because we are training from scratch, we can resize the
                token embeddings table freely.
        Returns:
            The transformer model.
        """
        config = hf_config_cls(**hf_config_params)
        transformer = hf_model_cls(config)
        self._maybe_resize_token_embeddings(transformer, vocab_size)
        return transformer

    def _maybe_resize_token_embeddings(self, transformer, vocab_size: int) -> None:
        """Resizes the token embeddings if the vocab size is different from the transformer's vocab size.

        This should only happen if we are instantiating a model from scratch (i.e. not loading from a pretrained model
        or checkpoint). Pretrained models update the vocab size stored in the config. This means if we are loading a
        pretrained model from a checkpoint, the config vocab size should match the model's vocab size.

        It is important that pretrained models update the vocab size stored in the config because sometimes the
        pretrained models will have an embeddings table that is a different size than the vocab size. Examples:

        CamemBERT:  https://github.com/huggingface/tokenizers/issues/900#issue-1122256698
        T5:         https://github.com/huggingface/transformers/issues/4875#issue-635471552

        Args:
            transformer: The transformer model.
            vocab_size: The vocab size of the dataset.
        """
        if vocab_size != transformer.config.vocab_size:
            transformer.resize_token_embeddings(vocab_size)

    def _wrap_transformer(
        self, transformer: nn.Module, adapter: Optional[BaseAdapterConfig], trainable: bool
    ) -> nn.Module:
        if adapter is not None:
            from peft import get_peft_model

            peft_config = adapter.to_config()
            transformer = get_peft_model(transformer, peft_config)

            logger.info("==================================================")
            logger.info("Trainable Parameter Summary For Fine-Tuning:")
            transformer.print_trainable_parameters()
            logger.info("==================================================")
        return FreezeModule(transformer, frozen=not trainable)

    def get_embedding_layer(self) -> nn.Module:
        return next(self.transformer.module.children())


HFModelT = TypeVar("HFModelT", bound="PreTrainedModel")
HFConfigT = TypeVar("HFConfigT", bound="PretrainedConfig")
ConfigT = TypeVar("ConfigT", bound="HFEncoderConfig")


class HFTextEncoderImpl(HFTextEncoder):
    def __init__(
        self,
        model_cls: Type[HFModelT],
        config_cls: Type[HFConfigT],
        schema_cls: Type[ConfigT],
        max_sequence_length: int,
        use_pretrained: bool,
        pretrained_model_name_or_path: str,
        saved_weights_in_checkpoint: bool,
        reduce_output: str,
        trainable: bool,
        adapter: Optional[BaseAdapterConfig],
        pretrained_kwargs: Dict,
        encoder_config: Optional[ConfigT],
        **kwargs,
    ):
        super().__init__()

        # TODO(travis): get_hf_config_param_names should be implemented as abstract in HFEncoderConfig
        vocab_size = kwargs["vocab_size"]
        hf_config_params = {k: v for k, v in kwargs.items() if k in schema_cls.get_hf_config_param_names()}
        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                model_cls, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(model_cls, config_cls, hf_config_params, vocab_size)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs["pooler_output"]
        else:
            hidden = transformer_outputs["last_hidden_state"][:, 1:-1, :]  # bos + [sent] + sep
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length - 2, self.transformer.module.config.hidden_size])
        if self.reduce_output == "concat":
            return torch.Size(
                [
                    (self.max_sequence_length - 2) * self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("albert", TEXT)
class ALBERTEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "albert-base-v2"

    def __init__(
        self,
        max_sequence_length,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        reduce_output: str = "cls_pooled",
        vocab_size: int = 30000,
        embedding_size: int = 128,
        hidden_size: int = 4096,
        num_hidden_layers: int = 12,
        num_hidden_groups: int = 1,
        num_attention_heads: int = 64,
        intermediate_size: int = 16384,
        inner_group_num: int = 1,
        hidden_act: str = "gelu_new",
        hidden_dropout_prob: float = 0,
        attention_probs_dropout_prob: float = 0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        classifier_dropout_prob: float = 0.1,
        position_embedding_type: str = "absolute",
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import AlbertConfig, AlbertModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_hidden_groups=num_hidden_groups,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            inner_group_num=inner_group_num,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            classifier_dropout_prob=classifier_dropout_prob,
            position_embedding_type=position_embedding_type,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                AlbertModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(AlbertModel, AlbertConfig, hf_config_params, vocab_size)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return ALBERTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by BERT tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 2,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("mt5", TEXT)
class MT5Encoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "google/mt5-base"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        reduce_output: str = "sum",
        vocab_size: int = 250112,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 1024,
        num_layers: int = 8,
        num_decoder_layers: int = None,
        num_heads: int = 6,
        relative_attention_num_buckets: int = 32,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-06,
        initializer_factor: float = 1.0,
        feed_forward_proj: str = "gated-gelu",
        is_encoder_decoder: bool = True,
        use_cache: bool = True,
        tokenizer_class: str = "T5Tokenizer",
        tie_word_embeddings: bool = False,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        decoder_start_token_id: int = 0,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import MT5Config, MT5EncoderModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_factor=initializer_factor,
            feed_forward_proj=feed_forward_proj,
            is_encoder_decoder=is_encoder_decoder,
            use_cache=use_cache,
            tokenizer_class=tokenizer_class,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                MT5EncoderModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(MT5EncoderModel, MT5Config, hf_config_params, vocab_size)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.reduce_output = reduce_output
        if reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
        )
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return MT5Config

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by MT5 tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 2,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("xlmroberta", TEXT)
class XLMRoBERTaEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "xlm-roberta-base"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "cls_pooled",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = None,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        max_position_embeddings: int = 514,
        type_vocab_size: int = 1,
        add_pooling_layer: bool = True,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import XLMRobertaConfig, XLMRobertaModel

        hf_config_params = dict(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                XLMRobertaModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(
                XLMRobertaModel, XLMRobertaConfig, hf_config_params, vocab_size
            )

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return XLMRoBERTaConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by XLMRoberta tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 2,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("bert", TEXT)
class BERTEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "bert-base-uncased"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        reduce_output: str = "cls_pooled",
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: Union[str, Callable] = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        gradient_checkpointing: bool = False,
        position_embedding_type: str = "absolute",
        classifier_dropout: float = None,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import BertConfig, BertModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            position_embedding_type=position_embedding_type,
            classifier_dropout=classifier_dropout,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                BertModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(BertModel, BertConfig, hf_config_params, vocab_size)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)

        self.transformer = self._wrap_transformer(transformer, adapter, trainable)

        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return BERTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    # TODO(shreya): Confirm that this is it
    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by BERT tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 2,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("xlm", TEXT)
class XLMEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "xlm-mlm-en-2048"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        reduce_output: str = "sum",
        vocab_size: int = 30145,
        emb_dim: int = 2048,
        n_layers: int = 12,
        n_heads: int = 16,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        gelu_activation: bool = True,
        sinusoidal_embeddings: bool = False,
        causal: bool = False,
        asm: bool = False,
        n_langs: int = 1,
        use_lang_emb: bool = True,
        max_position_embeddings: int = 512,
        embed_init_std: float = 2048**-0.5,
        layer_norm_eps: float = 1e-12,
        init_std: float = 0.02,
        bos_index: int = 0,
        eos_index: int = 1,
        pad_index: int = 2,
        unk_index: int = 3,
        mask_index: int = 5,
        is_encoder: bool = True,
        start_n_top: int = 5,
        end_n_top: int = 5,
        mask_token_id: int = 0,
        lang_id: int = 0,
        pad_token_id: int = 2,
        bos_token_id: int = 0,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import XLMConfig, XLMModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            gelu_activation=gelu_activation,
            sinusoidal_embeddings=sinusoidal_embeddings,
            causal=causal,
            asm=asm,
            n_langs=n_langs,
            use_lang_emb=use_lang_emb,
            max_position_embeddings=max_position_embeddings,
            embed_init_std=embed_init_std,
            layer_norm_eps=layer_norm_eps,
            init_std=init_std,
            bos_index=bos_index,
            eos_index=eos_index,
            pad_index=pad_index,
            unk_index=unk_index,
            mask_index=mask_index,
            is_encoder=is_encoder,
            start_n_top=start_n_top,
            end_n_top=end_n_top,
            mask_token_id=mask_token_id,
            lang_id=lang_id,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                XLMModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(XLMModel, XLMConfig, hf_config_params, vocab_size)

        self.config = self._init_config(transformer, hf_config_params, encoder_config)

        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return XLMConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    # TODO(shreya): Confirm that this is it
    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by BERT tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 2,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("gpt", TEXT)
class GPTEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "openai-gpt"

    def __init__(
        self,
        max_sequence_length: int,
        reduce_output: str = "sum",
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 30522,
        n_positions: int = 40478,
        n_ctx: int = 512,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        afn: str = "gelu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import OpenAIGPTConfig, OpenAIGPTModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            afn=afn,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                OpenAIGPTModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(
                OpenAIGPTModel, OpenAIGPTConfig, hf_config_params, vocab_size
            )

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return GPTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.module.config.hidden_size])
        elif self.reduce_output == "concat":
            return torch.Size([self.transformer.module.config.hidden_size * self.max_sequence_length])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("gpt2", TEXT)
class GPT2Encoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "gpt2"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        reduce_output: str = "sum",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_ctx: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import GPT2Config, GPT2Model

        hf_config_params = dict(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
        )

        if use_pretrained:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                GPT2Model, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(GPT2Model, GPT2Config, hf_config_params, vocab_size)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return GPT2Config

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.module.config.hidden_size])
        elif self.reduce_output == "concat":
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("deberta", TEXT)
class DeBERTaEncoder(HFTextEncoderImpl):
    def __init__(self, *args, **kwargs):
        from transformers import DebertaV2Config as _DebertaV2Config
        from transformers import DebertaV2Model

        super().__init__(DebertaV2Model, _DebertaV2Config, DebertaV2Config, *args, **kwargs)

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return DebertaV2Config


@DeveloperAPI
@register_encoder("roberta", TEXT)
class RoBERTaEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "roberta-base"

    def __init__(
        self,
        max_sequence_length,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "cls_pooled",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = None,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        max_position_embeddings: int = 514,
        type_vocab_size: int = 1,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import RobertaConfig, RobertaModel

        hf_config_params = dict(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                RobertaModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(RobertaModel, RobertaConfig, hf_config_params, vocab_size)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]  # bos + [sent] + sep
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return RoBERTaConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length - 2, self.transformer.module.config.hidden_size])
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("transformer_xl", TEXT)
class TransformerXLEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "transfo-xl-wt103"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 267735,
        cutoffs: List[int] = [20000, 40000, 200000],
        d_model: int = 1024,
        d_embed: int = 1024,
        n_head: int = 16,
        d_head: int = 64,
        d_inner: int = 4096,
        div_val: int = 4,
        pre_lnorm: bool = False,
        n_layer: int = 18,
        mem_len: int = 1600,
        clamp_len: int = 1000,
        same_length: bool = True,
        proj_share_all_but_first: bool = True,
        attn_type: int = 0,
        sample_softmax: int = -1,
        adaptive: bool = True,
        dropout: float = 0.1,
        dropatt: float = 0.0,
        untie_r: bool = True,
        init: str = "normal",
        init_range: float = 0.01,
        proj_init_std: float = 0.01,
        init_std: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        eos_token_id: int = 0,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import TransfoXLConfig, TransfoXLModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            cutoffs=cutoffs,
            d_model=d_model,
            d_embed=d_embed,
            n_head=n_head,
            d_head=d_head,
            d_inner=d_inner,
            div_val=div_val,
            pre_lnorm=pre_lnorm,
            n_layer=n_layer,
            mem_len=mem_len,
            clamp_len=clamp_len,
            same_length=same_length,
            proj_share_all_but_first=proj_share_all_but_first,
            attn_type=attn_type,
            sample_softmax=sample_softmax,
            adaptive=adaptive,
            dropout=dropout,
            dropatt=dropatt,
            untie_r=untie_r,
            init=init,
            init_range=init_range,
            proj_init_std=proj_init_std,
            init_std=init_std,
            layer_norm_epsilon=layer_norm_epsilon,
            eos_token_id=eos_token_id,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                TransfoXLModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = TransfoXLConfig(**hf_config_params)
            transformer = TransfoXLModel(config)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        transformer_outputs = self.transformer.module(inputs)
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TransformerXLConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.module.config.d_model])
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.d_model * self.max_sequence_length])
        return torch.Size([self.transformer.module.config.d_model])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("xlnet", TEXT)
class XLNetEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "xlnet-base-cased"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 32000,
        d_model: int = 1024,
        n_layer: int = 24,
        n_head: int = 16,
        d_inner: int = 4096,
        ff_activation: str = "gelu",
        untie_r: bool = True,
        attn_type: str = "bi",
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.1,
        mem_len: Optional[int] = 512,
        reuse_len: Optional[int] = None,
        use_mems_eval: bool = True,
        use_mems_train: bool = False,
        bi_data: bool = False,
        clamp_len: int = -1,
        same_length: bool = False,
        summary_type: str = "last",
        summary_use_proj: bool = True,
        summary_activation: str = "tanh",
        summary_last_dropout: float = 0.1,
        start_n_top: int = 5,
        end_n_top: int = 5,
        pad_token_id: int = 5,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import XLNetConfig, XLNetModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            d_inner=d_inner,
            ff_activation=ff_activation,
            untie_r=untie_r,
            attn_type=attn_type,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            mem_len=mem_len,
            reuse_len=reuse_len,
            use_mems_eval=use_mems_eval,
            use_mems_train=use_mems_train,
            bi_data=bi_data,
            clamp_len=clamp_len,
            same_length=same_length,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_activation=summary_activation,
            summary_last_dropout=summary_last_dropout,
            start_n_top=start_n_top,
            end_n_top=end_n_top,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                XLNetModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(XLNetModel, XLNetConfig, hf_config_params, vocab_size)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return XLNetConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.module.config.d_model])
        elif self.reduce_output == "concat":
            return torch.Size([self.transformer.module.config.d_model * self.max_sequence_length])
        return torch.Size([self.transformer.module.config.d_model])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("distilbert", TEXT)
class DistilBERTEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "distilbert-base-uncased"

    def __init__(
        self,
        max_sequence_length: int,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        use_pretrained: bool = True,
        vocab_size: int = 30522,
        max_position_embeddings: int = 512,
        sinusoidal_pos_embds: bool = False,
        n_layers: int = 6,
        n_heads: int = 12,
        dim: int = 768,
        hidden_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: Union[str, Callable] = "gelu",
        initializer_range: float = 0.02,
        qa_dropout: float = 0.1,
        seq_classif_dropout: float = 0.2,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import DistilBertConfig, DistilBertModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            sinusoidal_pos_embds=sinusoidal_pos_embds,
            n_layers=n_layers,
            n_heads=n_heads,
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            initializer_range=initializer_range,
            qa_dropout=qa_dropout,
            seq_classif_dropout=seq_classif_dropout,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                DistilBertModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(
                DistilBertModel, DistilBertConfig, hf_config_params, vocab_size
            )

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.max_sequence_length = max_sequence_length
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.last_inputs = None
        self.last_hidden = None

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)

        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
        )
        hidden = transformer_outputs[0][:, 1:-1, :]
        self.last_inputs = inputs
        self.last_hidden = hidden
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return DistilBERTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by BERT tokenizer.
            return torch.Size([self.max_sequence_length - 2, self.transformer.module.config.dim])
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.dim * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.dim])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("ctrl", TEXT)
class CTRLEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "ctrl"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 246534,
        n_positions: int = 256,
        n_ctx: int = 256,
        n_embd: int = 1280,
        dff: int = 8192,
        n_layer: int = 48,
        n_head: int = 16,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import CTRLConfig, CTRLModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_ctx,
            n_embd=n_embd,
            dff=dff,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                CTRLModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
            self.vocab_size = transformer.config.vocab_size
        else:
            transformer = self._init_transformer_from_scratch(CTRLModel, CTRLConfig, hf_config_params, vocab_size)
            self.vocab_size = vocab_size

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.max_sequence_length = max_sequence_length
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls():
        return CTRLConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.module.config.n_embd])
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.n_embd * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.n_embd])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("camembert", TEXT)
class CamemBERTEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "camembert-base"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "cls-pooled",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: Union[str, Callable] = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        gradient_checkpointing: bool = False,
        position_embedding_type: str = "absolute",
        classifier_dropout: float = None,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import CamembertConfig, CamembertModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            position_embedding_type=position_embedding_type,
            classifier_dropout=classifier_dropout,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                CamembertModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(
                CamembertModel, CamembertConfig, hf_config_params, vocab_size
            )

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return CamemBERTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by BERT tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 2,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("t5", TEXT)
class T5Encoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "t5-small"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 32128,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_decoder_layers: Optional[int] = None,
        num_heads: int = 8,
        relative_attention_num_buckets: int = 32,
        dropout_rate: float = 0.1,
        layer_norm_eps: float = 1e-6,
        initializer_factor: float = 1,
        feed_forward_proj: str = "relu",
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import T5Config, T5Model

        hf_config_params = dict(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            dropout_rate=dropout_rate,
            layer_norm_eps=layer_norm_eps,
            initializer_factor=initializer_factor,
            feed_forward_proj=feed_forward_proj,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                T5Model, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(T5Model, T5Config, hf_config_params, vocab_size)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            inputs,
            decoder_input_ids=inputs,
            attention_mask=mask,
        )
        hidden = transformer_outputs[0][:, 0:-1, :]  # [eos token]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return T5Config

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 1 to remove EOS token added by T5 tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 1,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -1 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 1)])
        return torch.Size([self.transformer.module.config.d_model])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("flaubert", TEXT)
class FlauBERTEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "flaubert/flaubert_small_cased"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 30145,
        pre_norm: bool = False,
        layerdrop: float = 0.0,
        emb_dim: int = 2048,
        n_layers: int = 12,
        n_heads: int = 16,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        gelu_activation: bool = True,
        sinusoidal_embeddings: bool = False,
        causal: bool = False,
        asm: bool = False,
        n_langs: int = 1,
        use_lang_emb: bool = True,
        max_position_embeddings: int = 512,
        embed_init_std: float = 2048**-0.5,
        init_std: int = 0.02,
        layer_norm_eps: float = 1e-12,
        bos_index: int = 0,
        eos_index: int = 1,
        pad_index: int = 2,
        unk_index: int = 3,
        mask_index: int = 5,
        is_encoder: bool = True,
        mask_token_id: int = 0,
        lang_id: int = 1,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import FlaubertConfig, FlaubertModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            pre_norm=pre_norm,
            layerdrop=layerdrop,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            attention_dropout=dropout,
            gelu_activation=gelu_activation,
            sinusoidal_embeddings=sinusoidal_embeddings,
            causal=causal,
            asm=asm,
            n_langs=n_langs,
            use_lang_emb=use_lang_emb,
            max_position_embeddings=max_position_embeddings,
            embed_init_std=embed_init_std,
            init_std=init_std,
            layer_norm_eps=layer_norm_eps,
            bos_index=bos_index,
            eos_index=eos_index,
            pad_index=pad_index,
            unk_index=unk_index,
            mask_index=mask_index,
            is_encoder=is_encoder,
            mask_token_id=mask_token_id,
            lang_id=lang_id,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                FlaubertModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(
                FlaubertModel, FlaubertConfig, hf_config_params, vocab_size
            )

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return FlauBERTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 2,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.emb_dim])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("electra", TEXT)
class ELECTRAEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "google/electra-small-discriminator"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 30522,
        embedding_size: int = 128,
        hidden_size: int = 256,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        hidden_act: Union[str, Callable] = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: str = "absolute",
        classifier_dropout: Optional[float] = None,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import ElectraConfig, ElectraModel

        hf_config_params = dict(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            position_embedding_type=position_embedding_type,
            classifier_dropout=classifier_dropout,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                ElectraModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(ElectraModel, ElectraConfig, hf_config_params, vocab_size)

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        if self.reduce_output == "cls_pooled":
            _cls_pooled_error_message(self.__class__.__name__)
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return ELECTRAConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 2,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("longformer", TEXT)
class LongformerEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = "allenai/longformer-base-4096"

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        attention_window: Union[List[int], int] = 512,
        sep_token_id: int = 2,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        saved_weights_in_checkpoint: bool = False,
        reduce_output: Optional[str] = "cls_pooled",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: int = 50265,
        num_tokens: Optional[int] = None,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import LongformerConfig, LongformerModel

        hf_config_params = dict(
            attention_window=attention_window,
            sep_token_id=sep_token_id,
            vocab_size=vocab_size,
            **kwargs,
        )

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            transformer, _ = load_pretrained_hf_model_with_hub_fallback(
                LongformerModel, pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            transformer = self._init_transformer_from_scratch(
                LongformerModel, LongformerConfig, hf_config_params, vocab_size
            )

        if encoder_config is not None:
            self.config = self._init_config(transformer, hf_config_params.keys(), encoder_config)
        else:
            self.config = None

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer.module(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]  # bos + [sent] + sep
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return LongformerConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by Longformer (== Roberta) tokenizer.
            return torch.Size(
                [
                    self.max_sequence_length - 2,
                    self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("auto_transformer", TEXT)
class AutoTransformerEncoder(HFTextEncoder):
    DEFAULT_MODEL_NAME = None

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_sequence_length: int,
        reduce_output: str = "sum",
        trainable: bool = False,
        adapter: Optional[BaseAdapterConfig] = None,
        vocab_size: Optional[int] = None,
        pretrained_kwargs: Dict = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()

        from transformers import AutoModel

        pretrained_kwargs = pretrained_kwargs or {}
        transformer, _ = load_pretrained_hf_model_with_hub_fallback(
            AutoModel, pretrained_model_name_or_path, **pretrained_kwargs
        )
        self._maybe_resize_token_embeddings(transformer, vocab_size)

        self.config = self._init_config(transformer, [], encoder_config)

        # Precompute the set of params that are included in the forward signature of the AutoModel implementation so
        # we can filter out unused params during the `forward` call.
        self.forward_kwargs = set(inspect.signature(transformer.forward).parameters.keys())

        self.transformer = self._wrap_transformer(transformer, adapter, trainable)
        self.reduce_output = reduce_output
        if self.reduce_output != "cls_pooled":
            self.reduce_sequence = SequenceReducer(
                reduce_mode=reduce_output, encoding_size=self.transformer.module.config.hidden_size
            )
        self.max_sequence_length = max_sequence_length

    def _maybe_resize_token_embeddings(self, transformer, vocab_size: Optional[int] = None):
        """Overridden because AutoModel should use its own vocab size unless vocab size is explicitly specified."""
        if vocab_size is not None:
            transformer.resize_token_embeddings(vocab_size)
            self.vocab_size = vocab_size
        else:
            self.vocab_size = transformer.config.vocab_size

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        if mask is not None:
            mask = mask.to(torch.int32)

        # The forward signature of AutoModel is not consistent across implementations, so we need to make sure we're
        # only passing in params included in the forward signature.
        kwargs = dict(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        kwargs = {k: v for k, v in kwargs.items() if k in self.forward_kwargs}

        transformer_outputs = self.transformer.module(**kwargs)
        if self.reduce_output == "cls_pooled":
            # this works only if the user know that the specific model
            # they want to use has the same outputs of
            # the BERT base class call() function
            hidden = transformer_outputs["pooler_output"]
        else:
            hidden = transformer_outputs["last_hidden_state"]
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return AutoTransformerConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # TODO(justin): This may need to be conditioned on which AutoModel gets chosen.
            return torch.Size([self.max_sequence_length, self.transformer.module.config.hidden_size])
        if self.reduce_output == "concat":
            return torch.Size(
                [
                    self.max_sequence_length * self.transformer.module.config.hidden_size,
                ]
            )
        elif self.reduce_output == "concat":
            # add the -2 to account of start and end tokens.
            return torch.Size([self.transformer.module.config.hidden_size * (self.max_sequence_length - 2)])
        return torch.Size([self.transformer.module.config.hidden_size])

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int32


@DeveloperAPI
@register_encoder("tf_idf", [TEXT])
class TfIdfEncoder(Encoder):
    def __init__(
        self,
        max_sequence_length: int,
        encoder_config=None,
        str2idf=None,
        vocab=None,
        vocab_size: int = None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size

        logger.debug(f" {self.name}")

        # Convert mapping of token -> frequency to a dense array
        idf = np.zeros(vocab_size)
        for i, s in enumerate(vocab):
            idf[i] = str2idf[s]
        self.idf = torch.from_numpy(idf).float().unsqueeze(0)

    def forward(self, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        # Compute the term frequency within each row
        tf = torch.stack([t_i.bincount(minlength=self.vocab_size) for t_i in torch.unbind(t.long())])

        # Normalize the term frequency by the number of tokens in each row
        tf = tf / tf.sum(dim=1).unsqueeze(-1)

        # Multiply the term frequency by the inverse document frequency
        tfidf = tf * self.idf

        return {ENCODER_OUTPUT: tfidf}

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TfIdfEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.vocab_size])

    def get_embedding_layer(self) -> nn.Module:
        return self
