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
import logging
import sys
from typing import Dict, Optional

import torch

from ludwig.constants import TEXT
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.encoders.text_encoders import (
    ALBERTConfig,
    AutoTransformerConfig,
    BERTConfig,
    CamemBERTConfig,
    CTRLConfig,
    DistilBERTConfig,
    ELECTRAConfig,
    FlauBERTConfig,
    GPT2Config,
    GPTConfig,
    LongformerConfig,
    MT5Config,
    RoBERTaConfig,
    T5Config,
    TransformerXLConfig,
    XLMConfig,
    XLMRoBERTaConfig,
    XLNetConfig,
)
from ludwig.utils.pytorch_utils import freeze_parameters

logger = logging.getLogger(__name__)


@register_encoder("albert", TEXT)
class ALBERTEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "albert-base-v2",
    }

    def __init__(self, encoder_config: ALBERTConfig = ALBERTConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import AlbertConfig, AlbertModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = AlbertModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = AlbertConfig(
                vocab_size=encoder_config.vocab_size,
                embedding_size=encoder_config.embedding_size,
                hidden_size=encoder_config.hidden_size,
                num_hidden_layers=encoder_config.num_hidden_layers,
                num_hidden_groups=encoder_config.num_hidden_groups,
                num_attention_heads=encoder_config.num_attention_heads,
                intermediate_size=encoder_config.intermediate_size,
                inner_group_num=encoder_config.inner_group_num,
                hidden_act=encoder_config.hidden_act,
                hidden_dropout_prob=encoder_config.hidden_dropout_prob,
                attention_probs_dropout_prob=encoder_config.attention_probs_dropout_prob,
                max_position_embeddings=encoder_config.max_position_embeddings,
                type_vocab_size=encoder_config.type_vocab_size,
                initializer_range=encoder_config.initializer_range,
                layer_norm_eps=encoder_config.layer_norm_eps,
                classifier_dropout_prob=encoder_config.classifier_dropout_prob,
                position_embedding_type=encoder_config.position_embedding_type,
                pad_token_id=encoder_config.pad_token_id,
                bos_token_id=encoder_config.bos_token_id,
                eos_token_id=encoder_config.eos_token_id,
            )
            self.transformer = AlbertModel(config)

        self.reduce_output = encoder_config.reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("mt5", TEXT)
class MT5Encoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "google/mt5-base",
    }

    def __init__(self, encoder_config: MT5Config = MT5Config()):
        super().__init__(encoder_config)
        try:
            from transformers import MT5Config, MT5EncoderModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = MT5EncoderModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = MT5Config(
                vocab_size=encoder_config.vocab_size,
                d_model=encoder_config.d_model,
                d_kv=encoder_config.d_kv,
                d_ff=encoder_config.d_ff,
                num_layers=encoder_config.num_layers,
                num_decoder_layers=encoder_config.num_decoder_layers,
                num_heads=encoder_config.num_heads,
                relative_attention_num_buckets=encoder_config.relative_attention_num_buckets,
                dropout_rate=encoder_config.dropout_rate,
                layer_norm_epsilon=encoder_config.layer_norm_epsilon,
                initializer_factor=encoder_config.initializer_factor,
                feed_forward_proj=encoder_config.feed_forward_proj,
                is_encoder_decoder=encoder_config.is_encoder_decoder,
                use_cache=encoder_config.use_cache,
                tokenizer_class=encoder_config.tokenizer_class,
                tie_word_embeddings=encoder_config.tie_word_embeddings,
                pad_token_id=encoder_config.pad_token_id,
                eos_token_id=encoder_config.eos_token_id,
                decoder_start_token_id=encoder_config.decoder_start_token_id,
            )
            self.transformer = MT5EncoderModel(config)

        self.reduce_output = encoder_config.reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("xlmroberta", TEXT)
class XLMRoBERTaEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "xlm-roberta-base",
    }

    def __init__(self, encoder_config: XLMRoBERTaConfig = XLMRoBERTaConfig()):
        super().__init__(encoder_config)

        try:
            from transformers import XLMRobertaConfig, XLMRobertaModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = XLMRobertaModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = XLMRobertaConfig(
                pad_token_id=encoder_config.pad_token_id,
                bos_token_id=encoder_config.bos_token_id,
                eos_token_id=encoder_config.eos_token_id,
            )

            self.transformer = XLMRobertaModel(config, encoder_config.add_pooling_layer)

        self.reduce_output = encoder_config.reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("bert", TEXT)
class BERTEncoder(Encoder):
    # TODO(justin): Use official class properties.
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "bert-base-uncased",
    }

    def __init__(self, encoder_config: BERTConfig = BERTConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import BertConfig, BertModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = BertModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = BertConfig(
                vocab_size=encoder_config.vocab_size,
                hidden_size=encoder_config.hidden_size,
                num_hidden_layers=encoder_config.num_hidden_layers,
                num_attention_heads=encoder_config.num_attention_heads,
                intermediate_size=encoder_config.intermediate_size,
                hidden_act=encoder_config.hidden_act,
                hidden_dropout_prob=encoder_config.hidden_dropout_prob,
                attention_probs_dropout_prob=encoder_config.attention_probs_dropout_prob,
                max_position_embeddings=encoder_config.max_position_embeddings,
                type_vocab_size=encoder_config.type_vocab_size,
                initializer_range=encoder_config.initializer_range,
                layer_norm_eps=encoder_config.layer_norm_eps,
                pad_token_id=encoder_config.pad_token_id,
                gradient_checkpointing=encoder_config.gradient_checkpointing,
                position_embedding_type=encoder_config.position_embedding_type,
                classifier_dropout=encoder_config.classifier_dropout,
            )
            self.transformer = BertModel(config)

        self.reduce_output = encoder_config.reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)

        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)

        self.transformer.resize_token_embeddings(encoder_config.vocab_size)
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("xlm", TEXT)
class XLMEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "xlm-mlm-en-2048",
    }

    def __init__(self, encoder_config: XLMConfig = XLMConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import XLMConfig, XLMModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = XLMModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
            if encoder_config.trainable:
                self.transformer.train()
        else:
            config = XLMConfig(
                vocab_size=encoder_config.vocab_size,
                emb_dim=encoder_config.emb_dim,
                n_layers=encoder_config.n_layers,
                n_heads=encoder_config.n_heads,
                dropout=encoder_config.dropout,
                attention_dropout=encoder_config.attention_dropout,
                gelu_activation=encoder_config.gelu_activation,
                sinusoidal_embeddings=encoder_config.sinusoidal_embeddings,
                causal=encoder_config.causal,
                asm=encoder_config.asm,
                n_langs=encoder_config.n_langs,
                use_lang_emb=encoder_config.use_lang_emb,
                max_position_embeddings=encoder_config.max_position_embeddings,
                embed_init_std=encoder_config.embed_init_std,
                layer_norm_eps=encoder_config.layer_norm_eps,
                init_std=encoder_config.init_std,
                bos_index=encoder_config.bos_index,
                eos_index=encoder_config.eos_index,
                pad_index=encoder_config.pad_index,
                unk_index=encoder_config.unk_index,
                mask_index=encoder_config.mask_index,
                is_encoder=encoder_config.is_encoder,
                start_n_top=encoder_config.start_n_top,
                end_n_top=encoder_config.end_n_top,
                mask_token_id=encoder_config.mask_token_id,
                lang_id=encoder_config.lang_id,
                pad_token_id=encoder_config.pad_token_id,
                bos_token_id=encoder_config.bos_token_id,
            )
            self.transformer = XLMModel(config)

        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("gpt", TEXT)
class GPTEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "openai-gpt",
    }

    def __init__(self, encoder_config: GPTConfig = GPTConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import OpenAIGPTConfig, OpenAIGPTModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = OpenAIGPTModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = OpenAIGPTConfig(
                vocab_size=encoder_config.vocab_size,
                n_positions=encoder_config.n_positions,
                n_ctx=encoder_config.n_ctx,
                n_embd=encoder_config.n_embd,
                n_layer=encoder_config.n_layer,
                n_head=encoder_config.n_head,
                afn=encoder_config.afn,
                resid_pdrop=encoder_config.resid_pdrop,
                embd_pdrop=encoder_config.embd_pdrop,
                attn_pdrop=encoder_config.attn_pdrop,
                layer_norm_epsilon=encoder_config.layer_norm_epsilon,
                initializer_range=encoder_config.initializer_range,
            )
            self.transformer = OpenAIGPTModel(config)

        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return GPTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.config.hidden_size])
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("gpt2", TEXT)
class GPT2Encoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "gpt2",
    }

    def __init__(self, encoder_config: GPT2Config = GPT2Config()):
        super().__init__(encoder_config)
        try:
            from transformers import GPT2Config, GPT2Model
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = GPT2Model.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = GPT2Config(
                vocab_size=encoder_config.vocab_size,
                n_positions=encoder_config.n_positions,
                n_ctx=encoder_config.n_ctx,
                n_embd=encoder_config.n_embd,
                n_layer=encoder_config.n_layer,
                n_head=encoder_config.n_head,
                n_inner=encoder_config.n_inner,
                activation_function=encoder_config.activation_function,
                resid_pdrop=encoder_config.resid_pdrop,
                embd_pdrop=encoder_config.embd_pdrop,
                attn_pdrop=encoder_config.attn_pdrop,
                layer_norm_epsilon=encoder_config.layer_norm_epsilon,
                initializer_range=encoder_config.initializer_range,
                scale_attn_weights=encoder_config.scale_attn_weights,
            )
            self.transformer = GPT2Model(config)

        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.max_sequence_length = encoder_config.max_sequence_length
        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return GPT2Config

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.config.hidden_size])
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("roberta", TEXT)
class RoBERTaEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "roberta-base",
    }

    def __init__(self, encoder_config: RoBERTaConfig = RoBERTaConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import RobertaConfig, RobertaModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = RobertaModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = RobertaConfig(
                pad_token_id=encoder_config.pad_token_id,
                bos_token_id=encoder_config.bos_token_id,
                eos_token_id=encoder_config.eos_token_id,
            )
            self.transformer = RobertaModel(config)
        if encoder_config.rainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.reduce_output = encoder_config.reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]  # bos + [sent] + sep
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return RoBERTaConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.config.hidden_size])
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("transformer_xl", TEXT)
class TransformerXLEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "transfo-xl-wt103",
    }

    def __init__(self, encoder_config: TransformerXLConfig = TransformerXLConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import TransfoXLConfig, TransfoXLModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = TransfoXLModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = TransfoXLConfig(
                vocab_size=encoder_config.vocab_size,
                cutoffs=encoder_config.cutoffs,
                d_model=encoder_config.d_model,
                d_embed=encoder_config.d_embed,
                n_head=encoder_config.n_head,
                d_head=encoder_config.d_head,
                d_inner=encoder_config.d_inner,
                div_val=encoder_config.div_val,
                pre_lnorm=encoder_config.pre_lnorm,
                n_layer=encoder_config.n_layer,
                mem_len=encoder_config.mem_len,
                clamp_len=encoder_config.clamp_len,
                same_length=encoder_config.same_length,
                proj_share_all_but_first=encoder_config.proj_share_all_but_first,
                attn_type=encoder_config.attn_type,
                sample_softmax=encoder_config.sample_softmax,
                adaptive=encoder_config.adaptive,
                dropout=encoder_config.dropout,
                dropatt=encoder_config.dropatt,
                untie_r=encoder_config.untie_r,
                init=encoder_config.init,
                init_range=encoder_config.init_range,
                proj_init_std=encoder_config.proj_init_std,
                init_std=encoder_config.init_std,
                layer_norm_epsilon=encoder_config.layer_norm_epsilon,
                eos_token_id=encoder_config.eos_token_id,
            )
            self.transformer = TransfoXLModel(config)
        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        transformer_outputs = self.transformer(inputs)
        hidden = transformer_outputs[0]

        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return TransformerXLConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.config.d_model])
        else:
            return torch.Size([self.transformer.config.d_model])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("xlnet", TEXT)
class XLNetEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "xlnet-base-cased",
    }

    def __init__(self, encoder_config: XLNetConfig = XLNetConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import XLNetConfig, XLNetModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = XLNetModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = XLNetConfig(
                vocab_size=encoder_config.vocab_size,
                d_model=encoder_config.d_model,
                n_layer=encoder_config.n_layer,
                n_head=encoder_config.n_head,
                d_inner=encoder_config.d_inner,
                ff_activation=encoder_config.ff_activation,
                untie_r=encoder_config.untie_r,
                attn_type=encoder_config.attn_type,
                initializer_range=encoder_config.initializer_range,
                layer_norm_eps=encoder_config.layer_norm_eps,
                dropout=encoder_config.dropout,
                mem_len=encoder_config.mem_len,
                reuse_len=encoder_config.reuse_len,
                use_mems_eval=encoder_config.use_mems_eval,
                use_mems_train=encoder_config.use_mems_train,
                bi_data=encoder_config.bi_data,
                clamp_len=encoder_config.clamp_len,
                same_length=encoder_config.same_length,
                summary_type=encoder_config.summary_type,
                summary_use_proj=encoder_config.summary_use_proj,
                summary_activation=encoder_config.summary_activation,
                summary_last_dropout=encoder_config.summary_last_dropout,
                start_n_top=encoder_config.start_n_top,
                end_n_top=encoder_config.end_n_top,
                pad_token_id=encoder_config.pad_token_id,
                bos_token_id=encoder_config.bos_token_id,
                eos_token_id=encoder_config.eos_token_id,
            )
            self.transformer = XLNetModel(config)
        self.max_sequence_length = encoder_config.max_sequence_length
        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return XLNetConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.config.d_model])
        else:
            return torch.Size([self.transformer.config.d_model])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("distilbert", TEXT)
class DistilBERTEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "distilbert-base-uncased",
    }

    def __init__(self, encoder_config: DistilBERTConfig = DistilBERTConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import DistilBertConfig, DistilBertModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = DistilBertModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = DistilBertConfig(
                vocab_size=encoder_config.vocab_size,
                max_position_embeddings=encoder_config.max_position_embeddings,
                sinusoidal_pos_embds=encoder_config.sinusoidal_pos_embds,
                n_layers=encoder_config.n_layers,
                n_heads=encoder_config.n_heads,
                dim=encoder_config.dim,
                hidden_dim=encoder_config.hidden_dim,
                dropout=encoder_config.dropout,
                attention_dropout=encoder_config.attention_dropout,
                activation=encoder_config.activation,
                initializer_range=encoder_config.initializer_range,
                qa_dropout=encoder_config.qa_dropout,
                seq_classif_dropout=encoder_config.seq_classif_dropout,
            )
            self.transformer = DistilBertModel(config)

        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)

        self.reduce_output = encoder_config.reduce_output
        self.max_sequence_length = encoder_config.max_sequence_length
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
        )
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return DistilBERTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # Subtract 2 to remove CLS and PAD tokens added by BERT tokenizer.
            return torch.Size([self.max_sequence_length - 2, self.transformer.config.dim])
        return torch.Size([self.transformer.config.dim])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("ctrl", TEXT)
class CTRLEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "ctrl",
    }

    def __init__(self, encoder_config: CTRLConfig = CTRLConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import CTRLConfig, CTRLModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = CTRLModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = CTRLConfig(
                vocab_size=encoder_config.vocab_size,
                n_positions=encoder_config.n_positions,
                n_ctx=encoder_config.n_ctx,
                n_embd=encoder_config.n_embd,
                dff=encoder_config.dff,
                n_layer=encoder_config.n_layer,
                n_head=encoder_config.n_head,
                resid_pdrop=encoder_config.resid_pdrop,
                embd_pdrop=encoder_config.embd_pdrop,
                attn_pdrop=encoder_config.attn_pdrop,
                layer_norm_epsilon=encoder_config.layer_norm_epsilon,
                initializer_range=encoder_config.initializer_range,
            )
            self.transformer = CTRLModel(config)

        self.vocab_size = encoder_config.vocab_size
        self.max_sequence_length = encoder_config.max_sequence_length
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        self.transformer.resize_token_embeddings(self.vocab_size)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return CTRLConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return torch.Size([self.max_sequence_length, self.transformer.config.n_embd])
        return torch.Size([self.transformer.config.n_embd])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("camembert", TEXT)
class CamemBERTEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "jplu/camembert-base",
    }

    def __init__(self, encoder_config: CamemBERTConfig = CamemBERTConfig()):
        super().__init__()
        try:
            from transformers import CamembertConfig, CamembertModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = CamembertModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = CamembertConfig(
                vocab_size=encoder_config.vocab_size,
                hidden_size=encoder_config.hidden_size,
                num_hidden_layers=encoder_config.num_hidden_layers,
                num_attention_heads=encoder_config.num_attention_heads,
                intermediate_size=encoder_config.intermediate_size,
                hidden_act=encoder_config.hidden_act,
                hidden_dropout_prob=encoder_config.hidden_dropout_prob,
                attention_probs_dropout_prob=encoder_config.attention_probs_dropout_prob,
                max_position_embeddings=encoder_config.max_position_embeddings,
                type_vocab_size=encoder_config.type_vocab_size,
                initializer_range=encoder_config.initializer_range,
                layer_norm_eps=encoder_config.layer_norm_eps,
                pad_token_id=encoder_config.pad_token_id,
                gradient_checkpointing=encoder_config.gradient_checkpointing,
                position_embedding_type=encoder_config.position_embedding_type,
                classifier_dropout=encoder_config.classifier_dropout,
            )
            self.transformer = CamembertModel(config)

        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.reduce_output = encoder_config.reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("t5", TEXT)
class T5Encoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "t5-small",
    }

    def __init__(self, encoder_config: T5Config = T5Config()):
        super().__init__(encoder_config)
        try:
            from transformers import T5Config, T5Model
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = T5Model.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = T5Config(
                vocab_size=encoder_config.vocab_size,
                d_model=encoder_config.d_model,
                d_kv=encoder_config.d_kv,
                d_ff=encoder_config.d_ff,
                num_layers=encoder_config.num_layers,
                num_decoder_layers=encoder_config.num_decoder_layers,
                num_heads=encoder_config.num_heads,
                relative_attention_num_buckets=encoder_config.relative_attention_num_buckets,
                dropout_rate=encoder_config.dropout_rate,
                layer_norm_eps=encoder_config.layer_norm_eps,
                initializer_factor=encoder_config.initializer_factor,
                feed_forward_proj=encoder_config.feed_forward_proj,
            )
            self.transformer = T5Model(config)

        self.max_sequence_length = encoder_config.max_sequence_length
        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            inputs,
            decoder_input_ids=inputs,
            attention_mask=mask,
        )
        hidden = transformer_outputs[0][:, 0:-1, :]  # [eos token]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.d_model])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("flaubert", TEXT)
class FlauBERTEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "flaubert/flaubert_small_cased",
    }

    def __init__(self, encoder_config: FlauBERTConfig = FlauBERTConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import FlaubertConfig, FlaubertModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = FlaubertModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = FlaubertConfig(
                vocab_size=encoder_config.vocab_size,
                pre_norm=encoder_config.pre_norm,
                layerdrop=encoder_config.layerdrop,
                emb_dim=encoder_config.emb_dim,
                n_layer=encoder_config.n_layer,
                n_head=encoder_config.n_head,
                dropout=encoder_config.dropout,
                attention_dropout=encoder_config.dropout,
                gelu_activation=encoder_config.gelu_activation,
                sinusoidal_embeddings=encoder_config.sinusoidal_embeddings,
                causal=encoder_config.causal,
                asm=encoder_config.asm,
                n_langs=encoder_config.n_langs,
                use_lang_emb=encoder_config.use_lang_emb,
                max_position_embeddings=encoder_config.max_position_embeddings,
                embed_init_std=encoder_config.embed_init_std,
                init_std=encoder_config.init_std,
                layer_norm_eps=encoder_config.layer_norm_eps,
                bos_index=encoder_config.bos_index,
                eos_index=encoder_config.eos_index,
                pad_index=encoder_config.pad_index,
                unk_index=encoder_config.unk_index,
                mask_index=encoder_config.mask_index,
                is_encoder=encoder_config.is_encoder,
                mask_token_id=encoder_config.mask_token_id,
                lang_id=encoder_config.lang_id,
            )
            self.transformer = FlaubertModel(config)

        self.max_sequence_length = encoder_config.max_sequence_length
        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.emb_dim])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("electra", TEXT)
class ELECTRAEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "google/electra-small-discriminator",
    }

    def __init__(self, encoder_config: ELECTRAConfig = ELECTRAConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import ElectraConfig, ElectraModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = ElectraModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, **pretrained_kwargs
            )
        else:
            config = ElectraConfig(
                vocab_size=encoder_config.vocab_size,
                embedding_size=encoder_config.embedding_size,
                hidden_size=encoder_config.hidden_size,
                num_hidden_layers=encoder_config.num_hidden_layers,
                num_attention_heads=encoder_config.num_attention_heads,
                intermediate_size=encoder_config.intermediate_size,
                hidden_act=encoder_config.hidden_act,
                hidden_dropout_prob=encoder_config.hidden_dropout_prob,
                attention_probs_dropout_prob=encoder_config.attention_probs_dropout_prob,
                max_position_embeddings=encoder_config.max_position_embeddings,
                type_vocab_size=encoder_config.type_vocab_size,
                initializer_range=encoder_config.initializer_range,
                layer_norm_eps=encoder_config.layer_norm_eps,
                position_embedding_type=encoder_config.position_embedding_type,
                classifier_dropout=encoder_config.classifier_dropout,
            )
            self.transformer = ElectraModel(config)

        self.max_sequence_length = encoder_config.max_sequence_length
        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("longformer", TEXT)
class LongformerEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    default_params = {
        "pretrained_model_name_or_path": "allenai/longformer-base-4096",
    }

    def __init__(self, encoder_config: LongformerConfig = LongformerConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import LongformerConfig, LongformerModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            pretrained_kwargs = encoder_config.pretrained_kwargs or {}
            self.transformer = LongformerModel.from_pretrained(
                encoder_config.pretrained_model_name_or_path, pretrained_kwargs
            )
        else:
            config = LongformerConfig(encoder_config.attention_window, encoder_config.sep_token_id)
            self.transformer = LongformerModel(config)
        self.reduce_output = encoder_config.reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.num_tokens)
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]  # bos + [sent] + sep
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
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
                    self.transformer.config.hidden_size,
                ]
            )
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32


@register_encoder("auto_transformer", TEXT)
class AutoTransformerEncoder(Encoder):
    fixed_preprocessing_parameters = {
        "tokenizer": "hf_tokenizer",
        "pretrained_model_name_or_path": "feature.pretrained_model_name_or_path",
    }

    def __init__(self, encoder_config: AutoTransformerConfig = AutoTransformerConfig()):
        super().__init__(encoder_config)
        try:
            from transformers import AutoModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        pretrained_kwargs = encoder_config.pretrained_kwargs or {}
        self.transformer = AutoModel.from_pretrained(encoder_config.pretrained_model_name_or_path, **pretrained_kwargs)
        self.reduce_output = encoder_config.reduce_output
        if self.reduce_output != "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)
        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(encoder_config.vocab_size)
        self.vocab_size = encoder_config.vocab_size
        self.max_sequence_length = encoder_config.max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            mask = mask.to(torch.int32)
        transformer_outputs = self.transformer(
            input_ids=inputs,
            attention_mask=mask,
            token_type_ids=torch.zeros_like(inputs),
        )
        if self.reduce_output == "cls_pooled":
            # this works only if the user know that the specific model
            # they want to use has the same outputs of
            # the BERT base class call() function
            hidden = transformer_outputs["pooler_output"]
        else:
            hidden = transformer_outputs["last_hidden_state"]
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return AutoTransformerConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            # TODO(justin): This may need to be conditioned on which AutoModel gets chosen.
            return torch.Size([self.max_sequence_length, self.transformer.config.hidden_size])
        return torch.Size([self.transformer.config.hidden_size])

    @property
    def input_dtype(self):
        return torch.int32
