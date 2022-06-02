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
from typing import Callable, Dict, List, Optional, Union

import torch

from ludwig.constants import TEXT
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.reduction_modules import SequenceReducer
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

    def __init__(
        self,
        max_sequence_length,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "albert-base-v2",
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import AlbertConfig, AlbertModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = AlbertModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = AlbertConfig(
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
            self.transformer = AlbertModel(config)

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(vocab_size)
        self.max_sequence_length = max_sequence_length

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "google/mt5-base",
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
        reduce_output: str = "cls_pooled",
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import MT5Config, MT5EncoderModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = MT5EncoderModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = MT5Config(
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
            self.transformer = MT5EncoderModel(config)

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(vocab_size)
        self.max_sequence_length = max_sequence_length

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "xlm-roberta-base",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "cls_pooled",
        trainable: bool = False,
        vocab_size: int = None,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        add_pooling_layer: bool = True,
        pretrained_kwargs: Dict = None,
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import XLMRobertaConfig, XLMRobertaModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = XLMRobertaModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = XLMRobertaConfig(
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            )

            self.transformer = XLMRobertaModel(config, add_pooling_layer)

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(vocab_size)
        self.max_sequence_length = max_sequence_length

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import BertConfig, BertModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = BertModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = BertConfig(
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
            self.transformer = BertModel(config)

        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)

        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)

        self.transformer.resize_token_embeddings(vocab_size)
        self.max_sequence_length = max_sequence_length

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "xlm-mlm-en-2048",
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
        reduce_output: str = "cls_pooled",
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import XLMConfig, XLMModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = XLMModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
            if trainable:
                self.transformer.train()
        else:
            config = XLMConfig(
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
            self.transformer = XLMModel(config)

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.resize_token_embeddings(vocab_size)
        self.max_sequence_length = max_sequence_length

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

    def __init__(
        self,
        max_sequence_length: int,
        reduce_output: str = "sum",
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "openai-gpt",
        saved_weights_in_checkpoint: bool = False,
        trainable: bool = False,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import OpenAIGPTConfig, OpenAIGPTModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = OpenAIGPTModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = OpenAIGPTConfig(
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
            self.transformer = OpenAIGPTModel(config)

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(vocab_size)
        self.max_sequence_length = max_sequence_length

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "gpt2",
        reduce_output: str = "sum",
        trainable: bool = False,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import GPT2Config, GPT2Model
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = GPT2Model.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = GPT2Config(
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
            self.transformer = GPT2Model(config)

        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.resize_token_embeddings(vocab_size)

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

    def __init__(
        self,
        max_sequence_length,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "roberta-base",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "cls_pooled",
        trainable: bool = False,
        vocab_size: int = None,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pretrained_kwargs: Dict = None,
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import RobertaConfig, RobertaModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = RobertaModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = RobertaConfig(
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            )
            self.transformer = RobertaModel(config)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.resize_token_embeddings(vocab_size)

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "transfo-xl-wt103",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import TransfoXLConfig, TransfoXLModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = TransfoXLModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = TransfoXLConfig(
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
            self.transformer = TransfoXLModel(config)
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        transformer_outputs = self.transformer(inputs)
        hidden = transformer_outputs[0]

        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {"encoder_output": hidden}

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "xlnet-base-cased",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import XLNetConfig, XLNetModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = XLNetModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = XLNetConfig(
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
            self.transformer = XLNetModel(config)
        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(vocab_size)

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

    def __init__(
        self,
        max_sequence_length: int,
        pretrained_model_name_or_path: str = "distilbert-base-uncased",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = True,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import DistilBertConfig, DistilBertModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = DistilBertModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = DistilBertConfig(
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
            self.transformer = DistilBertModel(config)

        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)

        self.reduce_output = reduce_output
        self.max_sequence_length = max_sequence_length
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.resize_token_embeddings(vocab_size)

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "ctrl",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = True,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import CTRLConfig, CTRLModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = CTRLModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = CTRLConfig(
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
            self.transformer = CTRLModel(config)

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "ctrl",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "cls-pooled",
        trainable: bool = False,
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
        **kwargs
    ):
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

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = CamembertModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = CamembertConfig(
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
            self.transformer = CamembertModel(config)

        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.resize_token_embeddings(vocab_size)
        self.max_sequence_length = max_sequence_length

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "t5-small",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import T5Config, T5Model
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = T5Model.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = T5Config(
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
            self.transformer = T5Model(config)

        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(vocab_size)

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool,
        pretrained_model_name_or_path: str = "flaubert/flaubert_small_cased",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
        vocab_size: int = 30145,
        pre_norm: bool = False,
        layerdrop: float = 0.0,
        emb_dim: int = 2048,
        n_layer: int = 12,
        n_head: int = 16,
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
        init_std: int = 50257,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import FlaubertConfig, FlaubertModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = FlaubertModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = FlaubertConfig(
                vocab_size=vocab_size,
                pre_norm=pre_norm,
                layerdrop=layerdrop,
                emb_dim=emb_dim,
                n_layer=n_layer,
                n_head=n_head,
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
            self.transformer = FlaubertModel(config)

        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(vocab_size)

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        pretrained_model_name_or_path: str = "google/electra-small-discriminator",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: str = "sum",
        trainable: bool = False,
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
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import ElectraConfig, ElectraModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = ElectraModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        else:
            config = ElectraConfig(
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
            self.transformer = ElectraModel(config)

        self.max_sequence_length = max_sequence_length
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(vocab_size)

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

    def __init__(
        self,
        max_sequence_length: int,
        use_pretrained: bool = True,
        attention_window: Union[List[int], int] = 512,
        sep_token_id: int = 2,
        pretrained_model_name_or_path: str = "allenai/longformer-base-4096",
        saved_weights_in_checkpoint: bool = False,
        reduce_output: Optional[str] = "cls_pooled",
        trainable: bool = False,
        num_tokens: Optional[int] = None,
        pretrained_kwargs: Dict = None,
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import LongformerConfig, LongformerModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        if use_pretrained and not saved_weights_in_checkpoint:
            pretrained_kwargs = pretrained_kwargs or {}
            self.transformer = LongformerModel.from_pretrained(pretrained_model_name_or_path, pretrained_kwargs)
        else:
            config = LongformerConfig(attention_window, sep_token_id, **kwargs)
            self.transformer = LongformerModel(config)
        self.reduce_output = reduce_output
        if not self.reduce_output == "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(num_tokens)
        self.max_sequence_length = max_sequence_length

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

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_sequence_length: int,
        reduce_output: str = "sum",
        trainable: bool = False,
        vocab_size: int = None,
        pretrained_kwargs: Dict = None,
        **kwargs
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ModuleNotFoundError:
            logger.error(
                " transformers is not installed. "
                "In order to install all text feature dependencies run "
                "pip install ludwig[text]"
            )
            sys.exit(-1)

        pretrained_kwargs = pretrained_kwargs or {}
        self.transformer = AutoModel.from_pretrained(pretrained_model_name_or_path, **pretrained_kwargs)
        self.reduce_output = reduce_output
        if self.reduce_output != "cls_pooled":
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)
        self.transformer.resize_token_embeddings(vocab_size)
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

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
