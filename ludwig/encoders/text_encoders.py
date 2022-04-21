#! /usr/bin/env python
# coding=utf-8
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
from abc import ABC
from typing import Optional

import tensorflow as tf
from transformers import TFXLNetModel

from ludwig.encoders import sequence_encoders
from ludwig.encoders.base import Encoder
from ludwig.utils.registry import Registry, register
from ludwig.modules.reduction_modules import SequenceReducer

logger = logging.getLogger(__name__)


ENCODER_REGISTRY = Registry(sequence_encoders.ENCODER_REGISTRY)


class TextEncoder(Encoder, ABC):
    @classmethod
    def register(cls, name):
        ENCODER_REGISTRY[name] = cls


@register(name='bert')
class BERTEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }


    default_params = {
        'pretrained_model_name_or_path': 'bert-base-uncased',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'bert-base-uncased',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'cls_pooled',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFBertModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFBertModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFBertModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        if not self.reduce_output == 'cls_pooled':
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)

        return {'encoder_output': hidden}


@register(name='gpt')
class GPTEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'openai-gpt',
    }

    def __init__(
            self,
            reduce_output: str = 'sum',
            pretrained_model_name_or_path: str = 'openai-gpt',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFOpenAIGPTModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFOpenAIGPTModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFOpenAIGPTModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='gpt2')
class GPT2Encoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'gpt2',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'gpt2',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFGPT2Model
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFGPT2Model.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFGPT2Model.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


# @register(name='transformer_xl')
class TransformerXLEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'transfo-xl-wt103',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'transfo-xl-wt103',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFTransfoXLModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFTransfoXLModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFTransfoXLModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs,
            training=training,
        )
        hidden = transformer_outputs[0]

        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='xlnet')
class XLNetEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'xlnet-base-cased',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'xlnet-base-cased',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFXLNetModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFXLNetModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFXLNetModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='xlm')
class XLMEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'xlm-mlm-en-2048',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'xlm-mlm-en-2048',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFXLMModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFXLMModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFXLMModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        hidden = transformer_outputs[0][:, 1:-1, :]  # bos + [sent] + sep
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='roberta')
class RoBERTaEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'roberta-base',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'roberta-base',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'cls_pooled',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFRobertaModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFRobertaModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFRobertaModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        if not self.reduce_output == 'cls_pooled':
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]  # bos + [sent] + sep
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='distilbert')
class DistilBERTEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'distilbert-base-uncased',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'distilbert-base-uncased',
            pretrained_weights: Optional[str] = None,
            saved_weights_in_checkpoint: bool = False,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFDistilBertModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFDistilBertModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFDistilBertModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask
        }, training=training)
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='ctrl')
class CTRLEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'ctrl',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'ctrl',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[str] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFCTRLModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFCTRLModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFCTRLModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        hidden = transformer_outputs[0]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='camembert')
class CamemBERTEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'jplu/tf-camembert-base',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'jplu/tf-camembert-base',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'cls_pooled',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFCamembertModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFCamembertModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFCamembertModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        if not self.reduce_output == 'cls_pooled':
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='albert')
class ALBERTEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'albert-base-v2',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'albert-base-v2',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'cls_pooled',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFAlbertModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFAlbertModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFAlbertModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        if not self.reduce_output == 'cls_pooled':
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='t5')
class T5Encoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 't5-small',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 't5-small',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFT5Model
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFT5Model.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFT5Model.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer(
            inputs,
            decoder_input_ids=inputs,
            training=training,
            attention_mask=mask,
        )
        hidden = transformer_outputs[0][:, 0:-1, :]  # [sent] + [eos token]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}

@register(name='mt5')
class MT5Encoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'google/mt5-small',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'google/mt5-small',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFMT5Model
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFMT5Model.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFMT5Model.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer(
            inputs,
            decoder_input_ids=inputs,
            training=training,
            attention_mask=mask,
        )
        hidden = transformer_outputs[0][:, 0:-1, :]  # [sent] + [eos token]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='xlmroberta')
class XLMRoBERTaEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'jplu/tf-xlm-roberta-base',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'jplu/tf-xlm-roberta-base',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'cls_pooled',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFXLMRobertaModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFXLMRobertaModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFXLMRobertaModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        if not self.reduce_output == 'cls_pooled':
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='flaubert')
class FlauBERTEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'jplu/tf-flaubert-small-cased',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'jplu/tf-flaubert-small-cased',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFFlaubertModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFFlaubertModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFFlaubertModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            'input_ids': inputs,
            'attention_mask': mask,
            'token_type_ids': tf.zeros_like(inputs),
        }, training=training)
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='electra')
class ELECTRAEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'google/electra-small-discriminator',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'google/electra-small-discriminator',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFElectraModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFElectraModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFElectraModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        hidden = transformer_outputs[0][:, 1:-1, :]
        hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='longformer')
class LongformerEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'allenai/longformer-base-4096',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str = 'allenai/longformer-base-4096',
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'cls_pooled',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFLongformerModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFLongformerModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFLongformerModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        if not self.reduce_output == 'cls_pooled':
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs),
        }, training=training)
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:, 1:-1, :]  # bos + [sent] + sep
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


@register(name='auto_transformer')
class AutoTransformerEncoder(TextEncoder):
    fixed_preprocessing_parameters = {
        'word_tokenizer': 'hf_tokenizer',
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            saved_weights_in_checkpoint: bool = False,
            pretrained_weights: Optional[str] = None,
            reduce_output: str = 'sum',
            trainable: bool = True,
            num_tokens: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        try:
            from transformers import TFAutoModel
        except ModuleNotFoundError:
            logger.error(
                ' transformers is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        if not saved_weights_in_checkpoint:
            self.transformer = TFAutoModel.from_pretrained(
                pretrained_model_name_or_path
            )
        else:
            self.transformer = TFAutoModel.from_pretrained(
                pretrained_weights
            )
        self.reduce_output = reduce_output
        if not self.reduce_output == 'cls_pooled':
            self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.transformer.trainable = trainable
        self.transformer.resize_token_embeddings(num_tokens)

    def call(self, inputs, training=None, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.int32)
        transformer_outputs = self.transformer({
            "input_ids": inputs,
            "training": training,
            "attention_mask": mask,
            "token_type_ids": tf.zeros_like(inputs)
        }, return_dict=True)
        if self.reduce_output == 'cls_pooled':
            # this works only if the user know that the specific model
            # they want to use has the same outputs of
            # the BERT base class call() function
            hidden = transformer_outputs['cls_pooled']
        else:
            hidden = transformer_outputs['last_hidden_state']
            hidden = self.reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}
