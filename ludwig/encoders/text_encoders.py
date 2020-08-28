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

import tensorflow as tf
from tensorflow.keras.layers import Layer
from transformers import TFBertModel, TFOpenAIGPTModel, TFGPT2Model, \
    TFTransfoXLModel, TFXLNetModel, TFXLMModel, \
    TFRobertaModel, TFDistilBertModel, TFCTRLModel, TFCamembertModel, \
    TFAlbertModel, TFT5Model, TFXLMRobertaModel, \
    TFFlaubertModel, TFElectraModel, TFAutoModel

from ludwig.modules.reduction_modules import reduce_sequence

logger = logging.getLogger(__name__)


class BERTEncoder(Layer):
    fixed_preprocessing_params = {
        'pretrained_model_name_or_path': 'feature.pretrained_model_name_or_path',
    }

    default_params = {
        'pretrained_model_name_or_path': 'bert-base-uncased',
    }

    def __init__(
            self,
            pretrained_model_name_or_path='bert-base-uncased',
            reduce_output='cls_pooled',
            trainable=False,
            **kwargs
    ):
        super(BERTEncoder, self).__init__()
        self.transformer = TFBertModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:,1:-1,:]
            hidden = reduce_sequence(hidden, self.reduce_output)

        return {'encoder_output': hidden}


class GPTEncoder(Layer):

    def __init__(
            self,
            reduce_output='sum',
            pretrained_model_name_or_path='openai-gpt',
            trainable=False,
            **kwargs
    ):
        super(GPTEncoder, self).__init__()
        self.transformer = TFOpenAIGPTModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable


    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        hidden = transformer_outputs[0]
        hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class GPT2Encoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='gpt2',
            reduce_output='sum',
            trainable=False,
            **kwargs
    ):
        super(GPT2Encoder, self).__init__()
        self.transformer = TFGPT2Model.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable


    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        hidden = transformer_outputs[0]
        hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class TransformerXLEncoder(Layer):
    def __init__(
            self,
            pretrained_model_name_or_path='transfo-xl-wt103',
            reduce_output='sum',
            trainable=False,
            **kwargs
    ):
        super(TransformerXLEncoder, self).__init__()
        self.transformer = TFTransfoXLModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable


    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
        )
        hidden = transformer_outputs[0]
        hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class XLNetEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='xlnet-base-cased',
            reduce_output='sum',
            trainable=False,
            **kwargs
    ):
        super(XLNetEncoder, self).__init__()
        self.transformer = TFXLNetModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        hidden = transformer_outputs[0]
        hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class XLMEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='xlm-mlm-en-2048',
            reduce_output='sum',
            trainable=False,
            **kwargs
    ):
        super(XLMEncoder, self).__init__()
        self.transformer = TFXLMModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        hidden = transformer_outputs[0][:,1:-1,:] #bos + [sent] + sep
        hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class RoBERTaEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='roberta-base',
            reduce_output='cls_pooled',
            trainable=False,
            **kwargs
    ):
        super(RoBERTaEncoder, self).__init__()
        self.transformer = TFRobertaModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:,1:-1,:] #bos + [sent] + sep
            hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class DistilBERTEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='distilbert-base-uncased',
            reduce_output='cls_pooled',
            trainable=False,
            **kwargs
    ):
        super(DistilBERTEncoder, self).__init__()
        self.transformer = TFDistilBertModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask
        )
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:,1:-1,:]
            hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class CTRLEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='ctrl',
            reduce_output='sum',
            trainable=False,
            **kwargs
    ):
        super(CTRLEncoder, self).__init__()
        self.transformer = TFCTRLModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        hidden = transformer_outputs[0]
        hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class CamemBERTEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='jplu/tf-camembert-base',
            reduce_output='cls_pooled',
            trainable=False,
            **kwargs
    ):
        super(CamemBERTEncoder, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.transformer = TFCamembertModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:,1:-1,:]
            hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class ALBERTEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='albert-base-v2',
            reduce_output='cls_pooled',
            trainable=False,
            **kwargs
    ):
        super(ALBERTEncoder, self).__init__()
        self.transformer = TFAlbertModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:,1:-1,:]
            hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class T5Encoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='t5-small',
            reduce_output='sum',
            trainable=False,
            **kwargs
    ):
        super(T5Encoder, self).__init__()
        self.transformer = TFT5Model.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):

        transformer_outputs = self.transformer({
            'inputs' : inputs, 
            'decoder_input_ids' : inputs,
            'training' : training,
            'attention_mask' : mask,
            'token_type_ids' : tf.zeros_like(inputs)
        })
        hidden = transformer_outputs[0]
        hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class XLMRoBERTaEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='jplu/tf-xlm-roberta-base',
            reduce_output='cls_pooled',
            trainable=False,
            **kwargs
    ):
        super(XLMRoBERTaEncoder, self).__init__()
        self.transformer = TFXLMRobertaModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:,1:-1,:]
            hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class FlauBERTEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='jplu/tf-flaubert-base-uncased',
            reduce_output='cls_pooled',
            trainable=False,
            **kwargs
    ):
        super(FlauBERTEncoder, self).__init__()
        self.transformer = TFFlaubertModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        if self.reduce_output == 'cls_pooled':
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0][:,1:-1,:]
            hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}

class ELECTRAEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path='google/electra-small-discriminator',
            reduce_output='sum',
            trainable=False,
            **kwargs
    ):
        super(ELECTRAEncoder, self).__init__()
        self.transformer = TFElectraModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        hidden = transformer_outputs[0][:,1:-1,:]
        hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}


class AutoTransformerEncoder(Layer):

    def __init__(
            self,
            pretrained_model_name_or_path,
            reduce_output='sum',
            trainable=False,
            **kwargs
    ):
        super(AutoTransformerEncoder, self).__init__()
        self.transformer = TFAutoModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.reduce_output = reduce_output
        self.transformer.trainable = trainable

    def call(self, inputs, training=None, mask=None):
        transformer_outputs = self.transformer(
            inputs, 
            training=training,
            attention_mask=mask,
            token_type_ids=tf.zeros_like(inputs)
        )
        if self.reduce_output == 'cls_pooled':
            # this works only if the user know that the specific model
            # they want to use has the same outputs of
            # the BERT base class call() function
            hidden = transformer_outputs[1]
        else:
            hidden = transformer_outputs[0]
            hidden = reduce_sequence(hidden, self.reduce_output)
        return {'encoder_output': hidden}
