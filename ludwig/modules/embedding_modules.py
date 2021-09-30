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
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ludwig.constants import TYPE
from ludwig.modules.initializer_modules import get_initializer
from ludwig.utils.data_utils import load_pretrained_embeddings
from ludwig.utils.torch_utils import reg_loss, LudwigModule

logger = logging.getLogger(__name__)


def embedding_matrix(
        vocab: List[str],
        embedding_size: int,
        representation: str = 'dense',
        embeddings_trainable: bool = True,
        pretrained_embeddings: Optional[str] = None,
        force_embedding_size: bool = False,
        embedding_initializer: Optional[Union[str, Dict]] = None,
) -> Tuple[nn.Module, int]:
    """ Returns initialized torch.nn.Embedding module and embedding size. """

    vocab_size = len(vocab)
    if representation == 'dense':
        if pretrained_embeddings:
            embeddings_matrix = load_pretrained_embeddings(
                pretrained_embeddings, vocab
            )
            if embeddings_matrix.shape[-1] != embedding_size:
                if not force_embedding_size:
                    embedding_size = embeddings_matrix.shape[-1]
                    logger.info(
                        f'Setting embedding size to be equal to {embeddings_matrix.shape[-1]}.')
                else:
                    raise ValueError(
                        f'The size of the pretrained embeddings is '
                        f'{embeddings_matrix.shape[-1]}, but the specified '
                        f'embedding_size is {embedding_size}. Please change '
                        f'the embedding_size accordingly.')
            embedding_initializer_obj = torch.tensor(
                embeddings_matrix, dtype=torch.float32)

        else:
            if vocab_size < embedding_size and not force_embedding_size:
                logger.info(
                    f'  embedding_size ({embedding_size}) is greater than '
                    f'vocab_size ({vocab_size}). Setting embedding size to be '
                    f'equal to vocab_size.')
                embedding_size = vocab_size

            if embedding_initializer is not None:
                embedding_initializer_obj_ref = get_initializer(
                    embedding_initializer)
            else:
                embedding_initializer_obj_ref = get_initializer(
                    {TYPE: 'uniform', 'a': -1.0, 'b': 1.0})
            embedding_initializer_obj = embedding_initializer_obj_ref(
                [vocab_size, embedding_size])

        embeddings = embedding_initializer_obj

    elif representation == 'sparse':
        embedding_size = vocab_size
        embeddings = torch.tensor(
            get_initializer('identity')([vocab_size, embedding_size]),
            requires_grad=False)
    else:
        raise Exception(
            f'Embedding representation {representation} not supported.')

    embeddings = nn.Embedding.from_pretrained(
        embeddings, freeze=not embeddings_trainable
    )
    return embeddings, embedding_size


def embedding_matrix_on_device(
        vocab: List[str],
        embedding_size: int,
        representation: str = 'dense',
        embeddings_trainable: bool = True,
        pretrained_embeddings: Optional[str] = None,
        force_embedding_size: bool = False,
        embeddings_on_cpu: bool = False,
        embedding_initializer: Optional[str] = None
) -> Tuple[nn.Module, int]:
    embeddings, embedding_size = embedding_matrix(
        vocab,
        embedding_size,
        representation=representation,
        embeddings_trainable=embeddings_trainable,
        pretrained_embeddings=pretrained_embeddings,
        force_embedding_size=force_embedding_size,
        embedding_initializer=embedding_initializer
    )
    if embeddings_on_cpu:
        embeddings.to('cpu')
    elif not embeddings_on_cpu and torch.cuda.is_available():
        embeddings.to(device='cuda:0')

    return embeddings, embedding_size


class Embed(LudwigModule):
    def __init__(
            self,
            vocab: List[str],
            embedding_size: int,
            representation: str = 'dense',
            embeddings_trainable: bool = True,
            pretrained_embeddings: Optional[str] = None,
            force_embedding_size: bool = False,
            embeddings_on_cpu: bool = False,
            dropout: float = 0.0,
            embedding_initializer: Optional[Union[str, Dict]] = None,
            embedding_regularizer: str = None
    ):
        super().__init__()
        self.supports_masking = True

        self.vocab_size = len(vocab)
        self.embeddings, self.embedding_size = embedding_matrix_on_device(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            embedding_initializer=embedding_initializer,
        )

        if embedding_regularizer:
            self.add_loss(lambda: reg_loss(self.embeddings, embedding_regularizer))

        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, inputs: torch.Tensor):
        embedded = self.embeddings(inputs.long())
        embedded = torch.sum(embedded, dim=1)
        if self.dropout:
            embedded = self.dropout(embedded)
        return embedded

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.vocab_size])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.embedding_size])

class EmbedWeighted(LudwigModule):
    def __init__(
            self,
            vocab: List[str],
            embedding_size: int,
            representation: str = 'dense',
            embeddings_trainable: bool = True,
            pretrained_embeddings: Optional[str] = None,
            force_embedding_size: bool = False,
            embeddings_on_cpu: bool = False,
            dropout: float = 0.0,
            embedding_initializer: Optional[str] = None,
    ):
        super().__init__()

        self.embeddings, self.embedding_size = embedding_matrix_on_device(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            embedding_initializer=embedding_initializer,
        )
        self.vocab_size = len(vocab)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, inputs):
        signed_input = (inputs != 0).type(torch.int32)

        # TODO(shreya): Check correctness
        multiple_hot_indexes = (
            signed_input * torch.arange(self.vocab_size, dtype=torch.int32))

        embedded = self.embeddings(multiple_hot_indexes)

        # Get the multipliers to embeddings
        weights_mask = torch.unsqueeze(inputs, -1)

        weighted_embedded = embedded * weights_mask

        embedded_reduced = torch.sum(weighted_embedded, dim=1)

        if self.dropout:
            embedded_reduced = self.dropout(embedded_reduced)

        return embedded_reduced

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.vocab_size])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.embedding_size])


# TODO(shreya): Implement sparse embedding lookup.
# class EmbedSparse(LudwigModule):
#     def __init__(
#             self,
#             vocab: List[str],
#             embedding_size: int = 50,
#             representation: str = 'dense',
#             embeddings_trainable: bool = True,
#             pretrained_embeddings: Optional[str] = None,
#             force_embedding_size: bool = False,
#             embeddings_on_cpu: bool = False,
#             dropout: float = 0.0,
#             embedding_initializer: Optional[str] = None,
#             reduce_output: str = 'sum'
#     ):
#         super().__init__()

#         self.embeddings, self.embedding_size = embedding_matrix_on_device(
#             vocab,
#             embedding_size,
#             representation=representation,
#             embeddings_trainable=embeddings_trainable,
#             pretrained_embeddings=pretrained_embeddings,
#             force_embedding_size=force_embedding_size,
#             embeddings_on_cpu=embeddings_on_cpu,
#             embedding_initializer=embedding_initializer,
#         )

#         if dropout > 0:
#             self.dropout = nn.Dropout(dropout)
#         else:
#             self.dropout = None

#         self.reduce_output = reduce_output

#     def forward(self, inputs: torch.Tensor):
#         # idx = tf.where(tf.equal(inputs, True))
#         # TODO(shreya): Check if this is equivalent
#         idx = torch.nonzero(inputs)

#         # sparse_multiple_hot_indexes = tf.SparseTensor(
#         #     idx,
#         #     idx[:, 1],
#         #     tf.shape(inputs, out_type=tf.int64)
#         # )
#         sparse_multiple_hot_index = torch.sparse_coo_tensor(
#             idx, idx[:, 1], inputs.shape
#         )

#         # TODO(shreya): Check if supported in torch
#         # embedded_reduced = tf.nn.embedding_lookup_sparse(
#         #     self.embeddings,
#         #     sparse_multiple_hot_indexes,
#         #     sp_weights=None,
#         #     combiner=self.reduce_output
#         # )

#         # if self.dropout:
#         #     embedded_reduced = self.dropout(embedded_reduced)

#         # return embedded_reduced
#         return None

class EmbedSequence(LudwigModule):
    def __init__(
            self,
            vocab,
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=False,
            embeddings_on_cpu=False,
            dropout=0.0,
            embedding_initializer=None,
            embedding_regularizer=None
    ):
        super().__init__()
        self.supports_masking = True

        self.embeddings, self.embedding_size = embedding_matrix_on_device(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            embedding_initializer=embedding_initializer,
        )

        if embedding_regularizer:
            embedding_regularizer_obj = tf.keras.regularizers.get(
                embedding_regularizer)
            self.add_loss(lambda: embedding_regularizer_obj(self.embeddings))

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, inputs, training=None, mask=None):
        embedded = self.embeddings(inputs)

        # todo: convert to torch with mask support
        # if mask is not None:
        #     mask_matrix = tf.cast(
        #         tf.expand_dims(mask, -1),
        #         dtype=tf.float32
        #     )
        #     embedded = tf.multiply(embedded, mask_matrix)

        if self.dropout:
            embedded = self.dropout(
                embedded)  # todo: neeeded for torch? ...., training=training)

        return embedded


class TokenAndPositionEmbedding(LudwigModule):
    def __init__(self,
                 max_length,
                 vocab,
                 embedding_size,
                 representation='dense',
                 embeddings_trainable=True,
                 pretrained_embeddings=None,
                 force_embedding_size=False,
                 embeddings_on_cpu=False,
                 dropout=0.0,
                 embedding_initializer=None,
                 embedding_regularizer=None
                 ):
        super().__init__()
        self.token_embed = EmbedSequence(
            vocab=vocab,
            embedding_size=embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=embedding_initializer,
            embedding_regularizer=embedding_regularizer
        )
        self.position_embed = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=self.token_embed.embedding_size
        )

    def forward(self, inputs, training=None, mask=None):
        max_length = inputs.shape[-1]
        positions = torch.arange(start=0, end=max_length, step=1)
        positions_hidden = self.position_embed(positions)
        token_hidden = self.token_embed(inputs)
        return token_hidden + positions_hidden
