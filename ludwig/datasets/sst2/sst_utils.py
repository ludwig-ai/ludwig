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
from abc import abstractmethod, ABC
from typing import Set

from pandas import DataFrame

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import ZipDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import *


class SST(ABC, ZipDownloadMixin, MultifileJoinProcessMixin, CSVLoadMixin,
          BaseDataset):
    """The SST2 dataset.

    This dataset is constructed using the Stanford Sentiment Treebank Dataset.
    This dataset contains binary labels (positive or negative) for each sample.

    The original dataset specified 5 labels:
    very negative, negative, neutral, positive, very positive with
    the following cutoffs:
    [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]

    This class pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming
    training data into a destination dataframe that can be use by Ludwig.
    """

    def __init__(self, dataset_name, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name=dataset_name, cache_dir=cache_dir)

    @staticmethod
    @abstractmethod
    def get_sentiment_label(id2sent, phrase_id):
        pass

    def process_downloaded_dataset(self):
        sentences_df = pd.read_csv(
            os.path.join(self.raw_dataset_path,
                         'stanfordSentimentTreebank/datasetSentences.txt'),
            sep=('\t'))

        datasplit_df = pd.read_csv(
            os.path.join(self.raw_dataset_path,
                        'stanfordSentimentTreebank/datasetSplit.txt'),
            sep=',')

        phrase2id = {}
        with open(os.path.join(self.raw_dataset_path,
                            'stanfordSentimentTreebank/dictionary.txt')) as f:
            Lines = f.readlines()
            for line in Lines:
                if line:
                    split_line = line.split('|')
                    phrase2id[split_line[0]] = int(split_line[1])

        id2sent = {}
        with open(os.path.join(self.raw_dataset_path,
                    'stanfordSentimentTreebank/sentiment_labels.txt')) as f:
            Lines = f.readlines()
            for line in Lines:
                if line:
                    split_line = line.split('|')
                    try:
                        id2sent[int(split_line[0])] = float(split_line[1])
                    except ValueError:
                        pass

        sentences_df['sentence'] = sentences_df['sentence'].apply(
            format_sentence
        )

        splits = {
            'train': 1,
            'test': 2,
            'dev': 3
        }

        for split_name, split_id in splits.items():
            sentence_idcs = get_sentence_idcs_in_split(datasplit_df, split_id)
            sentences = get_sentences_with_idcs(sentences_df, sentence_idcs)
            phrase_ids = [phrase2id[phrase] for phrase in sentences]

            pairs = []
            for sentence, phrase_id in zip(sentences, phrase_ids):
                label = self.get_sentiment_label(id2sent, phrase_id)
                if label != -1:  # only include non-neutral samples
                    pairs.append([sentence, label])

            final_csv = pd.DataFrame(pairs)
            final_csv.columns = ['sentence', 'label']
            final_csv.to_csv(os.path.join(self.raw_dataset_path,
                                          f'{split_name}.csv'),
                             index=False)

        super(SST, self).process_downloaded_dataset()


def format_sentence(sentence: str):
    """
    Formats raw sentences by decoding into utf-8 and replacing
    -LRB- and -RRB- tokens with their matching characters
    """
    formatted_sent = ' '.join(
        [w.encode('latin1').decode('utf-8')
         for w in sentence.strip().split(' ')])
    formatted_sent = formatted_sent.replace('-LRB-', '(')
    formatted_sent = formatted_sent.replace('-RRB-', ')')
    return formatted_sent


def get_sentence_idcs_in_split(datasplit: DataFrame, split_id: int):
    """
    Given a dataset split is (1 for train, 2 for test, 3 for dev),
    returns the set of corresponding sentence indices in sentences_df.
    """
    return set(
        datasplit[datasplit['splitset_label'] == split_id]['sentence_index']
    )


def get_sentences_with_idcs(sentences: DataFrame, sentences_idcs: Set[int]):
    """
    Given a set of sentence indices,
    returns the corresponding sentences texts in sentences
    """
    criterion = sentences['sentence_index'].map(
        lambda x: x in sentences_idcs
    )
    return sentences[criterion]['sentence'].tolist()
