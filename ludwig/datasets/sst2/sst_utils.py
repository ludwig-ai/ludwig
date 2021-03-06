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

    def __init__(self, dataset_name, cache_dir=DEFAULT_CACHE_LOCATION,
                 include_subtrees=False, discard_neutral=False,
                 convert_parentheses=True, remove_duplicates=False):
        super().__init__(dataset_name=dataset_name, cache_dir=cache_dir)
        self.include_subtrees = include_subtrees
        self.discard_neutral = discard_neutral
        self.convert_parentheses = convert_parentheses
        self.remove_duplicates = remove_duplicates

    @staticmethod
    @abstractmethod
    def get_sentiment_label(id2sent, phrase_id):
        pass

    def process_downloaded_dataset(self):
        sentences_df = pd.read_csv(
            os.path.join(self.raw_dataset_path,
                         'stanfordSentimentTreebank/datasetSentences.txt'),
            sep="\t",
            )
    
        sentences_df['sentence'] = sentences_df['sentence'].apply(format_text)

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
                    phrase = split_line[0]
                    phrase2id[phrase] = int(split_line[1])

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

        trees_pointers = None
        trees_phrases = None

        if self.include_subtrees:
            trees_pointers = []
            with open(os.path.join(self.raw_dataset_path,
                                   'stanfordSentimentTreebank/STree.txt')) as f:
                Lines = f.readlines()
                for line in Lines:
                    if line:
                        trees_pointers.append(
                            [int(s.strip()) for s in line.split('|')]
                        )

            trees_phrases = []
            with open(os.path.join(self.raw_dataset_path,
                                   'stanfordSentimentTreebank/SOStr.txt')) as f:
                Lines = f.readlines()
                for line in Lines:
                    if line:
                        trees_phrases.append(
                            [s.strip() for s in line.split('|')]
                        )

        splits = {
            'train': 1,
            'test': 2,
            'dev': 3
        }

        for split_name, split_id in splits.items():
            sentence_idcs = get_sentence_idcs_in_split(datasplit_df, split_id)

            pairs = []
            if split_name == 'train' and self.include_subtrees:
                phrases = []
                for sentence_idx in sentence_idcs:
                    # trees_pointers and trees_phrases are 0 indexed
                    # while sentence_idx starts from 1
                    # so we need to decrease sentence_idx value
                    sentence_idx -= 1
                    subtrees = sentence_subtrees(sentence_idx, trees_pointers,
                                                 trees_phrases)
            
                    sentence_idx += 1
                    sentence_phrase = list(sentences_df[
                        sentences_df['sentence_index'] == sentence_idx
                    ]['sentence'])[0]

                    sentence_phrase = convert_parentheses(sentence_phrase)
                    label = self.get_sentiment_label(id2sent, phrase2id[sentence_phrase])
                    # filter @ sentence level
                    # For SST-2, check subtrees only if sentence is not neutral
                    if not self.discard_neutral or label != -1:
                        for phrase in subtrees:
                            label = self.get_sentiment_label(id2sent, phrase2id[phrase])
                            if not self.discard_neutral or label != -1:
                                if not self.convert_parentheses:
                                    phrase = convert_parentheses_back(phrase)
                                    phrase = phrase.replace('\xa0', ' ')
                                pairs.append([phrase, label])
            else:
                phrases = get_sentences_with_idcs(sentences_df, sentence_idcs)
                for phrase in phrases:
                    phrase = convert_parentheses(phrase)
                    label = self.get_sentiment_label(id2sent, phrase2id[phrase])
                    if not self.discard_neutral or label != -1:
                        if not self.convert_parentheses:
                            phrase = convert_parentheses_back(phrase)
                            phrase = phrase.replace('\xa0', ' ')
                        pairs.append([phrase, label])

            final_csv = pd.DataFrame(pairs)
            final_csv.columns = ['sentence', 'label']
            if self.remove_duplicates:
                final_csv = final_csv.drop_duplicates(subset=['sentence'])
            final_csv.to_csv(
                os.path.join(self.raw_dataset_path, f'{split_name}.csv'),
                index=False
            )

        super(SST, self).process_downloaded_dataset()


def format_text(text: str):
    """
    Formats text by decoding into utf-8
    """
    return ' '.join(
        [w.encode('latin1').decode('utf-8')
         for w in text.strip().split(' ')]
    )


def convert_parentheses(text: str):
    """
    Replaces -LRB- and -RRB- tokens present in SST with ( and )
    """
    return text.replace('-LRB-', '(').replace('-RRB-', ')')


def convert_parentheses_back(text: str):
    """
    Replaces ( and ) tokens with -LRB- and -RRB-
    """
    return text.replace('(', '-LRB-').replace(')', '-RRB-')


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


def sentence_subtrees(sentence_idx, trees_pointers, trees_phrases):
    tree_pointers = trees_pointers[sentence_idx]
    tree_phrases = trees_phrases[sentence_idx]
    tree = SSTTree(tree_pointers, tree_phrases)
    return tree.subtrees()


def visit_postorder(node, visit_list):
    if node:
        visit_postorder(node.left, visit_list)
        visit_postorder(node.right, visit_list)
        visit_list.append(node.val)


class SSTTree:
    class Node:
        def __init__(self, key, val=None):
            self.left = None
            self.right = None
            self.key = key
            self.val = val

    def create_node(self, parent, i):
        if self.nodes[i] is not None:
            # already created
            return
        self.nodes[i] = self.Node(i)

        if parent[i] == -1:
            # is root
            self.root = self.nodes[i]
            return

        if self.nodes[parent[i]] is None:
            # parent not yet created
            self.create_node(parent, parent[i])

        # assign current node to parent
        parent = self.nodes[parent[i]]
        if parent.left is None:
            parent.left = self.nodes[i]
        else:
            parent.right = self.nodes[i]

    def create_tree(self, parents, tree_phrases):
        n = len(parents)
        self.nodes = [None for i in range(n)]
        self.root = [None]
        for i in range(n):
            self.create_node(parents, i)
        for i, phrase in enumerate(tree_phrases):
            self.nodes[i].val = phrase
        for node in self.nodes:
            if node.val is None:
                node.val = ' '.join((node.left.val, node.right.val))

    def __init__(self, tree_pointers, tree_phrases):
        self.create_tree(
            [int(elem) - 1 for elem in tree_pointers],
            tree_phrases
        )

    def subtrees(self):
        visit_list = []
        visit_postorder(self.root, visit_list)
        return visit_list
