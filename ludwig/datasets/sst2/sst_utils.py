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
                 include_subtrees=False):
        super().__init__(dataset_name=dataset_name, cache_dir=cache_dir)
        self.include_subtrees = include_subtrees

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

        if self.include_subtrees:
            trees_pointers = []
            with open(os.path.join(self.raw_dataset_path,
                                   'stanfordSentimentTreebank/STree.txt')) as f:
                Lines = f.readlines()
                for line in Lines:
                    if line:
                        trees_pointers.append(line.split('|'))

            trees_phrases = []
            with open(os.path.join(self.raw_dataset_path,
                                   'stanfordSentimentTreebank/SOStr.txt')) as f:
                Lines = f.readlines()
                for line in Lines:
                    if line:
                        trees_phrases.append(line.split('|'))

        else:
            trees_pointers = None
            trees_phrases = None

        splits = {
            'train': 1,
            'test': 2,
            'dev': 3
        }

        for split_name, split_id in splits.items():
            sentence_idcs = get_sentence_idcs_in_split(datasplit_df, split_id)

            if self.include_subtrees:
                phrases = []
                for sentence_idx in sentence_idcs:
                    subtrees = sentence_subtrees(sentence_idx, trees_pointers,
                                                 trees_phrases)
                    phrases.extend([format_sentence(st) for st in subtrees])
            else:
                phrases = get_sentences_with_idcs(sentences_df, sentence_idcs)

            phrase_ids = [phrase2id[phrase] for phrase in phrases]

            pairs = []
            for phrase, phrase_id in zip(phrases, phrase_ids):
                label = self.get_sentiment_label(id2sent, phrase_id)
                if label != -1:  # only include non-neutral samples
                    pairs.append([phrase, label])

            final_csv = pd.DataFrame(pairs)
            final_csv.columns = ['text', 'label']
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

    def __init__(self, tree_pointers, tree_phrases):
        self.tree_pointers = [int(elem) - 1 for elem in tree_pointers]
        self.tree_phrases = tree_phrases

        self.nodes = []
        phrases_cache = {}
        children_cache = {}
        for i, phrase in enumerate(self.tree_phrases):
            pointer = int(self.tree_pointers[i])
            self.nodes.append(Node(phrase))

            phrases = phrases_cache.get(pointer, [])
            phrases.append(phrase)
            phrases_cache[pointer] = phrases

            pointers = children_cache.get(pointer, [])
            pointers.append(i)
            children_cache[pointer] = pointers

        num_leaves = len(self.nodes)

        for i, pointer in enumerate(self.tree_pointers):
            if i >= num_leaves:
                phrase = ' '.join(phrases_cache[i])
                new_node = Node(phrase)
                children_pointers = children_cache[i]
                new_node.left = self.nodes[children_pointers[0]]
                new_node.right = self.nodes[children_pointers[1]]
                self.nodes.append(new_node)

                phrases = phrases_cache.get(pointer, [])
                phrases.insert(0, phrase)
                phrases_cache[pointer] = phrases

                pointers = children_cache.get(pointer, [])
                pointers.insert(0, i)
                children_cache[pointer] = pointers

        self.root = self.nodes[-1]

    def subtrees(self):
        visit_list = []
        visit_postorder(self.root, visit_list)
        return visit_list


class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key


if __name__ == '__main__':
    stree = "6|6|5|5|7|7|0"
    sostree = "Effective|but|too-tepid|biopic"

    tree_pointers = stree.split("|")
    tree_phrases = sostree.split("|")

    tree = SSTTree(tree_pointers, tree_phrases)
    print(tree.subtrees())

    stree = "70|70|68|67|63|62|61|60|58|58|57|56|56|64|65|55|54|53|52|51|49|47|47|46|46|45|40|40|41|39|38|38|43|37|37|69|44|39|42|41|42|43|44|45|50|48|48|49|50|51|52|53|54|55|66|57|59|59|60|61|62|63|64|65|66|67|68|69|71|71|0"
    sostree = "The|Rock|is|destined|to|be|the|21st|Century|'s|new|``|Conan|''|and|that|he|'s|going|to|make|a|splash|even|greater|than|Arnold|Schwarzenegger|,|Jean-Claud|Van|Damme|or|Steven|Segal|."

    tree_pointers = stree.split("|")
    tree_phrases = sostree.split("|")

    tree = SSTTree(tree_pointers, tree_phrases)
    print(tree.subtrees())
