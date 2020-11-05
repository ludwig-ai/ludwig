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
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.process import *
from ludwig.datasets.mixins.download import ZipDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = SST2(cache_dir=cache_dir)
    return dataset.load(split=split)


class SST2(ZipDownloadMixin, MultifileJoinProcessMixin, CSVLoadMixin, BaseDataset):
    """The SST2 dataset.

    This pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming training data into a destination
    dataframe that can fit into Ludwig's training API.
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="sst2", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        sentences_df = pd.read_csv(os.path.join(self.raw_dataset_path, 'SST-2/original/datasetSentences.txt'), sep=('\t'))
        datasplit_df = pd.read_csv(os.path.join(self.raw_dataset_path, 'SST-2/original/datasetSplit.txt'), sep=',')
        sen2id_df =  pd.read_csv(os.path.join(self.raw_dataset_path, "SST-2/original/dictionary.txt"), sep="|", names=['sentence', 'phrase_id'])
        sentiment_labels_df = pd.read_csv(os.path.join(self.raw_dataset_path, "SST-2/original/sentiment_labels.txt"), sep="|")
        sentiment_labels_df.columns = ["id", "sentiment"]

        def format_sentence(sent):
            formatted_sent = " ".join([w.encode('latin1').decode('utf-8').lower() for w in sent.strip().split(" ")])
            formatted_sent = formatted_sent.replace("-lrb-", "(")
            formatted_sent = formatted_sent.replace("-rrb-", ")")
            return formatted_sent

        def get_sentence_idxs(split):
            idxs = datasplit_df.index[datasplit_df['splitset_label'] == split]
            sentence_idxs = [datasplit_df['sentence_index'][i] for i in idxs]
            return sentence_idxs

        def get_sentences(sentences_idxs):
            row_idxs = [
                            sentences_df.index[sentences_df['sentence_index'] == int(idx)].tolist()[0] 
                            for idx in sentences_idxs
                        ]
            return [sentences_df.iloc[idx][1] for idx in row_idxs]

        def get_phrase_id(phrase):
            idx = sen2id_df.index[sen2id_df['sentence'] == phrase].tolist()
            phrase_id = sen2id_df.iloc[idx[0]][1]
            return int(phrase_id)

        def get_sentiment_label(phrase_id):
            sentiment = float(sentiment_labels_df.iloc[phrase_id][1])
            if sentiment <= 0.4: # negative
                return 0
            elif sentiment > 0.6: # positive
                return 1
            return -1 # neutral

        sentences_df['sentence'] = sentences_df['sentence'].apply(format_sentence)
        sen2id_df['sentence'] = sen2id_df['sentence'].apply(
            lambda sent: " ".join([t.lower() for t in sent.strip().split(" ")])
        )
        
        splits = {
            'train' : 1,
            'dev' : 3,
            'test' : 2
        }

        for split_name, split_id in splits.items():
            sent_idxs = get_sentence_idxs(split_id)
            sents = get_sentences(sent_idxs)
            phrase_ids = [get_phrase_id(phrase) for phrase in sents]
            
            pairs = []
            for sent, phrase_id in zip(sents, phrase_ids):
                sent_label = get_sentiment_label(phrase_id)
                if sent_label != -1: # only include non-neutral samples
                    pairs.append([sent, sent_label])
            
            final_csv = pd.DataFrame(pairs)
            final_csv.columns = ['sentence', 'label']
            final_csv.to_csv(os.path.join(self.raw_dataset_path, f"{split_name}.csv"))
        

        super(SST2, self).process_downloaded_dataset()
