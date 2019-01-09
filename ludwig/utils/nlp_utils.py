#! /usr/bin/env python
# coding=utf-8
# Copyright 2019 The Ludwig Authors. All Rights Reserved.
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

nlp_pipeline = None
default_characters = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                      'k', 'l', 'm', 'n', 'o', 'p',
                      'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0',
                      '1', '2', '3', '4', '5', '6',
                      '8', '9', '-', ',', ';', '.', '!', '?', ':', '\'', '\'',
                      '/', '\\', '|', '_', '@', '#',
                      '$', '%', '^', '&', '*', '~', '`', '+', '-', '=', '<',
                      '>', '(', ')', '[', ']', '{',
                      '}']
punctuation = {'.', ',', '@', '$', '%', '/', ':', ';', '+', '='}


def load_nlp_pipeline():
    global nlp_pipeline
    if nlp_pipeline is None:
        logging.info('Loading NLP pipeline')
        try:
            import en_core_web_sm
        except FileNotFoundError:
            logging.error("Unable to load spacy model en_core_web_sm. "
                          "Make sure to download it with: "
                          "python -m spacy download en")
        nlp_pipeline = en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])
    return nlp_pipeline


def pass_filters(
        token,
        filter_numbers=False,
        filter_punctuation=False,
        filter_short_tokens=False,
        filter_stopwords=False
):
    passes_filters = True
    if filter_numbers:
        passes_filters = not token.like_num
    if passes_filters and filter_punctuation:
        passes_filters = not bool(set(token.orth_) & punctuation)
    if passes_filters and filter_short_tokens:
        passes_filters = len(token) > 2
    if passes_filters and filter_stopwords:
        passes_filters = not token.is_stop
    return passes_filters


def process_text(
        text,
        nlp_pipeline,
        return_lemma=False,
        filter_numbers=False,
        filter_punctuation=False,
        filter_short_tokens=False,
        filter_stopwords=False
):
    doc = nlp_pipeline.tokenizer(text)
    return [token.lemma_ if return_lemma else token.text
            for token in doc if pass_filters(token,
                                             filter_numbers,
                                             filter_punctuation,
                                             filter_short_tokens,
                                             filter_stopwords)
            ]


# class Lemmatizer(object):
#     def __init__(self):
#         self.pipeline = load_nlp_pipeline()
#
#     def __call__(self, doc):
#         tokens = [t.lemma_ for t in self.pipeline(doc)
#                   if not t.like_num and len(t) > 2 and '.' not in t.orth_ and '/' not in t.orth_]
#         return tokens
#
#
# class Vectorizer():
#     def __init__(self, weighting, min_df=0.005, max_df=0.995):
#         self.tvectorizer = TfidfVectorizer(analyzer='word',
#                                            min_df=min_df,
#                                            max_df=max_df,
#                                            tokenizer=Lemmatizer(),
#                                            ngram_range=(1, 1))
#
#     def fit_transform(self, docs):
#         return self.tvectorizer.fit_transform(docs).toarray()
#
#     def transform(self, docs):
#         return self.tvectorizer.transform(docs).toarray()

if __name__ == '__main__':
    text = "Hello John, how are you doing my good old friend? Are you still number 732 in the list? Did you pay $32.43 or 54.21 for the book?"
    print(process_text(text, load_nlp_pipeline()))
    print(process_text(text, load_nlp_pipeline(),
                       filter_numbers=True,
                       filter_punctuation=True,
                       filter_short_tokens=True))
    print(process_text(text, load_nlp_pipeline(),
                       filter_stopwords=True))
    print(process_text(text, load_nlp_pipeline(),
                       return_lemma=True))
    print(process_text(text, load_nlp_pipeline(),
                       return_lemma=True,
                       filter_numbers=True,
                       filter_punctuation=True,
                       filter_short_tokens=True))
    print(process_text(text, load_nlp_pipeline(),
                       return_lemma=True,
                       filter_stopwords=True))
