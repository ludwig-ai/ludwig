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

logger = logging.getLogger(__name__)

nlp_pipelines = {
    'en': None,
    'it': None,
    'es': None,
    'de': None,
    'fr': None,
    'pt': None,
    'nl': None,
    'el': None,
    'nb': None,
    'lt': None,
    'da': None,
    'pl': None,
    'ro': None,
    'ja': None,
    'zh': None,
    'xx': None
}
language_module_registry = {
    'en': 'en_core_web_sm',
    'it': 'it_core_news_sm',
    'es': 'es_core_news_sm',
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm',
    'pt': 'pt_core_news_sm',
    'nl': 'nl_core_news_sm',
    'el': 'el_core_news_sm',
    'nb': 'nb_core_news_sm',
    'lt': 'lt_core_news_sm',
    'da': 'da_core_news_sm',
    'pl': 'pl_core_news_sm',
    'ro': 'ro_core_news_sm',
    'ja': 'ja_core_news_sm',
    'zh': 'zh_core_web_sm',
    'xx': 'xx_ent_wiki_sm'
}
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


def load_nlp_pipeline(language='xx'):
    if language not in language_module_registry:
        logger.error(
            'Language {} is not supported.'
            'Suported languages are: {}'.format(
                language,
                language_module_registry.keys()
            ))
        raise ValueError
    else:
        spacy_module_name = language_module_registry[language]
    global nlp_pipelines
    if nlp_pipelines[language] is None:
        logger.info('Loading NLP pipeline')
        try:
            import spacy
        except ImportError:
            logger.error(
                ' spacy is not installed. '
                'In order to install all text feature dependencies run '
                'pip install ludwig[text]'
            )
            sys.exit(-1)

        try:
            nlp_pipelines[language] = spacy.load(
                spacy_module_name,
                disable=['parser', 'tagger', 'ner']
            )
        except OSError:
            logger.info(
                ' spaCy {} model is missing, downloading it '
                '(this will only happen once)'
            )
            from spacy.cli import download
            download(spacy_module_name)
            nlp_pipelines[language] = spacy.load(
                spacy_module_name,
                disable=['parser', 'tagger', 'ner']
            )

    return nlp_pipelines[language]


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


if __name__ == '__main__':
    text = 'Hello John, how are you doing my good old friend? Are you still number 732 in the list? Did you pay $32.43 or 54.21 for the book?'
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
