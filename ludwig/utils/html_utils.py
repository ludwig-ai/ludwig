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
import re
from html.parser import HTMLParser

from ludwig.utils import strings_utils

logger = logging.getLogger(__name__)


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, data):
        self.fed.append(data)

    def get_data(self):
        return "".join(self.fed)

    def error(self, message):
        logger.error(message)


def strip_tags(html):
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_data()


# regular expressions for cleaning text
res_pre = [(re.compile(r"([^.:;\?\!>])(<br/?>)"), r"\1.\2"), (re.compile(r"<br/?>"), r" ")]
res_post = [
    (re.compile(r"[ \t\0]"), r" "),
    (re.compile(r"[–_]"), r"-"),
    (
        re.compile(r"[\’\‘]"),
        r"""),
    (re.compile(r'[”“]]'), r""",
    ),
    (re.compile(r"℅"), r"%"),
    (re.compile(r"([^.>])(<br/?>)"), r"\1.\2"),
    (re.compile(r"\\\\[NnRr]"), r" "),
    (re.compile(r"\\[NnRr]"), r" "),
    (re.compile(r"[\n\r]"), r" "),
    (re.compile(r"\\\\"), r" / "),
    (re.compile(r"<br/?>"), r" "),
    (re.compile(r"\\\\" ""), r"\'"),
    (re.compile(r"^\'([^\']+)$"), r"\1"),
    (re.compile(r"([\<\>\{\}\[\]\(\)\-\+\=:;,\./\?\!\$%&£#@\'₹ ])\1+"), r"\1"),
    (
        re.compile(
            r"[^qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890\<\>\{\}\[\]\(\)\-\+\=:;,\./\?\!\$%&£#@\'₹ ]"  # noqa
        ),
        r" ",
    ),
    (re.compile(r"\s{2,}"), r" "),
]


def clean_html(html_text):
    # print()
    # print(html_text)
    html_text, matched = strings_utils.match_replace(html_text, res_pre)
    # print(html_text)
    html_text = strip_tags(html_text)
    # print(html_text)
    html_text = strings_utils.strip_accents(html_text)
    # print(html_text)
    # result = html_text.strip(
    #     'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890\<\>\{\}\[\]\(\)\-\+\=:;,\./\?\!\$%&€£#@'₹\' ')
    # if result:
    #     print(result)
    html_text, matched = strings_utils.match_replace(html_text, res_post)
    # print(matched)
    # print(html_text)
    return html_text
