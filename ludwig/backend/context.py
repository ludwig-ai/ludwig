#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2020 Uber Technologies, Inc.
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

import threading


class Context(object):
    """Functionality for objects that put themselves in a context using
    the `with` statement.

    Adapted from: https://github.com/pymc-devs/pymc3/blob/10c9330e4c55e7c6c0b79dde47c498cdf637df02/pymc3/model.py#L149
    """
    contexts = threading.local()

    def __init__(self, cls):
        self.cls = cls

    def __enter__(self):
        self.cls.get_contexts().append(self)

    def __exit__(self, typ, value, traceback):
        self.cls.get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        # no race-condition here, cls.contexts is a thread-local object
        # be sure not to override contexts in a subclass however!
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []
        return cls.contexts.stack

    @classmethod
    def get_context(cls):
        """Return the deepest context on the stack.

        Raise `IndexError` if no context is on the stack.
        """
        return cls.get_contexts()[-1]
