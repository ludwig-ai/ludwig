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
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WithTimer(object):
    def __init__(self, title='', quiet=False):
        self.title = title
        self.quiet = quiet

    def elapsed(self):
        return time.time() - self.wall, time.clock() - self.proc

    def enter(self):
        """Manually trigger enter"""
        self.__enter__()

    def __enter__(self):
        self.proc = time.clock()
        self.wall = time.time()
        return self

    def __exit__(self, *args):
        if not self.quiet:
            elapsed_wp = self.elapsed()
            logger.info(
                'Elapsed {}: wall {:.06f}, sys {:.06f}'.format(self.title,
                                                               elapsed_wp[0],
                                                               elapsed_wp[1]))


class Timer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._proc = time.clock()
        self._wall = time.time()

    def elapsed(self):
        return self.wall(), self.proc()

    def elapsed_str(self):
        return strdelta(self.wall() * 1000.0), strdelta(self.proc() * 1000.0)

    def wall(self):
        return time.time() - self._wall

    def proc(self):
        return time.clock() - self._proc

    def tic(self):
        """Like Matlab tic/toc for wall time and processor time"""
        self.reset()

    def toc(self):
        """Like Matlab tic/toc for wall time"""
        return self.wall()

    def tocproc(self):
        """Like Matlab tic/toc, but for processor time"""
        return self.proc()


def timestamp():
    return '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())


def strdelta(tdelta):
    if isinstance(tdelta, (int, float)):
        tdelta = timedelta(milliseconds=tdelta)
    d = {'D': tdelta.days}
    d['H'], rem = divmod(tdelta.seconds, 3600)
    d['M'], d['S'] = divmod(rem, 60)
    d['f'] = str(tdelta.microseconds)[0:4]
    if d['D'] > 0:
        t = '{D}d {H}h {M}m {S}.{f}s'
    elif d['H'] > 0:
        t = '{H}h {M}m {S}.{f}s'
    elif d['M'] > 0:
        t = '{M}m {S}.{f}s'
    else:
        t = '{S}.{f}s'
    return t.format(**d)
