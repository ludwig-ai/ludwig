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
import math

import numpy as np

rng = np.random.RandomState()


def cartesian(s0, s1, reverse_order=False):
    ns0 = s0.shape[0]
    ns1 = s1.shape[0]
    if reverse_order:
        rs0 = np.tile(s0, [ns1, 1])
        rs1 = np.repeat(s1, ns0, axis=0)
    else:
        rs0 = np.repeat(s0, ns1, axis=0)
        rs1 = np.tile(s1, [ns0, 1])

    out = np.concatenate([rs0, rs1], axis=1)
    return out


# http://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
# (called cartesian_product2 there)
def cartesian1(arrays):
    '''cartesian of arb amount of 1d arrays'''
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def coord_grid(dim, res):
    return cartesian1([np.linspace(-1, 1, res)] * dim)


def prob_invert(x, p=0.5):
    assert x.max() <= 1., 'expects binary input'
    assert x.min() >= 0., 'expects binary input'
    invert = rng.rand(x.shape[0]) < p
    for ix, x0 in enumerate(x):
        x[ix] = x0 < 0.5 if invert[ix] else x0
    return x, invert.astype('float32')


def onehot(n, ii):
    return np.eye(n)[ii]


def unique_rows(a):
    # taken from http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    b = np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]


def get_range(sets, keys, ignore_min=False):
    low = float('inf')
    high = float('-inf')

    if not isinstance(keys, list):
        keys = [keys]

    for key in keys:
        for ii in range(len(sets)):
            low = min(low, int(sets[ii][key].min()))
            high = max(high, int(sets[ii][key].max()))

    # return the dif, 0 indexed so we good with range here
    if ignore_min:
        return int(high)
    else:
        return int(high) - int(low)


def radial_dist(arr):
    x, y = np.expand_dims(arr[:, 0], axis=0), np.expand_dims(arr[:, 1], axis=0)
    x2, y2 = np.square(x), np.square(y)
    r = np.sqrt(x2 + y2)
    return np.concatenate((x.T, y.T, r.T), axis=1)


def coord_rect(res1, res2):
    return cartesian1([np.linspace(-1, 1, res1), np.linspace(-1, 1, res2)])


def radial_grid(dim, res):
    return radial_dist(coord_grid(dim, res))


def radial_rect(res1, res2):
    return radial_dist(coord_rect(res1, res2))


def jaccard(sorted_list_1, sorted_list_2):
    max_jaccard_score = 0
    for path1 in sorted_list_1:
        for path2 in sorted_list_2:
            size_set_1 = len(path1)
            size_set_2 = len(path2)

            intersection = 0
            for i in range(min(size_set_1, size_set_2)):
                last_p1 = path1[-(i + 1)]
                last_p2 = path2[-(i + 1)]
                if last_p1 == last_p2:
                    intersection += 1
                else:
                    break

            jaccard_score = intersection / (
                        size_set_1 + size_set_2 - intersection)
            if jaccard_score > max_jaccard_score:
                max_jaccard_score = jaccard_score

    return max_jaccard_score


def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


def int_type(num_distinct):
    if num_distinct < 128:
        return np.int8
    elif num_distinct < 32768:
        return np.int16
    elif num_distinct < 2147483648:
        return np.int32
    else:
        return np.int64


def convert_size(size_bytes):
    if size_bytes == 0:
        return '0B'
    size_name = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return '{} {}'.format(s, size_name[i])
