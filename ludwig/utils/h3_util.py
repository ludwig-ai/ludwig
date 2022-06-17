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
from typing import List, NamedTuple


class H3Data(NamedTuple):
    mode: int
    edge: int
    resolution: int
    base_cell: int
    cells: List[int]


def set_bit(v, index, x):
    """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
    mask = 1 << index  # Compute mask, an integer with just bit 'index' set.
    v &= ~mask  # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask  # If x was True, set the bit indicated by the mask.
    return v  # Return the result, we're done.


def set_bits(v, start_bit, slice_length, x):
    bin_x = bin(x)
    for i, index in enumerate(range(start_bit, start_bit + slice_length)):
        val = int(bin_x[-(i + 1)]) if 2 + i < len(bin_x) else 0
        v = set_bit(v, index, val)
    return v


def components_to_h3(components):
    h3 = 18446744073709551615
    h3 = set_bits(h3, 64 - 5, 4, components["mode"])
    h3 = set_bits(h3, 64 - 8, 3, components["edge"])
    h3 = set_bits(h3, 64 - 12, 4, components["resolution"])
    h3 = set_bits(h3, 64 - 19, 7, components["base_cell"])
    for i, cell in enumerate(components["cells"]):
        h3 = set_bits(h3, 64 - 19 - (i + 1) * 3, 3, cell)
    h3 = set_bits(h3, 64 - 1, 4, 0)
    return h3


def bitslice(x: int, start_bit: int, slice_length: int) -> int:
    ones_mask: int = int(2**slice_length - 1)
    return (x & (ones_mask << start_bit)) >> start_bit


def h3_index_mode(h3_long: int) -> int:
    return bitslice(h3_long, 64 - 5, 4)


def h3_edge(h3_long: int) -> int:
    return bitslice(h3_long, 64 - 8, 3)


def h3_resolution(h3_long: int) -> int:
    return bitslice(h3_long, 64 - 12, 4)


def h3_base_cell(h3_long: int) -> int:
    return bitslice(h3_long, 64 - 19, 7)


def h3_octal_components(h3_long):
    res = h3_resolution(h3_long)
    return "{0:0{w}o}".format(bitslice(h3_long + 2**63, 64 - 19 - 3 * res, 3 * res), w=res)


def h3_component(h3_long: int, i: int) -> int:
    return bitslice(h3_long, 64 - 19 - 3 * i, 3)


def h3_components(h3_long: int) -> List[int]:
    return [h3_component(h3_long, i) for i in range(1, h3_resolution(h3_long) + 1)]


def h3_to_components(h3_value: int) -> H3Data:
    """Extract the values from an H3 hexadecimal value Refer to this for the bit layout:

    https://uber.github.io/h3/#/documentation/core-library/h3-index-representations
    """
    # lat_long = (0, 0)  # h3ToGeo(h3_value)
    return H3Data(
        mode=h3_index_mode(h3_value),
        edge=h3_edge(h3_value),
        resolution=h3_resolution(h3_value),
        base_cell=h3_base_cell(h3_value),
        cells=h3_components(h3_value),
    )


if __name__ == "__main__":
    value = 622236723497533439
    components = h3_to_components(value)
    h3 = components_to_h3(components)
    components2 = h3_to_components(h3)
    print(value)
    print(components)
    print(h3)
    print(components2)
