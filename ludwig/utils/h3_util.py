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
import json

from gmpy import mpz


def extract_bits(number, position, num_bits, start_from='left'):
    '''
    Function to extract num_bits bits from a given
    position in a number
    '''
    # convert number into binary first
    if not isinstance(number, str):  # it is str if called bin on it before
        binary = bin(number)
    else:
        binary = number
    # print('extract_bits before:', binary)

    # remove first two characters
    binary = binary[2:]

    if start_from == 'left':
        start = position
        end = position + num_bits
    else:  # if == 'right'
        end = len(binary) - position
        start = end - num_bits

    # extract k  bit sub-string
    bit_substring = binary[start:end]

    # print('extract_bits after:', bit_substring)

    # convert extracted sub-string into decimal again
    return int(bit_substring, 2)


def set_bits(number, position, num_bits, value, start_from='left', bits=64):
    '''
    Function to extract num_bits bits from a given
    position in a number
    '''
    # convert number into binary first
    if not isinstance(number, str):  # is str if previously binarized
        number = mpz(number).setbit(bits)
    binary = '0b' + bin(number)[-64:]

    # this is needed because mpz adds a 1 at the beginning
    # print('set_bits before:', binary)

    # remove first two characters
    binary = binary[2:]

    if start_from == 'left':
        start = position
        end = position + num_bits
    else:  # if == 'right'
        end = len(binary) - position
        start = end - num_bits

    # extract k  bit sub-string
    bin_value = bin(value)[2:]
    replacement = bin_value[max(0, len(bin_value) - (end - start)):]
    if len(replacement) < num_bits:
        pad = ''.join(['0' for _ in range(num_bits - len(replacement))])
        replacement = pad + replacement
    # print(replacement)
    binary = '0b' + binary[0:start] + replacement + binary[end:]

    # print('set_bits after:', binary)

    # convert into decimal again
    return int(binary, 2)


def h3_to_components(h3_value, mode='int'):
    '''
    Extract the values from an H3 hexadecimal value
    Refer to this for the bit layout:
    https://uber.github.io/h3/#/documentation/core-library/h3-index-representations
    '''
    # lat_long = (0, 0)  # h3ToGeo(h3_value)

    # from hex to integer
    if mode == 'hex':
        h3_value = int(h3_value, 16)
    # else assumes h3_value is a 64bit integer

    number = mpz(h3_value).setbit(64)
    binary = '0b' + bin(number)[-64:]

    mode = extract_bits(binary, 1, 4, start_from='left')
    edge = extract_bits(binary, 5, 3, start_from='left')
    resolution = extract_bits(binary, 8, 4, start_from='left')
    base_cell = extract_bits(binary, 12, 7, start_from='left')
    cells = []
    start = 19
    for _ in range(resolution):
        cells.append(extract_bits(binary, start, 3, start_from='left'))
        start += 3

    return {'mode': mode, 'edge': edge, 'resolution': resolution,
            'base_cell': base_cell, 'cells': cells}
    # 'lat_long': lat_long}


def components_to_h3(h3_components, output_mode='int'):
    number = mpz(0).setbit(64)

    number = set_bits(number, 1, 4, h3_components['mode'], start_from='left')
    number = set_bits(number, 5, 3, h3_components['edge'], start_from='left')
    number = set_bits(number, 8, 4, h3_components['resolution'],
                      start_from='left')
    number = set_bits(number, 12, 7, h3_components['base_cell'],
                      start_from='left')
    start = 19
    for cell in h3_components['cells']:
        number = set_bits(number, start, 3, cell, start_from='left')
        start += 3
    for _ in range(15 - len(h3_components['cells'])):
        number = set_bits(number, start, 3, 7, start_from='left')
        start += 3

    if output_mode == 'hex':
        number = hex(number)
    else:
        number = int(number)

    return number


if __name__ == "__main__":
    value = 171
    print(bin(value))

    print()
    print('Extract')
    position = 2
    num_bits = 5
    print('position right:', position)
    print('num_bits:', num_bits)
    print(extract_bits(value, position, num_bits, 'right'))
    print()
    position = 3
    num_bits = 5
    print('position left:', position)
    print('num_bits:', num_bits)
    print(extract_bits(value, position, num_bits, 'left'))

    print()
    print('Set')
    position = 2
    num_bits = 5
    replacement = 7
    print('position right:', position)
    print('num_bits:', num_bits)
    print('replacement:', replacement, bin(replacement))
    print(set_bits(value, position, num_bits, replacement))
    print()
    position = 3
    num_bits = 5
    print('position left:', position)
    print('num_bits:', num_bits)
    print('replacement:', replacement, bin(replacement))
    print(set_bits(value, position, num_bits, replacement, start_from='left'))

    print()
    print()
    print('H3 to component')
    value = 622236723497533439
    print(h3_to_components(value, mode='int'))
    print(h3_to_components(hex(value), mode='hex'))

    print()
    print()
    print('Component to H3')
    components = h3_to_components(value, mode='int')
    print(components)
    print(components_to_h3(components, output_mode='int'))
    print(components_to_h3(components, output_mode='hex'))

    print()
    print()
    print('Test')
    gt_value = 622236723497533439
    gt_components = {'mode': 1, 'edge': 0, 'resolution': 10, 'base_cell': 21,
                     'cells': [0, 2, 0, 0, 3, 2, 6, 2, 1, 3]}
    print('value:', gt_value, bin(gt_value))
    components = h3_to_components(gt_value, mode='int')
    gt_components_dump = json.dumps(gt_components, sort_keys=True, indent=2)
    components_dump = json.dumps(components, sort_keys=True, indent=2)
    print('components matches gt_components:',
          components_dump == gt_components_dump)
    value = components_to_h3(components, output_mode='int')
    print('values matches gt_value:', value == gt_value)

    print()
    print()
    print('Test 2')
    gt_value = 622580942897840127
    gt_components = {'mode': 1, 'edge': 0, 'resolution': 10, 'base_cell': 30,
                     'cells': [6, 4, 1, 0, 6, 5, 1, 2, 6, 4]}
    print('value:', gt_value, bin(gt_value))
    components = h3_to_components(gt_value, mode='int')
    gt_components_dump = json.dumps(gt_components, sort_keys=True, indent=2)
    components_dump = json.dumps(components, sort_keys=True, indent=2)
    print('components matches gt_components:',
          components_dump == gt_components_dump)
    value = components_to_h3(components, output_mode='int')
    print('values matches gt_value:', value == gt_value)

    print()
    print()
    print('Default value')
    components = {'mode': 1, 'edge': 0, 'resolution': 0, 'base_cell': 0,
                  'cells': []}
    print(components_to_h3(components, output_mode='int'))
