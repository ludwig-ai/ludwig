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
import argparse
import random


def split(input_path, output1, output2, split):
    with open(input_path) as file:
        lines = file.readlines()

    random.shuffle(lines)
    split_idx = int(len(lines) * split)

    with open(output1, "w") as f:
        for line in lines[:split_idx]:
            line = line if line.endswith("\n") else line + "\n"
            f.write(line)

    with open(output2, "w") as f:
        for line in lines[split_idx:]:
            line = line if line.endswith("\n") else line + "\n"
            f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a file based on its lines")

    parser.add_argument("-i", "--input", required=True, help="input file names")
    parser.add_argument("-o1", "--output1", required=True, help="output 1 file name")
    parser.add_argument("-o2", "--output2", required=True, help="output 2 file name")
    parser.add_argument("-s", "--split", required=True, type=float, default=0.8, help="percentage of the split")

    args = parser.parse_args()

    split(args.input, args.output1, args.output2, args.split)
