# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import argparse
from MMLUProMax import (
    MMLUProMaxNoCorrectAnswerConverter,
    MMLUProMaxBinaryConverter,
    MMLUProMaxMultiPromptConverter,
    MMLUProMaxChoiceOrderConverter,
    MMLUProMaxDropCorrectAnswerConverter,
    MMLUProMaxGenerativeConverter
)

ALL_TASKS = [
    MMLUProMaxDropCorrectAnswerConverter(),
    MMLUProMaxNoCorrectAnswerConverter(replace=False, random_loc=True),
    MMLUProMaxNoCorrectAnswerConverter(replace=True),
    MMLUProMaxBinaryConverter(),
    MMLUProMaxMultiPromptConverter(),
    MMLUProMaxChoiceOrderConverter(),
    MMLUProMaxGenerativeConverter()
]


def save_MMLUPro_MAX(save_dir, **kwargs):
    for task in ALL_TASKS:
        task.convert_and_save(save_dir, **kwargs)
        print(f"Modified dataset {task.__class__.__name__} saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    save_MMLUPro_MAX(args.save_dir)
