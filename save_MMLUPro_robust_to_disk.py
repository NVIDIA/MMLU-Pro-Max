# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
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

from MMLUProRobust import MMLUProRobustNoAnswerConverter
from MMLUProRobust import MMLUProRobustBinaryConverter

"""
this is the script that saves the MMLUPro Robust dataset json to disk
to run the script, you need to pass the following arguments:
1. dataset_name: the name of the dataset e.g. NoAnswer, Binary
2. save_dir: the directory to save the dataset json
3. replace: True or False, if NoAnswer was passed as dataset. Deaults to False
4. random_loc: True or False, if NoAnswer was passed as dataset. Defaults to True
5. no_answer_message: the message to replace the correct answer with
"""


def save_MMLUPro_robust(dataset_name, save_dir, **kwargs):
    if dataset_name == "NoAnswer":
        dataset = MMLUProRobustNoAnswerConverter()
    elif dataset_name == "Binary":
        dataset = MMLUProRobustBinaryConverter()
    else:
        raise ValueError("Invalid dataset name. Choose from NoAnswer, Binary")

    dataset.convert_and_save(save_dir, **kwargs)
    print(f"Dataset saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--replace", type=bool, required=False)
    parser.add_argument("--random_loc", type=bool, required=False)
    parser.add_argument("--no_answer_message", type=str, required=False)
    args = parser.parse_args()

    kwargs = {}
    if args.replace is not None:
        kwargs['replace'] = args.replace
    if args.random_loc is not None:
        kwargs['random_loc'] = args.random_loc
    if args.no_answer_message is not None:
        kwargs['no_answer_message'] = args.no_answer_message

    save_MMLUPro_robust(args.dataset_name, args.save_dir, **kwargs)
