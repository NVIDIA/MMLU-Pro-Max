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

from abc import abstractmethod
import os
from datetime import datetime
from datasets import load_dataset
import numpy as np
import json

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
NO_ANSWER_MESSAGE = "None of the answers is correct."


class MMLUProRobustConverterABC:
    def load_base_dataset(self, **kwargs):
        return load_dataset(path="TIGER-Lab/MMLU-Pro",
                            split="test", **kwargs)

    @abstractmethod
    def _process_data(self, data, **kwargs):
        pass

    @abstractmethod
    def _add_prompt(self):
        pass

    def convert(self, **kwargs):
        base = self.load_base_dataset()
        processed = self._process_data(base, **kwargs)
        processed = self._add_prompt(processed)
        return processed

    def convert_and_save(self, save_dir, **kwargs):
        processed = self.convert(**kwargs)
        class_name = self.__class__.__name__.replace('Converter', '')
        file_name = class_name + "_" + datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        path = f"{save_dir}/{file_name}.jsonl"
        # make directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for datapoint in processed:
                f.write(json.dumps(datapoint) + "\n")
        return processed


class MMLUProRobustNoAnswerConverter(MMLUProRobustConverterABC):
    PROMPT = "The following is a multiple-choice question " \
                "about {category}. Think step by step and then finish " \
                "your answer with \"The answer is (X)\" where X is " \
                "the correct letter choice. \n" \
                "Question: {question}\n" \
                "Options:\n{options}\n" \
                "Answer: Let’s think step by step. "

    def _process_data(self,
                      base,
                      no_answer_message=NO_ANSWER_MESSAGE,
                      replace=False,
                      random_loc=True):

        def __process_replace(datapoint):
            datapoint["options"][datapoint["answer_index"]] = no_answer_message
            return datapoint

        def __process_random_loc(datapoint):
            if random_loc:
                ind = np.random.randint(0, len(datapoint["options"]))
            else:
                ind = len(datapoint["options"]) - 1

            datapoint["options"].insert(ind, no_answer_message)

            if ind <= datapoint["answer_index"]:
                datapoint["answer_index"] += 1
                datapoint["answer"] = LABELS[datapoint["answer_index"]]
            return datapoint

        if replace:
            return base.map(__process_replace)
        else:
            return base.map(__process_random_loc)

    def _add_prompt(self, data):
        def __inner_add_prompt(datapoint):
            options = [f"{label}. {option}"
                       for label, option in zip(LABELS, datapoint["options"])]
            prompt = self.PROMPT.format(
                category=datapoint["category"],
                question=datapoint["question"],
                options="\n".join(options)
                )
            datapoint["prompt"] = prompt
            return datapoint

        return data.map(__inner_add_prompt)


class MMLUProRobustBinaryConverter(MMLUProRobustConverterABC):
    PROMPT = "The following is a question with it's answer " \
            "about {category}. You need to determine if the " \
            "provided answer is correct or not. "\
            "Think step by step and then finish " \
            "your answer with \"The answer is (X)\" where X is " \
            "either \"correct\" or \"incorrect\".\n" \
            "Question: {question}\n" \
            "Answer: {answer}\n" \
            "Let’s think step by step. "

    def _process_data(self, base):
        def __process(batched_docs):
            initial_len = len(next(iter(batched_docs.values())))
            keys = list(batched_docs.keys())
            new_batched_docs = {key: [] for key in keys}
            new_batched_docs["option_ind"] = []
            new_batched_docs["binary_answer"] = []

            for doc_ind in range(initial_len):
                for option_ind in range(len(batched_docs["options"][doc_ind])):
                    new_batched_docs["option_ind"].append(option_ind)
                    new_batched_docs["binary_answer"].append(
                        'correct'
                        if option_ind == batched_docs["answer_index"][doc_ind]
                        else 'incorrect'
                    )

                    for key in keys:
                        new_batched_docs[key].append(
                            batched_docs[key][doc_ind]
                            )
            return new_batched_docs
        return base.map(__process, batched=True)

    def _add_prompt(self, data):
        def __inner_add_prompt(datapoint):
            prompt = self.PROMPT.format(
                category=datapoint["category"],
                question=datapoint["question"],
                answer=datapoint["options"][datapoint["option_ind"]]
                )
            datapoint["prompt"] = prompt
            return datapoint

        return data.map(__inner_add_prompt)
