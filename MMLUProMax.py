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

from abc import abstractmethod
import copy
import os
from datetime import datetime
from datasets import load_dataset
import numpy as np
import json

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
NO_ANSWER_MESSAGE = "None of the answers is correct."


class MMLUProMaxConverterABC:
    def __init__(self):
        self.prompt_dict = json.load(open("prompts.json", "r"))

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


class MMLUProMaxNoCorrectAnswerConverter(MMLUProMaxConverterABC):
    PROMPT_KEY = "default_prompt"

    def __init__(self, replace=False, random_loc=True):
        super().__init__()
        self.prompt = self.prompt_dict[self.PROMPT_KEY]
        self.replace = replace
        self.random_loc = random_loc

    def _process_data(self,
                      base,
                      no_answer_message=NO_ANSWER_MESSAGE):

        def __process_replace(datapoint):
            datapoint["options"][datapoint["answer_index"]] = no_answer_message
            return datapoint

        def __process_random_loc(datapoint):
            if self.random_loc:
                ind = np.random.randint(0, len(datapoint["options"]))
            else:
                ind = len(datapoint["options"]) - 1

            datapoint["options"].insert(ind, no_answer_message)

            if ind <= datapoint["answer_index"]:
                datapoint["answer_index"] += 1
                datapoint["answer"] = LABELS[datapoint["answer_index"]]
            return datapoint

        if self.replace:
            return base.map(__process_replace)
        else:
            return base.map(__process_random_loc)

    def _add_prompt(self, data):
        def __inner_add_prompt(datapoint):
            options = [f"{label}. {option}"
                       for label, option in zip(LABELS, datapoint["options"])]
            prompt = self.prompt.format(
                category=datapoint["category"],
                question=datapoint["question"],
                options=" ".join(options)
                )
            datapoint["input"] = prompt
            return datapoint

        return data.map(__inner_add_prompt)

    def convert_and_save(self, save_dir, **kwargs):
        processed = self.convert(**kwargs)
        class_name = self.__class__.__name__.replace('Converter', '')
        if self.replace:
            file_name = class_name + "_answer_replace_" + datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        else:
            file_name = class_name + "_" + datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        path = f"{save_dir}/{file_name}.jsonl"
        # make directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for datapoint in processed:
                f.write(json.dumps(datapoint) + "\n")
        return processed


class MMLUProMaxDropCorrectAnswerConverter(MMLUProMaxConverterABC):
    PROMPT_KEY = "correct_answer_dropped_prompt"

    def __init__(self):
        super().__init__()
        self.prompt = self.prompt_dict[self.PROMPT_KEY]

    def _process_data(self,
                      base):

        def __drop_correct_answer(datapoint):
            options = datapoint["options"]
            options.pop(datapoint["answer_index"])
            return datapoint

        return base.map(__drop_correct_answer)

    def _add_prompt(self, data):
        def __inner_add_prompt(datapoint):
            options = [f"{label}. {option}"
                       for label, option in zip(LABELS, datapoint["options"])]
            prompt = self.prompt.format(
                category=datapoint["category"],
                question=datapoint["question"],
                options=" ".join(options)
                )
            datapoint["input"] = prompt
            return datapoint

        return data.map(__inner_add_prompt)


class MMLUProMaxMultiPromptConverter(MMLUProMaxConverterABC):
    PROMPT_KEY = "multiprompt"

    def __init__(self):
        super().__init__()
        self.prompts = self.prompt_dict[self.PROMPT_KEY]

    def _process_data(self, base):
        def __process(batched_docs):
            initial_len = len(next(iter(batched_docs.values())))
            keys = list(batched_docs.keys())
            new_batched_docs = {key: [] for key in keys}
            new_batched_docs["prompt_id"] = []

            for doc_ind in range(initial_len):
                for prompt_ind in range(len(self.prompts)):
                    new_batched_docs["prompt_id"].append(prompt_ind)
                    for key in keys:
                        new_batched_docs[key].append(
                            copy.deepcopy(batched_docs[key][doc_ind])
                            )
            return new_batched_docs
        return base.map(__process, batched=True)

    def _add_prompt(self, data):
        def __inner_add_prompt(datapoint):
            prompt_template = self.prompts[datapoint["prompt_id"]]
            prompt = prompt_template.format(
                category=datapoint["category"],
                question=datapoint["question"],
                answer=datapoint["options"][datapoint["option_ind"]]
                )
            datapoint["input"] = prompt
            return datapoint

        return data.map(__inner_add_prompt)

    def _add_prompt(self, data):
        def __inner_add_prompt(datapoint):
            prompt_template = self.prompts[datapoint["prompt_id"]]
            options = [f"{label}. {option}"
                       for label, option in zip(LABELS, datapoint["options"])]
            prompt = prompt_template.format(
                category=datapoint["category"],
                question=datapoint["question"],
                options=" ".join(options)
                )
            datapoint["input"] = prompt
            return datapoint

        return data.map(__inner_add_prompt)


class MMLUProMaxChoiceOrderConverter(MMLUProMaxConverterABC):
    PROMPT_KEY = "default_prompt"

    def __init__(self):
        super().__init__()
        self.prompt = self.prompt_dict[self.PROMPT_KEY]

    def _process_data(self, base):

        def repeat_doc_swap_correct_answer(batched_docs):
            initial_len = len(next(iter(batched_docs.values())))
            keys = list(batched_docs.keys())
            new_batched_docs = {key: [] for key in keys}
            new_batched_docs["always_same_option"] = []
            new_batched_docs["original_answer_index"] = []

            for doc_ind in range(initial_len):
                labels = LABELS[:len(batched_docs["options"][doc_ind])]
                for label_ind, label in enumerate(labels):
                    new_batched_docs["original_answer_index"].append(
                        batched_docs["answer_index"][doc_ind]
                    )
                    for key in keys:
                        new_batched_docs[key].append(copy.deepcopy(batched_docs[key][doc_ind]))
                        if key == "options":
                            # Swap correct answer with label_ind option
                            correct_answer = batched_docs["options"][doc_ind][batched_docs["answer_index"][doc_ind]]
                            replacement = batched_docs["options"][doc_ind][label_ind]

                            new_batched_docs["options"][-1][label_ind] = correct_answer
                            new_batched_docs["options"][-1][batched_docs["answer_index"][doc_ind]] = replacement

                        if key == "answer_index":
                            new_batched_docs[key][-1] = label_ind

                        if key == "answer":
                            new_batched_docs[key][-1] = label

                    new_batched_docs["always_same_option"].append(label)
            return new_batched_docs

        return base.map(repeat_doc_swap_correct_answer, batched=True)

    def _add_prompt(self, data):
        def __inner_add_prompt(datapoint):
            options = [f"{label}. {option}"
                       for label, option in zip(LABELS, datapoint["options"])]
            prompt = self.prompt.format(
                category=datapoint["category"],
                question=datapoint["question"],
                options=" ".join(options)
                )
            datapoint["input"] = prompt
            return datapoint

        return data.map(__inner_add_prompt)


class MMLUProMaxBinaryConverter(MMLUProMaxConverterABC):
    PROMPT_KEY = "binary_prompt"

    def __init__(self):
        super().__init__()
        self.prompt = self.prompt_dict[self.PROMPT_KEY]

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
                            copy.deepcopy(batched_docs[key][doc_ind])
                            )
            return new_batched_docs
        return base.map(__process, batched=True)

    def _add_prompt(self, data):
        def __inner_add_prompt(datapoint):
            prompt = self.prompt.format(
                category=datapoint["category"],
                question=datapoint["question"],
                answer=datapoint["options"][datapoint["option_ind"]]
                )
            datapoint["input"] = prompt
            return datapoint

        return data.map(__inner_add_prompt)


