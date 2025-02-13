Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under MIT Licence.

 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

# MMLU-Pro-Robust
Loads and processes the MMLU-Pro dataset from HuggingFace Datasets: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) to create the MMLU-Pro-Max.

To create MMLU-Pro-Max manifests, run
```bash
python convert_and_save.py --save_dir=SAVE_DIR
```
This will create 7 manifests where 'input' field should be used for inference. Notes about the data processing classes. </br>
## MMLUProMax Converters

Explore various ways to analyze and manipulate prompts for evaluating model performance. Below are the available converters:

- **<span style="color:teal;">MMLUProMaxMultiPromptConverter()</span>**
  *Does prompt matter?* Use 10 prompts to analyze accuracy variation.

- **<span style="color:teal;">MMLUProMaxChoiceOrderConverter()</span>**
  *Does the correct choice position matter?* The correct answer is always option A, option B, ... up to option J.

- **<span style="color:teal;">MMLUProMaxNoCorrectAnswerConverter(replace=True)</span>**
  *What if there is no correct answer?* Replace the correct answer with `"None of the answers is correct."`.

- **<span style="color:teal;">MMLUProMaxNoCorrectAnswerConverter(replace=False, random_loc=True)</span>**
  *What if there is no correct answer?* Remove the correct answer and randomly insert `"None of the answers is correct."`.

- **<span style="color:teal;">MMLUProMaxDropCorrectAnswerConverter()</span>**
  *What if there is no correct answer?* Drop the correct answer entirely.

- **<span style="color:teal;">MMLUProMaxBinaryConverter()</span>**
  *Can the model distinguish between correct and incorrect choices?* Convert a single question into multiple questions like:
  `"For the question, is this answer correct?"`.

- **<span style="color:teal;">MMLUProMaxGenerativeConverter()</span>**
  *Does performance in MCQ format transfer to real-life?* Leave questions that can be answered without options and directly ask the question.
