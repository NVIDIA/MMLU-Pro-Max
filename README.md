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
Scripts to process MMLU-Pro (HF link: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)) and create the MMLU-Pro-Max.

To create MMLU-Pro-Max manifests, run
```bash
python convert_and_save.py --save_dir=SAVE_DIR
```

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

# Running Inference
<br> The **<span style="color:teal;">input</span>** field of manifests should be used for inference. Only the chat template of respective model should be added before inference.</br>
We recommend using [Nemo Skills](https://github.com/NVIDIA/NeMo-Skills/tree/main).

```bash
ns generate \
       --cluster=cluster_config_path \
       --server_type=vllm \
       --model=model_path \
       --server_gpus=1 \
       --server_nodes=1 \
       --output_dir=output_dir \
       ++input_file=path_to_manifest \
       ++prompt_config=path_to_empty_prompt.yaml \
       ++prompt_template=model_chat_template \
       ++inference.tokens_to_generate=2048 \
       ++batch_size=512 \
       ++skip_filled=True
```

# Postprocessing
We provide scripts to postprocess model predictions and extract final answer.
 - **<span style="color:teal;">MCQPostprocessor()</span>** Can be used to extract answer from MCQ datasets
 - **<span style="color:teal;">BinaryPostprocessor()</span>** Can be used to extract answer from Binary evaluatin prediction
 - **<span style="color:teal;">GenPostprocessor()</span>** Can be used to extract answer from generative evaluation prediction

# Generative Evaluatin and LLM-as-a-Judge
Pipeline for Generative Evaluation
1. Run inference on corresponding manifest
2. Use postprocessors.GenPostprocessor to extract prediction
3. Run inference on the judge model using **<span style="color:teal;">./data/judge.yaml</span>** prompt. We recommend using [Qwen-2.5 72B](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) as a judge.
For running the judge using Nemo_Skills use
```bash
ns generate \
        --generation_type=math_judge \
        --cluster=cluster_config_path \
        --model=/hf_models/Qwen2.5-72B-Instruct \
        --server_type=vllm \
        --server_gpus=8 \
        --server_nodes=1 \
        --output_dir=output_dir \
        ++input_dir=path_to_mainest \
        ++prompt_config=path_to_judge.yaml \
        ++prompt_template=nemo_skills/prompt/template/qwen-instruct.yaml \
        ++inference.tokens_to_generate=2048 \
        ++batch_size=512 \
        ++skip_filled=True
```