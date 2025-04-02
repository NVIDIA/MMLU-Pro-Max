class Postprocessor:
    """
    Defines a process function that receives a pandas dataframe that must
    contain "generation_key" column that is "generate" by default and returns
    the same pandas dataframe with an added column "answer_key" containing the
    extracted answer from the generation.
    """
    def __init__(self,
                 generation_key="generation",
                 answer_key="extracted_answer",
                 **kwargs):
        self.generation_key = generation_key
        self.answer_key = answer_key
        self.kwargs = kwargs

    def process(self, data):
        return data


class BinaryPostprocessor(Postprocessor):
    def process(self, data):
        def extract_answer(generation):
            answer = generation.split("\n")
            answer = [a for a in answer if a.strip()]
            answer = answer[-1].lower().split("the answer is")[-1]
            if len(answer) == 1:
                answer = answer.lower().split("the answer provided is")[-1]
            if len(answer.split()) < 1:
                return "unparsed"
            answer = "".join([c for c in answer.split()[0] if c.isalpha()])
            return answer

        if self.answer_key in data:
            raise ValueError(f"Key {self.answer_key} already exists in data, the answers are already extracted.")

        data[self.answer_key] = data[self.generation_key].apply(extract_answer)
        return data


class MCQPostprocessor(Postprocessor):
    def process(self, data):
        LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

        def fetch_final_answer_MCQ(model_answer):
            split_strings = ["the answer is", "the best answer is", "the final answer is"]
            for split_by in split_strings:
                if split_by in model_answer.lower():
                    model_answer = model_answer.strip().lower().split(split_by)[-1]
                    for c in ["(", ")", "*", "boxed", "{", "}", "\\", ",", '$', ":", "."]:
                        model_answer = model_answer.replace(c, "").strip()
                    if len(model_answer) == 1:
                        if model_answer.upper() in LABELS:
                            return model_answer.upper()
                        print("101: Answer out of bounds of possible labels A-K.")
                        return model_answer

                    elif len(model_answer) == 0:
                        print("102: Generation ended without providing an answer.")
                        return model_answer
            return model_answer

        if self.answer_key in data:
            raise ValueError(f"Key {self.answer_key} already exists in data, the answers are already extracted.")

        data[self.answer_key] = data[self.generation_key].apply(fetch_final_answer_MCQ)
        return data


class GenPostprocessor(Postprocessor):
    def process(self, data):
        def extract_answer(generation):
            generation = generation.split('boxed{')[-1]
            if generation.endswith('}$'):
                generation = generation[:-2]
            return generation

        if self.answer_key in data:
            raise ValueError(f"Key {self.answer_key} already exists in data, the answers are already extracted.")

        data[self.answer_key] = data[self.generation_key].apply(extract_answer)
        return data