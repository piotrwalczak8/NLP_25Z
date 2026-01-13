# src/models/llama.py
from transformers import pipeline
import numpy as np
import re

class LlamaModel:
    def __init__(self, model_name: str, max_new_tokens: int, device_map: str, batch_size: int, few_shot_examples=None, classes=None):

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.batch_size = batch_size
        self.pipe = None
        self.few_shot_examples = few_shot_examples or []
        self.classes = classes or ["Positive", "Negative", "Neutral"]


    def _init_pipe(self):
        if self.pipe is None:
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=self.max_new_tokens,
                device_map=self.device_map
            )

    def _build_prompt(self, text: str):
        """
        Buduje prompt dla pojedynczego tekstu z opcjonalnymi przykładami few-shot
        Zoptymalizowany pod TinyLLaMA (krótki, restrykcyjny, bez gadania)
        """
        class_list = ", ".join(self.classes)
        prompt = (
                f"Classify the sentiment of the following text.\n"
                f"Choose exactly ONE label from: {class_list}.\n"
                f"Respond with ONLY the label name.\n\n"
        )

        if self.few_shot_examples:
            prompt += "Examples:\n"
            for t, l in self.few_shot_examples:
                prompt += f"Text: {t}\nLabel: {l}\n\n"

        prompt += (
            "Now classify the following text.\n"
            f"Text: {text}\n"
            "Label:"
        )

        return prompt


    def classify_single(self, text: str) -> int:
        self._init_pipe()
        prompt = self._build_prompt(text)
        out = self.pipe(prompt)[0]["generated_text"].lower()
        return self._parse_output(out)

    def classify_batch(self, texts):
        self._init_pipe()
        preds = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            prompts = [self._build_prompt(t) for t in batch]

            outs = self.pipe(prompts)  # zwraca listę list/dict
            for out_item in outs:
                if isinstance(out_item, list):
                    out_item = out_item[0]  # weź pierwszy element listy
                text = out_item["generated_text"] if isinstance(out_item, dict) else str(out_item)
                preds.append(self._parse_output(text.lower()))

        return preds


    def _parse_output(self, out: str) -> int:
        out = out.lower()

        # bierzemy TYLKO to, co model wygenerował po ostatnim "label:"
        if "label:" in out:
            out = out.split("label:")[-1]

        out = out.strip()

        for i, c in enumerate(self.classes):
            if out.startswith(c.lower()):
                return i

        return -1



