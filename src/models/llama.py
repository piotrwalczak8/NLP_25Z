# src/models/llama.py
from transformers import pipeline
import numpy as np

class LlamaZeroShot:
    def __init__(self, model_name: str, max_new_tokens: int = 32, device_map="auto"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.pipe = None

    def _init_pipe(self):
        if self.pipe is None:
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=self.max_new_tokens,
                device_map=self.device_map
            )

    def classify_single(self, text: str) -> int:
        self._init_pipe()
        prompt = (
            "Classify the sentiment of the following text as positive, neutral, or negative.\n"
            f"Text: {text}\nAnswer with exactly one of: Positive, Neutral, Negative.\nSentiment:"
        )
        out = self.pipe(prompt)[0]["generated_text"].lower()
        return self._parse_output(out)

    def classify_batch(self, texts, batch_size=8):
        self._init_pipe()
        preds = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prompts = [
                "Classify the sentiment of the following text as positive, neutral, or negative.\n"
                f"Text: {t}\nAnswer with exactly one of: Positive, Neutral, Negative.\nSentiment:"
                for t in batch
            ]

            outs = self.pipe(prompts)  # zwraca listę list
            for out_item in outs:
                # out_item może być listą lub dict
                if isinstance(out_item, list):
                    out_item = out_item[0]  # weź pierwszy element listy
                # out_item teraz powinien być dict lub string
                text = out_item["generated_text"] if isinstance(out_item, dict) else str(out_item)
                preds.append(self._parse_output(text.lower()))

        return preds

    @staticmethod
    def _parse_output(out: str) -> int:
        if "positive" in out:
            return 2
        if "neutral" in out:
            return 1
        return 0
