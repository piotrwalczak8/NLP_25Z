# src/models/llama.py
from transformers import pipeline
import numpy as np
import re

class LlamaZeroShot:
    def __init__(self, model_name: str, max_new_tokens: int, device_map: str, batch_size: int, few_shot_examples=None):
        """
        model_name: nazwa modelu LLaMA
        max_new_tokens: maksymalna liczba tokenów do wygenerowania
        device_map: "auto" lub konkretne urządzenie np. "cuda:0"
        batch_size: liczba tekstów przetwarzanych jednocześnie
        few_shot_examples: lista przykładowych tekstów z etykietami, np. [(tekst, etykieta), ...]
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.batch_size = batch_size
        self.pipe = None
        self.few_shot_examples = few_shot_examples or []

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

        prompt = (
            "Classify the following text as exactly one of: Positive, Negative."

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

    @staticmethod
    def _parse_output(out: str) -> int:
        """
        Parsuje TYLKO ostatnią odpowiedź modelu
        """
        # weź ostatnie wystąpienie "sentiment:"
        matches = re.findall(r"(positive|negative)", out, re.IGNORECASE)

        if not matches:
            return 0  # fallback: negative

        label = matches[-1].lower()
        if label == "positive":
            return 2
        #if label == "neutral":
        #    return 1
        return 0
