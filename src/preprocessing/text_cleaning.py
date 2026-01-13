# src/preprocessing/text_cleaning.py
import re

def clean_text(text: str) -> str:

    text = re.sub(r"<.*?>", "", str(text))
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text.lower().strip()
