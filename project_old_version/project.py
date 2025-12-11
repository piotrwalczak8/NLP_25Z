import json
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
import torch
from datasets import Dataset


# -------------------------
# 1. Jedna funkcja: czyszczenie
# -------------------------
def clean_text(text):
    text = re.sub(r"<.*?>", "", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()


# -------------------------
# 2. Ładowanie configu
# -------------------------
def load_config():
    with open("config_IMDB.json", "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# 3. SVM
# -------------------------
def run_svm(cfg, X_train, y_train, X_test, y_test):
    vectorizer = TfidfVectorizer(
        max_features=cfg["max_features"],
        ngram_range=tuple(cfg["ngram_range"])
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LinearSVC(C=cfg["C"])
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    print("\n===== WYNIKI SVM =====")
    print(classification_report(y_test, preds))

    return {"accuracy": accuracy_score(y_test, preds)}


# -------------------------
# 4. BERT (mBERT)
# -------------------------
def run_bert(cfg, X_train, y_train, X_test, y_test):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
    test_ds = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length",
                         truncation=True, max_length=cfg["max_length"])

    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text"])
    test_ds = test_ds.map(tokenize, batched=True).remove_columns(["text"])

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(set(y_train))
    )

    training_args = TrainingArguments(
        output_dir="bert_out",
        per_device_train_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"]
    )

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=test_ds)

    print("\n[mBERT] Trening…")
    trainer.train()

    logits = trainer.predict(test_ds).predictions
    preds = np.argmax(logits, axis=1)

    print("\n===== WYNIKI BERT =====")
    print(classification_report(y_test, preds))

    return {"accuracy": accuracy_score(y_test, preds)}


# -------------------------
# 5. LLaMA (zero-shot)
# -------------------------
def classify_llama(text, pipe):
    prompt = (
        "Classify the sentiment of the following text as positive, neutral, or negative.\n"
        f"Text: {text}\nSentiment:"
    )
    out = pipe(prompt)[0]["generated_text"].lower()

    if "positive" in out:
        return 2
    if "neutral" in out:
        return 1
    return 0  # negative (fallback)


def run_llama(cfg, X_test, y_test):
    pipe = pipeline(
        "text-generation",
        model=cfg["model_name"],
        max_new_tokens=cfg["max_new_tokens"],
        device_map="auto"
    )

    sample = X_test.sample(cfg["sample_size"], random_state=42)
    preds = [classify_llama(t, pipe) for t in sample]

    print("\n===== WYNIKI LLaMA =====")
    print(classification_report(y_test.loc[sample.index], preds))

    return {"accuracy": accuracy_score(y_test.loc[sample.index], preds)}


# -------------------------
# 6. Główna funkcja
# -------------------------
def main():
    cfg = load_config()

    ds = cfg["dataset"]
    df = pd.read_csv(ds["path"])

    df["label"] = df[ds["label_col"]].map(ds["label_mapping"])
    df["clean"] = df[ds["text_col"]].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=cfg["general"]["test_size"],
        random_state=cfg["general"]["random_state"],
        stratify=df["label"]
    )

    svm = run_svm(cfg["svm"], X_train, y_train, X_test, y_test)
    bert = run_bert(cfg["bert"], X_train, y_train, X_test, y_test)
    llama = run_llama(cfg["llama"], X_test, y_test)

    print("\n===== PODSUMOWANIE =====")
    print("SVM:", svm)
    print("BERT:", bert)
    print("LLaMA:", llama)


if __name__ == "__main__":
    main()
