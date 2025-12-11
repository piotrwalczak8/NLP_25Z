# data/dataset.py
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from src.preprocessing.text_cleaning import clean_text

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def prepare_data(df: pd.DataFrame, text_col: str, label_col: str, label_mapping: dict,
                 test_size: float, random_state: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = df.copy()
    df["label"] = df[label_col].map(label_mapping)
    df["clean"] = df[text_col].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )
    return X_train, X_test, y_train, y_test

def to_hf_dataset(X, y):
    """opcjonalnie: utworzenie HF Dataset z Pandas Series"""
    return Dataset.from_dict({"text": X.tolist(), "label": y.tolist()})
