# src/evaluation/metrics.py
from sklearn.metrics import classification_report, accuracy_score

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))

def accuracy(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred)}
