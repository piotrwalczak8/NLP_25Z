# src/models/svm.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

class SVMClassifier:
    def __init__(self, max_features=20000, ngram_range=(1,1), C=1.0):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LinearSVC(C=C)

    def fit(self, X_train, y_train):
        X_train_vec = self.vec.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)

    def predict(self, X):
        X_vec = self.vec.transform(X)
        return self.model.predict(X_vec)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        print("\n===== WYNIKI SVM =====")
        print(classification_report(y_test, preds))
        return {"accuracy": accuracy_score(y_test, preds)}
