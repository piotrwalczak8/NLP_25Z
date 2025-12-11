# src/models/bert.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from datasets import Dataset

class BertTrainer:
    def __init__(self, model_name: str, max_length: int, batch_size: int, epochs: int, learning_rate: float, output_dir: str = "bert_out"):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir

    def prepare_datasets(self, X_train, y_train, X_test, y_test):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
        test_ds = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

        def tokenize(batch):
            return tokenizer(batch["text"], padding="max_length",
                             truncation=True, max_length=self.max_length)

        train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text"])
        test_ds = test_ds.map(tokenize, batched=True).remove_columns(["text"])
        train_ds.set_format("torch")
        test_ds.set_format("torch")
        return tokenizer, train_ds, test_ds

    def build_model(self, num_labels: int):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        return model

    def train_and_eval(self, model, train_ds, test_ds):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds)
        print("\n[mBERT] Trening…")
        trainer.train()
        logits = trainer.predict(test_ds).predictions
        preds = np.argmax(logits, axis=1)
        return preds, trainer
