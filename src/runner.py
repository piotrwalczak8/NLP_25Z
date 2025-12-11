# src/runner.py
from src.utils.config_utils import load_config
from data.dataset import load_raw, prepare_data, to_hf_dataset
from src.models.svm import SVMClassifier
from src.models.bert import BertTrainer
from src.models.llama import LlamaZeroShot
from src.evaluation.metrics import print_classification_report, accuracy

def run(config_path: str, opts: dict = None):
    """
    Uruchamia eksperymenty sentymentu.
    opts: słownik z opcjonalnymi parametrami:
        - "only": uruchomić tylko wybrane modele ("svm", "bert", "llama", "all")
        - "dry_run": jeśli True, nie trenuje modeli, tylko wypisuje plan
        - "seed": opcjonalny seed dla reproducibility
    """
    opts = opts or {}
    only = opts.get("only", "all")
    dry_run = opts.get("dry_run", False)
    seed = opts.get("seed", None)

    cfg = load_config(config_path)

    ds_cfg = cfg["dataset"]
    df = load_raw(ds_cfg["path"])
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        text_col=ds_cfg["text_col"],
        label_col=ds_cfg["label_col"],
        label_mapping=ds_cfg["label_mapping"],
        test_size=cfg["general"]["test_size"],
        random_state=cfg["general"]["random_state"]
    )

    # ----------------- SVM -----------------
    svm_res = None
    if only in ["svm", "all"]:
        svm_cfg = cfg["svm"]
        svm = SVMClassifier(max_features=svm_cfg["max_features"],
                            ngram_range=tuple(svm_cfg["ngram_range"]),
                            C=svm_cfg["C"])
        if dry_run:
            print("[SVM] Dry run: przygotowany model z configu", svm_cfg)
        else:
            svm.fit(X_train, y_train)
            svm_res = svm.evaluate(X_test, y_test)

    # ----------------- BERT -----------------
    bert_res = None
    if only in ["bert", "all"]:
        bert_cfg = cfg["bert"]
        bert = BertTrainer(model_name=bert_cfg["model_name"],
                           max_length=bert_cfg["max_length"],
                           batch_size=bert_cfg["batch_size"],
                           epochs=bert_cfg["epochs"],
                           learning_rate=bert_cfg["learning_rate"])
        if dry_run:
            print("[BERT] Dry run: przygotowany model z configu", bert_cfg)
        else:
            tokenizer, train_ds, test_ds = bert.prepare_datasets(X_train, y_train, X_test, y_test)
            model = bert.build_model(num_labels=len(set(y_train)))
            preds_bert, trainer = bert.train_and_eval(model, train_ds, test_ds)
            print("\n===== WYNIKI BERT =====")
            print_classification_report(y_test, preds_bert)
            bert_res = accuracy(y_test, preds_bert)

    # ----------------- LLaMA -----------------
    llama_res = None
    if only in ["llama", "all"]:
        llama_cfg = cfg["llama"]
        llm = LlamaZeroShot(model_name=llama_cfg["model_name"],
                            max_new_tokens=llama_cfg["max_new_tokens"])
        if dry_run:
            print("[LLaMA] Dry run: przygotowany model z configu", llama_cfg)
        else:
            sample = X_test.sample(llama_cfg["sample_size"], random_state=42)
            preds_llama = llm.classify_batch(sample.tolist(),
                                             batch_size=llama_cfg.get("batch_size", 4))
            print("\n===== WYNIKI LLaMA =====")
            print_classification_report(y_test.loc[sample.index], preds_llama)
            llama_res = accuracy(y_test.loc[sample.index], preds_llama)

    # ----------------- Podsumowanie -----------------
    print("\n===== PODSUMOWANIE =====")
    print("SVM:", svm_res)
    print("BERT:", bert_res)
    print("LLaMA:", llama_res)
