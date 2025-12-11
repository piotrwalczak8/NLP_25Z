1. Na komputerze należy mieć zainstalowane python 3.10/3.11
2. W folderze projektu należy stworzyć wirtualne środowisko (venv) i aktywować
     python -m venv .venv
    .venv\Scripts\activate
Jeśli się uda, zobaczysz na początku linii w terminaly (.venv) C:\Users\..
4. Zaktualizuj pip
    python -m pip install --upgrade pip
5. Zainstaluj wszystko co jest zawarte w requirements.txt
     pip install -r requirements.txt
Jeśli jest problem z torch można dodatkowo samemu wpisać w terminalu: pip install torch --index-url https://download.pytorch.org/whl/cpu
6. Zawsze przed uruchomieniem programu, należy zedytować kod i wpisać plik json, który zbiór chcemy zbadać
W tym miejscu \/\/\/\/\/\/
  def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

lub wpisać w konsoli python project.py --config configIMBD.json
8. na koniec uruchomić program
    python project.py

Jak po uruchomieniu się przyblokuje to wpisz '3'

linki do datasetów (pamiętać że jak się je pobierze, by były w tym samym folderze co projekt):
https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/notebook?fbclid=IwY2xjawOldPJleHRuA2FlbQIxMABicmlkETBsREhtMTBIa3lQNnpZUG1rc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHqZqYiA_etETDHHTBtoW2H_Kt9iIJrusDC0OlqX4P_LPE0pYQnpOSOmmeVhJ_aem_4qg1KCPaxZIzs9R-EYUsuQ
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment?select=Tweets.csv&fbclid=IwY2xjawOldOlleHRuA2FlbQIxMABicmlkETBsREhtMTBIa3lQNnpZUG1rc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHu4smGjewcCKKP3p8Oq_lBo7j_85OyF4jDTp7In7IBYsLj3Qs-jp91B_rsWN_aem_85qKlxeb_M-8wbE54mLg9g


==============================wyniki============================

2025-12-10 09:28:51.710590: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1765358931.730471    3464 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1765358931.736452    3464 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1765358931.751697    3464 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1765358931.751721    3464 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1765358931.751726    3464 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1765358931.751729    3464 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-12-10 09:28:51.756252: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.93      0.88      1835
           1       0.66      0.57      0.61       620
           2       0.81      0.63      0.71       473

    accuracy                           0.80      2928
   macro avg       0.77      0.71      0.73      2928
weighted avg       0.80      0.80      0.80      2928

Map: 100% 11712/11712 [00:01<00:00, 7904.84 examples/s]
Map: 100% 2928/2928 [00:00<00:00, 7684.10 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-multilingual-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

[mBERT] Trening…
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 3
wandb: You chose "Don't visualize my results"
wandb: Tracking run with wandb version 0.23.1
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in /content/wandb/offline-run-20251210_092916-pdk8t5z3
{'loss': 0.6731, 'grad_norm': 18.162527084350586, 'learning_rate': 1.3183060109289618e-05, 'epoch': 0.34}
{'loss': 0.5355, 'grad_norm': 11.812284469604492, 'learning_rate': 6.352459016393443e-06, 'epoch': 0.68}
{'train_runtime': 522.2855, 'train_samples_per_second': 22.425, 'train_steps_per_second': 2.803, 'train_loss': 0.5693963889867231, 'epoch': 1.0}
100% 1464/1464 [08:29<00:00,  2.87it/s]
100% 366/366 [00:19<00:00, 18.36it/s]

===== WYNIKI BERT =====
              precision    recall  f1-score   support

           0       0.85      0.92      0.88      1835
           1       0.67      0.61      0.64       620
           2       0.79      0.63      0.70       473

    accuracy                           0.81      2928
   macro avg       0.77      0.72      0.74      2928
weighted avg       0.80      0.81      0.80      2928

config.json: 100% 608/608 [00:00<00:00, 1.09MB/s]
model.safetensors: 100% 2.20G/2.20G [00:23<00:00, 95.6MB/s]
generation_config.json: 100% 124/124 [00:00<00:00, 277kB/s]
tokenizer_config.json: 1.29kB [00:00, 2.65MB/s]
tokenizer.model: 100% 500k/500k [00:00<00:00, 980kB/s]
tokenizer.json: 1.84MB [00:00, 122MB/s]
special_tokens_map.json: 100% 551/551 [00:00<00:00, 2.52MB/s]
Device set to use cuda:0
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset

===== WYNIKI LLaMA =====
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        14
           1       0.00      0.00      0.00         3
           2       0.15      1.00      0.26         3

    accuracy                           0.15        20
   macro avg       0.05      0.33      0.09        20
weighted avg       0.02      0.15      0.04        20


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.8043032786885246}
BERT: {'accuracy': 0.8090846994535519}
LLaMA: {'accuracy': 0.15}
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /content/wandb/offline-run-20251210_092916-pdk8t5z3
wandb: Find logs at: wandb/offline-run-20251210_092916-pdk8t5z3/logs
