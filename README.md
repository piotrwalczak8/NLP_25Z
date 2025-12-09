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

