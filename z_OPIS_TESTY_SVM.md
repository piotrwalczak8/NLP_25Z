# SVM - Opis i eksperymenty

# Klasyfikator SVM w projekcie

## 1. Wprowadzenie do SVM

**Support Vector Machine (SVM)** to klasyfikator nadzorowany, który służy do oddzielania klas w przestrzeni cech.  
Działa na zasadzie znalezienia **hiperpłaszczyzny maksymalizującej margines oddzielający klasy**, czyli takiej, która maksymalnie zwiększa odległość między najbliższymi przykładami różnych klas (tzw. wektorami nośnymi).  
SVM sprawdza się dobrze zarówno przy problemach binarnych, jak i wieloklasowych, jest odporny na wysoką wymiarowość cech oraz stosunkowo szybki w trenowaniu przy średniej wielkości zbiorach danych.

## 2. Implementacja SVM w projekcie

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

class SVMClassifier:
    def __init__(self, max_features, ngram_range, C):
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

```

## 3. Analiza implementacji

### 3.1 Konstruktor __init__ -

- Tworzy wektor TF-IDF (`TfidfVectorizer`), który zamienia tekst w macierz liczbową, reprezentującą częstość słów i n-gramów.
- Inicjalizuje `LinearSVC`, liniowy klasyfikator SVM.
- Parametry:
  - `max_features` – maksymalna liczba cech w wektorze TF-IDF.
  - `ngram_range` – zakres n-gramów (np. `[1, 2]` uwzględnia pojedyncze słowa i pary słów).
  - `C` – parametr regularyzacji kontrolujący kompromis między dopasowaniem do danych a generalizacją.

### 3.2 Metoda `fit`
- Dopasowuje wektor TF-IDF do danych treningowych.
- Trenuje liniowy model SVM na przetworzonych danych.
- Proces polega na znalezieniu hiperpłaszczyzny maksymalizującej margines między klasami w przestrzeni cech.

### 3.3 Metoda `predict`
- Zamienia nowe teksty w wektory TF-IDF (korzystając ze słownika wyuczonego podczas treningu).
- Zwraca przewidywane klasy.

### 3.4 Metoda `evaluate`
- Oblicza metryki jakości klasyfikacji:
  - `accuracy` – dokładność predykcji,
  - `precision`, `recall`, `f1-score` – szczegółowe wskaźniki dla każdej klasy.
- Wyświetla czytelny raport klasyfikacji i zwraca słownik wyników do dalszej analizy.


## 4. Wyniki algorytmu na różnych datasetach przy domyślnej konfiguracji

Wszystkie eksperymenty przeprowadzono przy tej samej konfiguracji SVM:  
- `max_features = 30000`  
- `ngram_range = [1, 2]`  
- `C = 1.0`  

Testowano trzy zbiory danych: **Tweets (angielski)**, **Tweets_DE (niemiecki, przetłumaczone)** oraz **IMDB Dataset (opinie o filmach)**.


### 4.1 Tweets (angielski, 14873 rekordów)

```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.93      0.88      1835
           1       0.66      0.59      0.62       620
           2       0.81      0.62      0.70       473

    accuracy                           0.80      2928
   macro avg       0.77      0.71      0.74      2928
weighted avg       0.80      0.80      0.80      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.8049863387978142}
BERT: None
LLaMA: None
```

Analizując powysze wyniki moezemy zauwazyc, ze model najlepiej radzi sobie z klasą 0 (negative) co widać poprzez wysoki recall i precision. Klasa neutralna (1) jest trudniejsza do rozpoznania, posiada niższe f1-score. Ogólna dokładność: wynosi ~80%, co jest rozsądnym wynikiem dla problemu trójklasowego na krótkich tekstach typu tweet.


### 4.2 Tweets_DE (niemiecki, 14873 rekordów, przetłumaczone)

```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.93      0.88      1835
           1       0.66      0.59      0.62       620
           2       0.81      0.62      0.70       473

    accuracy                           0.80      2928
   macro avg       0.77      0.71      0.74      2928
weighted avg       0.80      0.80      0.80      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.8049863387978142}
```

Wyniki są praktycznie identyczne jak dla angielskiego zbioru. Jest to spowodowane tym, że dane DE są dokładnym tłumaczeniem angielskich tweetów i model SVM uczy się na wektorach TF-IDF które statystycznie zachowują podobną strukturę. Ukazuje to ciekawą zalezność modelu SVM. Nie "rozumie języka", a działa on jedynie na cechach statystycznych tekstu.

### 4.3 IMDB Dataset (opinie o filmach, 50 000 rekordów)

```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.91      0.90      0.90      5000
           1       0.90      0.91      0.91      5000

    accuracy                           0.91     10000
   macro avg       0.91      0.91      0.91     10000
weighted avg       0.91      0.91      0.91     10000


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.9054}
```

Model osiąga bardzo wysoką dokładność bi az 91% na danych binarnych (positive/negative). Wyniki są równomiernie rozłożone między klasy, precision i recall są zbliżone dla obu klas. Wyższa skuteczność niż na tweetach wynika z:  
  - dłuższych tekstów (opinie filmowe zawierają więcej słów i kontekstu),  
  - wyraźniejszych sygnałów semantycznych między klasami.

### 4.4 Podsumowanie wyników

Analiza wyników SVM na trzech różnych datasetach pokazuje wyraźne zależności między rodzajem danych a skutecznością klasyfikacji:  

- Tweets (angielski) i Tweets_DE (niemiecki, przetłumaczone) osiągnęły tę samą dokładność ~80%, co wynika z faktu, że niemieckie tweety są tłumaczeniem angielskich, a model opiera się na statystycznych cechach TF-IDF, a nie na semantyce języka.  
- IMDB Dataset, zawierający dłuższe i bardziej opisowe opinie filmowe, uzyskał znacznie wyższą dokładność ~91%, co pokazuje, że SVM lepiej radzi sobie z tekstami bogatszymi w informacje i kontekst.  
- Najtrudniejszą klasą do przewidzenia w przypadku tweetów była klasa neutralna, natomiast klasy pozytywne i negatywne były rozpoznawane znacznie lepiej.  

Ogólnie wyniki potwierdzają, że SVM z reprezentacją TF-IDF jest skuteczną i stabilną metodą klasyfikacji tekstu, a jej wydajność zależy głównie od długości i jakości sygnałów w danych.

### 5. Eksperymenty i wyniki eksperymentów

W tej sekcji przeprowadzono szereg eksperymentów mających na celu zbadanie wpływu różnych parametrów SVM i przetwarzania tekstu na jakość klasyfikacji. Jako przedmiot eksperymentów posłu  

Celem eksperymentów było zrozumienie, jak zmiana poszczególnych ustawień wpływa na metryki modelu, takie jak **accuracy**, **precision**, **recall** i **f1-score**, oraz interpretacja obserwowanych efektów w kontekście danych tekstowych i konstrukcji modelu SVM.

Wybrano trzy główne parametry do testowania:  

1. **Parametr C w SVM** – odpowiada za regularyzację, kontrolując kompromis między dopasowaniem do danych treningowych a generalizacją modelu.  
2. **Liczba cech w wektorze TF-IDF (`max_features`)** – określa, ile najważniejszych słów/n-gramów zostanie uwzględnionych przy reprezentacji tekstu w przestrzeni cech.  
3. **Zakres n-gramów (`ngram_range`)** – decyduje, czy model uwzględnia pojedyncze słowa (unigramy), pary słów (bigramy) czy inne kombinacje przy tworzeniu macierzy TF-IDF.

Każdy eksperyment został przeprowadzony niezależnie, przy zachowaniu pozostałych parametrów w wartości domyślnej, co pozwala na obserwację wpływu tylko jednego czynnika na wydajność modelu.  

W kolejnych podpunktach przedstawione zostaną wyniki dla poszczególnych eksperymentów wraz z interpretacją ich wpływu na działanie algorytmu.


### 5.1 Eksperyment 1: Wpływ parametru C w SVM

W pierwszym eksperymencie badaliśmy wpływ parametru **C** w klasyfikatorze **LinearSVC** na dokładność klasyfikacji danych z zbioru **Tweets**. Parametr **C** kontroluje siłę regularyzacji modelu:
- mała wartość C → większa regularyzacja, model jest prostszy, bardziej uogólniony,
- duża wartość C → mniejsza regularyzacja, model stara się dokładniej dopasować do danych treningowych.

#### Wyniki eksperymentu

**Wyniki dla C = 0.01**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.66      1.00      0.79      1835
           1       0.78      0.06      0.12       620
           2       0.88      0.21      0.34       473

    accuracy                           0.67      2928
   macro avg       0.77      0.42      0.42      2928
weighted avg       0.72      0.67      0.58      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.6717896174863388}
```
- **C = 0.01**  
  - Dokładność: 0.67  
  - Model jest bardzo mocno regularyzowany, co prowadzi do wysokiego recall dla klasy 0, ale bardzo niskiego dla klas 1 i 2.  
  - Model nie jest w stanie dobrze rozróżnić klas neutralnej i pozytywnej.


**Wyniki dla C = 0.1**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.78      0.97      0.87      1835
           1       0.73      0.43      0.54       620
           2       0.86      0.53      0.66       473

    accuracy                           0.78      2928
   macro avg       0.79      0.64      0.69      2928
weighted avg       0.79      0.78      0.76      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.7848360655737705}
```
- **C = 0.1**  
  - Dokładność: 0.78  
  - Wzrost C poprawił wyniki dla klas 1 i 2, nadal klasa 0 dominuje, ale model lepiej balansuje predykcje.


**Wyniki dla C = 1.0 - watość domyślna** 
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.93      0.88      1835
           1       0.66      0.59      0.62       620
           2       0.81      0.62      0.70       473

    accuracy                           0.80      2928
   macro avg       0.77      0.71      0.74      2928
weighted avg       0.80      0.80      0.80      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.8049863387978142}
```
- **C = 1.0 (domyślna)**  
  - Dokładność: 0.805  
  - Optymalny kompromis między dopasowaniem a regularyzacją.  
  - Klasyfikator osiąga najlepszą równowagę między precyzją i recall dla wszystkich klas.



**Wyniki dla C = 10.0**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.85      0.89      0.87      1835
           1       0.60      0.58      0.59       620
           2       0.77      0.65      0.71       473

    accuracy                           0.79      2928
   macro avg       0.74      0.71      0.72      2928
weighted avg       0.78      0.79      0.78      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.7855191256830601}
```
- **C = 10.0**  
  - Dokładność: 0.786  
  - Większe C powoduje, że model próbuje bardziej dopasować się do danych treningowych.  
  - Poprawa recall dla klasy 2, ale spadek dla klasy 1. Ogólna dokładność lekko spada.


**Wyniki dla C = 100.0**

```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.85      0.88      0.86      1835
           1       0.58      0.56      0.57       620
           2       0.73      0.64      0.68       473

    accuracy                           0.77      2928
   macro avg       0.72      0.69      0.70      2928
weighted avg       0.77      0.77      0.77      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.773224043715847}
```
- **C = 100.0**  
  - Dokładność: 0.773  
  - Bardzo wysokie C powoduje przeuczenie na klasie dominującej (0).  
  - Klasyfikacja dla mniejszych klas (1 i 2) pogarsza się, mimo że klasa 0 nadal dobrze klasyfikowana.  
  - Pojawiło się ostrzeżenie o braku konwergencji (nie wpłynęło na wynik, można je zignorować dla eksperymentu).

#### Wnioski

Eksperyment z wartością parametru C prowadzi do trzech kluczowych wniosków:

1. **Zbyt małe C (0.01)** powoduje nadmierną regularyzację, model nie uczy się rozróżniać rzadziej występujących klas.  
2. **C w okolicach 1.0** daje najlepsze wyniki, bo zachowuje kompromis między dopasowaniem, a generalizacją.  
3. **Zbyt duże C (>10)** prowadzi do przeuczenia modelu na dominującą klasę, co pogarsza wyniki dla mniejszych klas.  
4. Zmiana C ma wyraźny wpływ na równowagę między precyzją a recall dla poszczególnych klas, dlatego dobór parametru jest kluczowy w zadaniach wieloklasowych.

Eksperyment pokazuje, że parametr C należy stroić w zależności od charakterystyki danych, zwłaszcza przy niezbalansowanych zbiorach danych jak w przypadku Tweetów.



### 5.2 Eksperyment 2: Liczba cech w wektorze TF-IDF (`max_features`)

Celem drugiego eksperymentu było zbadanie wpływu liczby cech wykorzystywanych przez wektor TF-IDF na jakość klasyfikacji przy użyciu algorytmu SVM. Parametr `max_features` określa maksymalną liczbę najczęściej występujących cech (słów i n-gramów), które są uwzględniane w procesie wektoryzacji tekstu. Pozostałe parametry modelu pozostały bez zmian.

**Wynik dla max_features = 1000**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.90      0.87      1835
           1       0.63      0.60      0.61       620
           2       0.74      0.57      0.64       473

    accuracy                           0.78      2928
   macro avg       0.73      0.69      0.71      2928
weighted avg       0.78      0.78      0.78      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.782103825136612}
```
Przeprowadzone eksperymenty pokazują wyraźny trend wzrostu jakości klasyfikacji wraz ze zwiększaniem liczby cech. Dla niskiej wartości `max_features = 1000` model osiąga najniższą dokładność (accuracy ~ 0.78), co wskazuje na niedostateczną reprezentację danych tekstowych. W tym przypadku model opiera się jedynie na najbardziej ogólnych i najczęściej występujących słowach, co prowadzi do utraty informacji istotnych dla rozróżnienia klas, szczególnie klasy neutralnej.



**Wynik dla max_features = 5000**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.85      0.90      0.87      1835
           1       0.61      0.59      0.60       620
           2       0.77      0.62      0.69       473

    accuracy                           0.79      2928
   macro avg       0.74      0.70      0.72      2928
weighted avg       0.78      0.79      0.78      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.7879098360655737}
```

**Wynik dla max_features = 10000**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      1835
           1       0.64      0.60      0.62       620
           2       0.79      0.63      0.70       473

    accuracy                           0.80      2928
   macro avg       0.76      0.71      0.73      2928
weighted avg       0.79      0.80      0.79      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.7974726775956285}
```
Zwiększenie liczby cech do 5000 i 10 000 skutkuje stopniową poprawą wyników. Model zaczyna uwzględniać rzadsze, ale bardziej informatywne cechy, co przekłada się na wzrost wartości recall i f1-score dla klas pozytywnej oraz neutralnej. Najlepsze wyniki uzyskano dla `max_features = 30 000`, gdzie dokładność klasyfikacji osiągnęła około 0.80, a wartości metryk były najbardziej zbalansowane pomiędzy klasami.
**Wynik dla max_features = 30000**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.93      0.88      1835
           1       0.66      0.59      0.62       620
           2       0.81      0.62      0.70       473

    accuracy                           0.80      2928
   macro avg       0.77      0.71      0.74      2928
weighted avg       0.80      0.80      0.80      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.8049863387978142}
```

**Wynik dla max_features = 60000**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.93      0.88      1835
           1       0.67      0.57      0.61       620
           2       0.80      0.62      0.70       473

    accuracy                           0.80      2928
   macro avg       0.77      0.71      0.73      2928
weighted avg       0.80      0.80      0.80      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.8046448087431693}
```
Dalsze zwiększenie liczby cech do 60 000 nie przyniosło już istotnej poprawy jakości modelu. Osiągnięte wyniki były niemal identyczne jak dla 30 000 cech, co wskazuje na osiągnięcie punktu nasycenia informacji. W tym zakresie model posiada już wystarczająco bogatą reprezentację tekstu, a dodawanie kolejnych cech nie wnosi nowych, istotnych informacji, a jedynie zwiększa potencjalny szum.

Wyniki eksperymentu potwierdzają, że liczba cech w wektorze TF-IDF ma istotny wpływ na jakość klasyfikacji, jednak tylko do pewnego momentu. Dla analizowanego zbioru danych Tweets optymalnym kompromisem pomiędzy jakością modelu a jego złożonością okazała się wartość `max_features` w przedziale 10 000–30 000.

### 5.3 Eksperyment 3: Zakres n-gramów (`ngram_range`)

Celem trzeciego eksperymentu było zbadanie wpływu zakresu n-gramów używanych w wektoryzacji TF-IDF na jakość klasyfikacji sentymentu tweetów. Parametr `ngram_range` kontroluje, czy model analizuje pojedyncze słowa (unigramy), pary słów (bigramy) lub dłuższe sekwencje wyrazów, co bezpośrednio wpływa na zdolność modelu do uchwycenia kontekstu językowego.

**Wynik dla ngram_range = [1, 1]**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.91      0.88      1835
           1       0.65      0.58      0.61       620
           2       0.79      0.66      0.72       473

    accuracy                           0.80      2928
   macro avg       0.76      0.72      0.74      2928
weighted avg       0.80      0.80      0.80      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.8012295081967213}
```
Model wykorzystujący wyłącznie unigramy osiągnął dokładność na poziomie około 80%. Wynik ten pokazuje, że pojedyncze słowa niosą wystarczająco dużo informacji, aby skutecznie rozróżniać sentyment w krótkich tekstach, takich jak tweety. Klasy „neutralna” i „pozytywna” były klasyfikowane z umiarkowaną skutecznością, natomiast klasa „negatywna” cechowała się bardzo wysokim recallem, co sugeruje dominację słów jednoznacznie negatywnych.

**Wynik dla ngram_range = [1, 2] - wartość domyślna**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.93      0.88      1835
           1       0.66      0.59      0.62       620
           2       0.81      0.62      0.70       473

    accuracy                           0.80      2928
   macro avg       0.77      0.71      0.74      2928
weighted avg       0.80      0.80      0.80      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.8049863387978142}
```
Uwzględnienie zarówno unigramów, jak i bigramów przyniosło najlepszy wynik w całym eksperymencie. Dokładność klasyfikacji wzrosła nieznacznie w porównaniu do modelu opartego wyłącznie na unigramach. Bigramy pozwoliły modelowi lepiej rozpoznawać krótkie frazy kontekstowe, takie jak zaprzeczenia czy charakterystyczne zwroty emocjonalne, co przełożyło się na bardziej zbalansowane wartości precision i recall dla wszystkich klas.

**Wynik dla ngram_range = [2, 2]**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.79      0.91      0.84      1835
           1       0.60      0.45      0.52       620
           2       0.76      0.54      0.63       473

    accuracy                           0.75      2928
   macro avg       0.72      0.64      0.66      2928
weighted avg       0.74      0.75      0.74      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.7544398907103825}
```
Model wykorzystujący wyłącznie bigramy uzyskał najsłabsze wyniki spośród wszystkich testowanych konfiguracji. Spadek dokładności i metryk jakościowych wskazuje, że same pary słów nie są wystarczające do pełnego uchwycenia znaczenia wypowiedzi. Wiele tweetów zawiera bowiem krótkie, jednoznaczne słowa niosące silny ładunek emocjonalny, które w tej konfiguracji zostały całkowicie pominięte.

**Wynik dla ngram_range = [1, 3]**
```
===== WYNIKI SVM =====
              precision    recall  f1-score   support

           0       0.84      0.92      0.88      1835
           1       0.66      0.59      0.62       620
           2       0.81      0.63      0.71       473

    accuracy                           0.80      2928
   macro avg       0.77      0.71      0.74      2928
weighted avg       0.80      0.80      0.80      2928


===== PODSUMOWANIE =====
SVM: {'accuracy': 0.8032786885245902}
```
#### Wyniki dla `ngram_range = (1, 3)`
Rozszerzenie zakresu n-gramów o trigramy nie przyniosło istotnej poprawy wyników w porównaniu do konfiguracji `(1, 2)`. Otrzymane metryki były bardzo zbliżone, a dokładność pozostała na podobnym poziomie. Wskazuje to, że w przypadku krótkich tekstów trigramy rzadko występują wystarczająco często, aby wnieść dodatkową informację, a ich uwzględnienie może prowadzić do zwiększenia liczby cech bez realnej poprawy jakości modelu.

#### Wnioski
Przeprowadzony eksperyment potwierdza, że konfiguracja `ngram_range = (1, 2)` stanowi najlepszy kompromis pomiędzy zdolnością modelu do uchwycenia kontekstu a jego ogólną skutecznością. Unigramy dostarczają podstawowej informacji semantycznej, natomiast bigramy umożliwiają rozpoznawanie prostych zależności kontekstowych. Dalsze zwiększanie zakresu n-gramów nie prowadzi do zauważalnej poprawy wyników i może jedynie zwiększać złożoność modelu.