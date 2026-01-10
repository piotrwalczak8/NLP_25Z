import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

# Włączamy pasek postępu
tqdm.pandas()

# Wczytanie oryginalnego CSV
df = pd.read_csv("tweets.csv")  # <- podaj swoją ścieżkę do pliku

# Funkcja tłumaczenia
def translate_to_german(text):
    try:
        return GoogleTranslator(source='auto', target='de').translate(text)
    except Exception as e:
        print(f"Nie udało się przetłumaczyć: {text}\nBłąd: {e}")
        return text  # w razie błędu zwróć oryginał

# Tłumaczenie kolumny 'text'
df['text_de'] = df['text'].progress_apply(translate_to_german)

# Zapis nowego CSV
df.to_csv("tweets_german.csv", index=False)

print("Tłumaczenie zakończone, zapisano do tweets_german.csv")