**1. tweet full config**
  "llama": {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "sample_size": null,
    "max_new_tokens": 8,
    "batch_size": 16,
    "device_map": "auto",
    "classes": ["Negative", "Neutral", "Positive"],
  "few_shot_examples": [
    {
      "text": "@VirginAmerica All of group E was told there was no more room in the bins. when I got on the plane, was room for at least 4 bags in my row!",
      "label": "Negative"
    },
    {
      "text": "@VirginAmerica Thanks for making my flight from LAX to JFK a nightmare by forcing me to check my carry on bag at the gate.",
      "label": "Negative"
    },
    {
      "text": "@united Agent in LAS letting 20 customers know they can't help them rebook delayed flight to DEN #unfriendlyskies",
      "label": "Neutral"
    },
    {
      "text": "@united flight arrives 30 minutes early, but then have we to wait for an hour for our bags.",
      "label": "Neutral"
    },
    {
      "text": "@VirginAmerica of course! I work for @VirginAtlantic and I'm obsessed with the entire Virgin family!!",
      "label": "Positive"
    },
    {
      "text": "@VirginAmerica just bought tix for ATX - Dallas route - thanks for adding that! Love yr airline &amp",
      "label": "Positive"
    }
  ]

}
}

**WYNIKI 1 (tweet full)**
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.73      0.49      0.58      1835
           1       0.22      0.47      0.30       620
           2       0.36      0.27      0.31       473

    accuracy                           0.45      2928
   macro avg       0.33      0.31      0.30      2928
weighted avg       0.56      0.45      0.48      2928






**2. Tweet DE full config**
"llama": {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "sample_size": null,
    "max_new_tokens": 8,
    "batch_size": 16,
    "device_map": "auto",
    "classes": ["Negative", "Neutral", "Positive"],
    "few_shot_examples": [
    {
      "text": "@VirginAmerica Der gesamten Gruppe E wurde gesagt, dass kein Platz mehr in den Gepäckfächern sei. Als ich ins Flugzeug kam, war in meiner Reihe Platz für mindestens 4 Taschen!",
      "label": "Negative"
    },
    {
      "text": "@VirginAmerica Danke, dass ihr meinen Flug von LAX nach JFK zu einem Albtraum gemacht habt, indem ihr mich gezwungen habt, mein Handgepäck am Gate aufzugeben.",
      "label": "Negative"
    },
    {
      "text": "@united Ein Mitarbeiter in LAS informiert 20 Kunden darüber, dass man ihnen bei der Umbuchung ihres verspäteten Fluges nach DEN nicht helfen kann. #unfriendlyskies",
      "label": "Neutral"
    },
    {
      "text": "@united Der Flug kommt 30 Minuten früher an, aber dann müssen wir eine Stunde auf unser Gepäck warten.",
      "label": "Neutral"
    },
    {
      "text": "@VirginAmerica Natürlich! Ich arbeite für @VirginAtlantic und bin begeistert von der gesamten Virgin-Familie!!",
      "label": "Positive"
    },
    {
      "text": "@VirginAmerica Habe gerade Tickets für die Strecke ATX–Dallas gekauft – danke, dass ihr sie hinzugefügt habt! Ich liebe eure Airline.",
      "label": "Positive"
    }
  ]

  }

**WYNIKI 2 (tweet DE):**

                precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.73      0.35      0.47      1835
           1       0.22      0.60      0.33       620
           2       0.33      0.24      0.27       473

    accuracy                           0.38      2928
   macro avg       0.32      0.30      0.27      2928
weighted avg       0.56      0.38      0.41      2928


**3. IMDB config**
"llama": {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "sample_size": null,
    "max_new_tokens": 8,
    "batch_size": 16,
    "device_map": "auto",
    "classes": ["Negative", "Positive"],
    "few_shot_examples": [
      {
        "text": "This movie was absolutely amazing! The plot was engaging and the actors were fantastic.",
        "label": "Positive"
      },
      {
        "text": "I loved this film. Every scene was well done and it kept me hooked until the end.",
        "label": "Positive"
      },
      {
        "text": "Terrible movie. The story made no sense and the acting was horrible.",
        "label": "Negative"
      },
      {
        "text": "I really disliked this film. It was boring and way too long.",
        "label": "Negative"
      }
    ]

}

**wyniki 3 (imdb)**


              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.83      0.72      0.77      5000
           1       0.75      0.85      0.80      5000

    accuracy                           0.78     10000
   macro avg       0.53      0.52      0.52     10000
weighted avg       0.79      0.78      0.78     10000


**4.1 Prompt**
treść prompta:

class_list = ", ".join(self.classes)
        prompt = (
                f"Classify the sentiment of the following text.\n"
                f"Choose exactly ONE label from: {class_list}.\n"
                f"Respond with ONLY the label name.\n\n"
        )

        if self.few_shot_examples:
            prompt += "Examples:\n"
            for t, l in self.few_shot_examples:
                prompt += f"Text: {t}\nLabel: {l}\n\n"

        prompt += (
            "Now classify the following text.\n"
            f"Text: {text}\n"
            "Label:"
        )

**4.1 Wynik**
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.74      0.46      0.57       639
           1       0.20      0.44      0.27       209
           2       0.25      0.20      0.22       152

    accuracy                           0.42      1000
   macro avg       0.30      0.27      0.26      1000
weighted avg       0.55      0.42      0.45      1000

**4.2 Prompt**
treść prompta:
    f"Read the following text carefully and determine its sentiment.\n"
    f"The sentiment should be one of: {class_list}.\n"
    "Answer ONLY with the sentiment label (no extra words, punctuation, or explanation).\n\n"

**4.2 Wynik**
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.75      0.48      0.58       639
           1       0.23      0.50      0.32       209
           2       0.30      0.26      0.28       152

    accuracy                           0.45      1000
   macro avg       0.32      0.31      0.29      1000
weighted avg       0.57      0.45      0.48      1000


**4.3 Prompt**
treść prompta:
prompt = (
                    f"Here are examples of text and their sentiment labels:\n"
        )

        if self.few_shot_examples:
            prompt += "Examples:\n"
            for t, l in self.few_shot_examples:
                prompt += f"Text: {t}\nLabel: {l}\n\n"

        prompt += (
                f"\nNow, classify the sentiment of the following text.\n"
                f"Use one of these labels: {class_list}.\n"
                f"Text: {text}\n"
                "Sentiment:"
            )


**4.3 Wynik**
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.70      0.01      0.02       639
           1       0.00      0.00      0.00       209
           2       0.15      1.00      0.27       152

    accuracy                           0.16      1000
   macro avg       0.21      0.25      0.07      1000
weighted avg       0.47      0.16      0.05      1000


**5.1 few shots examples**
  "few_shot_examples": [
    {
      "text": "@VirginAmerica All of group E was told there was no more room in the bins. when I got on the plane, was room for at least 4 bags in my row!",
      "label": "Negative"
    },
    {
      "text": "@VirginAmerica Thanks for making my flight from LAX to JFK a nightmare by forcing me to check my carry on bag at the gate.",
      "label": "Negative"
    },
    {
      "text": "@united Agent in LAS letting 20 customers know they can't help them rebook delayed flight to DEN #unfriendlyskies",
      "label": "Neutral"
    },
    {
      "text": "@united flight arrives 30 minutes early, but then have we to wait for an hour for our bags.",
      "label": "Neutral"
    },
    {
      "text": "@VirginAmerica of course! I work for @VirginAtlantic and I'm obsessed with the entire Virgin family!!",
      "label": "Positive"
    },
    {
      "text": "@VirginAmerica just bought tix for ATX - Dallas route - thanks for adding that! Love yr airline &amp",
      "label": "Positive"
    }
  ]

  **5.1 Wyniki**
                precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.74      0.46      0.57       639
           1       0.20      0.44      0.27       209
           2       0.25      0.20      0.22       152

    accuracy                           0.42      1000
   macro avg       0.30      0.27      0.26      1000
weighted avg       0.55      0.42      0.45      1000

**5.2 few shots examples**

BRAK EXAMPLES

**5.2 Wyniki**
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.74      0.34      0.47       639
           1       0.22      0.25      0.24       209
           2       0.20      0.61      0.31       152

    accuracy                           0.36      1000
   macro avg       0.29      0.30      0.25      1000
weighted avg       0.55      0.36      0.40      1000

**5.3 few shots examples**
"few_shot_examples": [
    {
      "text": "@VirginAmerica All of group E was told there was no more room in the bins. when I got on the plane, was room for at least 4 bags in my row!",
      "label": "Negative"
    },
    {
      "text": "@VirginAmerica Thanks for making my flight from LAX to JFK a nightmare by forcing me to check my carry on bag at the gate.",
      "label": "Negative"
    },
    {
      "text": "@united Agent in LAS letting 20 customers know they can't help them rebook delayed flight to DEN #unfriendlyskies",
      "label": "Neutral"
    },
    {
      "text": "@united flight arrives 30 minutes early, but then have we to wait for an hour for our bags.",
      "label": "Neutral"
    },
    {
      "text": "@VirginAmerica of course! I work for @VirginAtlantic and I'm obsessed with the entire Virgin family!!",
      "label": "Positive"
    },
    {
      "text": "@VirginAmerica just bought tix for ATX - Dallas route - thanks for adding that! Love yr airline &amp",
      "label": "Positive"
    },
    {
      "text": "@DeltaSupport my luggage is missing and customer service doesn't answer the phone! #frustrated",
      "label": "Negative"
    },
    {
      "text": "@SouthwestAir flight delayed again, no explanation, very disappointed.",
      "label": "Negative"
    },
    {
      "text": "@JetBlue the boarding process was smooth and quick, no complaints.",
      "label": "Neutral"
    },
    {
      "text": "@AmericanAirlines flight was on time, nothing special, everything normal.",
      "label": "Neutral"
    },
    {
      "text": "@AlaskaAir I love the in-flight service and the friendly crew, made my day!",
      "label": "Positive"
    },
    {
      "text": "@SpiritAirlines had a great experience, crew was funny and helpful.",
      "label": "Positive"
    }
]

**5.3 Wyniki**
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.75      0.56      0.64       639
           1       0.23      0.47      0.31       209
           2       0.41      0.21      0.28       152

    accuracy                           0.49      1000
   macro avg       0.35      0.31      0.31      1000
weighted avg       0.59      0.49      0.52      1000

**6.1 sample size**
sample_size = 500

**6.1 Wyniki**
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.73      0.50      0.59       322
           1       0.23      0.48      0.31       103
           2       0.32      0.27      0.29        75

    accuracy                           0.46       500
   macro avg       0.32      0.31      0.30       500
weighted avg       0.56      0.46      0.49       500

**6.2 sample size**
sample_size = 1000


**6.2 Wyniki**
                precision    recall  f1-score   support

          -1       0.00      0.00      0.00         0
           0       0.74      0.46      0.57       639
           1       0.20      0.44      0.27       209
           2       0.25      0.20      0.22       152

    accuracy                           0.42      1000
   macro avg       0.30      0.27      0.26      1000
weighted avg       0.55      0.42      0.45      1000

**6.3 sample size**
sample_size = 1500

**6.3 Wyniki**
          -1       0.00      0.00      0.00         0
           0       0.70      0.47      0.56       943
           1       0.22      0.45      0.30       333
           2       0.34      0.27      0.30       224

    accuracy                           0.44      1500
   macro avg       0.32      0.30      0.29      1500
weighted avg       0.54      0.44      0.46      1500