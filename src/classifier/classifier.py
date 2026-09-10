import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


print("Classifier module ready!")


def train_classifier(df):

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_vec,
        y_train
    )

    predictions = model.predict(
        X_test_vec
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    return model, vectorizer


def predict_ticket(model, vectorizer, text):

    text_vec = vectorizer.transform(
        [text]
    )

    prediction = model.predict(
        text_vec
    )[0]

    return prediction