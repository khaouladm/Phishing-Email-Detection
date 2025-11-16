import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_dataset(path):
    return pd.read_csv(path)

def preprocess_dataset(df):
    # Assumes processed_text already exists in your CSV
    df["processed_text"] = df["processed_text"].astype(str)
    label_encoder = LabelEncoder()
    df["label_num"] = label_encoder.fit_transform(df["label"])
    return df, label_encoder

def vectorize_text(df):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df["processed_text"])
    return X_tfidf, vectorizer

def save_preprocessing_objects(X_train, X_test, y_train, y_test, vectorizer, encoder):
    joblib.dump(X_train, "/data/processed/X_train_tfidf.pkl")
    joblib.dump(X_test, "/data/processed/X_test_tfidf.pkl")
    joblib.dump(y_train, "/data/processed/y_train.pkl")
    joblib.dump(y_test, "/data/processed/y_test.pkl")
    joblib.dump(vectorizer, "data/vectorizer/tfidf_vectorizer.pkl")
    joblib.dump(encoder, "data/processed/label_encoder.pkl")
if __name__ == "__main__":
    df = load_dataset("data/processed_spam_dataset.csv")
    df, encoder = preprocess_dataset(df)

    from sklearn.model_selection import train_test_split
    X, vectorizer = vectorize_text(df)
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    save_preprocessing_objects(X_train, X_test, y_train, y_test, vectorizer, encoder)
    print("Preprocessing completed and saved.")
