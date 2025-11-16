import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_objects():
    model = joblib.load("models/best_spam_classifier.pkl")
    vectorizer = joblib.load("data/vectorizer/tfidf_vectorizer.pkl")
    df = pd.read_csv("/data/processed/processed_spam_dataset.csv")
    return model, vectorizer, df

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.show()

def run_cross_validation(model, vectorizer, df):
    X = vectorizer.transform(df["processed_text"])
    y = df["label_num"]
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print("CV F1 scores:", scores)
    print("Mean F1:", scores.mean())

if __name__ == "__main__":
    model, vectorizer, df = load_objects()
    X_test = joblib.load("/data/processed/X_test_tfidf.pkl")
    y_test = joblib.load("/data/processed/y_test.pkl")
    evaluate_model(model, X_test, y_test)
    run_cross_validation(model, vectorizer, df)
