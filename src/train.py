import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    X_train = joblib.load("/data/processed/X_train_tfidf.pkl")
    X_test = joblib.load("/data/processed/X_test_tfidf.pkl")
    y_train = joblib.load("/data/processed/y_train.pkl")
    y_test = joblib.load("/data/processed/y_test.pkl")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial NB": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=120),
        "Linear SVC": LinearSVC(max_iter=10000)
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
        models[name] = m
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds)
        }
    return results

def save_best_model(results):
    best = max(results, key=lambda x: results[x]["f1"])
    joblib.dump(results[best]["model"], "models/best_spam_classifier.pkl")
    return best, results[best]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    best_name, best_results = save_best_model(results)

    print("\nBest model:", best_name)
    print(best_results)
