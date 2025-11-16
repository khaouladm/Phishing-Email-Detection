import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def load_objects():
    model = joblib.load("models/best_spam_classifier.pkl")
    vectorizer = joblib.load("data/vectorizer/tfidf_vectorizer.pkl")
    return model, vectorizer

def plot_top_features(model, vectorizer, n=20):
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        top_idx = np.argsort(coefs)[-n:]
        top_features = feature_names[top_idx]
        top_values = coefs[top_idx]

    elif hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        top_idx = np.argsort(imps)[-n:]
        top_features = feature_names[top_idx]
        top_values = imps[top_idx]

    else:
        print("Model does not support feature importance.")
        return

    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_values)
    plt.title("Top Features")
    plt.show()

if __name__ == "__main__":
    model, vectorizer = load_objects()
    plot_top_features(model, vectorizer)
