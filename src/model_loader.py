# model_loader.py

import pickle

def load_model(path="models/best_spam_classifier.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
