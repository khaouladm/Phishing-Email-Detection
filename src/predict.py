import joblib

def load_inference_objects():
    model = joblib.load("models/best_spam_classifier.pkl")
    vectorizer = joblib.load("data/vectorizer/tfidf_vectorizer.pkl")
    encoder = joblib.load("data/processed/label_encoder.pkl")
    return model, vectorizer, encoder

def predict_email(text):
    model, vectorizer, encoder = load_inference_objects()

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = encoder.inverse_transform([pred])[0]

    return label

if __name__ == "__main__":
    email = input("Enter email text: ")
    print("Prediction:", predict_email(email))
