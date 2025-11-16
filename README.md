# Phishing-Email-Detection
AI-based phishing email detection using NLP and ML — classifies emails as phishing or legitimate.
Here is a **clean, professional, production-ready README.md** for your email spam classifier project.
It matches the architecture we created (`src/`, Streamlit UI, TF-IDF, preprocessing, predict.py, trained model, etc.).

---

# 📧 Email Spam Classifier — ML Project

A machine-learning application that classifies emails as **SPAM** or **HAM** using text preprocessing, TF-IDF vectorization, and a trained Naive Bayes model.
The project includes:

✅ Clean modular code in `src/`
✅ A Streamlit web interface
✅ Preprocessing + TF-IDF + model loading
✅ Easy prediction workflow
✅ Reusable training pipeline

---

# 🚀 Features

### 🔍 **1. Email Spam Detection**

Paste any email text → the model predicts:

* **SPAM**
* **HAM (not spam)**

### 🧹 **2. Text Preprocessing**

Uses the following steps:

* lowercase
* URL removal
* punctuation/number cleaning
* stopwords removal
* lemmatization (WordNet)

### 📊 **3. Machine Learning Pipeline**

* TF-IDF vectorizer
* Multinomial Naive Bayes (best for text classification)
* Train/test split
* Model persisted with pickle

### 🖥 **4. Streamlit Interface**

Simple interface:

* Paste email
* Click **Predict**
* See classification instantly


# 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourname/spam-classifier.git
cd spam-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

# 📌 Training the Model

You must have a dataset in:

```
data/emails.csv
```

Expected format:

| text          | label  |
| ------------- | ------ |
| email content | 0 or 1 |

Run training:

```bash
python src/train.py
```

This will generate:

```
models/best_spam_classifier.pkl
models/tfidf_vectorizer.pkl
```

---

# 🔮 Running Predictions (CLI)

```bash
python src/predict.py
```

Paste an email when prompted.

---

# 🖥 Running the Streamlit Interface (Recommended)

```
streamlit run src/email_classifier_interface.py
```

Your UI will open in the browser:

* Paste the email
* Click **Predict**
* See result (SPAM or HAM)

---

# 📦 Requirements

Add this to **requirements.txt**:

```
streamlit
pandas
scikit-learn
nltk
numpy
```

If WordNet is missing:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

---

# 🧠 Algorithm Used

### ✔ TF-IDF Vectorizer

Converts cleaned emails into numerical vectors.

### ✔ Multinomial Naive Bayes

Best for bag-of-words and TF-IDF text classification.


