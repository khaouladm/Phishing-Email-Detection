# 🛡️ Phishing Email Detection — NLP & Machine Learning

AI-based phishing email detection system that analyzes **email headers + text content** to classify messages as **Phishing** or **Legitimate**.

The project includes:

✔ Full **text preprocessing pipeline** (cleaning, lemmatization, stopwords)
✔ **Feature engineering** (URL count, suspicious words, caps ratio…)
✔ **TF-IDF vectorization + ML models**
✔ A clean **Streamlit web app** (copy–paste an email → instant classification)
✔ Modular code inside `src/`
✔ Saved trained model inside `models/`

---

# 🚀 Features

### 🔍 **1. Phishing Detection (Content + Header Analysis)**

The app extracts features from:

* **Email subject**
* **Sender (`From:`)**
* **Email body**
* **Embedded URLs**
* **Suspicious vocabulary**
* **Capital letters ratio**

And predicts:

* ⚠️ **Phishing**
* 🛡️ **Legitimate**

---

### 🧹 **2. Text Preprocessing Pipeline**

Includes:

* Lowercasing
* Removing URLs
* Removing special characters
* Stopwords removal (NLTK)
* Lemmatization
* Combining Subject + Body

---

### 📊 **3. Machine Learning Pipeline**

* TF-IDF vectorizer
* ColumnTransformer (NLP + numerical features)
* ML models compared (LogReg, SVM, RandomForest, etc.)
* Best model saved as:

```
models/best_phishing_model.pkl
```

---

### 🖥 **4. Streamlit Web App**

Features:

* Paste raw email text (with headers)
* Automatic feature extraction
* Fast prediction
* Risk indicators (URLs, suspicious words, caps ratio)

Run:

```
streamlit run src/app.py
```

---

# 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourname/Phishing-Email-Detection.git
cd Phishing-Email-Detection
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔧 Training the Model

Your dataset should be inside:

```
data/raw/Enron.csv
```

Expected columns:

| text_combined | label  |
| ------------- | ------ |
| email text    | 0 or 1 |

Train the model:

```bash
python src/train_model.py
```

This generates:

```
models/best_phishing_model.pkl
```

---

# 🔮 Running Predictions (CLI)

```bash
python src/predict.py
```

Paste any email → see prediction.

---

# 🖥️ Running the Streamlit App

```bash
streamlit run src/app.py
```

Then open:

```
http://localhost:8501
```

Paste an email → get classification.

---

# 📦 Requirements (requirements.txt)

```
streamlit
pandas
numpy
scikit-learn==1.5.0
nltk
joblib
matplotlib
seaborn
wordcloud
```

If needed:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

---

# 🤖 Model Overview

### ✔ NLP + TF-IDF Embedding

Transforms processed text to numerical vectors.

### ✔ Feature Engineering

Adds security-related features:

* URL count
* Suspicious word frequency
* Capital letters ratio
* Email subject and sender

### ✔ Best Model (Grid Search)

* Logistic Regression
* RandomForest
* Linear SVM
* Gradient Boosting

---

# 🚀 Deployment 

### 🔸 Hugging Face Spaces — free : https://huggingface.co/spaces/Khaoula34/Phishing-Email-Detection


