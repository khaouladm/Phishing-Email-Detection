import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# 1. CONFIGURATION & RESSOURCES
# ==========================================
st.set_page_config(
    page_title="Phishing Detector AI",
    page_icon="🛡️",
    layout="centered"
)

# Téléchargement silencieux des ressources NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ==========================================
# 2. CLASSE & FONCTIONS (Identiques à l'entraînement)
# ==========================================
class EmailPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 2]
        return " ".join(tokens)

def extract_features_from_text(raw_text):
    """
    Transforme le texte brut collé par l'utilisateur en DataFrame
    avec les mêmes colonnes que lors de l'entraînement.
    """
    # 1. Extraction Header (Regex simple)
    def get_header(text, field):
        match = re.search(rf"{field}\s*:(.*)", text, re.IGNORECASE)
        return match.group(1).strip() if match else "unknown"

    # Tentative de séparation Header/Body
    try:
        parts = raw_text.split("\n\n", 1)
        body = parts[1] if len(parts) > 1 else raw_text
    except:
        body = raw_text

    subj = get_header(raw_text, "Subject")
    from_ = get_header(raw_text, "From")
    
    # 2. Features Statistiques
    suspicious_words = ["urgent", "verify", "account", "update", "bank", "suspend", "click", "password", "security", "login"]
    
    url_count = len(re.findall(r"http\S+|www\S+", raw_text))
    susp_word_count = sum(raw_text.lower().count(w) for w in suspicious_words)
    caps_ratio = sum(1 for c in raw_text if c.isupper()) / len(raw_text) if len(raw_text) > 0 else 0
    
    # 3. NLP
    preprocessor = EmailPreprocessor()
    message_combined = f"{subj} {body}"
    processed_body = preprocessor.preprocess(message_combined)

    # 4. DataFrame (Ordre des colonnes CRUCIAL)
    df = pd.DataFrame([{
        'processed_body': processed_body,
        'from_': from_,
        'subject': subj,
        'url_count': url_count,
        'susp_word_count': susp_word_count,
        'caps_ratio': caps_ratio
    }])
    
    return df

# ==========================================
# 3. INTERFACE STREAMLIT
# ==========================================
st.title("🛡️ Phishing Email Detector")
st.markdown("Collez le contenu d'un email (avec les headers si possible) pour analyser s'il est malveillant.")

# Chargement du modèle avec chemin spécifique
@st.cache_resource
def load_model():
    # Monte d'un dossier → va dans /models
    base_dir = os.path.dirname(os.path.abspath(__file__))   # chemin vers src/
    model_path = os.path.join(base_dir, "..", "models", "best_phishing_model.pkl")
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        st.error(f"❌ Modèle introuvable : {model_path}")
        st.info("Assurez-vous que 'best_phishing_model.pkl' est dans /models.")
        return None

    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

# Zone de texte
email_input = st.text_area("Contenu de l'email", height=300, placeholder="From: ...\nSubject: ...\n\nDear Customer...")

if st.button("🔍 Analyser l'email"):
    if not email_input:
        st.warning("Veuillez coller un texte d'abord.")
    elif model:
        # Préparation
        input_df = extract_features_from_text(email_input)
        
        # Prédiction
        prediction = model.predict(input_df)[0]
        
        # Affichage Résultat
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.error("🚨 PHISHING DÉTECTÉ")
                # Icône Warning
                st.markdown("## ⚠️")
            else:
                st.success("✅ EMAIL LÉGITIME")
                # Icône Check
                st.markdown("## 🛡️")
        
        with col2:
            st.subheader("Détails de l'analyse")
            st.write(f"**Sujet détecté :** {input_df['subject'][0]}")
            st.write(f"**Expéditeur :** {input_df['from_'][0]}")
            st.write(f"**Mots suspects trouvés :** {input_df['susp_word_count'][0]}")
            st.write(f"**Liens détectés :** {input_df['url_count'][0]}")
            
            # Feature Importance (Si dispo)
            if prediction == 1:
                st.warning("⚠️ Cet email contient des indicateurs d'urgence et des liens suspects.")
            else:
                st.info("ℹ️ Le style et le vocabulaire correspondent à une communication normale.")

st.markdown("---")
st.caption("Développé par l'équipe Data Science - Modèle entraîné sur Enron & Phishing Corpus")
