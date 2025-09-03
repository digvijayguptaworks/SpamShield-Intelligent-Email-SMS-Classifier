import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')

ps = PorterStemmer()

# -------------------------
# Preprocessing Function
# -------------------------
def transform_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('english')]
    words = [ps.stem(word) for word in words]
    return " ".join(words)

# -------------------------
# Load vectorizer + model
# -------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# -------------------------
# Custom Premium Styling
# -------------------------
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background-color: #121212;
        color: #f5f5f5;
        font-family: "Segoe UI", sans-serif;
    }
    /* Title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }
    .sub-text {
        text-align: center;
        font-size: 16px;
        color: #bbbbbb;
        margin-bottom: 40px;
    }
    /* Text Area */
    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: #f5f5f5 !important;
        border-radius: 12px !important;
        border: 1px solid #444 !important;
        font-size: 16px !important;
        padding: 14px !important;
    }
    /* Button */
    .stButton>button {
        background: #2d2d2d;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #444;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #444;
        border: 1px solid #666;
    }
    /* Result Box */
    .result-box {
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 30px;
    }
    .spam {
        background-color: #2b0000;
        color: #ff4d4d;
        border: 1px solid #ff4d4d;
    }
    .ham {
        background-color: #003300;
        color: #66ff99;
        border: 1px solid #66ff99;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# UI
# -------------------------
st.markdown('<p class="main-title">📩 Spam Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Elegant • Minimal • Accurate</p>', unsafe_allow_html=True)

# Input
input_sms = st.text_area("✍️ Enter your message below:", height=140)

if st.button("🔍 Predict"):
    if input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown('<div class="result-box spam">🚨 Spam Message Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box ham">✅ Safe Message</div>', unsafe_allow_html=True)
