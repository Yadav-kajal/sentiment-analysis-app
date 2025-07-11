import streamlit as st
import nltk
import os
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Append local nltk_data path
nltk_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_path)

# Load required data (no downloading needed now)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load your models
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocess input
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered)

# Streamlit interface
st.title("🧠 Sentiment Analysis of Product Reviews")
user_input = st.text_area("✍️ Enter Review:")

if st.button("Analyze"):
    cleaned = preprocess(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    st.success(f"This review is **{prediction.upper()}**")
