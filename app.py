import streamlit as st
import nltk
import os
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Create a local path for nltk_data
nltk_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_path, exist_ok=True)

# Tell nltk to use this path
nltk.data.path.append(nltk_path)

# Download necessary resources if not already present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_path)
    stop_words = set(stopwords.words('english'))

try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download('punkt', download_dir=nltk_path)

stemmer = PorterStemmer()

# Load trained model and vectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.title("🧠 Sentiment Analysis of Product Reviews")
st.write("Enter a product review below to analyze its sentiment:")

user_input = st.text_area("✍️ Enter Review:")

if st.button("Analyze"):
    cleaned = preprocess(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    st.subheader("Result:")
    st.success(f"This review is **{prediction.upper()}**")
