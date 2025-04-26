# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from collections import Counter
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings("ignore")

# Setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ---- Streamlit App ----
st.set_page_config(layout="wide")
st.title("🎬 Movie Review Sentiment Analyzer")

# ---- Text Preprocessing ----
def clean_text(text):
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(filtered)

def no_of_words(text):
    return len(text.split())

# ---- Load and preprocess dataset ----
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('../archive/IMDB_Dataset.csv')
    df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 2})
    df['review'] = df['review'].apply(clean_text)
    df.drop_duplicates(subset='review', inplace=True)
    df['word count'] = df['review'].apply(no_of_words)
    return df

# ---- Model Training (cached) ----
@st.cache_resource
def train_model(df):
    X = df['review']
    y = df['sentiment']
    vect = TfidfVectorizer()
    X_vect = vect.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.3, random_state=42)
    model = LinearSVC()
    model.fit(x_train, y_train)
    acc = accuracy_score(model.predict(x_test), y_test)

    # Save vectorizer and model for reuse
    joblib.dump(vect, 'vectorizer.pkl')
    joblib.dump(model, 'sentiment_model.pkl')
    return model, vect, acc

# ---- Load everything ----
df = load_and_prepare_data()
model, vect, accuracy = train_model(df)

# ---- Sidebar Info ----
st.sidebar.header("📊 Model Info")
st.sidebar.write(f"LinearSVC Accuracy: **{accuracy * 100:.2f}%**")
st.sidebar.write("Dataset size:", df.shape)

# ---- Review Analyzer ----
st.subheader("💬 Try it Yourself: Sentiment Prediction")

user_input = st.text_area("Enter a movie review below:", height=200)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Clean and predict
        cleaned = clean_text(user_input)
        vectorized = vect.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        sentiment_label = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.subheader(f"Predicted Sentiment: **{sentiment_label}**")

        # WordCloud
        wc = WordCloud(max_words=200, width=800, height=400).generate(cleaned)
        st.image(wc.to_array(), caption="🌀 Word Cloud of the Review")

        # Bar Chart of Top Words
        word_list = cleaned.split()
        word_freq = Counter(word_list)
        top_words_df = pd.DataFrame(word_freq.most_common(15), columns=["word", "count"])

        st.plotly_chart(
            px.bar(top_words_df, x="count", y="word", title="📌 Top Words in the Review", color="word"),
            use_container_width=True
        )


