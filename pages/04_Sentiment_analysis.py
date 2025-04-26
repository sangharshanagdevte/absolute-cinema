import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import streamlit as st

# Custom stopwords (optional, since TfidfVectorizer also handles this)
CUSTOM_STOPWORDS = set([
    "the", "and", "is", "in", "to", "of", "it", "this", "that", "was", "for",
    "on", "with", "as", "but", "be", "at", "by", "an", "have", "not", "are"
])

# Load IMDB dataset for training
df = pd.read_csv('archive/IMDB_Dataset.csv')  # Use the relative path to the dataset

# Preview the data
df.head()

# Clean the text data (remove unnecessary characters, punctuation, etc.)
def clean_text(text):
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    tokens = text.split()
    filtered = [w for w in tokens if w not in CUSTOM_STOPWORDS]
    return " ".join(filtered)

df['cleaned_review'] = df['review'].apply(clean_text)

# Convert sentiment labels to binary (positive = 1, negative = 0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Feature and target variable
X = df['cleaned_review']
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert the text data into a matrix of TF-IDF features
vect = TfidfVectorizer()
X_train_tfidf = vect.fit_transform(X_train)
X_test_tfidf = vect.transform(X_test)

# Train a Linear Support Vector Classifier (SVC)
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')


# Save the model and vectorizer for future use
joblib.dump(vect, 'vectorizer.pkl')
joblib.dump(model, 'sentiment_model.pkl')

# Streamlit UI
st.title("🎬 Movie Sentiment Analyzer")


# Sidebar Accuracy Info
st.sidebar.header("📊 Model Info")
st.sidebar.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")

# Create a text area to get user input
user_input = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    cleaned = clean_text(user_input)  # Clean the user input
    vectorized_input = vect.transform([cleaned])  # Vectorize the input using the pre-trained vectorizer
    prediction = model.predict(vectorized_input)  # Get the sentiment prediction

    # Show sentiment result
    if prediction == 1:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😞")

    # Generate a WordCloud for the review text
    st.subheader("WordCloud of frequent words in your review:")
    
    wordcloud = WordCloud(width=800, height=400, stopwords=CUSTOM_STOPWORDS).generate(cleaned)
    
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)

    # Display top words from the review
    words = cleaned.split()
    word_freq = {}
    
    # Count word frequency
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by most common
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Display top words
    st.subheader("Top words from the review:")
    top_words = sorted_word_freq[:10]  # Show top 10 frequent words
    for word, count in top_words:
        st.write(f"{word}: {count}")

