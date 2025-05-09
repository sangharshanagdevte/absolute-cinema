# -*- coding: utf-8 -*-
"""Big Data CS661 Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1E6qeaaPtzccoKYApLsFnvhO7iGcrT63t
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(page_title="Letterboxd Genre Analysis", layout="wide")
st.title("Letterboxd Genre-Based Movie Analysis")

# --- Data Loading ---
# st.sidebar.header("Upload your JSON data")
file = "archive/feb2021.json"

if file:
    df = pd.read_json(file)

    # Preprocessing
    df = df.dropna(subset=['year', 'genre', 'language'])
    df['year'] = df['year'].astype(int)
    df['genre'] = df['genre'].apply(lambda x: x if isinstance(x, list) else [x])
    df = df[(df['year'] >= 1950) & (df['year'] <= 2021)]

    def extract_viewers(text):
        if isinstance(text, str):
            match = re.search(r'based on ([\d,]+)', text)
            if match:
                return int(match.group(1).replace(',', ''))
        return np.nan

    def extract_rating(text):
        if isinstance(text, str):
            match = re.search(r'average of ([\d.]+)', text)
            if match:
                return float(match.group(1))
        return np.nan

    if 'viewers' in df.columns:
        df['viewers_clean'] = df['viewers'].apply(extract_viewers)

    if 'rating' in df.columns:
        df['rating_clean'] = df['rating'].apply(extract_rating)

    exploded_df = df.explode('genre')

    # --- Sidebar options ---
    st.sidebar.header("Select Visualization")
    option = st.sidebar.selectbox("Choose one:", (
        "PDF-style Genre Distribution",
        "Top Genres Over Time",
        "Genre Distribution by Language",
        "Heatmap of Genre by Language",
        "Top Genres by Average Viewers",
        "Number of Unique Genres Per Year",
        "Genre Co-occurrence Matrix",
        "Genre Distribution Pie Chart (2020)",
        "Genre Popularity by Decade"
    ))

    # --- Visualizations ---
    if option == "PDF-style Genre Distribution":
        genre_year = exploded_df.groupby(['year', 'genre']).size().reset_index(name='count')
        total_per_year = genre_year.groupby('year')['count'].transform('sum')
        genre_year['pdf'] = genre_year['count'] / total_per_year
        pdf_pivot = genre_year.pivot(index='year', columns='genre', values='pdf').fillna(0)

        st.subheader("PDF-style Genre Distribution Over Years")
        fig, ax = plt.subplots(figsize=(16, 8))
        pdf_pivot.plot.area(stacked=True, colormap='tab20', alpha=0.9, ax=ax)
        ax.set_title("PDF-style Genre Distribution Over Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("Relative Popularity")
        ax.grid(True)
        st.pyplot(fig)

    elif option == "Top Genres Over Time":
        top_genres = exploded_df['genre'].value_counts().head(6).index
        top_df = exploded_df[exploded_df['genre'].isin(top_genres)]
        line_df = top_df.groupby(['year', 'genre']).size().reset_index(name='count')

        st.subheader("Top Genres Over Time")
        fig, ax = plt.subplots()
        sns.lineplot(data=line_df, x='year', y='count', hue='genre', ax=ax)
        ax.grid(True)
        st.pyplot(fig)

    elif option == "Genre Distribution by Language":
        lang_genre = exploded_df.groupby(['language', 'genre']).size().reset_index(name='count')
        lang_pivot = lang_genre.pivot(index='language', columns='genre', values='count').fillna(0)
        top_langs = lang_pivot.sum(axis=1).sort_values(ascending=False).head(10).index

        st.subheader("Genre Distribution by Language (Top 10 Languages)")
        fig, ax = plt.subplots(figsize=(14, 6))
        lang_pivot.loc[top_langs].plot(kind='bar', stacked=True, colormap='tab20', ax=ax)
        ax.set_xlabel("Language")
        ax.set_ylabel("Number of Movies")
        st.pyplot(fig)

    elif option == "Heatmap of Genre by Language":
        top_langs = exploded_df['language'].value_counts().head(8).index
        heat_df = exploded_df[exploded_df['language'].isin(top_langs)]
        heatmap_data = heat_df.groupby(['language', 'genre']).size().unstack().fillna(0)

        st.subheader("Genre Frequency Heatmap by Language")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    elif option == "Top Genres by Average Viewers":
        if 'viewers_clean' in exploded_df.columns:
            viewer_df = exploded_df.dropna(subset=['viewers_clean'])
            viewer_avg = viewer_df.groupby('genre')['viewers_clean'].mean().sort_values(ascending=False).head(10)

            st.subheader("Top 10 Genres by Average Viewers")
            fig, ax = plt.subplots()
            viewer_avg.plot(kind='bar', color='coral', ax=ax)
            st.pyplot(fig)

    elif option == "Number of Unique Genres Per Year":
        genre_diversity = exploded_df.groupby('year')['genre'].nunique()

        st.subheader("Number of Unique Genres Per Year")
        fig, ax = plt.subplots(figsize=(14, 6))
        genre_diversity.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Genre Count")
        st.pyplot(fig)

    elif option == "Genre Co-occurrence Matrix":
        grouped = df.groupby(['title', 'year'])['genre'].apply(lambda x: list(set([i for sublist in x for i in sublist]) if all(isinstance(i, list) for i in x) else x))
        mlb = MultiLabelBinarizer()
        co_matrix = pd.DataFrame(mlb.fit_transform(grouped), columns=mlb.classes_)
        co_matrix = co_matrix.T.dot(co_matrix)

        st.subheader("Genre Co-occurrence Matrix")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(co_matrix, cmap='coolwarm', xticklabels=True, yticklabels=True, ax=ax)
        st.pyplot(fig)

    elif option == "Genre Distribution Pie Chart (2020)":
        year_data = exploded_df[exploded_df['year'] == 2020]['genre'].value_counts().head(10)

        st.subheader("Genre Distribution in 2020")
        fig, ax = plt.subplots(figsize=(8, 8))
        year_data.plot.pie(autopct='%1.1f%%', startangle=140, ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    elif option == "Genre Popularity by Decade":
        exploded_df['decade'] = (exploded_df['year'] // 10) * 10
        decade_genre = exploded_df.groupby(['decade', 'genre']).size().reset_index(name='count')

        st.subheader("Genre Popularity by Decade")
        fig = sns.catplot(data=decade_genre, x='decade', y='count', hue='genre', kind='bar', height=6, aspect=2)
        st.pyplot(fig)

else:
    st.info("👆 Upload a JSON file to get started!")