import streamlit as st
import pandas as pd
import plotly.express as px
import random

# Page Config
st.set_page_config(
    page_title='Absolute Cinema',
    page_icon='🎥',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Inject Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

html, body, [class*=\"css\"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: #111;
    color: #fff;
}

.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(0,0,0,0.8);
    padding: 1rem 2rem;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.2);
}

.navbar h1 {
    color: #00FFFF;
    margin: 0;
    font-size: 30px;
    font-weight: 800;
}

.section {
    background: rgba(0,0,0,0.7);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}

.viz-container {
    padding: 1rem;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
}

</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("<div class='navbar'><h1>Absolute Cinema 🎥</h1></div>", unsafe_allow_html=True)

# Generate synthetic movie dataset
movie_list = [
    {"title": "Inception",        "year": 2010, "rating": 8.8, "revenue": 829},
    {"title": "Interstellar",     "year": 2014, "rating": 8.6, "revenue": 677},
    {"title": "The Shawshank Redemption", "year": 1994, "rating": 9.3, "revenue": 58},
    {"title": "The Godfather",    "year": 1972, "rating": 9.2, "revenue": 246},
    {"title": "The Dark Knight",  "year": 2008, "rating": 9.0, "revenue": 1004},
    {"title": "Parasite",        "year": 2019, "rating": 8.6, "revenue": 258},
    {"title": "Avengers: Endgame","year": 2019, "rating": 8.4, "revenue": 2798},
    {"title": "Joker",           "year": 2019, "rating": 8.5, "revenue": 1074},
    {"title": "Mad Max: Fury Road","year":2015, "rating": 8.1, "revenue": 378},
    {"title": "La La Land",      "year": 2016, "rating": 8.0, "revenue": 446},
]
df_movies = pd.DataFrame(movie_list)

# Section: Scatter plot of Rating vs Year
st.markdown("<div class='section viz-container'>", unsafe_allow_html=True)
st.subheader("Scientific Visualization: Rating vs. Release Year")
fig_scatter = px.scatter(
    df_movies,
    x='year',
    y='rating',
    size='revenue',
    hover_name='title',
    labels={'year':'Release Year', 'rating':'IMDb Rating', 'revenue':'Revenue (M)'},
    title='Movie Ratings Over Time (bubble size ~ revenue)'
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Section: Bar chart of Revenue
st.markdown("<div class='section viz-container'>", unsafe_allow_html=True)
st.subheader("Scientific Visualization: Revenue by Movie")
fig_bar = px.bar(
    df_movies.sort_values('revenue', ascending=False),
    x='title',
    y='revenue',
    text='revenue',
    labels={'title':'Movie Title', 'revenue':'Revenue (M)'},
    title='Box Office Revenue Comparison'
)
fig_bar.update_traces(texttemplate='%{text:.0f}M', textposition='outside')
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Call to Action
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("**Explore these interactive scientific visuals to discover how cinematic success evolves!**")
st.markdown("</div>", unsafe_allow_html=True)
