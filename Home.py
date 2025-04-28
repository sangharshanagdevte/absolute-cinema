import streamlit as st
import random
import base64
import os
import pandas as pd
import plotly.express as px
from google_images_search import GoogleImagesSearch

# Page Config
st.set_page_config(
    page_title='Absolute Cinema',
    page_icon='🎥',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Google Images Search setup (env vars)
GCS_DEVELOPER_KEY = os.getenv('GCS_DEVELOPER_KEY')
GCS_CX = os.getenv('GCS_CX')
if not GCS_DEVELOPER_KEY or not GCS_CX:
    st.error("Set GCS_DEVELOPER_KEY and GCS_CX env vars for Google Image Search.")
    st.stop()

# Dynamic sample data for visualization
genres = ['Action', 'Drama', 'Comedy', 'Horror', 'Sci-Fi']
years = list(range(2000, 2025))
data = {'year': [], 'genre': [], 'count': []}
for genre in genres:
    for year in years:
        data['year'].append(year)
        data['genre'].append(genre)
        data['count'].append(random.randint(20, 200))
df_viz = pd.DataFrame(data)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: #000; color: #fff; }
.navbar { position: sticky; top: 0; background: rgba(0,0,0,0.8); padding: 1rem; text-align: center; }
.navbar h1 { color: #00FFFF; }
.section { padding: 2rem; margin-bottom: 2rem; background: rgba(0,0,0,0.7); border-radius: 10px; }
""", unsafe_allow_html=True)

# Navbar
st.markdown("<div class='navbar'><h1>Absolute Cinema 🎥</h1></div>", unsafe_allow_html=True)

# Interactive Visualization Section
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("Genre Trends Over Time")
selected_genres = st.multiselect("Select Genres", genres, default=genres)
filtered = df_viz[df_viz['genre'].isin(selected_genres)]
fig = px.line(filtered, x='year', y='count', color='genre', title='Number of Films by Genre per Year')
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Featured Posters via Google Search
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("Featured Film Posters")

movies = ["Inception", "Interstellar", "Parasite", "The Matrix", "Gladiator", "Whiplash", "Avengers: Endgame", "Joker", "La La Land", "Mad Max: Fury Road"]
cols = st.columns(5)
gis = GoogleImagesSearch(GCS_DEVELOPER_KEY, GCS_CX)
for idx, title in enumerate(random.sample(movies, 10)):
    params = {'q': f"{title} poster", 'num': 1, 'safe': 'high', 'fileType': 'jpg', 'imgType': 'photo', 'imgSize': 'medium'}
    gis.search(search_params=params)
    url = gis.results()[0].url if gis.results() else None
    with cols[idx%5]:
        if url:
            st.image(url, caption=title, use_column_width=True)
        else:
            st.write(f"No poster for {title}")
st.markdown("</div>", unsafe_allow_html=True)

# Call to Action
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("**Join us as we decode the secrets of cinematic success with interactive insights and live poster demos!**")
st.markdown("</div>", unsafe_allow_html=True)
