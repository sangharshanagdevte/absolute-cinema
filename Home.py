import streamlit as st
import random
import base64
import os
import pandas as pd

# Page Config
st.set_page_config(
    page_title='Absolute Cinema',
    page_icon='🎥',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Background image load
def get_base64(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img = get_base64('c.jpeg')

# Inject Global CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif;
}}

.stApp {{
    background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("data:image/png;base64,{bg_img}") center center / cover no-repeat fixed;
}}

.navbar {{
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(0,0,0,0.8);
    padding: 1rem 2rem;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.2);
}}

.navbar h1 {{
    color: #00FFFF;
    margin: 0;
    font-size: 30px;
    font-weight: 800;
}}

.container {{
    padding: 4rem 2rem;
    animation: fadeInUp 1s ease-out;
}}

.section {{
    background: rgba(0,0,0,0.7);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 50px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6);
}}

h2 {{
    color: #FFD700;
    text-align: center;
    margin-bottom: 20px;
}}

.intro-text {{
    color: #dddddd;
    text-align: justify;
    font-size: 18px;
    line-height: 1.7;
}}

.team-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 25px;
    margin-top: 30px;
}}

.team-card {{
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    transition: 0.3s ease;
}}

.team-card:hover {{
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-8px);
}}

.team-card h4 {{
    color: #00e6e6;
    margin-bottom: 8px;
}}

.team-card p {{
    color: #FFD700;
    font-size: 15px;
}}

.poster-container img {{
    width: 100%;
    border-radius: 10px;
    max-height: 180px;
    object-fit: cover;
    transition: all 0.4s ease;
}}

.poster-container:hover img {{
    transform: scale(1.05);
    box-shadow: 0px 6px 12px rgba(0,0,0,0.9);
}}

@keyframes fadeInUp {{
    0% {{
        opacity: 0;
        transform: translateY(30px);
    }}
    100% {{
        opacity: 1;
        transform: translateY(0);
    }}
}}
</style>
""", unsafe_allow_html=True)

# Top Navbar
st.markdown("<div class='navbar'><h1>Absolute Cinema 🎥</h1></div>", unsafe_allow_html=True)

# Main Content
st.markdown("<div class='container'>", unsafe_allow_html=True)

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h2>Behind the Scenes of Movie Industry</h2>", unsafe_allow_html=True)
intro = """
In the dynamic world of cinema, understanding what resonates with audiences is crucial.  
Our data-driven journey helps uncover patterns and insights that drive iconic achievements.  
From genre trends to cultural shifts, we visualize and celebrate the art of storytelling.
"""
st.markdown(f"<div class='intro-text'>{intro}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Project Team Section
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h2>Meet Our Team</h2>", unsafe_allow_html=True)

credits = [
    ("Pushpesh Kumar Srivastava", "24111045", "Data Cleaning & Preprocessing"),
    ("Sagar Kumar", "24111060", "UI Design & CSS Styling"),
    ("Krishanu Ray", "24111037", "Data Visualization Modules"),
    ("Uddeshya Raj", "24111046", "Sentiment Analysis Implementation"),
    ("Souravdip Das", "23110051", "Recommendation Engine"),
    ("Sayak Bera", "24111068", "Backend & Streamlit Integration"),
    ("Praveen Patel", "24111004", "Data Collection & ETL"),
    ("Srinjoy Srikan", "23110050", "Statistical Analysis & Reporting"),
]

st.markdown("<div class='team-grid'>", unsafe_allow_html=True)
for name, roll, role in credits:
    st.markdown(f"""
    <div class="team-card">
        <h4>{name}</h4>
        <p>{roll}</p>
        <p>{role}</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Posters Section
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h2>Featured Films</h2>", unsafe_allow_html=True)

df = pd.read_csv("data/movies.csv")
movie_info = df.set_index('id').to_dict(orient='index')
cols = st.columns(6)
for idx, movie_id in enumerate(random.sample(list(movie_info.keys()), 10)):
    path = f"data/posters/{movie_id}.jpg"
    if os.path.exists(path):
        with cols[idx % 6]:
            with open(path, "rb") as pf:
                img_b64 = base64.b64encode(pf.read()).decode()
            st.markdown(f"<div class='poster-container'><img src='data:image/jpeg;base64,{img_b64}'></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # End container

# TODO: Homepage components
st.markdown("""
**Join us as we decode the secrets of musical success, shaping the future of music creation and engagement.**
""", unsafe_allow_html=False)
