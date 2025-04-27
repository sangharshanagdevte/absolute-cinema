import streamlit as st
import os
import base64
from female_centric.data_loader import load_all_data
from female_centric.viz_intro import render_intro
from female_centric.viz_production import render_production
from female_centric.viz_genre_heatmap import render_genre_heatmap
from female_centric.viz_revenue import render_revenue
from female_centric.viz_bechdel_trends import render_bechdel_trends
from female_centric.viz_global_map import render_global_map

# Set wide layout and page title
st.set_page_config(layout="wide", page_title="Female-Centric Visualizations")

# Embed Broadway font for main title & set tab font-size
def _embed_heading_and_tab_font():
    font_path = os.path.join(os.getcwd(), 'broadway-font', 'BROADWAY.ttf')
    try:
        with open(font_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        css = f"""
        <style>
        /* Define Broadway font */
        @font-face {{
            font-family: 'Broadway';
            src: url(data:font/truetype;base64,{b64}) format('truetype');
        }}
        /* Apply Broadway to only the main title wrapper */
        #main-title h1 {{
            font-family: 'Broadway', sans-serif !important;
            font-size: 72px !important;
        }}
        #data-card {{
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: right;
            min-width: 200px;
        }}
        #data-card p {{
            margin: 0;
            padding: 3px 0;
            font-size: 16px;
        }}
        /* Increase tab label font size */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
            font-size: 18px !important;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Font load error: {e}")

# Initialize fonts before rendering title
_embed_heading_and_tab_font()

# Main page title wrapped for scoped styling
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <div id="main-title">
            <h1>Female-Centric Visualizations</h1>
        </div>
        <div style="background-color: #f0f2f6; padding: 10px 15px; border-radius: 5px; text-align: left; min-width: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-size: 16px;">#Movies: 45,466</p>
            <p style="margin: 0; font-size: 16px;">#Female_Centric: 13,492</p>
            <p style="margin: 0; font-size: 16px;">Data: MovieLens & TMDb</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# st.markdown(
#     """
#     <div id='main-title'>
#       <h1>Female-Centric Visualizations</h1>
#     </div>
#     <div id="data-card">
#             <p><strong>#movies        :</strong> 45,466</p>
#             <p><strong>#female centric:</strong> 13,492</p>
#             <p><strong>Data :</strong> MovieLens & TMDb</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# Load data
data = load_all_data()

# Render with tabs
tab_names = [
    "Intro",
    "Production Volume",
    "Genre Penetration Heatmap",
    "Budget & Revenue Analysis",
    "Bechdel Test Trends",
    "Global Production & Revenue Map",
]
tabs = st.tabs(tab_names)

for name, tab in zip(tab_names, tabs):
    with tab:
        if name == "Intro":
            render_intro(data)
        elif name == "Production Volume":
            render_production(data['female_centric'],data['metadata'])
        elif name == "Genre Penetration Heatmap":
            render_genre_heatmap(data['female_centric'])
        elif name == "Budget & Revenue Analysis":
            render_revenue(data['female_centric'])
        elif name == "Bechdel Test Trends":
            render_bechdel_trends(data)
        elif name == "Global Production & Revenue Map":
            render_global_map(data)

# Footer note
st.markdown("---")
st.markdown("*Data loaded from MovieLens & TMDb datasets*")
