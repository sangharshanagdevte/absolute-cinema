import streamlit as st
import pandas as pd
import ast
import numpy as np
import plotly.graph_objects as go
from matplotlib.ticker import MaxNLocator
import os
# ----------------------
# Genre Penetration Heatmap + Single-Genre Trend + Top Film per Genre
# ----------------------
def render_genre_heatmap(df):
    """
    Render a hoverable heatmap showing counts of female-centric titles by genre and year,
    with an adjustable year-range slider, plus a line chart for a selected genre
    and the top female-centric film per genre.
    """
    st.header("Genre Penetration Heatmap & Trend")

    # Clean dataset: drop rows with empty-region
    df_clean = df[(df['region'] != '[]') & df['region'].notnull()].copy()

    # Parse genres column (stringified JSON) into actual lists
    df_clean['parsed_genres'] = df_clean['genres'].dropna().apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    # Explode into one row per genre
    exploded = df_clean.explode('parsed_genres').dropna(subset=['parsed_genres'])
    exploded['genre_name'] = exploded['parsed_genres'].apply(
        lambda d: d.get('name') if isinstance(d, dict) and 'name' in d else None
    )
    exploded = exploded.dropna(subset=['genre_name'])

    # Ensure release_year is integer
    exploded['release_year'] = exploded['release_year'].astype(int)

    # Build pivot table: genre_name x release_year
    counts = (
        exploded
        .groupby(['genre_name', 'release_year'])
        .size()
        .unstack(fill_value=0)
    )

    # Year bounds for slider
    all_years = counts.columns.astype(int)
    slider_min = int(all_years.min())
    slider_max = int(all_years.max())

    # Year-range selector
    start_year, end_year = st.slider(
        "Select year range:",
        min_value=slider_min,
        max_value=slider_max,
        value=(slider_min, slider_max),
        step=1
    )

    # Subset data for heatmap
    years = [y for y in all_years if start_year <= y <= end_year]
    z = counts[years].values
    y_labels = counts.index.tolist()

    # Plotly heatmap for hover
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=years,
        y=y_labels,
        colorscale='Plasma',
        hovertemplate='Genre: %{y}<br>Year: %{x}<br>Count: %{z}<extra></extra>'
    ))
    fig.update_layout(
        height=600,
        xaxis_title='Release Year',
        yaxis_title='Genre',
        title=f'Genre Penetration Heatmap ({start_year}–{end_year})'
    )
    st.plotly_chart(fig, use_container_width=True)

    # -- Single-Genre Trend --
    st.subheader("Genre Trend Over Time")
    genre_list = counts.index.tolist()
    selected_genre = st.selectbox("Select a genre:", genre_list)

    # New slider specifically for the single-genre trend
    trend_start, trend_end = st.slider(
        "Select trend year range:",
        min_value=slider_min,
        max_value=slider_max,
        value=(start_year, end_year),
        step=1,
        key="trend_slider"
    )

    # Time series for selected genre
    ts = counts.loc[selected_genre]
    ts = ts[(ts.index >= trend_start) & (ts.index <= trend_end)]

    fig2 = go.Figure(data=go.Scatter(
        x=ts.index, y=ts.values, mode='lines+markers',
        hovertemplate='Year: %{x}<br>Count: %{y}<extra></extra>'
    ))
    fig2.update_layout(
        height=400,
        xaxis_title='Release Year',
        yaxis_title='Count',
        title=f'{selected_genre} Film Count ({trend_start}–{trend_end})'
    )
    st.plotly_chart(fig2, use_container_width=True)

            # -- Top film per genre --
    st.subheader("Top Female-Centric Film by Genre with Poster")
    # Display four genres per row
    genres = list(counts.index)
    rows = [genres[i:i+4] for i in range(0, len(genres), 4)]
    for quartet in rows:
        cols = st.columns(4)
        for col, genre in zip(cols, quartet):
            grp = exploded[exploded['genre_name'] == genre]
            # grp = grp[(grp['bechdel'] == True) & (grp['lead'] == True) & \
            #           (grp['ensemble'] == True) & (grp['community'] == True)]
            with col:
                st.markdown(f"**{genre}**")
                if grp.empty:
                    st.write("No female-centric film found")
                else:
                    best = grp.sort_values(
                        ['popularity', 'revenue', 'vote_average'],
                        ascending=False
                    ).iloc[0]
                    # Load local poster from 'posters' folder by title
                    title = best.get('title') or best.get('original_title')
                    file_name = f"female_centric/posters/{title}.jpg"
                    if os.path.exists(file_name):
                        st.image(file_name, use_container_width=True)
                    else:
                        st.write("Poster not available")
                    year = int(best.get('release_year', np.nan)) if pd.notna(best.get('release_year')) else ''
                    st.markdown(f"**Title**: {title}")
                    st.markdown(f"**Year**: {year}")
        st.markdown("---")

