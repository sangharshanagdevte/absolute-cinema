import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image

def render_revenue(df):
    st.header("Revenue & Budget Analysis for Female-Centric Films")

    # Pre-filter: Only movies with budget > 0 and revenue > 0
    df_filtered = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    df_filtered['release_year'] = df_filtered['release_year'].astype(int)

    # --------- 1. Avg Annual Budget & Revenue (All Movies) ----------
    st.subheader("Overall Average Budget vs Revenue Over Time")

    yearly = df_filtered.groupby('release_year').agg({
        'budget': 'mean',
        'revenue': 'mean'
    }).reset_index()

    fig1 = px.line(yearly, x='release_year', y=['budget', 'revenue'],
                   labels={'value': 'USD', 'release_year': 'Year'},
                   title='Average Annual Budget vs Revenue (All Female-Centric Films)',
                   markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    # --------- 2. Country-Specific Avg Budget vs Revenue ----------
    st.subheader("Country-Specific Analysis")

    # Expand the 'production_countries' field (stringified JSON)
    df_filtered['parsed_countries'] = df_filtered['production_countries'].dropna().apply(
        lambda x: [d.get('name') for d in eval(x)] if isinstance(x, str) else []
    )
    exploded_countries = df_filtered.explode('parsed_countries').dropna(subset=['parsed_countries'])

    country_list = sorted(exploded_countries['parsed_countries'].dropna().unique())
    selected_country = st.selectbox("Select a country:", country_list)

    country_df = exploded_countries[exploded_countries['parsed_countries'] == selected_country]

    if not country_df.empty:
        country_yearly = country_df.groupby('release_year').agg({
            'budget': 'mean',
            'revenue': 'mean'
        }).reset_index()

        fig2 = px.line(country_yearly, x='release_year', y=['budget', 'revenue'],
                       labels={'value': 'USD', 'release_year': 'Year'},
                       title=f'Average Budget vs Revenue Over Time ({selected_country})',
                       markers=True)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No data available for the selected country.")

    # --------- 3. Genre-Specific Avg Budget vs Revenue ----------
    st.subheader("Genre-Specific Analysis")

    # Parse and explode genres
    df_filtered['parsed_genres'] = df_filtered['genres'].dropna().apply(
        lambda x: [d.get('name') for d in eval(x)] if isinstance(x, str) else []
    )
    exploded_genres = df_filtered.explode('parsed_genres').dropna(subset=['parsed_genres'])

    genre_list = sorted(exploded_genres['parsed_genres'].dropna().unique())
    selected_genre = st.selectbox("Select a genre:", genre_list)

    genre_df = exploded_genres[exploded_genres['parsed_genres'] == selected_genre]

    if not genre_df.empty:
        genre_yearly = genre_df.groupby('release_year').agg({
            'budget': 'mean',
            'revenue': 'mean'
        }).reset_index()

        fig3 = px.line(genre_yearly, x='release_year', y=['budget', 'revenue'],
                       labels={'value': 'USD', 'release_year': 'Year'},
                       title=f'Average Budget vs Revenue Over Time ({selected_genre})',
                       markers=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No data available for the selected genre.")

    # --------- 4. Top 5 Countries by Net Income ----------
    st.subheader("Top 5 Countries by Net Income (Revenue - Budget) from Female Centric Movies")

    exploded_countries['net_income'] = exploded_countries['revenue'] - exploded_countries['budget']
    top_countries = (exploded_countries.groupby('parsed_countries')['net_income']
                     .sum()
                     .sort_values(ascending=False)
                     .head(5)
                     .reset_index())

    st.table(top_countries.rename(columns={"parsed_countries": "Country", "net_income": "Net Income (USD)"}))

    # --------- 5. Top Movies ----------
    st.subheader("Top 5 Movies")

    df_filtered['net_income'] = df_filtered['revenue'] - df_filtered['budget']

    def display_top_movies(df, column, title):
        st.markdown(f"### {title}")

        top5 = df.sort_values(by=column, ascending=False).head(5)

        # 4 posters in a row
        cols = st.columns(5)
        for idx, (_, row) in enumerate(top5.iterrows()):
            col = cols[idx % 5]

            movie_title = row['title']
            release_year = row['release_year']
            value = row[column]
            imdb_id = row.get('imdb_id')  # <-- get imdb_id

            poster_path = os.path.join('female_centric','posters', f"{imdb_id}.jpg") if imdb_id else None

            with col:
                if poster_path and os.path.exists(poster_path):
                    image = Image.open(poster_path)
                    st.image(image, caption=f"{movie_title} ({release_year})\n${value:,.0f}", use_container_width=True)
                else:
                    st.write(f"{movie_title} ({release_year})\n${value:,.0f}")
                    st.write("Poster not available")

    display_top_movies(df_filtered, 'budget', 'Highest Budget Female Centric Movies')
    #display_top_movies(df_filtered, 'revenue', 'Highest Revenue Female Centric Movies')
    display_top_movies(df_filtered, 'net_income', 'Highest Net Income Female Centric Movies')
