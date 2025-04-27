import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests


def render_bechdel_trends(data):
    """
    Render stacked bar charts of Bechdel pass vs fail proportions by region and by decade.
    """
    st.header("Bechdel Test Passage Trends")

    # Unpack data
    meta = data.get('metadata').copy()
    bech = data.get('bechdel').copy()

    # Prepare metadata
    meta['release_date'] = pd.to_datetime(meta['release_date'], errors='coerce')
    meta['release_year'] = meta['release_date'].dt.year

    # parse primary country
    def parse_primary_country(country_str):
        if isinstance(country_str, str) and country_str.strip():
            if country_str == '[]':  # Explicitly check for empty array string
                return np.nan
            if country_str.startswith('['):
                try:
                    arr = eval(country_str)
                    if arr and isinstance(arr, list):
                        return arr[0].get('name', 'Unknown')
                except:
                    pass
            return country_str.split(',')[0].strip()
        return np.nan  # Return NaN for empty regions

    meta['region'] = meta['production_countries'].apply(parse_primary_country)
    
    # Remove rows with empty region field
    meta = meta.dropna(subset=['region'])

    # assign decade
    def assign_decade(y):
        try:
            d = int(y) // 10 * 10
            return f"{d}s"
        except:
            return np.nan

    meta['decade'] = meta['release_year'].apply(assign_decade)

    # Prepare bechdel
    bech['imdb_id'] = bech['imdbid'].astype(str).str.zfill(7).radd('tt')
    bech['pass'] = bech['rating'] == 3

    # Merge
    df = meta.merge(bech[['imdb_id','pass']], on='imdb_id', how='inner')
    df = df.dropna(subset=['release_year','region','decade'])
    
    # Create a category column for plotting
    df['test_result'] = df['pass'].apply(lambda x: 'Pass' if x else 'Fail')
    
    # Aggregation by region
    region_agg = df.groupby(['region', 'test_result']).size().reset_index(name='count')
    region_totals = region_agg.groupby('region')['count'].sum().reset_index(name='total')
    region_agg = region_agg.merge(region_totals, on='region')
    region_agg['proportion'] = region_agg['count'] / region_agg['total']
    
    # Sort by total count to get most represented regions first
    region_order = region_totals['region'].tolist()
    
    # Plot by region using Plotly with HORIZONTAL bars
    st.subheader("Pass vs Fail by Region")
    fig1 = px.bar(
        region_agg,
        y='region',
        x='proportion',
        color='test_result',
        color_discrete_map={'Fail': 'salmon', 'Pass': 'teal'},
        custom_data=['count', 'total'],
        barmode='stack',
        height=max(500, len(region_order) * 20)
    )
    fig1.update_traces(
        hovertemplate='<b>%{y}</b><br>' +
                     'Result: %{customdata[0]} %{fullData.name} (%{x:.1%})<br>' +
                     'Total films: %{customdata[1]}'
    )
    fig1.update_layout(
        xaxis_title='Proportion',
        yaxis={'categoryorder': 'total ascending'},
        yaxis_title='',
        title='Proportion of Films Passing vs Failing Bechdel Test by Region',
        legend_title='Bechdel Test'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Aggregation by decade
    decade_agg = df.groupby(['decade', 'test_result']).size().reset_index(name='count')
    decade_totals = decade_agg.groupby('decade')['count'].sum().reset_index(name='total')
    decade_agg = decade_agg.merge(decade_totals, on='decade')
    decade_agg['proportion'] = decade_agg['count'] / decade_agg['total']
    
    # Sort decades chronologically
    decade_order = sorted(decade_agg['decade'].unique(), key=lambda d: int(d.rstrip('s')) if isinstance(d, str) else -1)
    decade_agg['decade'] = pd.Categorical(decade_agg['decade'], categories=decade_order, ordered=True)
    decade_agg = decade_agg.sort_values('decade')
    
    # Plot by decade using Plotly
    st.subheader("Pass vs Fail by Decade")
    fig2 = px.bar(
        decade_agg, 
        x='decade', 
        y='proportion', 
        color='test_result',
        color_discrete_map={'Fail': 'salmon', 'Pass': 'teal'},
        custom_data=['count', 'total'],
        barmode='stack',
        height=500
    )
    fig2.update_traces(
        hovertemplate='<b>%{x}</b><br>' +
                     'Result: %{customdata[0]} %{fullData.name} (%{y:.1%})<br>' +
                     'Total films: %{customdata[1]}'
    )
    fig2.update_layout(
        xaxis_title='Decade',
        yaxis_title='Proportion',
        title='Proportion of Films Passing vs Failing Bechdel Test by Decade',
        legend_title='Bechdel Test'
    )
    st.plotly_chart(fig2, use_container_width=True)
    # ===== Updated Section: 10 Recent Female-Centric Bechdel-Pass Movies with Posters (API method) =====
    st.header("10 Most Recent Female-Centric, Bechdel-Passing Movies with Posters")
    female_df = data.get('female_centric').copy()
    passed = female_df[female_df['bechdel'] == True]
    passed['release_date'] = pd.to_datetime(passed['release_date'], errors='coerce')
    recent = passed.sort_values('release_date', ascending=False).head(10)

    # Your TMDB API key (replace 'YOUR_API_KEY' with the real one)
    TMDB_API_KEY = "ed2e35fa097949e807f98615c4e9a79d"

    def get_poster_from_api(imdb_id):
        """
        Query TMDB API to get poster path given an IMDb ID.
        """
        url = f"https://api.themoviedb.org/3/find/{imdb_id}"
        params = {
            "api_key": TMDB_API_KEY,
            "external_source": "imdb_id"
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                movie_results = data.get('movie_results', [])
                if movie_results:
                    poster_path = movie_results[0].get('poster_path')
                    return poster_path
        except Exception as e:
            st.warning(f"Failed to fetch poster for {imdb_id}: {e}")
        return None

    cols = st.columns(5)
    for idx, row in recent.iterrows():
        imdb_id = row['imdb_id']
        title = row.get('title', '')
        year = pd.to_datetime(row.get('release_date')).year if pd.notnull(row.get('release_date')) else ''
        poster_path = get_poster_from_api(imdb_id)
        col = cols[idx % 5]
        with col:
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}"
                st.image(poster_url, caption=f"{title} ({year})", use_container_width=True)
            else:
                st.write(f"No poster for {title} ({year})")
    
  