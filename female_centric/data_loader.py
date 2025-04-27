#!/usr/bin/env python
# coding: utf-8

# In[2]:


# data_loader.py
# --------------------------------
# Centralized data-loading utilities for Female-Centric Analysis
# Assumes working directory is 'Project' containing 'Puspesh_45K_MovieLens' folder

import pandas as pd
import os
import streamlit as st
# from female_centric.Puspesh_45K_MovieLens import 
# Base directory for CSV files
BASE_DIR = os.path.join(os.getcwd(),'female_centric' ,'Puspesh_45K_MovieLens')

# Helper to build file paths
def _path(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)

@st.cache_data
def load_metadata() -> pd.DataFrame:
    """
    Load movies_metadata.csv with basic parsing.
    """
    df = pd.read_csv(_path('movies_metadata.csv'), low_memory=False)
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    return df

@st.cache_data
def load_credits() -> pd.DataFrame:
    """
    Load credits.csv (cast & crew JSON strings).
    """
    df = pd.read_csv(_path('credits.csv'))
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    return df

@st.cache_data
def load_keywords() -> pd.DataFrame:
    """
    Load keywords.csv and parse JSON string into list of names.
    """
    df = pd.read_csv(_path('keywords.csv'))
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df['keywords_list'] = df['keywords'].apply(
        lambda x: [kw['name'] for kw in eval(x)] if pd.notna(x) else []
    )
    return df[['id', 'keywords_list']]

@st.cache_data
def load_links() -> pd.DataFrame:
    """
    Load links.csv mapping MovieLens movieId to TMDb id and IMDb id.
    """
    df = pd.read_csv(_path('links.csv'))
    df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce')
    df['tmdbId']  = pd.to_numeric(df['tmdbId'], errors='coerce')
    df['imdbId']  = pd.to_numeric(df['imdbId'], errors='coerce')
    df = df.rename(columns={'tmdbId':'id','imdbId':'imdb_id_raw'})
    return df[['movieId','id','imdb_id_raw']]

@st.cache_data
def load_ratings_small() -> pd.DataFrame:
    """
    Load ratings_small.csv (MovieLens user ratings sample).
    """
    df = pd.read_csv(_path('ratings_small.csv'))
    return df

@st.cache_data
def load_bechdel() -> pd.DataFrame:
    """
    Load bechdel.csv (raw ratings & flags for the Bechdel test).
    """
    df = pd.read_csv(_path('bechdel.csv'))
    return df

@st.cache_data
def load_female_centric() -> pd.DataFrame:
    """
    Load precomputed female_centric_movies.csv.
    """
    df = pd.read_csv(_path('female_centric_movies.csv'))
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    return df

@st.cache_data
def load_all_data() -> dict:
    """
    Load all data sources into a single dict.
    """
    return {
        'metadata': load_metadata(),
        #'credits': load_credits(),
        #'keywords': load_keywords(),
        #'links': load_links(),
        #'ratings_small': load_ratings_small(),
        'bechdel': load_bechdel(),
        'female_centric': load_female_centric(),
    }


# In[ ]:




