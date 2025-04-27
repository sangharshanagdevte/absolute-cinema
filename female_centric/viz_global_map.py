import streamlit as st
import pandas as pd
import pycountry
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ----------------------
# Global Production & Revenue Map
# ----------------------
def render_global_map(data):
    """
    Render two choropleth maps:
      1. Count of female-centric films by country (USA highlighted).
      2. Average revenue of female-centric films by country.
    """
    st.header("Global Production & Revenue Map")

    # Load female-centric DataFrame
    df = data['female_centric'].copy()
    df = df[df['region'].notna() & (df['region'] != '[]')]

    # Aggregate
    counts = df.groupby('region').size().reset_index(name='film_count')
    revenue = df.groupby('region')['revenue'].mean().reset_index(name='avg_revenue')
    dag = counts.merge(revenue, on='region')

    # Normalize region names for ISO lookup
    region_override = {
        'United States of America': 'United States',
        'Kyrgyz Republic':          'Kyrgyzstan',
        'Soviet Union':             'Russian Federation',
        'East Germany':             'Germany',
        'Congo':                    'Congo, The Democratic Republic of the',
        'South Korea':              'Korea, Republic of',
        'Czech Republic':           'Czechia'
    }
    dag['region_norm'] = dag['region'].replace(region_override)

    # Map to ISO3
    def country_to_iso(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except:
            return None

    dag['iso_alpha'] = dag['region_norm'].apply(country_to_iso)
    dag_clean = dag.dropna(subset=['iso_alpha'])

    # Add log-transformed columns for better visualization
    dag_clean['log_film_count'] = np.log1p(dag_clean['film_count'])
    dag_clean['formatted_revenue'] = dag_clean['avg_revenue'].apply(lambda x: f"${x:,.0f}")

    # Split USA vs Others
    others = dag_clean[dag_clean['region_norm'] != 'United States']
    usa = dag_clean[dag_clean['region_norm'] == 'United States']

    # Determine max for color scale
    max_non_usa = others['film_count'].max()

    # 1) Production Count Map
    st.subheader("Number of Female-Centric Films Produced by Country")
 
    # Others layer with enhanced hover template
    fig_count = go.Figure()
    fig_count.add_trace(go.Choropleth(
        locations=others['iso_alpha'],
        z=others['film_count'],
        zmin=0, zmax=max_non_usa,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text="Film Count",
                side="top"
            ),
            thicknessmode="pixels", 
            thickness=20,
            lenmode="fraction", 
            len=0.75
        ),
        marker_line_color='white', 
        marker_line_width=0.7,
        name='Other Countries'
    ))
     
    '''fig_count.add_trace(go.Choropleth(
        locations=others['iso_alpha'],
        z=others['film_count'],
        zmin=0, zmax=max_non_usa,
        colorscale='Viridis',  # Changed from Turbo for better perception
        colorbar=dict(
            title='Film Count',
            titleside='right',
            thicknessmode='pixels', thickness=20,
            lenmode='fraction', len=0.75,
            ticks='outside'
        ),
        marker_line_color='white', marker_line_width=0.7,
        name='Other Countries',
        hovertemplate='<b>%{hovertext}</b><br>' +
                     'Films Produced: %{z}<extra></extra>',
        hovertext=others['region']
    ))
    
    # USA layer with enhanced styling
    fig_count.add_trace(go.Choropleth(
        locations=usa['iso_alpha'],
        z=usa['film_count'],
        showscale=False,
        colorscale=[[0,'#FF4136'],[1,'#FF4136']],  # Brighter red for USA
        marker_line_color='white', marker_line_width=0.7,
        name='United States',
        hovertemplate='<b>United States</b><br>' +
                     'Films Produced: %{z}<extra></extra>',
        hovertext=usa['region']
    ))
    '''
    # USA layer
    fig_count.add_trace(go.Choropleth(
        locations=usa['iso_alpha'],
        z=usa['film_count'],
        showscale=False,
        colorscale=[[0,'#FF4136'],[1,'#FF4136']],
        marker_line_color='white', 
        marker_line_width=0.7,
        name='United States'
    ))
    fig_count.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='LightGray',
            projection_type='natural earth',
            landcolor='rgb(240, 240, 240)',
            showocean=True,
            oceancolor='aliceblue',
            showlakes=True,
            lakecolor='aliceblue'
        ),
        legend=dict(
            title='Country Groups',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        width=1000,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        title=dict(
            text='Global Distribution of Female-Centric Film Production',
            x=0.5,
            font=dict(size=18)
        ),
        font=dict(family='Arial', size=12),
        hoverlabel=dict(bgcolor='white', font_size=14, font_family='Arial')
    )
    
    st.plotly_chart(fig_count, use_container_width=True)
    
    # Add contextual note
    with st.expander("About this visualization"):
        st.markdown("""
        This map shows the global distribution of female-centric films by country of production.
        - The United States (highlighted in red) leads in production volume
        - The color intensity of other countries indicates their relative production volume
        - Hover over any country to see the exact number of films produced
        """)

    # 2) Average Revenue Map with improved aesthetics
    st.subheader("Average Revenue of Female-Centric Films by Country")
    
    fig_revenue = px.choropleth(
        dag_clean,
        locations='iso_alpha',
        color='avg_revenue',
        hover_name='region',
        color_continuous_scale='Viridis',  # Changed from Turbo
        labels={'avg_revenue': 'Avg Revenue ($)'},
        width=1000,
        height=600,
        custom_data=['film_count', 'formatted_revenue']  # For enhanced hover info
    )
    
    fig_revenue.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                     'Average Revenue: %{customdata[1]}<br>' +
                     'Films Produced: %{customdata[0]}<extra></extra>'
    )
    
    fig_revenue.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='LightGray',
        projection_type='natural earth',
        landcolor='rgb(240, 240, 240)',
        showocean=True,
        oceancolor='aliceblue',
        showlakes=True,
        lakecolor='aliceblue'
    )
    
    fig_revenue.update_layout(
        coloraxis_colorbar=dict(
            title=dict(
             text='Avg Revenue ($)',
             side='right'),
            thicknessmode='pixels', thickness=20,
            lenmode='fraction', len=0.75,
            ticks='outside',
            tickprefix='$',
            tickformat=',.0f'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        title=dict(
            text='Average Revenue of Female-Centric Films by Country',
            x=0.5,
            font=dict(size=18)
        ),
        font=dict(family='Arial', size=12),
        hoverlabel=dict(bgcolor='white', font_size=14, font_family='Arial')
    )
    
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Interactive filter option
    st.subheader("Interactive Data Exploration")
    min_films = st.slider("Minimum number of films for country inclusion:", 1, int(dag_clean['film_count'].max()), 1)
    
    filtered_data = dag_clean[dag_clean['film_count'] >= min_films]
    
    # Bar chart of top countries by film count
    fig_bar = px.bar(
        filtered_data.sort_values('film_count', ascending=False).head(20),
        x='region',
        y='film_count',
        color='avg_revenue',
        color_continuous_scale='Viridis',
        labels={'film_count': 'Number of Films', 'region': 'Country', 'avg_revenue': 'Avg Revenue ($)'},
        title=f'Top Countries by Film Count (Minimum {min_films} films)',
        hover_data=['formatted_revenue']
    )
    
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        height=700,
        coloraxis_colorbar=dict(
            title='Avg Revenue ($)',
            tickprefix='$',
            tickformat=',.0f'
        )
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

'''#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import pycountry
import plotly.express as px
import plotly.graph_objects as go

# ----------------------
# Global Production & Revenue Map
# ----------------------
def render_global_map(data):
    """
    Render two choropleth maps:
      1. Count of female-centric films by country (USA highlighted).
      2. Average revenue of female-centric films by country.
    """
    st.header("Global Production & Revenue Map")

    # Load female-centric DataFrame
    df = data['female_centric'].copy()
    df = df[df['region'].notna() & (df['region'] != '[]')]

    # Aggregate
    counts = df.groupby('region').size().reset_index(name='film_count')
    revenue = df.groupby('region')['revenue'].mean().reset_index(name='avg_revenue')
    dag = counts.merge(revenue, on='region')

    # Normalize region names for ISO lookup
    region_override = {
        'United States of America': 'United States',
        'Kyrgyz Republic':          'Kyrgyzstan',
        'Soviet Union':             'Russian Federation',
        'East Germany':             'Germany',
        'Congo':                    'Congo, The Democratic Republic of the',
        'South Korea':              'Korea, Republic of',
        'Czech Republic':           'Czechia'
    }
    dag['region_norm'] = dag['region'].replace(region_override)

    # Map to ISO3
    def country_to_iso(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except:
            return None

    dag['iso_alpha'] = dag['region_norm'].apply(country_to_iso)
    dag_clean = dag.dropna(subset=['iso_alpha'])

    # Split USA vs Others
    others = dag_clean[dag_clean['region_norm'] != 'United States']
    usa    = dag_clean[dag_clean['region_norm'] == 'United States']

    # Determine max for color scale
    max_non_usa = others['film_count'].max()

    # 1) Production Count Map
    st.subheader("Number of Female-Centric Films Produced by Country")
    fig_count = go.Figure()
    # Others layer
    fig_count.add_trace(go.Choropleth(
        locations=others['iso_alpha'],
        z=others['film_count'],
        zmin=0, zmax=max_non_usa,
        colorscale='Turbo',
        colorbar=dict(title='Film Count'),
        marker_line_color='white', marker_line_width=0.5,
        name='Others'
    ))
    # USA layer in red
    fig_count.add_trace(go.Choropleth(
        locations=usa['iso_alpha'],
        z=usa['film_count'],
        showscale=False,
        colorscale=[[0,'red'],[1,'red']],
        marker_line_color='white', marker_line_width=0.5,
        name='USA'
    ))
    fig_count.update_layout(
        geo=dict(showframe=False, showcoastlines=False),
        width=1000, height=600,
        legend_title_text='Country Group'
    )
    st.plotly_chart(fig_count, use_container_width=True)

    # 2) Average Revenue Map
    st.subheader("Average Revenue of Female-Centric Films by Country")
    fig_revenue = px.choropleth(
        dag_clean,
        locations='iso_alpha',
        color='avg_revenue',
        hover_name='region',
        color_continuous_scale='Turbo',
        labels={'avg_revenue':'Avg Revenue'},
        width=1000, height=600
    )
    fig_revenue.update_geos(showframe=False, showcoastlines=False)
    st.plotly_chart(fig_revenue, use_container_width=True)


# In[ ]:

'''


