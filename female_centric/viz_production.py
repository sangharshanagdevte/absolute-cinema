import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
import pandas as pd
import plotly.graph_objects as go
import os, base64
def render_production(df_female, df_all):
    """
    Display:
      1. Top 5 and Bottom 5 countries by total female-centric film production.
      2. Ridgeline plot for top 10 regions.
      3. Dropdown selector with a stacked ridgeline for female-centric vs Other movies in the chosen country.
      4. Decadal bar chart for the chosen country.
    """
    st.header("Production Volume Over Time & Region")

    # --- Extract 'region' from production_countries in full metadata ---
    df_all['region'] = df_all['production_countries'].apply(
        lambda x: x[0]['name'] if isinstance(x, list) and x and isinstance(x[0], dict) and 'name' in x[0] else None
    )

        # Clean female-centric and full data (drop rows with missing or empty-region)
    df_fc_clean = df_female[(df_female['region'].notnull()) & (df_female['region'] != '[]')].copy()
    df_all_clean = df_all[(df_all['region'].notnull()) & (df_all['region'] != '[]')].copy()
    df_fc_clean['release_year'] = df_fc_clean['release_year'].astype(int)
    # Extract release_year from release_date for full metadata
    df_all_clean['release_year'] = pd.to_datetime(df_all_clean['release_date'], errors='coerce').dt.year.fillna(0).astype(int)

    # Aggregate counts by year & region for female-centric
    counts = (
        df_fc_clean
        .groupby(['release_year', 'region'])
        .size()
        .unstack(fill_value=0)
    )

    # Overall totals
    region_totals = counts.sum()
    top5 = region_totals.nlargest(5)
    bottom5 = region_totals.nsmallest(5)

    # Display top/bottom tables
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Countries by Female-Centric Production")
        st.table(top5.rename("Total Films").to_frame())
    with col2:
        st.subheader("Bottom 5 Countries by Female-Centric Production")
        st.table(bottom5.rename("Total Films").to_frame())

    # Ridgeline for top 10 regions
    st.subheader("Ridgeline: Female-Centric Production by Top Regions")
    top_regions = region_totals.nlargest(10).index
    counts_top = counts[top_regions]
    years = counts_top.index.values
    max_counts = counts_top.max()
    spacing = max_counts.mean() * 0.1
    baselines = []
    current = 0.0
    for reg in top_regions:
        baselines.append(current)
        current += max_counts[reg] + spacing

    fig , ax = plt.subplots(figsize=(16, 7))
    # Plot ridgeline
    for i, reg in enumerate(top_regions):
        y = counts_top[reg].values
        ax.fill_between(years, baselines[i], y + baselines[i], alpha=0.6)
        total = counts_top[reg].sum()
        # place label further to the right by adding 0.1% of the year span
        x_offset = years[-1] + (years.max() - years.min()) * 0.02
        ax.text(x_offset, baselines[i] + y[-1] / 2, f"{reg} ({total:,})", va='center')

    ax.set_yticks([])
    ax.set_xlabel("Release Year")
    ax.set_title("Ridgeline Plot: Female-Centric Movie Production by Top Regions")
    # extend x-axis limit to make room for labels
    ax.set_xlim(years.min(), years.max() + (years.max() - years.min()) * .01)
    # adjust subplot parameters for more right margin
    fig.subplots_adjust(right=0.4)
    plt.tight_layout()
    st.pyplot(fig)

        # Interactive selector for individual country
    st.subheader("Ridgeline: Female-Centric Production for a Country")
    country = st.selectbox("Select a country:", sorted(counts.columns.tolist()))

    # Female-centric for selected country
    df_country_fc = df_fc_clean[df_fc_clean['region'] == country]

    # Year range based on female-centric data
    years_all = np.arange(df_country_fc['release_year'].min(), df_country_fc['release_year'].max() + 1)
    fc = df_country_fc.groupby('release_year').size().reindex(years_all, fill_value=0)

    # Compute baselines for ridgeline
    peak = fc.max()
    spacing_country = peak * 0.1
    baselines_country = [0]  # single ridge baseline

        # Plot single ridgeline with hover using Plotly
    # Build cumulative baseline and trace for interactivity
    y_vals = list(fc.values)
    baseline = 0.0
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=years_all,
        y=[baseline + v for v in y_vals],
        mode='lines+markers',
        line_shape='linear',  # smooth curve
        fill='tozeroy',
        fillcolor='rgba(255, 100, 100, 0.6)',  # semi-transparent ridge color
        marker=dict(size=6),
        hovertemplate='Year: %{x}<br>Count: %{y}<extra></extra>',
        name=f'{country}'
    ))
    fig2.update_layout(
        title=f"Female-Centric Production — {country}",
        xaxis_title='Release Year',
        yaxis_title='Cumulative Count',
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)


    # Decadal bar chart for female-centric only

    dec = (
        df_country_fc
        .groupby(df_country_fc['release_year'] // 10 * 10)
        .size()
        .sort_index()
    )
    dec.index = dec.index.astype(int).astype(str) + 's'
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(dec.index, dec.values)
    ax3.set_title(f"Decadal Production (Female-Centric): {country}")
    ax3.set_xlabel("Decade")
    ax3.set_ylabel("Film Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)
