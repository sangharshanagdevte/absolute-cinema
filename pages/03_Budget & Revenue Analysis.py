import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import streamlit as st
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import summary_table

st.set_page_config(page_title='Budget and Revenue analysis',page_icon="💰💰",layout="wide")

# # Creating a sidebar menu-------
# with st.sidebar:
#     selected = option_menu(
#         "Budget and Revenue Analysis",
#         ["Overall", "Genre", "Runtime", "Rating", "Month of Release"],
#         icons=["bar-chart", "film", "clock", "star", "calendar"],
#         menu_icon="cast",
#         default_index=0,
#         styles={
#             "container": {"font-family": "Roboto","background-color": "#A3C3DF"},
#             "nav-link": {"font-size": "18px", "font-family": "Roboto"},
#             "nav-link-selected": {"font-size": "18px", "font-family": "Roboto","background-color": "#4b8dffb0"},
#             "icon": {"font-size": "18px"}
#
#         }
#     )
# Background Image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://i.postimg.cc/bv95s6TW/background-streamlit.png");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
     <link rel="preconnect" href="https://fonts.googleapis.com">
     <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
     <link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100..900;1,100..900&display=swap" rel="stylesheet">
     <link href="https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
     h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #E8E8E8; /* Light blue for sidebar */
    }
    .stApp {
        background-color: #F8F8F8; /* Light gray or any color you want */
        font-family: 'Montserrat', sans-serif !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: #F8EEDF; /* Blue */
    }

    </style>
""", unsafe_allow_html=True)

# Loading the data --------
myclient = pymongo.MongoClient(
    "mongodb+srv://prav906715:prG9rsEp8VWWml7R@cluster0.ql4uvtn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = myclient["imdb"]
imdb = db['imdb_cleaned']


# Helper Functions-------
def getMovie(imdb_id):
    for x in imdb.find({'imdb_id': imdb_id}):
        return x
    return None


def my_subheader(text):
    st.subheader(text, divider=True)


# Beautifying the ols results
def extract_ols_summary(results):
    # Calculate Durbin-Watson separately
    dw_stat = durbin_watson(results.resid)

    # Extract general model information
    model_info = {
        'R-squared': results.rsquared,
        'Adj. R-squared': results.rsquared_adj,
        'F-statistic': results.fvalue,
        'Prob (F-statistic)': results.f_pvalue,
        'No. Observations': int(results.nobs),
        'AIC': results.aic,
        'BIC': results.bic,
        'Durbin-Watson': dw_stat
    }

    # Extract coefficients
    coef_df = pd.DataFrame({
        'coef': results.params,
        'p-value': results.pvalues,
        'CI_lower (2.5%)': results.conf_int()[0],
        'CI_upper (97.5%)': results.conf_int()[1]
    })

    # Create two tables
    model_info_df = pd.DataFrame(model_info.items(), columns=['Metric', 'Value'])
    coef_info_df = coef_df.reset_index().rename(columns={'index': 'Variable'})

    return model_info_df, coef_info_df


def highlight_pvalues(s):
    # Highlight p-values < 0.05
    return ['background-color: lightgreen' if (s.name == 'p-value' and v < 0.05) else '' for v in s]


def highlight_coefficients(df):
    # Highlight large positive and large negative coefficients
    styles = []
    for col in df.columns:
        if col == 'coef':
            styles.append(['color: green' if v > 0 else 'color: red' for v in df[col]])
        else:
            styles.append(['' for _ in df[col]])
    return pd.DataFrame({col: style for col, style in zip(df.columns, styles)})


# -----------------------------------------------------------------------------
# ---------------1.Overall Analysis--------------------------------------------
# -----------------------------------------------------------------------------

# ---Overall Helper functions------------
def plotDistribution_overall(revenue_arr, budget_arr):
    revenue_arr = np.array(revenue_arr)
    budget_arr = np.array(budget_arr)
    million = 1000000
    margin = 50 * million
    # mx = max(revenue_arr.max(), budget_arr.max())
    mx = 499 * million

    current = 0

    revenue_dict = {}
    budget_dict = {}
    while (current <= mx):
        revenue_dict[current] = None
        budget_dict[current] = None
        current += margin

    for rev in revenue_arr:
        if (rev > mx):
            continue
        rng = (rev // margin) * margin
        if (revenue_dict[rng] == None):
            revenue_dict[rng] = 0
        revenue_dict[rng] += 1

    for bud in budget_arr:
        if (bud > mx):
            continue
        rng = (bud // margin) * margin
        # print(bud, rng)
        if (budget_dict[rng] == None):
            budget_dict[rng] = 0
        budget_dict[rng] += 1

    label_arr = []
    revenue_dist = []
    budget_dist = []

    for key in revenue_dict:
        label_arr.append('{}M - {}M'.format(key // million, (key + margin) // million))
        revenue_dist.append(revenue_dict[key])
        budget_dist.append(budget_dict[key])

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=label_arr,
            y=revenue_dist,
            text=revenue_dist,
            name="Revenue",
            mode='lines + markers',
            line_color='#5CB338',
            line_shape='spline'
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=label_arr,
            y=budget_dist,
            name="Budget",
            mode='lines + markers',
            line_color='#FB4141',
            line_shape='spline'
        ),
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(
        height=600,
        width=800,
    )
    return fig


def plot_regression_line_overall(results, xrr, yrr):
    st, data, ss2 = summary_table(results, alpha=0.05)

    fittedvalues = data[:, 2]
    predict_mean_se = data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    predict_ci_low, predict_ci_upp = data[:, 6:8].T

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=yrr,
            name="Data points",
            mode='markers',
            line_color='rgba(153, 153, 255, .6)'
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=fittedvalues,
            name="Regression line",
            # mode='lines',
            # line_color='green'
            line=dict(color='green', width=3)
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_ci_low,
            name="95% prediction band",
            line=dict(color='rgba(153, 0, 51, .5)', width=1, dash='dot')
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_ci_upp,
            line=dict(color='rgba(153, 0, 51, .5)', width=1, dash='dot'),
            showlegend=False
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_mean_ci_low,
            name="95% prediction region",
            mode='lines',
            line=dict(color='rgba(0, 153, 51, .5)', width=1, dash='dot'),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_mean_ci_upp,
            line=dict(color='rgba(0, 153, 51, .5)', width=1, dash='dot'),
            showlegend=False
        ),
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(
        height=600,
        width=800,
    )
    # fig.update_layout(
    #     # margin=dict(l=10, r=10, t=10, b=10),
    #     paper_bgcolor="LightSteelBlue",
    # )
    fig.update_layout(
        # title="Plot Title",
        xaxis_title="Budget",
        yaxis_title="Revenue",
        # legend_title="Legend Title",
    )

    return fig


# ----Overall Main Function--------------
def overall_analysis():
    my_subheader("Movie Revenue Distribution (Log Binned)")
    budget_arr = []
    revenue_arr = []

    for movie in imdb.find():
        budget = movie['cleaned_Budget']
        revenue = movie['cleaned_Revenue']

        budget_arr.append(budget)
        revenue_arr.append(revenue)
    rev_arr = np.array(revenue_arr)
    rev_arr = rev_arr[rev_arr > 0]
    num_bins = 30
    log_bins = np.logspace(np.log10(rev_arr.min()), np.log10(rev_arr.max()), num=num_bins)
    counts, bins = np.histogram(rev_arr, bins=log_bins)
    x_labels = [f"${int(lo) // 1_000_000}M–${int(hi) // 1_000_000}M" for lo, hi in zip(bins[:-1], bins[1:])]
    fig = plt.figure(figsize=(14, 6))
    plt.bar(x_labels, counts, width=1.0, color='#e2d810', linewidth=4, edgecolor='w')
    plt.xticks(rotation=40, ha='right')
    plt.xlabel('Revenue Range')
    plt.ylabel('Number of Movies')
    plt.title('Movie Revenue Distribution (Log Binned)')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

    my_subheader("Movie Budget Distribution (Log Binned)")
    bud_arr = np.array(budget_arr)
    bud_arr = bud_arr[bud_arr > 0]
    num_bins = 30
    log_bins = np.logspace(np.log10(bud_arr.min()), np.log10(bud_arr.max()), num=num_bins)
    counts, bins = np.histogram(bud_arr, bins=log_bins)
    x_labels = [f"${int(lo) // 1_000_000}M–${int(hi) // 1_000_000}M" for lo, hi in zip(bins[:-1], bins[1:])]
    fig = plt.figure(figsize=(14, 6))
    plt.bar(x_labels, counts, width=1.0, color='#F8B55F', linewidth=4, edgecolor='w')
    plt.xticks(rotation=40, ha='right')
    plt.xlabel('Budget Range')
    plt.ylabel('Number of Movies')
    plt.title('Movie Budget Distribution (Log Binned)')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

    my_subheader("Combined Budget and Revenue plot")
    fig = plotDistribution_overall(revenue_arr, budget_arr)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        # paper_bgcolor="LightSteelBlue",
    )

    fig.update_layout(
        font_family="Times New Roman",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="red",
        legend_title_font_color="green",
    )

    fig.update_layout(
        title="",
        xaxis_title="USD",
        yaxis_title="Count",
        # legend_title="Legend Title",
    )

    fig.update_layout(font=dict(size=18))
    st.plotly_chart(fig, use_container_width=True, key="Combined Budget and Revenue plot")

    my_subheader("Regression Plot")
    # budget_x = sm.add_constant(budget_arr)
    # model = sm.OLS(revenue_arr, budget_x)
    # results_overall = model.fit()
    #
    # fig = plot_regression_line_overall(results_overall, budget_arr, revenue_arr)
    # fig.update_layout(
    #     margin=dict(l=10, r=10, t=10, b=10),
    #     # paper_bgcolor="LightSteelBlue",
    # )
    #
    # fig.update_layout(
    #     font_family="Times New Roman",
    #     font_color="black",
    #     title_font_family="Times New Roman",
    #     title_font_color="red",
    #     legend_title_font_color="green",
    # )
    # fig.update_layout(font=dict(size=18))
    # st.plotly_chart(fig, use_container_width=True, key="Combined Budget and Revenue Regrsssion plot")

    fig = plt.figure(figsize=(12, 9))
    sns.regplot(x=budget_arr, y=revenue_arr, scatter_kws={'alpha': 0.3}, line_kws={"color": "darkgreen"})
    plt.title('Budget vs Worldwide Revenue')  # Set title.
    plt.xlabel("Budget \n in hundred millions")  # Set x-axis label.
    plt.ylabel('Worldwide Revenue \n in billions')  # Set y-axis label.
    sns.despine()
    st.pyplot(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# ---------------2.Genre Analysis---------------------------------------------
# -----------------------------------------------------------------------------

# ---Genre Helper functions------------

# finding top genre actors name
def get_top_cast_by_genre(genre_name):
    # MongoDB query to filter by genre
    query = {"genre": genre_name, "actor.0.name": {"$exists": True}, "cleaned_Revenue": {"$exists": True},
             "cleaned_year": {"$exists": True}}

    cursor = imdb.find(query, {
        "name": 1,
        "actor": 1,
        "cleaned_Revenue": 1,
        "cleaned_year": 1
    })

    data = []
    for doc in cursor:
        try:
            movie_name = doc['name']
            name = doc['actor'][0]['name']
            revenue = doc['cleaned_Revenue']
            year = doc['cleaned_year']
            data.append({"Top Actors": name, "Highest Revenue": revenue, "Year": year, "Movie": movie_name})
        except Exception as e:
            continue  # skip entries with unexpected structure
    df = pd.DataFrame(data)

    # Group by actor and find the year with maximum revenue
    top_actors = (
        df.sort_values(by="Highest Revenue", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return top_actors


def getValues(st_year, nd_year, entity):
    info = {}
    info['total_revenue'] = 0
    info['total_budget'] = 0
    info['num_movies'] = 0
    info['num_raters'] = 0
    for year in range(st_year, nd_year + 1):
        year = str(year)
        info['total_revenue'] += entity['year-wise-performance'][year]['sum_revenue']
        info['total_budget'] += entity['year-wise-performance'][year]['sum_budget']
        info['num_movies'] += entity['year-wise-performance'][year]['num_movies']
        info['num_raters'] += entity['year-wise-performance'][year]['num_raters']
    return info


def plotFigure_genre(xrr, yrr, zrr):
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=yrr,
            name="Revenue",
            mode='lines+markers',
            line_color="rgba(102, 0, 204, .8)"
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=zrr,
            name="Budget",
            mode='lines+markers',
            line_color='rgba(153, 0, 51, .8)'
        ),
    )
    return fig


def plotBars_genre(dct, isList=False):
    xrr = []
    yrr = []
    for key in dct:
        xrr.append(key)
        if (isList == True):
            yrr.append(len(dct[key]))
        else:
            yrr.append(dct[key])
    fig = go.Figure([go.Bar(x=xrr, y=yrr, marker_color='#bfd8b8')])
    return fig


# --Displaying a single genre--
def top_genre(genre_features, fav_gen, key=None):
    genre = genre_features[fav_gen]
    # finding the first movie year
    rolling_year = 5
    present = 2020
    first_movie_year = 1967

    for year in genre['year-wise-performance']:
        if (genre['year-wise-performance'][year]['num_movies'] != 0):
            first_movie_year = int(year)
            break

    # Now finding the budget and revenue of the genre year by year
    year_label = []
    revenue_arr = []
    budget_arr = []
    num_movie_arr = []

    for year in range(first_movie_year + rolling_year - 1, present):
        st_year = year - rolling_year + 1
        nd_year = year
        info = getValues(st_year, nd_year, genre)
        year_label.append('{} - {}'.format(st_year, nd_year))

        if (info['num_movies'] != 0):
            revenue_arr.append(info['total_revenue'] / info['num_movies'])
            budget_arr.append(info['total_budget'] / info['num_movies'])
            num_movie_arr.append(info['num_movies'])
        else:
            revenue_arr.append(0)
            budget_arr.append(0)
            num_movie_arr.append(0)

            # Plotting the figure
    fig = plotFigure_genre(year_label, revenue_arr, budget_arr)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(
        height=600,
        width=800,
    )

    fig.update_layout(
        title=fav_gen,
        xaxis_title="Year",
        yaxis_title="Revenue",
        # legend_title="Legend Title",
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        # paper_bgcolor="LightSteelBlue",
    )

    fig.update_layout(
        font_family="Times New Roman",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="black",
        legend_title_font_color="green",
    )
    fig.update_layout(font=dict(size=18))
    st.plotly_chart(fig, use_container_width=True, key=key)


# ---Genre-Main Function----
def genre_analysis():
    genre_features = {}
    with open('SavedFeatures/genre_features.json', 'r') as f:
        genre_features = json.load(f)
    col1, col2 = st.columns((2, 1))

    # First column: Plot genre vs count
    with col1:
        my_subheader("Genre vs Movie Count")
        genre_track = {}
        with open('SavedFeatures/genre_track.json', 'r') as f:
            genre_track = json.load(f)

        fig = plotBars_genre(genre_track, isList=True)
        fig.update_layout(
            title="Movie count plot",
            height=600,
            width=800,
            yaxis_title="Count",
            legend_title="Legend Title",
            margin=dict(l=10, r=10, t=10, b=10),
            font_family="Times New Roman",
            font_color="black",
            title_font_family="Times New Roman",
            title_font_color="Black",
            legend_title_font_color="green",
            font=dict(size=18)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Second column: Show table
    with col2:
        my_subheader("Genre Count Table")
        genre_df = pd.DataFrame([
            {'Genre': genre,
             'Total Movies': data['total_movies'],
             'Total Revenue': data['total_revenue']}
            for genre, data in genre_features.items()
        ])
        genre_df = genre_df.sort_values(by='Total Movies', ascending=False).reset_index(drop=True)
        st.dataframe(genre_df)

    st.header("Top Genre Analysis")
    col3, col4 = st.columns(2)
    with col3:
        top_genre(genre_features, "Mystery", key="Mystery_plot")
    with col4:
        top_genre(genre_features, "Comedy", key="Comedy_plot")

    col5, col6 = st.columns(2)
    with col5:
        top_genre(genre_features, "Drama", key="Drama_plot")
    with col6:
        top_genre(genre_features, "Thriller", key="Thriller_plot")

    selected_genre = st.selectbox(
        "Choose a Genre to display the analysis",
        ('Comedy', 'Romance', 'Crime', 'Drama', 'Action', 'Thriller', 'Adventure', 'Fantasy', 'Family', 'Sci-Fi',
         'Mystery', 'Horror', 'Western', 'Biography', 'Sport', 'War', 'Music', 'Musical', 'Animation', 'History',
         'Documentary', 'News', 'Short', 'Talk-Show'),
    )
    col7, col8 = st.columns((2, 1))
    with col7:
        top_genre(genre_features, selected_genre, key="SelectedGenre_plot")

    with col8:
        top_genre_actors = get_top_cast_by_genre(selected_genre)
        st.dataframe(top_genre_actors)


# -----------------------------------------------------------------------
# -------------3.Runtime Analysis----------------------------------------
# -----------------------------------------------------------------------

# -----  Runtime Helper Functions-------
def plot_runtime_regression_line(results, xrr, yrr, title):
    st, data, ss2 = summary_table(results, alpha=0.05)

    fittedvalues = data[:, 2]
    predict_mean_se = data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    predict_ci_low, predict_ci_upp = data[:, 6:8].T

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=yrr,
            name="Data",
            mode='markers',
            line_color='rgba(153, 153, 255, .6)'
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=fittedvalues,
            name="regression line",
            # mode='lines',
            # line_color='green'
            line=dict(color='green', width=3)
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_ci_low,
            name="95% prediction band",
            line=dict(color='rgba(153, 0, 51, .5)', width=1, dash='dot')
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_ci_upp,
            line=dict(color='rgba(153, 0, 51, .5)', width=1, dash='dot'),
            showlegend=False
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_mean_ci_low,
            name="95% prediction region",
            mode='lines',
            line=dict(color='rgba(0, 153, 51, .5)', width=1, dash='dot'),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_mean_ci_upp,
            line=dict(color='rgba(0, 153, 51, .5)', width=1, dash='dot'),
            showlegend=False
        ),
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(
        height=600,
        width=800,
    )
    # fig.update_layout(
    #     # margin=dict(l=10, r=10, t=10, b=10),
    #     paper_bgcolor="LightSteelBlue",
    # )
    fig.update_layout(
        title=title,
        xaxis_title="Runtime(in minutes)",
        yaxis_title="Money(in millions)",
        legend_title="Legends",
    )

    return fig


def plotCurve_runtime(x, y, mymodel):
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name="Movies",
            mode='markers',
            line_color='rgba(153, 153, 255, .6)'
        ),
    )
    myline = np.linspace(1, x.max(), 100)
    fig.add_trace(
        go.Scatter(
            x=myline,
            y=mymodel(myline),
            name="regression curve",
            # mode='lines',
            # line_color='green'
            line=dict(color='green', width=3)
        ),
    )

    return fig


# ---Runtime- Main Function---
def runtime_analysis():
    runtime_arr = []
    revenue_arr = []
    budget_arr = []

    for movie in imdb.find():
        runtime = movie['cleaned_Runtime_min']
        revenue = movie['cleaned_Revenue']
        budget = movie['cleaned_Budget']

        runtime_arr.append(runtime)
        revenue_arr.append(revenue)
        budget_arr.append(budget)

    # If your arrays are not already in numpy format, convert them
    runtime_arr = np.array(runtime_arr)
    revenue_arr = np.array(revenue_arr) / 1e6  # Convert to millions
    budget_arr = np.array(budget_arr) / 1e6  # Convert to millions

    # Create scatter traces
    revenue_trace = go.Scatter(
        x=runtime_arr,
        y=revenue_arr,
        mode='markers',
        name='Revenue (in millions)',
        marker=dict(color='green')
    )

    budget_trace = go.Scatter(
        x=runtime_arr,
        y=budget_arr,
        mode='markers',
        name='Budget (in millions)',
        marker=dict(color='blue')
    )

    # Layout
    layout = go.Layout(
        title='Runtime vs Revenue and Budget',
        xaxis=dict(title='Runtime (minutes)'),
        yaxis=dict(title='Money (in millions USD)'),
        template='plotly_white'
    )

    # Plot
    fig = go.Figure(data=[revenue_trace, budget_trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True, key="Runtime analysis")

    col1, col2 = st.columns(2)
    with col1:
        my_subheader("Runtime vs Revenue OLS regression Results")
        runtime_x = sm.add_constant(runtime_arr)
        model = sm.OLS(revenue_arr, runtime_x)
        results1 = model.fit()
        model_info1, coef_info1 = extract_ols_summary(results1)
        st.dataframe(model_info1.style.format(precision=4))
        styled_coef = coef_info1.style.apply(highlight_pvalues, axis=0).apply(highlight_coefficients, axis=None)
        st.dataframe(styled_coef.format(precision=4))
        # st.code(results1.summary().as_text(), language='text')

    with col2:
        my_subheader("Runtime vs Budget OLS regression Results")
        runtime_x = sm.add_constant(runtime_arr)
        model = sm.OLS(budget_arr, runtime_x)
        results2 = model.fit()
        model_info2, coef_info2 = extract_ols_summary(results2)
        st.dataframe(model_info2.style.format(precision=4))
        styled_coef = coef_info2.style.apply(highlight_pvalues, axis=0).apply(highlight_coefficients, axis=None)
        st.dataframe(styled_coef.format(precision=4))
        # st.code(results2.summary().as_text(), language='text')

    col3, col4 = st.columns(2)

    with col3:
        fig = plot_runtime_regression_line(results1, runtime_arr, revenue_arr, title="Revenue Regression Plot")
        st.plotly_chart(fig, use_container_width=True, key="runtime_revenue_regression_line")
    with col4:
        fig = plot_runtime_regression_line(results2, runtime_arr, budget_arr, title="Budget Regression Plot")
        st.plotly_chart(fig, use_container_width=True, key="runtime_budget_regression_line")

    my_subheader("Polynomial Curve Fitting")

    x = np.array(runtime_arr)
    y = np.array(revenue_arr)
    mymodel = np.poly1d(np.polyfit(x, y, 6))

    fig = plotCurve_runtime(x, y, mymodel)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(
        height=600,
        width=800,
    )

    fig.update_layout(
        title="Curve Fitting",
        xaxis_title="Runtime",
        yaxis_title="Revenue",
        # legend_title="Legend Title",
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        # paper_bgcolor="LightSteelBlue",
    )

    fig.update_layout(
        font_family="Times New Roman",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="red",
        legend_title_font_color="green",
    )
    fig.update_layout(font=dict(size=18))
    st.plotly_chart(fig, use_container_width=True, key="runtime_mannual_curve_fitted_line")


# -----------------------------------------------------------------------
# -------------3.Rating Analysis----------------------------------------
# -----------------------------------------------------------------------

# -----  Rating Helper Functions-------
def plot_rating_regression_line(results, xrr, yrr):
    st, data, ss2 = summary_table(results, alpha=0.05)

    fittedvalues = data[:, 2]
    predict_mean_se = data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    predict_ci_low, predict_ci_upp = data[:, 6:8].T

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=yrr,
            name="Data",
            mode='markers',
            line_color='rgba(153, 153, 255, .6)'
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=fittedvalues,
            name="regression line",
            # mode='lines',
            # line_color='green'
            line=dict(color='green', width=3)
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_ci_low,
            name="95% prediction band",
            line=dict(color='rgba(153, 0, 51, .5)', width=1, dash='dot')
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_ci_upp,
            line=dict(color='rgba(153, 0, 51, .5)', width=1, dash='dot'),
            showlegend=False
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_mean_ci_low,
            name="95% prediction region",
            mode='lines',
            line=dict(color='rgba(0, 153, 51, .5)', width=1, dash='dot'),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=xrr,
            y=predict_mean_ci_upp,
            line=dict(color='rgba(0, 153, 51, .5)', width=1, dash='dot'),
            showlegend=False
        ),
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(
        height=600,
        width=800,
    )
    # fig.update_layout(
    #     # margin=dict(l=10, r=10, t=10, b=10),
    #     paper_bgcolor="LightSteelBlue",
    # )

    return fig


def plotBars_mpaa(dct, isList=False):
    xrr = []
    yrr = []
    for key in dct:
        xrr.append(key)
        if (isList == True):
            yrr.append(len(dct[key]))
        else:
            yrr.append(dct[key])
    fig = go.Figure([go.Bar(x=xrr, y=yrr, marker_color="#00dbff")])
    return fig


def getCluster(key, clusters):
    for cluster in clusters:
        if (key in clusters[cluster]):
            return cluster
    return key


# ----- Rating Main function-------
def rating_analysis():
    rating_arr = []
    raters_arr = []
    revenue_arr = []

    for movie in imdb.find():
        rating = float(movie['aggregateRating']['ratingValue'])
        raters = movie['aggregateRating']['ratingCount']
        revenue = movie['cleaned_Revenue']

        rating_arr.append(rating)
        raters_arr.append(raters)
        revenue_arr.append(revenue)

    rating_arr = np.array(rating_arr)
    revenue_arr = np.array(revenue_arr)
    raters_arr = np.array(raters_arr)
    rating_multiply_arr = raters_arr * rating_arr

    rating_arr = rating_arr / 10.0
    revenue_arr = (revenue_arr - revenue_arr.min()) / (revenue_arr.max() - revenue_arr.min())
    raters_arr = (raters_arr - raters_arr.min()) / (raters_arr.max() - raters_arr.min())
    rating_multiply_arr = (rating_multiply_arr - rating_multiply_arr.min()) / (
            rating_multiply_arr.max() - rating_multiply_arr.min())

    rating_x = sm.add_constant(rating_arr)
    model1 = sm.OLS(revenue_arr, rating_x)
    results_rating = model1.fit()

    raters_x = sm.add_constant(raters_arr)
    model2 = sm.OLS(revenue_arr, raters_x)
    results_raters = model2.fit()

    rating_multiply_x = sm.add_constant(rating_multiply_arr)
    model3 = sm.OLS(revenue_arr, rating_multiply_x)
    results_rating_multiply = model3.fit()

    st.header("User Based Analysis")
    col1, col2 = st.columns((1.5, 1))

    with col1:
        my_subheader("Rating vs Revenue regression plot")
        fig = plot_rating_regression_line(results_rating, rating_arr, revenue_arr)
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            # paper_bgcolor="LightSteelBlue",
        )

        fig.update_layout(
            # title="Rating vs Revenue regression plot",
            yaxis_title="Revenue (Normalized)",
            xaxis_title="IMDb rating (Normalized)",
            # legend_title="Legend Title",
        )

        fig.update_layout(
            font_family="Times New Roman",
            font_color="black",
            # title_font_family="Times New Roman",
            # title_font_color="black",
            legend_title_font_color="green",
        )
        fig.update_layout(font=dict(size=18))
        st.plotly_chart(fig, use_container_width=True, key="Rating vs Revenue regression plot")

    with col2:
        my_subheader("                                 ")
        model_info1, coef_info1 = extract_ols_summary(results_rating)
        st.dataframe(model_info1.style.format(precision=4))
        styled_coef = coef_info1.style.apply(highlight_pvalues, axis=0).apply(highlight_coefficients, axis=None)
        st.dataframe(styled_coef.format(precision=4))
        # st.code(results_rating.summary().as_text(), language='text')

    col3, col4 = st.columns((1.5, 1))

    with col3:
        my_subheader("Raters vs Revenue regression plot")
        fig = plot_rating_regression_line(results_raters, raters_arr, revenue_arr)
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            # paper_bgcolor="LightSteelBlue",
        )

        fig.update_layout(
            # title="Plot Title",
            xaxis_title="Number of raters (Normalized)",
            yaxis_title="Revenue (Normalized)",
            # legend_title="Legend Title",
        )

        fig.update_layout(
            font_family="Times New Roman",
            font_color="black",
            # title_font_family="Times New Roman",
            # title_font_color="black",
            legend_title_font_color="green",
        )
        fig.update_layout(font=dict(size=18))
        st.plotly_chart(fig, use_container_width=True, key="Raters vs Revenue regression plot")

    with col4:
        my_subheader("                                 ")
        model_info2, coef_info2 = extract_ols_summary(results_raters)
        st.dataframe(model_info2.style.format(precision=4))
        styled_coef = coef_info2.style.apply(highlight_pvalues, axis=0).apply(highlight_coefficients, axis=None)
        st.dataframe(styled_coef.format(precision=4))
        # st.code(results_raters.summary().as_text(), language='text')

    col5, col6 = st.columns((1.5, 1))

    with col5:
        my_subheader("Rating*Raters vs Revenue regression plot")
        fig = plot_rating_regression_line(results_rating_multiply, rating_multiply_arr, revenue_arr)
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            # paper_bgcolor="LightSteelBlue",
        )

        fig.update_layout(
            # title="Plot Title",
            yaxis_title="Revenue (Normalized)",
            xaxis_title="Rating*Raters(Normalized)",
            # legend_title="Legend Title",
        )

        fig.update_layout(
            font_family="Times New Roman",
            font_color="black",
            # title_font_family="Times New Roman",
            # title_font_color="black",
            legend_title_font_color="green",
        )
        fig.update_layout(font=dict(size=18))
        st.plotly_chart(fig, use_container_width=True, key="Rating*Raters vs Revenue regression plot")

    with col6:
        my_subheader("                                 ")
        model_info3, coef_info3 = extract_ols_summary(results_rating_multiply)
        st.dataframe(model_info3.style.format(precision=4))
        styled_coef = coef_info3.style.apply(highlight_pvalues, axis=0).apply(highlight_coefficients, axis=None)
        st.dataframe(styled_coef.format(precision=4))
        # st.code(results_rating_multiply.summary().as_text(), language='text')

    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(15, 10))  # Create figure object
    sns.regplot(
        x=rating_arr,
        y=revenue_arr,
        scatter_kws={'alpha': 0.2},
        line_kws={"color": "darkgreen", 'linewidth': 3},
        order=5
    )
    plt.title('Rating vs Worldwide Revenue')
    plt.xlabel("IMDb rating (Normalized)")
    plt.ylabel('Revenue (Normalized)')
    sns.despine()

    st.pyplot(fig, use_container_width=True)

    st.header("MPAA Rating Analysis")
    content_dict = {}

    for movie in imdb.find():
        rev = movie['cleaned_Revenue']
        cr = movie['contentRating']

        if (cr not in content_dict):
            content_dict[cr] = []
        content_dict[cr].append(rev)

    col7, col8 = st.columns((2, 1))

    with col7:
        my_subheader("Histogram Plot")
        fig = plotBars_mpaa(content_dict, isList=True)
        fig.update_layout(
            title="MPAA Rating Movies count",
            yaxis_title="Number of movies",
            xaxis_title="MPAA Rating",
        )

        st.plotly_chart(fig, use_container_width=True, key="Histogram Plot")

    with col8:
        my_subheader("                                 ")
        mpaa_df = pd.DataFrame([
            {'Rating': cr,
             'Total Movies': len(content_dict[cr])}
            for cr in content_dict
        ])
        st.dataframe(mpaa_df)

    clusters = {
        'PG': ['PG-13', 'PG'],
        'R': ['R', 'NC-17'] + ["Approved", "X", "M", "M/PG", "GP", "Passed", "Passed"],
        'TV': ["TV-MA", "TV-PG", "TV-14", "TV-Y7", "TV-G"] + ["Unrated", "Not Rated"],

    }
    content_dict2 = {}

    for key in content_dict:
        cluster = getCluster(key, clusters)
        if (cluster not in content_dict2):
            content_dict2[cluster] = content_dict[key]
        else:
            content_dict2[cluster] += content_dict[key]

    col9, col10 = st.columns((2, 1))

    with col9:
        my_subheader("Histogram Plot")
        fig = plotBars_mpaa(content_dict2, isList=True)
        fig.update_layout(
            title="MPAA Rating(Clustered) Movies count",
            yaxis_title="Number of movies",
            xaxis_title="MPAA Rating(Clustered)",
        )

        st.plotly_chart(fig, use_container_width=True, key="Histogram Plot 2")

    with col10:
        my_subheader("                                 ")
        mpaa_df2 = pd.DataFrame([
            {'Rating': cr,
             'Total Movies': len(content_dict2[cr])}
            for cr in content_dict2
        ])
        st.dataframe(mpaa_df2)

    col11, col12 = st.columns(2)
    with col11:
        avg_rev = {}
        for key in content_dict:
            content_dict[key] = np.array(content_dict[key])
            avg_rev[key] = content_dict[key].mean()
        fig = plotBars_mpaa(avg_rev)
        fig.update_layout(
            title="MPAA Rating Avg Revenue",
            yaxis_title="Revenue",
            xaxis_title="MPAA Rating",
        )
        st.plotly_chart(fig, use_container_width=True, key="Revenue vs MPAA rating")

    with col12:
        avg_rev2 = {}
        for key in content_dict2:
            content_dict2[key] = np.array(content_dict2[key])
            avg_rev2[key] = content_dict2[key].mean()

        fig = plotBars_mpaa(avg_rev2)
        fig.update_layout(
            title="MPAA Rating(Clustered) Avg Revenue",
            yaxis_title="Revenue",
            xaxis_title="MPAA Rating(Clustered)",
        )
        st.plotly_chart(fig, use_container_width=True, key="MPAA Rating(Clustered) Avg Revenue")


# -----------------------------------------------------------------------
# -------------3.Release Month Analysis----------------------------------------
# -----------------------------------------------------------------------

# -----  Release Month Functions-------
def plotBars_month(dct, color, isList=False):
    xrr = []
    yrr = []
    for key in dct:
        xrr.append(key)
        if (isList == True):
            yrr.append(len(dct[key]))
        else:
            yrr.append(dct[key])
    fig = go.Figure([go.Bar(x=xrr, y=yrr, marker_color=color)])
    return fig


# ----- Release Month Main function-------
def monthwise_analysis():
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    month_dict = {}
    budget_dict = {}
    for month in months:
        month_dict[month] = []
        budget_dict[month] = []

    for movie in imdb.find():
        rev = movie['cleaned_Revenue']
        month = movie['cleaned_month']
        month_dict[month].append(rev)
        budget_dict[month].append(movie['cleaned_Budget'])

    col1, col2 = st.columns((2, 1))

    with col1:
        my_subheader("Number of Movies Released in every month")
        fig = plotBars_month(month_dict, "limegreen", isList=True)
        fig.update_layout(
            # title="Number of Movies Released in every month",
            xaxis_title="Months",
            yaxis_title="Released movies",
            legend_title="Legend Title",
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            # paper_bgcolor="LightSteelBlue",
        )

        fig.update_layout(
            font_family="Times New Roman",
            font_color="black",
            # title_font_family="Times New Roman",
            # title_font_color="red",
            legend_title_font_color="green",
        )
        fig.update_layout(font=dict(size=18))
        st.plotly_chart(fig, use_container_width=True, key="Total movies per month")

    with col2:
        my_subheader("Total Count")
        total_release = pd.DataFrame([
            {'Month': month,
             'Total Movies': len(month_dict[month])}
            for month in month_dict
        ])
        st.dataframe(total_release)

    col3, col4 = st.columns(2)
    with col3:
        my_subheader("Average Revenue per Month")
        avg_rev = {}
        track = []
        rev_arr = []
        for key in month_dict:
            month_dict[key] = np.array(month_dict[key])
            avg_rev[key] = month_dict[key].mean()
            track.append((avg_rev[key], key))
            rev_arr.append(avg_rev[key])

        fig = plotBars_month(avg_rev, "red")
        fig.update_layout(
            # title="Monthly Revenue",
            xaxis_title="Months",
            yaxis_title="Average Revenue per Month",
            legend_title="Legend Title",
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            # paper_bgcolor="LightSteelBlue",
        )

        fig.update_layout(
            font_family="Times New Roman",
            font_color="black",
            # title_font_family="Times New Roman",
            # title_font_color="red",
            legend_title_font_color="steelblue",
        )
        fig.update_layout(font=dict(size=18))
        st.plotly_chart(fig, use_container_width=True, key="Average Revenue per Month")

    with col4:
        my_subheader("Average Budget per Month")
        avg_bjt = {}
        for key in budget_dict:
            avg_bjt[key] = np.array(budget_dict[key]).mean()
        fig = plotBars_month(avg_bjt, "yellow")
        fig.update_layout(
            title="Monthly Budget",
            xaxis_title="Months",
            yaxis_title="Average Budget per Month",
            legend_title="Legends",
        )
        st.plotly_chart(fig, use_container_width=True, key="Average Budget per Month")


# -------------------------- Calling funtions after selection----------------
# if selected == "Overall":
#     st.title("Overall Budget and Revenue Analysis")
#     overall_analysis()
#
# elif selected == "Genre":
#     st.title("Genre Analysis")
#     genre_analysis()
#
# elif selected == "Runtime":
#     st.title("Runtime Analysis")
#     runtime_analysis()
#
# elif selected == "Rating":
#     st.title("Rating Analysis")
#     rating_analysis()
#
# elif selected == "Month of Release":
#     st.title("Month of Release Analysis")
#     monthwise_analysis()
# Create horizontal tabs
st.title("Budget and Revenue Analysis")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊Overall", "🎥Genre", "⌚Runtime", "🌟Rating", "📆Month of Release"])

with tab1:
    st.header("Overall Budget and Revenue Analysis", divider=True)
    overall_analysis()

with tab2:
    st.header("Genre Analysis", divider=True)
    genre_analysis()

with tab3:
    st.header("Runtime Analysis", divider=True)
    runtime_analysis()

with tab4:
    st.header("Rating Analysis", divider=True)
    rating_analysis()

with tab5:
    st.header("Month of Release Analysis", divider=True)
    monthwise_analysis()
