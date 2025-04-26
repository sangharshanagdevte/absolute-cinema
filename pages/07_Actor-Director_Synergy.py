import pandas as pd
import plotly.express as px
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(
    layout='wide',
)
# Load the dataset
file_path = 'archive/Indian_6Lan_Movies.csv'
try:
    movies_df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"Error: CSV file not found at '{file_path}'. Please make sure the path is correct.")
    st.stop()

# Rename columns
movies_df.rename(columns={'name': 'movie_title', 'date': 'year', 'actor': 'actors'}, inplace=True)

# Drop NaN values
movies_df.dropna(subset=['movie_title', 'year', 'director', 'actors', 'language'], inplace=True)

# Convert year to numeric
movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce').dropna().astype(int)

# Explode actors column
movies_df['actors'] = movies_df['actors'].str.split(',')
movies_df = movies_df.explode('actors').reset_index(drop=True)
movies_df['actors'] = movies_df['actors'].str.strip()

# Get unique values for dropdowns and multiselect
unique_years = sorted(movies_df['year'].unique())
unique_languages = sorted(movies_df['language'].unique().tolist())
unique_languages.insert(0, 'All')
unique_all_actors = sorted(movies_df['actors'].unique())
unique_all_directors = sorted(movies_df['director'].unique())

# Streamlit app title
st.title("Actor-Director Chemistry Analysis")

# User input for time duration and language (applied to most sections)
st.sidebar.header("Filter Options (General)")
general_year_range = st.sidebar.slider(
    "Select Year Range (General)",
    min_value=min(unique_years),
    max_value=max(unique_years),
    value=(min(unique_years), max(unique_years)),
    key="general_year_slider"
)
start_year_filter, end_year_filter = general_year_range
selected_languages_filter = st.sidebar.multiselect("Select Language(s)", unique_languages, default=['All'], key="general_languages")

# Filter the DataFrame based on user input (general filters)
filtered_df_general = movies_df[
    (movies_df['year'] >= start_year_filter) & (movies_df['year'] <= end_year_filter)
]
if 'All' not in selected_languages_filter:
    filtered_df_general = filtered_df_general[filtered_df_general['language'].isin(selected_languages_filter)]

if filtered_df_general.empty:
    st.info("No data available based on the selected general filters.")
    st.stop()

# 1. Top 20 Actor-Director Chemistry
st.subheader("1. Top 20 Actor-Director Chemistry (by Number of Movies)")
top_collaborations = filtered_df_general.groupby(['actors', 'director']).size().nlargest(40).reset_index(name='num_movies')
top_actors = top_collaborations['actors'].unique()[:20]
top_directors = top_collaborations['director'].unique()[:20]
top_collaborations_filtered = top_collaborations[top_collaborations['actors'].isin(top_actors) & top_collaborations['director'].isin(top_directors)]

if not top_collaborations_filtered.empty:
    fig1 = px.scatter(top_collaborations_filtered, x='director', y='actors', size='num_movies', color='num_movies',
                      color_continuous_scale=px.colors.sequential.Greens,
                      labels={'num_movies': 'Number of Movies', 'director': 'Director', 'actors': 'Actor'},
                      title="Top 20 Actor-Director Collaborations")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("No collaborations found for the top actors and directors within the selected general filters.")

# 2. User Specified Actor-Director Chemistry
st.subheader("2. User Specified Actor-Director Chemistry")
selected_actors_user = st.multiselect("Select Actors", unique_all_actors, default=unique_all_actors[:5])
selected_directors_user = st.multiselect("Select Directors", unique_all_directors, default=unique_all_directors[:3])

user_collaborations = filtered_df_general[filtered_df_general['actors'].isin(selected_actors_user) & filtered_df_general['director'].isin(selected_directors_user)]
user_chemistry = user_collaborations.groupby(['actors', 'director']).size().reset_index(name='num_movies')

if not user_chemistry.empty:
    fig2 = px.scatter(user_chemistry, x='director', y='actors', size='num_movies', color='num_movies',
                      color_continuous_scale=px.colors.sequential.Reds,
                      labels={'num_movies': 'Number of Movies', 'director': 'Director', 'actors': 'Actor'},
                      title="User Specified Actor-Director Collaborations")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No collaborations found for the selected actors and directors within the specified general filters.")

st.divider()

# 3. Top Directors for a Selected Actor
st.subheader("3. Top Directors for a Selected Actor")
selected_actor_focus = st.selectbox("Select an Actor", unique_all_actors)
actor_director_counts = filtered_df_general[filtered_df_general['actors'] == selected_actor_focus]['director'].value_counts().nlargest(10).reset_index(name='num_movies')
actor_director_counts.columns = ['director', 'num_movies']

if not actor_director_counts.empty:
    fig3 = px.bar(actor_director_counts, x='director', y='num_movies',
                  labels={'num_movies': 'Number of Movies', 'director': 'Director'},
                  title=f"Top 10 Directors {selected_actor_focus} Worked With")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info(f"No collaborations found for {selected_actor_focus} within the specified general filters.")

# 4. Top Actors for a Selected Director
st.subheader("4. Top Actors for a Selected Director")
selected_director_focus = st.selectbox("Select a Director", unique_all_directors)
director_actor_counts = filtered_df_general[filtered_df_general['director'] == selected_director_focus]['actors'].value_counts().nlargest(10).reset_index(name='num_movies')
director_actor_counts.columns = ['actor', 'num_movies']

if not director_actor_counts.empty:
    fig4 = px.bar(director_actor_counts, x='actor', y='num_movies',
                  labels={'num_movies': 'Number of Movies', 'actor': 'Actor'},
                  title=f"Top 10 Actors {selected_director_focus} Worked With")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info(f"No collaborations found for {selected_director_focus} within the specified general filters.")

st.divider()

# 5. Movies by a Specific Director and Actor
st.subheader("5. Movies by a Specific Director and Actor")
selected_director_movie_list = st.selectbox("Select Director", unique_all_directors, key="director_movie_list")
selected_actor_movie_list = st.selectbox("Select Actor", unique_all_actors, key="actor_movie_list")

specific_movies = filtered_df_general[
    (filtered_df_general['director'] == selected_director_movie_list) &
    (filtered_df_general['actors'] == selected_actor_movie_list)
]['movie_title'].unique().tolist()

if specific_movies:
    st.subheader(f"Movies by {selected_actor_movie_list} directed by {selected_director_movie_list}:")
    for movie in specific_movies:
        st.markdown(f"- {movie}")
else:
    st.info(f"No movies found for {selected_actor_movie_list} directed by {selected_director_movie_list} within the specified general filters.")

st.divider()

# 6. Interactive Network Plot of Actor-Director Chemistry
st.subheader("6. Interactive Network Plot of Actor-Director Chemistry")

# Slider for the time duration specific to the network plot
network_year_range = st.slider(
    "Select Year Range for Network Plot",
    min_value=min(unique_years),
    max_value=max(unique_years),
    value=(min(unique_years), max(unique_years)),
    key="network_year_slider"
)
network_start_year_slider, network_end_year_slider = network_year_range

# Filter the DataFrame for the network plot based on the slider and general language filter
filtered_df_network = movies_df[
    (movies_df['year'] >= network_start_year_slider) & (movies_df['year'] <= network_end_year_slider)
]
if 'All' not in selected_languages_filter:
    filtered_df_network = filtered_df_network[filtered_df_network['language'].isin(selected_languages_filter)]

if not filtered_df_network.empty:
    collab_network = filtered_df_network.groupby(['actors', 'director']).size().reset_index(name='weight')
    top_n_network = 20  # Consider top 20 actors and directors for the network (total 40)
    top_actors_network = filtered_df_network['actors'].value_counts().nlargest(top_n_network).index
    top_directors_network = filtered_df_network['director'].value_counts().nlargest(top_n_network).index
    collab_network_filtered = collab_network[
        collab_network['actors'].isin(top_actors_network) & collab_network['director'].isin(top_directors_network)
    ]

    if not collab_network_filtered.empty:
        G = nx.Graph()
        actor_nodes = list(top_actors_network)
        director_nodes = list(top_directors_network)
        G.add_nodes_from(actor_nodes, node_type='actor')
        G.add_nodes_from(director_nodes, node_type='director')

        for row in collab_network_filtered.itertuples(index=False):
            G.add_edge(row.actors, row.director, weight=row.weight)

        pos = nx.spring_layout(G, k=0.5)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            if G.nodes[node]['node_type'] == 'actor':
                node_color.append('skyblue')
                degree = filtered_df_network[filtered_df_network['actors'] == node].shape[0]
                node_size.append(degree * 1.5)
            else:
                node_color.append('salmon')
                degree = filtered_df_network[filtered_df_network['director'] == node].shape[0]
                node_size.append(degree * 1.5)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(size=node_size, color=node_color),
            text=node_text,
            textposition="bottom center"
        )

        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))

        node_trace.hovertext = [f'{node}<br># of Collaborations: {adj}' for node, adj in zip(G.nodes(), node_adjacencies)]

        fig6 = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title=dict(text='Actor-Director Collaboration Network ({} - {}, Top 40)'.format(network_start_year_slider, network_end_year_slider),
                                       font=dict(size=16)),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(text="Actor (Skyblue), Director (Salmon)", showarrow=False,
                                        xref="paper", yref="paper", x=0.005, y=-0.002 ) ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        st.plotly_chart(fig6, use_container_width=True)

    else:
        st.info("Not enough collaboration data to generate the network plot based on the selected time range.")

else:
    st.info("No movie data available for the selected time range for the network plot.")