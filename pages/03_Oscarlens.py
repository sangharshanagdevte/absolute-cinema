import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
st.set_page_config(layout="wide")
# Background image with CSS trick
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                    url("https://wallpapercave.com/wp/wp1855040.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)




# Load the Oscar dataset (replace 'oscar_data.csv' with your actual file)
@st.cache_data
def load_data():
    df1 = pd.read_csv("archive/the_oscar_award.csv")  # Ensure this CSV contains 'Year' and 'Category' columns
    return df1

# Load data
df1 = load_data()

coarse_label_map = {
    "ACTOR IN A LEADING ROLE":"Acting",
    "ACTRESS IN A LEADING ROLE":"Acting",
    "ART DIRECTION":"Production",
    "CINEMATOGRAPHY":"Production",
    "DIRECTING (Comedy Picture)":"Directing",
    "DIRECTING (Dramatic Picture)":"Directing",
    "VISUAL EFFECTS":"Post Production",
    "BEST PICTURE":"Title",
    "UNIQUE AND ARTISTIC PICTURE":"Title",
    "WRITING (Adapted Screenplay)":"Writing",
    "WRITING (Original Story)":"Writing",
    "WRITING (Title Writing)":"Writing",
    "SPECIAL AWARD":"Special",
    "DIRECTING":"Directing",
    "SOUND RECORDING":"Production",
    "SHORT FILM (Animated)":"Animation",
    "SHORT SUBJECT (Comedy)":"Title",
    "SHORT SUBJECT (Novelty)":"Title",
    "ASSISTANT DIRECTOR":"Directing",
    "FILM EDITING":"Post Production",
    "MUSIC (Original Song Score or Adaptation Score)":"Music",
    "MUSIC (Original Song)":"Music",
    "DANCE DIRECTION":"Music",
    "ACTOR IN A SUPPORTING ROLE":"Acting",
    "ACTRESS IN A SUPPORTING ROLE":"Acting",
    "SHORT SUBJECT (Color)":"Title",
    "SHORT SUBJECT (One-reel)":"Title",
    "SHORT SUBJECT (Two-reel)":"Title",
    "IRVING G. THALBERG MEMORIAL AWARD":"Special",
    "MUSIC (Original Score)":"Music",
    "CINEMATOGRAPHY (Black-and-White)":"Production",
    "CINEMATOGRAPHY (Color)":"Production",
    "ART DIRECTION (Black-and-White)":"Production",
    "ART DIRECTION (Color)":"Production",
    "WRITING (Original Screenplay)":"Writing",
    "DOCUMENTARY (Short Subject)":"Title",
    "DOCUMENTARY (Feature)":"Title",
    "COSTUME DESIGN (Black-and-White)":"Production",
    "COSTUME DESIGN (Color)":"Production",
    "SPECIAL FOREIGN LANGUAGE FILM AWARD":"Special",
    "INTERNATIONAL FEATURE FILM":"International",
    "HONORARY AWARD":"Special",
    "JEAN HERSHOLT HUMANITARIAN AWARD":"Special",
    "COSTUME DESIGN":"Production",
    "SHORT FILM (Live Action)":"Title",
    "SOUND MIXING":"Post Production",
    "SOUND EDITING":"Post Production",
    "SPECIAL ACHIEVEMENT AWARD (Visual Effects)":"Special",
    "SPECIAL ACHIEVEMENT AWARD (Sound Effects)":"Special",
    "SPECIAL ACHIEVEMENT AWARD":"Special",
    "SPECIAL ACHIEVEMENT AWARD (Sound Effects Editing)":"Special",
    "MEDAL OF COMMENDATION":"Special",
    "SPECIAL ACHIEVEMENT AWARD (Sound Editing)":"Special",
    "MAKEUP AND HAIRSTYLING":"Production",
    "GORDON E. SAWYER AWARD":"Special",
    "AWARD OF COMMENDATION":"Special",
    "JOHN A. BONNER MEDAL OF COMMENDATION":"Special",
    "ANIMATED FEATURE FILM":"Animation",
}

df1['coarse_category'] = df1['canon_category'].map(coarse_label_map)
option="None"
# Streamlit UI
st.title("🎭 :orange[Oscarlens]")
st.header("📈 Number of Award Categories Per Year",divider=True)
st.write("")
st.markdown("### <em>\" The number of Oscar award categories has definitely evolved since the very first Academy Awards in 1928, and the rise reflects how the film industry has grown in complexity, diversity, and technological sophistication.\"</em>", unsafe_allow_html=True)
st.write("")
# Create side-by-side column
option = st.selectbox(
    "Select a subcategory",
        ("None", "Acting", "Production","Directing","Title","Writing","Special","Music","Post Production","Animation","International"),
    )

# Right column: Raw data table

# Left column: Line chart

if option=="None":
    category_counts = df1.groupby("year_ceremony")["canon_category"].nunique().reset_index()
    # st.line_chart(category_counts.set_index("year_ceremony"))


    ############################3
    fig = px.line(category_counts, x='year_ceremony', y='canon_category',labels={
        'year_ceremony': 'Year',
        'canon_category': 'Unique Categories'
    }, markers=True)
    fig.update_layout(
    xaxis=dict(
        tickmode='linear',     # Show every tick
        tick0=0,               # Starting tick (adjust if needed)
        dtick=5,               # Interval between ticks (1 = show every label)
        tickangle=0            # Set angle (0 = horizontal)
    )
)
    
    event = st.plotly_chart(fig, use_container_width=True,on_select="rerun",selection_mode="points")
    if event["selection"]["points"]:
        year = event["selection"]["points"][0]["x"]
        tab1, tab2 = st.tabs(["More details", "Award names"])
        with tab1:
            filter_data = df1[df1["year_ceremony"]==year]
            col1,col2 = st.columns([1,2])
            bool_counts = filter_data['winner'].value_counts().rename(index={True:"Winner",False:"Not Winner"})
            subfig1 = px.pie(bool_counts,values=bool_counts.values,names=bool_counts.index,title=f"Percentage of Winners and Not Winners",color_discrete_sequence=px.colors.sequential.Sunsetdark)
            col1.plotly_chart(subfig1,use_container_width=True)

            winner_films = filter_data[filter_data["winner"]==True]
            movie_counts = winner_films['film'].value_counts().reset_index()
            movie_counts.columns = ['film','count']
            fig = px.bar(
                movie_counts,
                x='count',
                y='film',
                title="Winning Films by Number of Appearances",
                labels={'film': 'Film', 'count': 'Count'},
                color='count',
                color_continuous_scale='Viridis'
            )

            # fig.update_layout(xaxis_tickangle=-45)  # Tilt x-axis labels if long

            # Show in Streamlit
            col2.plotly_chart(fig)

        with tab2:
            filter_data = df1[df1["year_ceremony"]==year]
            mylist = filter_data["canon_category"].unique().tolist()
            mylist.sort()
            earliest = []
            for item in mylist:
                earliest.append(df1[df1['canon_category']==item]["year_ceremony"].min())
            st.markdown(f"#### <em>The following awards were given in \"{year}\":</em>",unsafe_allow_html=True)
            # for count,item in enumerate(mylist):
            #     earliest = df1[df1['canon_category']==item]["year_ceremony"].min()
            #     st.markdown(f"##### <em>{count+1}) {item.capitalize()} - First introduced in {earliest}</em>",unsafe_allow_html=True)
            table_df = pd.DataFrame({"Award name":mylist,"First awarded":earliest})
            st.table(table_df)




else:
    
    sub_df = df1.loc[df1["coarse_category"]==option]
    sub_counts = sub_df.groupby("year_ceremony")["canon_category"].nunique().reset_index()
    # st.line_chart(sub_counts.set_index("year_ceremony"))

    ###############

    fig = px.line(sub_counts, x='year_ceremony', y='canon_category',labels={
        'year_ceremony': 'Year',
        'canon_category': 'Unique Categories'
    }, markers=True)
    fig.update_layout(
    xaxis=dict(
        tickmode='linear',     # Show every tick
        tick0=0,               # Starting tick (adjust if needed)
        dtick=5,               # Interval between ticks (1 = show every label)
        tickangle=0            # Set angle (0 = horizontal)
    ))
    
    event = st.plotly_chart(fig, use_container_width=True,on_select="rerun",selection_mode="points")
    

    if event["selection"]["points"]:
        year = event["selection"]["points"][0]["x"]
        tab1, tab2 = st.tabs(["More details", "Award names"])
        with tab1:
            filter_data = df1[df1["year_ceremony"]==year]
            filter_data = filter_data[filter_data["coarse_category"]==option]
            col1,col2 = st.columns([1,2])
            bool_counts = filter_data['winner'].value_counts().rename(index={True:"Winner",False:"Not Winner"})
            subfig1 = px.pie(bool_counts,values=bool_counts.values,names=bool_counts.index,title=f"Percentage of Winners and Not Winners",color_discrete_sequence=px.colors.sequential.Sunsetdark)
            col1.plotly_chart(subfig1,use_container_width=True)

            winner_films = filter_data[filter_data["winner"]==True]
            movie_counts = winner_films['film'].value_counts().reset_index()
            movie_counts.columns = ['film','count']
            fig = px.bar(
                movie_counts,
                x='count',
                y='film',
                title="Winning Films by Number of Appearances",
                labels={'film': 'Film', 'count': 'Count'},
                color='count',
                color_continuous_scale='Viridis'
            )

            # fig.update_layout(xaxis_tickangle=-45)  # Tilt x-axis labels if long

            # Show in Streamlit
            col2.plotly_chart(fig)

        with tab2:
            filter_data = df1[df1["year_ceremony"]==year]
            filter_data = filter_data[filter_data["coarse_category"]==option]
            mylist = filter_data["canon_category"].unique().tolist()
            mylist.sort()
            earliest = []
            for item in mylist:
                earliest.append(df1[df1['canon_category']==item]["year_ceremony"].min())
            st.markdown(f"#### <em>The following awards were given in \"{year}\" for \"{option}\" category:</em>",unsafe_allow_html=True)
            # for count,item in enumerate(mylist):
            #     earliest = df1[df1['canon_category']==item]["year_ceremony"].min()
            #     st.markdown(f"#### <em>{count+1}) {item.capitalize()} - First introduced in {earliest}</em>",unsafe_allow_html=True)
            table_df = pd.DataFrame({"Award name":mylist,"First awarded":earliest})
            st.table(table_df)

    


# st.write(":heavy_minus_sign:" * 50)

# Optional: Show/hide the table with a checkbox
# if st.checkbox("Show full dataset"):
#     st.write(df1)

st.divider()
st.header("🌟Rating Distribution of Oscar Winning Films",divider=True)

# winner_count = df1[df1['winner']==True]['film'].nunique()
# winner_films = df1[df1['winner']==True]['film'].unique()
# picture_winners = df1[(df1['winner']==True)&(df1['canon_category']=="BEST PICTURE")]['film'].unique()
# picture_count = df1[(df1['winner']==True)&(df1['canon_category']=="BEST PICTURE")]['film'].nunique()
# direction_count = df1[(df1['winner']==True)&(df1['coarse_category']=="Directing")]
# st.write(direction_count)
# writing_count = df1[(df1['winner']==True)&(df1['coarse_category']=="Writing")]['film'].nunique()
# ratings = []
# # for film in picture_winners:
# #     url = f'http://www.omdbapi.com/?apikey=c0a886a6&t={film}'
# #     response = requests.get(url)
# #     data = response.json()
# #     ratings.append(data['imdbRating'])

# # combo = np.column_stack((picture_winners,ratings))
# # win_rating = pd.DataFrame(combo,columns=['film','rating'])
# # st.write(win_rating)

plot_type = st.selectbox("Choose plot type", ["Box Plot", "Violin Plot"])
df2 = pd.read_csv("archive/oscar_rate_rev.csv")
filtered = df2[["Title","Rating","category"]]
if plot_type == "Box Plot":
        fig = px.box(
            filtered,
            x="category",
            y="Rating",
            color="category",
            points="all",
            hover_data=["Title"],
            title="Box Plot of Movie Ratings"
        )
        fig.update_layout(
            xaxis_title="Category"
        )
else:
        fig = px.violin(
            filtered,
            x="category",
            y="Rating",
            color="category",
            box=True,         # show embedded boxplot
            points="all",     # show all data points
            hover_data=["Title"],
            title="Violin Plot of Movie Ratings"
        )
        fig.update_layout(
            xaxis_title="Category"
        )

fig.update_layout(yaxis_title="Rating")
st.plotly_chart(fig, use_container_width=True)

# st.markdown(f"### <em>\"The number of unique film winners : {writing_count}\"</em>", unsafe_allow_html=True)

st.divider()
st.header("💸Best Picture Winners Budget vs Box Office",divider=True)

filtered_df2 = df2[(df2["Budget"]!=0)&(df2["Box Office"]!=0)]

st.write(filtered_df2)

fig = px.scatter(
    filtered_df2,
    x="Budget",
    y="Box Office",
    log_x=True, log_y=True,
    symbol="category",
    color="category",
    hover_data=["Title"],  # optional: show movie names
    title="Correlation between Movie Budget and Revenue",
    labels={"Budget": "Budget ($)", "Box Office": "Revenue ($)"},
    # trendline="ols"  # adds a linear regression line
)
st.plotly_chart(fig, use_container_width=True)
if st.checkbox("Show full dataset"):
    st.write(filtered_df2)

st.divider()
st.header("🌎Countries with their number of awards",divider=True)
country_data = df1[(df1['canon_category']=="INTERNATIONAL FEATURE FILM")]
country_data['award_status'] = country_data['winner'].map({True: 'Winner', False: 'Nominee'})
# country_counts = country_data['name'].value_counts().reset_index()
# country_counts.columns = ['country', 'count']
grouped = country_data.groupby(['name', 'award_status']).size().unstack(fill_value=0)

# Optional: reset index to get a clean DataFrame
grouped = grouped.reset_index()
# st.write(grouped)

grouped['total'] = grouped['Winner'] + grouped['Nominee']
grouped['winner_pct'] = grouped['Winner'] / grouped['total']

# # Optional: Geocode countries to lat/lon
# if 'lat' not in grouped.columns or 'lon' not in grouped.columns:
#     geolocator = Nominatim(user_agent="streamlit_app")

#     @st.cache_data(show_spinner=False)
#     def geocode_country(country):
#         try:
#             location = geolocator.geocode(country)
#             return location.latitude, location.longitude
#         except:
#             return None, None

#     st.info("Geocoding country names... This runs once and caches.")
#     grouped[['lat', 'lon']] = grouped['name'].apply(lambda c: pd.Series(geocode_country(c)))

# # Drop countries that failed geocoding
# grouped = grouped.dropna(subset=['lat', 'lon'])

# # Create Bubble Map with hover info showing both winners and nominees
# fig = go.Figure(go.Scattergeo(
#     lon=grouped['lon'],
#     lat=grouped['lat'],
#     text=grouped.apply(lambda row: f"<b>{row['name']}</b><br>🏆 Winners: {row['Winner']}<br>🎖 Nominees: {row['Nominee']}", axis=1),
#     marker=dict(
#         size=grouped['total'] * 2,  # Adjust size scale as needed
#         color=grouped['winner_pct'],
#         colorscale='Viridis',
#         colorbar=dict(title='% Winners'),
#         sizemode='area',
#         line=dict(width=0.5, color='white'),
#         showscale=True
#     )
# ))

# fig.update_layout(
#     title='Award Winners and Nominees by Country (Bubble Map)',
#     geo=dict(
#         showframe=False,
#         showcoastlines=True,
#         projection_type='natural earth'
#     ),
#     margin=dict(l=0, r=0, t=40, b=0)
# )

# st.plotly_chart(fig, use_container_width=True)
fig = px.scatter_geo(grouped, locations="name",locationmode='country names', color="winner_pct",hover_data=['Winner', 'Nominee'],
                     hover_name="name", size="total",
                     projection="natural earth",color_continuous_scale='Viridis',
                     title='Award Winners and Nominees by Country')
fig.update_geos(
    showcountries=True
)
fig.update_layout(height=800)  # Increase height
st.plotly_chart(fig, use_container_width=True)

agree= st.checkbox("Show full dataset")
if agree:
     st.write(grouped)