import streamlit as st
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', 50)

# Importing Plotly Packages

import plotly 
import plotly.offline as py
import plotly.graph_objs as go
import plotly_express as px

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.plotting import gmap


from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

#importing python scripts
KM = __import__("cluster")
LOC= __import__("location")
CT = __import__("content")
#CF = __import__("collaborate")
HF = __import__("light")


st.title("Restaurant Recommendation platform")
st.markdown("This application is for recommending restaurants to visit for users in Portland")


st.sidebar.title("Restaurant Recommendation platform")
# st.sidebar.subheader("TCS Project")
# st.sidebar.markdown("By: V Priyanka")
# st.sidebar.markdown("This application is for recommending restaurants to visit for users in Portland  🍔🍕🍹🍺")

business_URL= "D:/TCS/restaurant-recommendation-system-main/business_clean.csv"
final_URL="D:/TCS/restaurant-recommendation-system-main/final_reviews.csv"
portland_URL= "portland_data2.csv"



@st.cache(persist=True, allow_output_mutation=True)
def load_data(url):
    data = pd.read_csv(url)
    return data

def clean(data):
    data.drop(['Unnamed: 0'], axis=1, inplace = True)
    data['business_id'] = data['business_id ']
    data = data[['business_id', 'name', 'categories','stars','review_count','latitude','longitude','postal_code']]
    return data

business_data = load_data(business_URL)
portland_data = pd.read_csv(r'D:\TCS\restaurant-recommendation-system-main\portland_data2.csv')
final_reviews = load_data(final_URL)
top=pd.read_csv(r'D:\TCS\restaurant-recommendation-system-main\top_restaurants_portland.csv')
address_data=pd.read_csv(r'D:\TCS\restaurant-recommendation-system-main\adress_portland.csv')
# st.write(all_data.head())

# create a list of our conditions
portland_data['super_score'] = portland_data['super_score'].round(0)
conditions = [
    (portland_data['super_score'] <=2),
    (portland_data['super_score'] == 3),
    (portland_data['super_score'] >= 4)
    ]

# create a list of the values we want to assign for each condition
values = ['negative', 'neutral', 'positive']

# create a new column and use np.select to assign values to it using our lists as arguments
portland_data['sentiment'] = np.select(conditions, values)

@st.cache(persist=True)
def plot_sentiment(restaurant):
    df = portland_data[portland_data['name']==restaurant]
    count = df['sentiment'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'text':count.values.flatten()})
    return count

def main():
    st.sidebar.markdown("### Recommendation type")
    section = st.sidebar.selectbox('choose recommendation type', ['Pick a Value', 'Locations', 'Restaurants','User login'], key= 1)

    #fig.update_layout(mapbox_style="dark")
    #fig.show()
    # 
        

    if section == "Pick a Value":
        #st.markdown("## How to get the most out of this platform")
        st.markdown('This platform contains 3 recommendation system models to recommend to you restaurants based on Yelp reviews in portland city')
        #st.markdown("- If you're a new user of this platform or in this city and you have never tried any restaurant around portland, please select the **location based** recommender on the sidebar to get recommended top restaurants around where you are.")
        #st.markdown("- If you want recommendations of restaurants similar to one you have previously visited and liked, please select **content-based** on the sidebar.")
        #st.markdown("- If this isn't your first time using this platform and would like to get recommendations based on previous restaurants you have visited and rated please select the **collaborative filtering** option on the sidebar.")
        #st.markdown("- If you just want to compare the ratings of different restaurants you have in mind, please select **Restaurant Analytics** on the sidebar.")


        st.subheader("Graphical Overview of Restaurants in portland City")
        px.set_mapbox_access_token("pk.eyJ1Ijoic2hha2Fzb20iLCJhIjoiY2plMWg1NGFpMXZ5NjJxbjhlM2ttN3AwbiJ9.RtGYHmreKiyBfHuElgYq_w")
        fig = px.scatter_mapbox(business_data, lat="latitude", lon="longitude", color="stars", size='review_count',
                        size_max=15, zoom=10, width=1000, height=700)
        st.plotly_chart(fig)

    if section == "Locations":

        st.subheader('Location Based Recommendation System')

        #st.markdown("please enter your location")
        st.markdown("please select your current location")
        location = st.selectbox('select location',address_data['address'].unique())
        #location = st.text_area('Input your location here')

        if location:
            # URL = "https://geocode.search.hereapi.com/v1/geocode"
            # api_key = 'ODfYgIX45wrL41qboC3F_z2hg8e5_ABJYi71Pu6o948' # Acquire from developer.here.com
            # PARAMS = {'apikey':api_key,'q':location}
            geolocator = Nominatim(user_agent="my_app")
            location_1 = geolocator.geocode(location)
            latitude=location_1.latitude
            longitude=location_1.longitude
            #lat_long = LOC.get_location(URL, PARAMS)
            # latitude = lat_long[0]
            # longitude = lat_long[1]

            df = KM.location_based_recommendation(top, latitude, longitude)

            if st.sidebar.checkbox("Show data", False):
                st.write(df)

            st.markdown("## Geographical Plot of Nearby Recommended Restaurants from "+ location)
            px.set_mapbox_access_token("pk.eyJ1Ijoic2hha2Fzb20iLCJhIjoiY2plMWg1NGFpMXZ5NjJxbjhlM2ttN3AwbiJ9.RtGYHmreKiyBfHuElgYq_w")
            fig = px.scatter_mapbox(df, lat="latitude", lon="longitude",  
                            zoom=10, width=1000, height=700, hover_data= ['name', 'latitude', 'longitude', 'categories', 'stars', 'review_count'])
            fig.add_scattermapbox(lat=[latitude], lon=[longitude]).update_traces(dict(mode='markers', marker = dict(size = 15)))
            fig.update_layout(mapbox_style="dark")
            st.plotly_chart(fig)
    
    if section == 'Restaurants':
        st.subheader('Content based recommendation system')
        st.markdown("please select a restaurant")
        restaurant = st.selectbox('select restaurant',portland_data['name'].unique())

        if restaurant:
            restaurant_recommendations = CT.content_based_recommendations(restaurant)
            restaurant1 = portland_data[portland_data['name'] == restaurant_recommendations[0]][['name','categories','super_score']].groupby(['name', 'categories'], as_index=False).mean()
            restaurant2 = portland_data[portland_data['name'] == restaurant_recommendations[1]][['name','categories','super_score']].groupby(['name', 'categories'], as_index=False).mean()
            restaurant3 = portland_data[portland_data['name'] == restaurant_recommendations[2]][['name','categories','super_score']].groupby(['name', 'categories'], as_index=False).mean()
            restaurant4 = portland_data[portland_data['name'] == restaurant_recommendations[3]][['name','categories','super_score']].groupby(['name', 'categories'], as_index=False).mean()
            restaurant5 = portland_data[portland_data['name'] == restaurant_recommendations[4]][['name','categories','super_score']].groupby(['name', 'categories'], as_index=False).mean()


            rest_merged = pd.concat([restaurant1.head(1), restaurant2.head(1), restaurant3.head(1), restaurant4.head(1), restaurant5.head(1)])
            st.write(rest_merged)

    #     st.subheader('Collaborative Filtering recommendation system')

    #     if restaurant:
    #          collab_recommendations = CT.content_based_recommendations(restaurant)
    #          collab_recommendations = pd.DataFrame(data = restaurant_recommendations)

    #          st.write(restaurant_recommendations)


        if section != 'Pick a value':
            if st.sidebar.checkbox("Compare restaurants by sentiments", False):
                choice = st.sidebar.multiselect('Pick restaurants', portland_data['name'].unique())
                if len(choice) > 0:
                    st.subheader("Breakdown restaurant by sentiment")
                    fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
                    for i in range(1):
                        for j in range(len(choice)):
                            fig_3.add_trace(
                                go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).text, showlegend=False),
                                row=i+1, col=j+1
                            )
                    fig_3.update_layout(height=600, width=800)
                    st.plotly_chart(fig_3)

                # st.write(portland_data.head())
                # st.sidebar.header("Word Cloud")
                # word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
                # if not st.sidebar.checkbox("Close", True, key='3'):
                #      st.subheader('Word cloud for %s sentiment' % (word_sentiment))
                #      df = portland_data[portland_data['sentiment']==word_sentiment]
                #      words = ' '.join(df['text'])
                #      processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
                #      wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
                #      plt.imshow(wordcloud)
                #      plt.xticks([])
                #      plt.yticks([])
                #      st.pyplot()
           

    # if section == 'Collaborative Filtering':
    #      st.subheader("Collaborative Filtering Recommendation System")

    #      st.markdown("please select a restaurant you've visited before")
    #      restaurant = st.selectbox('select restaurant', ['Pai Northern Thai Kitchen', 'Sabor Del Pacifico'])

    if section == 'User login':
        st.subheader("Hybrid Filtering Recommendation System")
        st.markdown("Please enter user id")
        user_id = st.number_input('Input your User_id here',step=1)
        if user_id : 
           top_items= HF.hyb_recommendation(user_id)
        #    df_top=pd.DataFrame(top_items)
        #    for x in range(5):
        #        top_items_1=df_top.values[x]
           st.dataframe(top_items)
           #st.write(df_top)

   

if __name__ == "__main__":
    main()
