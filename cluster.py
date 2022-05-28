import numpy as np
import pandas as pd 
import sklearn
from sklearn.cluster import KMeans




# Creating Location-Based Recommendation Function
def location_based_recommendation(data, latitude, longitude):

    # Putting the Coordinates of Restaurants together into a dataframe
    coordinates = data[['longitude','latitude']]

    kmeans = KMeans(n_clusters = 10, init = 'k-means++')
    kmeans.fit(coordinates)
    y = kmeans.labels_

    data['cluster'] = kmeans.predict(data[['longitude','latitude']])
    top_restaurants_portland = data.sort_values(by=['stars', 'review_count'], ascending=False)

    
    """Predict the cluster for longitude and latitude provided"""
    cluster = kmeans.predict(np.array([longitude,latitude]).reshape(1,-1))[0]
    
   
    """Get the best restaurant in this cluster along with the relevant information for a user to make a decision"""
    return top_restaurants_portland[top_restaurants_portland['cluster']==cluster].iloc[0:10][['name', 'latitude','longitude','categories','stars', 'review_count','cluster']]


#location_based_recommendation(top_restaurants_portland, 43.6677, -79.3948)

