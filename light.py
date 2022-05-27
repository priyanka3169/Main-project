from unicodedata import category
import numpy as np
import pandas as pd
from lightfm import LightFM
import scipy
from math import sqrt
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
import pickle
import Rec_fx as rf
# df_train = "D:\TCS\restaurant-recommendation-system-main\df_train.csv"
# df_test = "D:\TCS\restaurant-recommendation-system-main\df_test.csv"
# review_link="D:\TCS\restaurant-recommendation-system-main\ratings1.csv"
# business_link="D:\TCS\restaurant-recommendation-system-main\business3.csv"
#user_link='D:\TCS\restaurant-recommendation-system-main\user1.csv'

#@st.cache(persist=True, allow_output_mutation=True)
#def load_data(url):
    # data = pd.read_csv(url)
    # return data

train1 = pd.read_csv(r"D:\TCS\restaurant-recommendation-system-main\df_train.csv",low_memory=False)
test1 = pd.read_csv(r"D:\TCS\restaurant-recommendation-system-main\df_test.csv",low_memory=False)

review_df =pd.read_csv(r"D:\TCS\restaurant-recommendation-system-main\ratings1.csv",low_memory=False)
business_df =pd.read_csv(r"D:\TCS\restaurant-recommendation-system-main\business3.csv",low_memory=False)
user_df = pd.read_csv(r"D:\TCS\restaurant-recommendation-system-main\user1.csv",low_memory=False)


#model establishment
dataset = Dataset()
dataset.fit(review_df.user_id,review_df.business_id)
type(dataset)
num_users, num_items = dataset.interactions_shape()
# n_users = review_df.user_id.unique().shape[0]
# n_items = review_df.business_id.unique().shape[0]

#fitting item and user features
dataset.fit_partial(items=business_df.business_id,
                    item_features=['stars'])
dataset.fit_partial(items=business_df.business_id,
                    item_features=['review_count'])


tar_cols = [x for x in business_df.columns[26:]]

dataset.fit_partial(items = business_df.business_id,
                item_features = tar_cols)                                    
                                                

user_cols = [x for x in user_df[['compliment_cool', 'compliment_cute',
        'compliment_funny', 'compliment_hot', 'compliment_list',
        'compliment_more', 'compliment_note', 'compliment_photos',
        'compliment_plain', 'compliment_profile', 'compliment_writer', 
            'review_count', 'useful','is_elite']]]
# user_cols = [x for x in user_df[['review_count', 'useful', 'is_elite',
#                                 'Fast Food', 'Chinese',
#     'Cocktail Bars', 'Barbeque', 'Delis', 'Food Delivery Services',
#     'Vietnamese', 'Ethnic Food', 'Soup', 'Thai', 'Sports Bars', 'Grocery',
#     'Dive Bars', 'Caterers', 'Desserts', 'Korean', 'Gluten-Free', 'Vegan',
#     'Middle Eastern', 'Mediterranean', 'Comfort Food', 'Salad',
#     'Vegetarian', 'Convenience Stores', 'Bagels', 'Seafood', 'Asian Fusion','Beer Bar', 'Wine Bars', 'Arts & Entertainment', 'Italian', 'Lounges',
#     'Chicken Wings', 'Japanese', 'Juice Bars & Smoothies', 'Local Flavor',
#     'Venues & Event Spaces', 'Shopping', 'Pubs', 'Sushi Bars', 'Breweries',
#     'Ice Cream & Frozen Yogurt', 'Street Vendors']]]
dataset.fit_partial(users=user_df.user_id,
                    user_features = user_cols)

#building train interactions
# (trainin, weights) = dataset.build_interactions([(x['user_id'],
#                                                        x['business_id'],
#                                                        x['rating']) for index,x in train1.iterrows()])

# #building test interactions
# (testin, weights) = dataset.build_interactions([(x['user_id'],
#                                                        x['business_id'],
#                                                        x['rating']) for index,x in test1.iterrows()])
#build interaction
(interactions, weights) = dataset.build_interactions([(x['user_id'],
                                                    x['business_id'],
                                                    x['rating']) for index,x in review_df.iterrows()])

unwanted = {'business_id', 'name', 'address', 'city', 'state', 'postal_code', 'category', 'RestaurantsDelivery', 'RestaurantsTakeOut', 'WiFi', 'RestaurantsAttire', 'Caters', 'RestaurantsReservations', 'NoiseLevel', 'BikeParking', 'RestaurantsGoodForGroups', 'HasTV', 'Alcohol', 'RestaurantsPriceRange2', 'OutdoorSeating', 'BusinessAcceptsCreditCards', 'GoodForKids'}
tar_cols = [e for e in tar_cols if e not in unwanted]
seed = 123
from lightfm.cross_validation import random_train_test_split
train,test=random_train_test_split(interactions,test_percentage=0.2,random_state=np.random.RandomState(seed))
# build item features
def build_dict(df,tar_cols,val_list):
    rst = {}
    for col in tar_cols:
        rst[col] = df[col]
    sum_val = sum(list(rst.values())) # get sum of all the tfidf values
    
    if(sum_val == 0):
        return rst
    else:
        
        w = (2-sum(val_list))/sum_val # weight for each tag to be able to sum to 1
        for key,value in rst.items():
            rst[key] = value * w
    return rst
# def itemfeature(df,tar_cols,val_list):
#     rst = {}
#     for col in tar_cols:
#         rst[col] = df[col]
#     sum_val = sum(list(rst.values())) # get sum of all the tfidf values
    
#     if(sum_val == 0):
#         return rst
#     else:
        
#         w = (2-sum(val_list))/sum_val # weight for each tag to be able to sum to 1
#         for key,value in rst.items():
#             rst[key] = value * w
#     return rst

# get max of each column to regularize value to [0,1]
max_star = max(business_df.stars)
max_b_rc = max(business_df.review_count)

item_features = dataset.build_item_features(((x['business_id'], 
                                            {'stars':0.5*x['stars']/max_star,
                                            'review_count':0.5*x['review_count']/max_b_rc,
                                            **build_dict(x,tar_cols,[0.5*x['stars']/max_star,
                                                        0.5*x['review_count']/max_b_rc])})
                                            for index,x in business_df.iterrows()))

#build user featuress

# def userfeature(df,tar_cols,val_list):
#     rst = {}
#     for col in tar_cols:
#         rst[col] = df[col]
#     sum_val = sum(list(rst.values())) # get sum of all the tfidf values
    
#     if(sum_val == 0):
#         return rst
#     else:
#         w = (2-sum(val_list))/sum_val # weight for each tag to be able to sum to 1
#         for key,value in rst.items():
#             rst[key] = value * w
#     return rst
def user_build_dict(df,tar_cols,val_list):
    rst = {}
    for col in tar_cols:
        rst[col] = df[col]
    sum_val = sum(list(rst.values())) # get sum of all the tfidf values
    
    if(sum_val == 0):
        return rst
    else:
        w = (2-sum(val_list))/sum_val # weight for each tag to be able to sum to 1
        for key,value in rst.items():
            rst[key] = value * w
    return rst
max_u_rc = max(user_df.review_count)
max_useful = max(user_df.useful)

user_features = dataset.build_user_features(((x['user_id'],
                                            {'review_count':0.35*x['review_count']/max_u_rc,'is_elite':0.35*int(x['is_elite']),
                                            'useful':0.35*x['useful']/max_useful,
                                            **user_build_dict(x,user_cols,[0.35*x['review_count']/max_u_rc,
                                                                            0.35*int(x['is_elite']),
                                                                            0.35*x['useful']/max_useful])})
                                        for index, x in user_df.iterrows()))

best_model = pickle.load(open(r'D:\TCS\restaurant-recommendation-system-main\savefile.pkl','rb'))
def hyb_recommendation(user_id):
    n_users = review_df.user_id.unique().shape[0]
    n_items = review_df.business_id.unique().shape[0]
    k=5
    data_meta = business_df
    name = 'name'
    u_idx = [x for x in train.tocsr()[user_id].indices]
    known_positives = data_meta.loc[u_idx, name]
    tag = 'category'
    if tag is not None:
      known_tags = data_meta.loc[u_idx,name]
    num_threads = 2
    scores = best_model.predict(user_id, np.arange(n_items), user_features=user_features, item_features=item_features,
                                num_threads=num_threads)
    i_idx = [x for x in np.argsort(-scores)]
    top_items = data_meta.loc[i_idx,name]
    if tag is not None:
                top_tags = data_meta.loc[i_idx, tag]  # get item tags.
    return top_items
#n_users,n_items=interactions.shape
# a=rf.sample_train_recommendation(best_model,train,business_df,[user_id],5,'name',mapping=dataset.mapping()[2],tag='category',
#                             user_features = user_features,item_features=item_features)

   

# c=hyb_recommendation(500)
# c
# n_business=business_df.set_index('business_id')
# mapping=dataset.mapping()[2]
# train=trainin
# data_meta=business_df
# name='name'
# tag='category'
# model=best_model
# num_threads=2
# def hyb_recommendation(user_id): 
#     # #recom_hybrid=rf.sample_test_recommendation(best_model,trainin,testin,business_df,[201],5,'name',mapping=dataset.mapping()[2],train_interactions=trainin,tag='category',user_features = user_features,item_features=item_features)
#     # hybrid_r=best_model.predict(user_id,np.arange(n_items))
#     # rank=np.argsort(-hybrid_r)
#     # selected=np.array(list(dataset.mapping()[2].keys()))[rank]
#     # top_items=n_business.loc[selected]
#     # recom_hybrid=top_items['name'][:5].values
#     # recom_hybrid=business_df['name'][np.argsort(-hybrid_r)].head(5)
#     # recom_hybrid=rf.sample_train_recommendation(best_model,trainin,business_df,[user_id],5,'name',mapping=dataset.mapping()[2],tag='category',
#     #                           user_features = user_features,item_features=item_features)
#     #for user_id in user_ids:

#     t_idx = {value: key for key, value in mapping.items()}
#     u_idx = [x for x in train.tocsr()[user_id].indices]
#     known_positives = data_meta.loc[u_idx, name]  # may need change
#     if tag is not None:
#         known_tags = data_meta.loc[u_idx, tag]  # get item tags.

#     if (len(known_positives) < 5):
#         return 0 
#     scores = model.predict(user_id, np.arange(n_items), user_features=user_features, item_features=item_features,
#                             num_threads=num_threads)
#     i_idx = [x for x in np.argsort(-scores)]
#     top_items = data_meta.loc[i_idx, name]
#     if tag is not None:
#         top_tags = data_meta.loc[i_idx, tag]  # get item tags.
#     if tag is not None:
#         for x in range(len(known_positives)):
#             known_pos= known_positives.values[x]
#             known_tag= known_tags.values[x]
#     cnt = 0
#     if tag is not None:
#         for x in range(5):
#             top_itms=top_items.values[x] 
#             top_tag=top_tags.values[x]
#             if (top_items.values[x] in known_positives.values):
#                 cnt += 1
            
#     else:
#         for x in top_items[:5]:
            
#             if (x in known_positives.values):
#                 cnt += 1
                
#     return known_pos,known_tag,top_itms,top_tag


# a=hyb_recommendation(user_id=10)
# print(a)