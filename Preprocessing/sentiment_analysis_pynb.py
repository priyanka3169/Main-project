# -*- coding: utf-8 -*-
"""Sentiment Analysis.pynb


"""

from google.colab import drive
drive.mount('/content/drive')


!pip install stylecloud
!pip install plotly_express
import nltk
nltk.download('stopwords')

## Importing Basic Packages
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
pd.set_option('display.max_columns', 50)

# Import Plotly Packages
import plotly 
import plotly.offline as py
import plotly.graph_objs as go
import plotly_express as px
from plotly.offline import init_notebook_mode, plot, iplot


## sklearn Packages
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Import NLP Packages
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import stylecloud

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
from pprint import pprint
import matplotlib.colors as mcolors

"""# Reading in CSV Files"""

# Reading in Restaurant Businesses Final CSV File

business_final = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Project/business_clean.csv")
business_final.drop(['Unnamed: 0'], axis=1, inplace = True)
print(business_final.shape)
business_final.head(10)

business_final.columns

# Reading in Reviews Final CSV File

reviews_final = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Project/review_clean.csv")
reviews_final.drop(['Unnamed: 0','review_id'], axis=1, inplace = True)

#Drop key column as it contains the same values in the user_id column
#reviews_final.drop(['key'], axis=1, inplace = True) 
print(reviews_final.shape)
reviews_final.head(3)

# Reading in Users Final CSV File

users_final = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Project/users_clean.csv")
users_final.drop(['year','Unnamed: 0'], axis=1, inplace = True)
print(users_final.shape)
users_final.head(3)

business_final.head()

# Number of Unique Restaurant Categories

len(set(restaurant_categories['categories']))

# Sorting portland Restaurant Businesses based on the number of reviews per business

portland_review_count = business_final.sort_values(by = 'review_count', ascending = False)
portland_review_count

portland_review_count.review_count.value_counts()

"""# Sentiment Analysis"""

reviews_final['text'].head()

"""To further make sense of the data in our reviews dataset, we will be performing sentiment analysis on the text content of the reviews to understand user sentiment on their overall emotional experience dining at different restaurants.

Based on the overal reviews distribution plot in the previous section, we will be classifying negative reviews as those whose Star/Rating was `1` or `2` while classifying positive reviews as those whose Star/Rating was `5`. The remainder of the reviews whose Star/Rating was `3` or `4` will be classified as neutral reviews.

## Negative Reviews
"""

# Filtering for negative reviews (one and two star)

one_star_reviews = reviews_final[reviews_final['stars'] == 1.0]
two_star_reviews = reviews_final[reviews_final['stars'] == 2.0]
negative_reviews = [one_star_reviews, two_star_reviews]
negative_reviews = pd.concat(negative_reviews)
print(negative_reviews.shape)
negative_reviews.sample(5)

"""We have `48,496` rows of reviews data classified as **Negative Reviews.**"""

## Sentiment Analysis for Negative Reviews

def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words for negative reviews')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
    
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(negative_reviews['text'])

# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)

"""Words like `food`, `like`, `good`, `restaurant`, `place` are freqeuent words in negative reviews but on their own they are not very informative about the sentiment of negative reviews. Hence, we will look to remove these words before plotting our word cloud."""

# Let's use the stop_words argument to remove words like "and, the, a"

stopwords = set(stopwords.words('english'))
cvec = CountVectorizer(min_df = 2, stop_words= stopwords, max_features = 1000)
cvec.fit(negative_reviews['text'])

# Transforming using our fitted cvec and converting the result to a DataFrame

negative_words = pd.DataFrame(cvec.transform(negative_reviews['text']).todense(),
                       columns=cvec.get_feature_names())

negative_top = negative_words.sum(axis = 0).sort_values(ascending = False).head(150)
negative_pd = pd.DataFrame(data=negative_top)
negative_top = negative_words[negative_pd.index]
negative_top.drop(columns=['food','like', 'restaurant', 'place', 'good'], inplace = True)
negative_top.head()

# Generate Word Cloud

wordcloud_low = WordCloud(background_color="white").generate(' '.join(negative_top))
plt.figure(figsize = (12,10))
plt.imshow(wordcloud_low, interpolation='bilinear')
plt.title('Word Cloud - Negative Yelp Restaurant Reviews', fontsize=16, y=1.01)
plt.axis("off");

negative_reviews['counter'] = 1
negative_reviews.head()

# Sorting Restaurants based on number of reviews

negative_reviews.groupby('business_id')['counter'].sum().sort_values(ascending = False)

# Restaurant with most negative reviews

worst_restaurant = business_final[business_final['business_id'] == '4CxF8c3MB7VAdY8zFb2cZQ']
worst_restaurant

# Filtering Reviews from the Worst Restaurant

worst_restaurant_reviews = negative_reviews[negative_reviews['business_id'] == '4CxF8c3MB7VAdY8zFb2cZQ']
worst_restaurant_reviews

negative_top.columns

# Let's use the stop_words argument to remove words like "and, the, a"

cvec = CountVectorizer(min_df = 2, stop_words= stopwords, max_features = 1000)
cvec.fit(worst_restaurant_reviews['text'])

# Transforming using our fitted cvec and converting the result to a DataFrame

negative_words = pd.DataFrame(cvec.transform(worst_restaurant_reviews['text']).todense(),
                       columns=cvec.get_feature_names())

negative_top = negative_words.sum(axis = 0).sort_values(ascending = False).head(300)
negative_pd = pd.DataFrame(data=negative_top)
negative_top = negative_words[negative_pd.index]
negative_top.drop(columns=['better', 'great','nice','bar',  'really', 'much','one','donuts', 'donut', 'voodoo', 'line', 'doughnuts', 'place', 'doughnut','even', 'us', 'got', 'go', 'came',], inplace = True)
negative_top.head()

column_list = list(negative_top)
negative_words = negative_top[column_list].sum(axis=0)
negative_words = negative_words.to_frame(name = 'sum').reset_index()
negative_words.set_index('index',inplace = True)
negative_words.to_csv("/content/drive/MyDrive/Colab Notebooks/Project/negative_words.csv")

negative_words.shape

thumbs_down = stylecloud.gen_stylecloud(file_path="/content/drive/MyDrive/Colab Notebooks/Project/negative_words.csv", 
                                        icon_name = "fas fa-thumbs-down",
                                        size = 550,
                                        palette="colorbrewer.sequential.RdPu_3", 
                                        background_color="white",
                                        output_name = 'Negative_Reviews.png')

from IPython.display import Image
Image(filename='Negative_Reviews.png')

"""## Positive Reviews"""

# Filtering for positive reviews (five star)

positive_reviews = reviews_final[reviews_final['stars'] == 5.0]
print(positive_reviews.shape)
positive_reviews.sample(5)

"""We have `86,573` rows of reviews data classified as **Positive Reviews.**"""

def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words for positive reviews')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

## Sentiment Analysis for high rated reviews
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(positive_reviews['text'])

# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)

# Let's use the stop_words argument to remove words like "and, the, a"
cvec = CountVectorizer(min_df = 2, stop_words= stopwords, max_features = 1000)

cvec.fit(positive_reviews['text'])

positive_words = pd.DataFrame(cvec.transform(positive_reviews['text']).todense(),
                       columns=cvec.get_feature_names())

positive_top = positive_words.sum(axis = 0).sort_values(ascending = False).head(100)
positive_pd = pd.DataFrame(data=positive_top)
positive_top = positive_words[positive_pd.index]
positive_top.drop(columns=['food','like', 'restaurant', 'place', 'really'], inplace = True)
positive_top.head()

# Generate Word Cloud

wordcloud_high = WordCloud(background_color="white").generate(' '.join(positive_top))
plt.figure(figsize = (12,10))
plt.imshow(wordcloud_high, interpolation='bilinear')
plt.title('Word Cloud - Positive Yelp Restaurant Reviews', fontsize=16, y=1.01)
plt.axis("off");

positive_reviews['counter'] = 1
positive_reviews.head()

# Sorting Restaurants Based on Most Reviews

positive_reviews.groupby('business_id')['counter'].sum().sort_values(ascending = False)

# Restaurant with most positive reviews

best_restaurant = business_final[business_final['business_id'] == 'OQ2oHkcWA8KNC1Lsvj1SBA ']
best_restaurant

best_restaurant_reviews = positive_reviews[positive_reviews['business_id'] == 'OQ2oHkcWA8KNC1Lsvj1SBA ']
best_restaurant_reviews

# Let's use the stop_words argument to remove words like "and, the, a"
cvec = CountVectorizer(min_df = 2, stop_words= stopwords, max_features = 1000)

cvec.fit(best_restaurant_reviews['text'])

positive_words = pd.DataFrame(cvec.transform(best_restaurant_reviews['text']).todense(),
                       columns=cvec.get_feature_names())

positive_top = positive_words.sum(axis = 0).sort_values(ascending = False).head(300)
positive_pd = pd.DataFrame(data=positive_top)
positive_top = positive_words[positive_pd.index]
positive_top.drop(columns=['food','like', 'restaurant', 'place', 'really', 'thai', 'pad', 'pai', 'khao'], inplace = True)
positive_top.head()

column_list = list(positive_top)
positive_words = positive_top[column_list].sum(axis=0)
positive_words = positive_words.to_frame(name = 'sum').reset_index()
positive_words.set_index('index',inplace = True)
positive_words.to_csv("/content/drive/MyDrive/Colab Notebooks/Project/positive_words.csv")

thumbs_up = stylecloud.gen_stylecloud(file_path="/content/drive/MyDrive/Colab Notebooks/Project/positive_words.csv", 
                                        icon_name = "fas fa-thumbs-up",
                                        size = 550,
                                        palette="colorbrewer.sequential.Greens_5", 
                                        background_color="white",
                                        output_name = 'Positive_Reviews.png')

Image(filename='Positive_Reviews.png')

"""## Creating New Feature Columns"""

reviews_final.head(3)

## Creating new feature columns
# Calculate reviews word count

reviews_final['word_count'] = reviews_final['text'].apply(lambda x: len(str(x).split(" ")))

# Calculate reviews character count

reviews_final['char_count'] = reviews_final['text'].str.len()

# Calculate average review length

def avg_word(review):
  words = review.split()
  return (sum(len(word) for word in words) / len(words))

reviews_final['avg_word_len'] = reviews_final['text'].apply(lambda x: avg_word(x))

# Calculate number of stop words in reviews

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
reviews_final['stopword_count'] = reviews_final['text'].apply(lambda x: len([x for x in x.split() if x in stopwords]))

reviews_final.head()

# Histogram of Word Count of Yelp Reviews

sns.set_style("darkgrid")
plt.figure(figsize=(14,8))
sns.distplot(reviews_final['word_count'])
plt.title('Histogram of Word Count in Yelp Reviews', fontsize = 16)
plt.xlabel('Word Count of Reviews', fontsize=14)
plt.ylabel('Percentage of Reviews', fontsize=14)

# Histogram of Character Count of Yelp Reviews

sns.set_style("darkgrid")
plt.figure(figsize=(14,8))
sns.distplot(reviews_final['char_count'])
plt.title('Histogram of Character Count of Yelp Reviews', fontsize = 16)
plt.xlabel('Character Count of Reviews', fontsize=14)
plt.ylabel('Percentage of Reviews', fontsize=14)

# Histogram of Stopword Count of Yelp Reviews

sns.set_style("darkgrid")
plt.figure(figsize=(14,8))
sns.distplot(reviews_final['stopword_count'])
plt.title('Histogram of Stopword Count of Yelp Reviews', fontsize = 16)
plt.xlabel('Stopword Count of Reviews', fontsize=14)
plt.ylabel('Percentage of Restaurants', fontsize=14)

# Average Word Length of Review for different Stars/Ratings
reviews_final.groupby('stars')['word_count'].mean()

stars_wordcount = reviews_final.groupby('stars')['word_count'].mean()
stars_wordcount = stars_wordcount.to_frame(name = 'sum').reset_index()
stars_wordcount

# Distribution of Length of Reviews vs Ratings of Reviews

sns.set_style("darkgrid")
plt.figure(figsize=(14,8))
plt.bar(stars_wordcount['stars'],stars_wordcount['sum'])
plt.title('Length of Reviews vs Ratings of Reviews ', fontsize = 16)
plt.xlabel('Ratings of Reviews (Stars)', fontsize=14)
plt.ylabel('Average Word Count of Reviews', fontsize=14)
plt.ylim((0,175))
plt.show()

"""## Text Processing"""

import nltk
nltk.download('wordnet')

# Import Textblob

from textblob import Word

# Splitting up words in reviews

reviews_final['cleaned_text'] = reviews_final['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Remove Punctuation

reviews_final['cleaned_text'] = reviews_final['cleaned_text'].str.replace('[^\w\s]', '')

# Remove Stopwords

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
reviews_final['cleaned_text'] = reviews_final['cleaned_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))

# Lemmatizing cleaned text

reviews_final['cleaned_text'] = reviews_final['cleaned_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
reviews_final.head()

# Calculate polarity and subjectivity score

from textblob import TextBlob

reviews_final['polarity'] = reviews_final['cleaned_text'].apply(lambda x: TextBlob(x).sentiment[0])
reviews_final['subjectivity'] = reviews_final['cleaned_text'].apply(lambda x: TextBlob(x).sentiment[1])
reviews_final.head(3)

!pip install vaderSentiment

# Calculate Vader Sentiment Analysis Scores

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

sentiment = reviews_final['text'].apply(lambda x: analyzer.polarity_scores(x))
reviews_final = pd.concat([reviews_final,sentiment.apply(pd.Series)],1)
reviews_final.sample(5)

# Histogram of Compound Score of Yelp Reviews

sns.set_style("darkgrid")
plt.figure(figsize=(14,8))
sns.distplot(reviews_final['compound'])
plt.title('Histogram of Compound Score of Yelp Reviews', fontsize = 16)
plt.xlabel('Compound Score of Reviews', fontsize=14)
plt.ylabel('Percentage of Restaurants', fontsize=14)

# Histogram of Polarity Score of Yelp Reviews

sns.set_style("darkgrid")
plt.figure(figsize=(14,8))
sns.distplot(reviews_final['polarity'])
plt.title('Histogram of Polarity Score of Yelp Reviews', fontsize = 16)
plt.xlabel('Polarity Score of Reviews', fontsize=14)
plt.ylabel('Percentage of Restaurants', fontsize=14)

# Histogram of Subjectivity Score of Yelp Reviews

sns.set_style("darkgrid")
plt.figure(figsize=(14,8))
sns.distplot(reviews_final['subjectivity'])
plt.title('Histogram of Subjectivity Score  of Yelp Reviews', fontsize = 16)
plt.xlabel('Subjectivity Score of Reviews', fontsize=14)
plt.ylabel('Percentage of Restaurants', fontsize=14)

reviews_final.shape

reviews_final.head()

# create a list of our conditions
conditions = [
    (reviews_final['stars'] <= 2.0),
    (reviews_final['stars'] == 3.0 ),
    (reviews_final['stars'] >= 4.0)
    ]

# create a list of the values we want to assign for each condition
values = ['negative', 'neutral', 'positive']

# create a new column and use np.select to assign values to it using our lists as arguments
reviews_final['sentiment'] = np.select(conditions, values)

# display updated DataFrame
reviews_final.head()

# # Save combined_reviews into csv file

reviews_final.to_csv("/content/drive/MyDrive/Colab Notebooks/Project/sentiment_reviews.csv")

combined_reviews = reviews_final.copy()

# Filtering for reviews that have 0.0 scores across 3 metrics: polarity, subjectivity, compound

dropped_reviews = combined_reviews[(combined_reviews.polarity ==0.0) & (combined_reviews.compound==0.0) & (combined_reviews.subjectivity==0.0)]
print(dropped_reviews.shape)
dropped_reviews.sample(5)

# Dropping these filtered rows

combined_reviews = combined_reviews.drop(index = dropped_reviews.index)
print(combined_reviews.shape)
combined_reviews.sample(3)

# Converting words in reviews to a list

def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

# Convert to list

data = combined_reviews.text.values.tolist()
data_words = list(sent_to_words(data))
print(data_words[:1])

# Text Processing

import spacy

def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

data_ready = process_words(data_words)

# Save final_reviews into csv file

final_reviews.to_csv("/content/drive/MyDrive/Colab Notebooks/Project/final_reviews.csv")
