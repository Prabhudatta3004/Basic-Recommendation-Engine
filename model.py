# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 18:12:48 2021

@author: prabhu
"""
##importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#importing the dataset
movies_data=pd.read_csv("tmdb_5000_movies.csv")
credits_data=pd.read_csv("tmdb_5000_credits.csv")
credits_data.head(10)
movies_data.head(10)
movies_data.shape
credits_data.shape
##both datasets are balanced
##credits dataset has 4 columns whereas movies dataset have 20 columns
#we need to combine the two datasets based on a similar column
##the id colums are similar
credits_new=credits_data.rename(index=str, columns={"movie_id":"id"})
data_final=movies_data.merge(credits_new,on='id')
data_final.head()
movies_cleaned=data_final.drop(columns=["homepage","production_countries","title_x","title_y","status"])
##now the unnecessary data has been removed
##we will go for weighted average method to devlop the recommendation engine
movies_cleaned.info()
##no null values
# Calculate all the components based on the above formula
v=movies_cleaned['vote_count']
R=movies_cleaned['vote_average']
C=movies_cleaned['vote_average'].mean()
m=movies_cleaned['vote_count'].quantile(0.70)
movies_cleaned['weighted_average']=((R*v)+ (C*m))/(v+m)
movie_sorted_ranking=movies_cleaned.sort_values('weighted_average',ascending=False)
movie_sorted_ranking[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20)
import matplotlib.pyplot as plt
import seaborn as sns
weight_average=movie_sorted_ranking.sort_values('weighted_average',ascending=False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=weight_average['weighted_average'].head(10), y=weight_average['original_title'].head(10), data=weight_average)
plt.xlim(4, 10)
plt.title('Best Movies by average votes', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')
plt.savefig('best_movies.png')
from sklearn.preprocessing import MinMaxScaler
scaling=MinMaxScaler()
movie_scaled_df=scaling.fit_transform(movies_cleaned[['weighted_average','popularity']])
movie_normalized_df=pd.DataFrame(movie_scaled_df,columns=['weighted_average','popularity'])
movie_normalized_df.head()
movies_cleaned[['normalized_weight_average','normalized_popularity']]= movie_normalized_df
movies_cleaned.head()
movies_cleaned['score'] = movies_cleaned['normalized_weight_average'] * 0.5 + movies_cleaned['normalized_popularity'] * 0.5
movies_scored_df = movies_cleaned.sort_values(['score'], ascending=False)
movies_scored_df[['original_title', 'normalized_weight_average', 'normalized_popularity', 'score']].head(20)
scored_df = movies_cleaned.sort_values('score', ascending=False)
plt.figure(figsize=(16,6))
ax = sns.barplot(x=scored_df['score'].head(10), y=scored_df['original_title'].head(10), data=scored_df, palette='deep')
#plt.xlim(3.55, 5.25)
plt.title('Best Rated & Most Popular Blend', weight='bold')
plt.xlabel('Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')
plt.savefig('scored_movies.png')
movies_cleaned.head(1)['overview']
from sklearn.feature_extraction.text import TfidfVectorizer
# Using Abhishek Thakur's arguments for TF-IDF
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
movies_cleaned['overview'] = movies_cleaned['overview'].fillna('')
# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'])
tfv_matrix.shape
from sklearn.metrics.pairwise import sigmoid_kernel
# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
# Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()
indices
def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_cleaned['original_title'].iloc[movie_indices]
# Testing our content-based recommendation system with the seminal film Spy Kidsg
give_rec('Spy Kids')