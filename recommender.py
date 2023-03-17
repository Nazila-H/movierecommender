"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import movies
import pickle
from sklearn.decomposition import NMF 


with open('nmf_model1.pkl','rb') as file:
    loaded_model = pickle.load(file) 

rating = pd.read_csv('raiting.csv', index_col =0)     

def recommend_random(k=3):
    return movies['title'].sample(k).to_list()

def recommend_with_NMF(query, model = loaded_model, Ratings=rating, k=10):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model

    OUTPUT
    - a list of movieIds
    """

     
    #it needs Ratings 
    Q_matrix = model.components_
    print(Q_matrix)
    recommendations = []
    # 1. candidate generation
    
    new_user_query = query
    
    # 2. construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(new_user_query, columns=Ratings.columns.to_list(), index=["new_user"])
    new_user_dataframe_imputed = new_user_dataframe.fillna(Ratings.mean())
    print(new_user_dataframe_imputed.isna().sum())
    P_new_user_matrix = model.transform(new_user_dataframe_imputed)
    #P_new_user = pd.DataFrame(data=P_new_user_matrix, index = ['new_user'])
    R_hat_new_user_matrix = np.dot(P_new_user_matrix,Q_matrix)
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix, columns=Ratings.columns.to_list(),index = ['new_user'])
    sorted_list = R_hat_new_user.transpose().sort_values(by="new_user", ascending=False).index.to_list()
    
    # filter out movies already seen by the user
    rated_movies = list(new_user_query.keys())
    recommended = [movie for movie in sorted_list if movie not in rated_movies]
    
    # return the top-k highest rated movie ids or titles
    recommendations = recommended[0:k]
    
    return recommendations




def recommend_neighborhood(query, model, k=3):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   
    pass
    

