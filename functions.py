#basic libraries
from math import *
import random
import pandas as pd
import numpy as np
import numpy.matlib

#creation of dictionaries
from collections import defaultdict
from collections import Counter

#file saving and format
import pickle
import ast

#clustering
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
#SVD method
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

#preprocesisng
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

#data visualization
import seaborn as sns
import matplotlib.pyplot as plt


####################################################################################################
                            # ANALYSIS OF THE DATA & PRE-PROCESSING #
####################################################################################################

'''this function takes in input a string of text (the review) and returns the
pre-processed text'''

def pre_processing_data(text):           
    
    stop_words = set(stopwords.words('english'))
    text = (text.lower()).replace('\\n',' ')        #converts to lower case and removes extra space
    
    text = text.replace("\r","")
    text = text.replace("\n","")
    text = text.replace("<br />","")
    
    text = text.encode("ascii", "ignore")            #removes non ASCII characters
    text = text.decode()
    
    punctuation = RegexpTokenizer(r'\w+')               #identifies punctuation
    tokens = punctuation.tokenize(text)               #create a list of all words

    ps = PorterStemmer()

    filtered_text = []
    for word in tokens:
        if word not in stop_words:
            filtered_text.append(ps.stem(word))
    
    return filtered_text


'''this function takes in input a whole dataframe and returns a dictonary 
   where the keys are all the unique words found in the column 'text_words'
   of the dataframe, and for values an identificative number'''

def build_dictionary(df):                            

    vocabulary_list = []
    vocabs_in_reviews = defaultdict(list)

    for i in range(len(df)):

        text_filtered = df['text_words'][i]
        
        vocabs_review_i=set()
        for word in text_filtered:
            vocabs_review_i.add(word)

        vocabulary_list.append(vocabs_review_i)
        vocabs_in_reviews[i]=list(vocabs_review_i)

    bag_of_words = set.union(*vocabulary_list)
    
    dictionary = {}
    for num, word in enumerate(bag_of_words):
        dictionary[word] = num
    
    return dictionary




def union_of_reviews_for_same_product(unique_products, data)

    new_data = dict.fromkeys(unique_products)

    for product in unique_products:
        index = (np.where(data['ProductId']==product))[0].tolist()

        reviews = []
        for i in index:
            text = data['text_words'][i]
            reviews += text

        new_data[product] = reviews
        
    return new_data



def frequency(vocabulary,frequency_of_word):
    f = defaultdict()
    for word in vocabulary.keys():
        f[word] = sum(frequency_of_word[i][word] for i in range(len(frequency_of_word)) if word in frequency_of_word[i])
    return f



def filter_words(frequencies):
    
    useful_words = []                             #list that contains all the words that we think are useful for clustering
    frequent_words = []                           #list that contains all the words over a certain treshold 

    for key, value in frequencies.items():
        if value > 20 and value < 200000:         #we decided to only consider the words that are inside this range of frequencies 
            useful_words.append(key)
        elif value > 200000:
            frequent_words.append(key)
    
    return(useful_words, frequent_words)
        
useful_words = filter_words(frequencies)[0]
frequent_words = filter_words(frequencies)[1]



def new_dictionary(useful_words):
    
    dictionary = defaultdict()
    
    for i, word in enumerate(useful_words):
        dictionary[word] = i
    
    return dictionary

dictionary_filtered = new_dictionary(useful_words)


####################################################################################################
                                           # SVD METHOD #
####################################################################################################


def get_relevant_words(components,features):
    
    components_features = {i: [] for i in range(len(components))} 
    n_comp = len(components)
    for i in range(n_comp):
        ith_comp = components[i]
        for coef,feat in zip(ith_comp,features):
            if  coef > 10**(-2):
                components_features[i].append(feat)

    relevant_words = list(set(list(components_features.values())[0]))

    return relevant_words


####################################################################################################
                                           # TF-IDF SCORES #
####################################################################################################

def tf(frequency_of_word,review_per_products):      
 
    tfs = []

    for i, item in enumerate(reviews_per_product.items()):
        text_filtered = reviews_per_product[item[0]]

        tf = dict.fromkeys(text_filtered,0)
        tot_number_of_words = len(text_filtered)

        for key,item in frequency_of_word[i].items():
            frequency = item
            tf_score = frequency / tot_number_of_words
            tf[key] = tf_score    
        tfs.append(tf)
        
    return tfs



def idf(relevant_words,frequency_of_word,reviews_per_product):    
    
    N = len(reviews_per_product)
    idf = dict.fromkeys(relevant_words, 0)

    #calculate df by looking at the number of documents each token appears
    for i in range(len(reviews_per_product)):
        for key,item in frequency_of_word[i].items():
            if item > 0:
                idf[key] += 1
                
    #calculate idf 
    for key,item in idf.items():
        idf[key] = np.log(N/(item))
        
    return idf


def tf_idf_score(tf_score,idf_score,reviews_per_product):    

    tf_idf_scores = []
    for i in range(len(reviews_per_product)):
        tf_idf_score = {}
        for key,item in tf_score[i].items():
            tf_idf_score[key] = round(item*idf_score[key],5) 
        tf_idf_scores.append(tf_idf_score)

    return tf_idf_scores


####################################################################################################
                                           # K-MEANS #
####################################################################################################


def elbow_method(data):
    total_variance = []
    for k in range(1, 40):
        print(k)
        kmeans = KMeans(n_clusters=k, max_iter = 100)# init='k-means++')
        kmeans.fit(data)
        total_variance.append(kmeans.inertia_)   #Sum of distances of samples to their closest cluster center
    
    #visualization of the curve
    fig = plt.figure(figsize=(15, 10))
    plt.xticks(range(1,41))
    plt.plot(range(1, 40), total_variance, linewidth = 3)
    plt.grid(color = 'lightgray', linestyle = '-.')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Cost')
    
    

#fucntion that computes the euclidea distance between vectors 
def distance_between_products(x1,x2):
    return euclidean_distances(x1,x2)


'''function that assigns each point to a cluster. In input has:
- the centroids used in the specific iteration we are in
- the array 'vectors' that has as many rows as the unique products and as many columns as the significant words 
  that we have selected for the analysis
'''

def assign_points_to_clusters(centroids, vectors):
    clusters = []
    for i in range(vectors.shape[0]):
        distances = []
        for centroid in centroids:
            distance = distance_between_products([centroid], [vectors[i]])
            distances.append(distance)
            
        cluster_selected = [distances.index(min(distances)) + 1]    #to determine in which cluster to put that element
        clusters.append(cluster_selected)
    
    return clusters


'''function that calculates new centroids based on each cluster's mean'''

def define_new_centroids(clusters, vectors):
    new_centroids = []
    cluster_df = pd.concat([pd.DataFrame(vectors),pd.DataFrame(clusters, columns=['cluster'])], axis=1)
    
    for c in set(cluster_df['cluster']):
        
        current_cluster = cluster_df[cluster_df['cluster']==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    
    return new_centroids



'''function that calculates the variance within each cluster at each iteration'''

def variance(clusters, vectors):
    sum_squares = []
    cluster_df = pd.concat([pd.DataFrame(vectors),pd.DataFrame(clusters, columns=['cluster'])], axis=1)
    
    for c in set(cluster_df['cluster']):
        current_cluster = cluster_df[cluster_df['cluster']==c][cluster_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        mean_repmat = np.matlib.repmat(cluster_mean, current_cluster.shape[0],1)
        sum_squares.append(np.sum(np.sum((current_cluster - mean_repmat)**2)))
    
    return sum_squares