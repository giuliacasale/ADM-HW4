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
import wordcloud


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




def union_of_reviews_for_same_product(unique_products, data):

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



def new_dictionary(useful_words):
    
    dictionary = defaultdict()
    
    for i, word in enumerate(useful_words):
        dictionary[word] = i
    
    return dictionary


def get_occurrency_matrix(unique_products,dictionary_filtered,frequency_of_word):
    matrix = np.zeros((len(unique_products),len(dictionary_filtered)))
    for i, product in enumerate(unique_products):           #per ogni prodotto con indice i   
        for word,j in dictionary_filtered.items():          #per ogni parola con indice j nel dizionario
            if word in frequency_of_word[i].keys():         #se la parola si trova nel dizionario di frequenze del prodotto
                matrix[i][j] += frequency_of_word[i][word]
    return matrix



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



def get_tfidf_scores_matrix(unique_products,relevant_words, tf_idf_scores, final_dictionary):
    n = len(unique_products)
    m = len(relevant_words)

    product_vector = np.zeros((n,m))        #matrix that has for each unique productID (row), all the scores for each word in
                                            #the final_dictionary (columns): 0 if the word is not present, tf-idf score otherwise

    for i in range(n):                                   #for every productID
        for j,word in enumerate(tf_idf_scores[i]):       #for every word in his list of review words         
                product_vector[i][final_dictionary[word]] = tf_idf_scores[i][word]      #add to that word column of that product                                                                                
                                                                                        #its tf-idf score 
    return product_vector


####################################################################################################
                                           # K-MEANS #
####################################################################################################

'''function needed to understand the best number of clusters '''

def elbow_method(data):
    total_variance = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, max_iter = 100)
        kmeans.fit(data)
        total_variance.append(kmeans.inertia_)   #Sum of distances of samples to their closest cluster center
    
    benchmark = (total_variance[0]-(total_variance[0]-total_variance[-1])*0.80)
    best_k = [i for i in range(len(total_variance)) if (total_variance[i]>= benchmark and total_variance[i+1] <= benchmark)][0] 
    
    #visualization of the curve
    fig = plt.figure(figsize=(15, 5))
    plt.xticks(range(1,21))
    plt.plot(range(1, 20), total_variance, linewidth = 3)
    plt.grid(color = 'lightgray', linestyle = '-.')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Cost')
    plt.show()
    
    return best_k
    
    
'''fucntion that computes the euclidea distance between vectors'''

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


''' functioon that computes our k-mean algorithm for clusterization of the products. In input needs the tf-idf-scores dataset
    that we have created and then coverted into an array for easier computation'''

def k_means(vectors,d,k):
    
    print('INITIALIZATION\n')
    cluster_variance = []

    initial_centroids = random.sample(range(0, len(d)), k)    # I choose k numbers random between 0 and the length of our dataset

    #we use these numbers as indices to get the coordinates of the products associated to those centroids
    centroids = []      
    for i in initial_centroids:
        centroids.append(vectors[i])
    
    print(f'our initial centroids are the products:\n')
    
    for i in initial_centroids:
        print(d.index[i])

    clusters = assign_points_to_clusters(centroids, vectors)
    initial_clusters = clusters
    print(f'\niteration 0: cluster variance: {round(np.mean(variance(clusters, vectors)),1)}\n')
    
    print('ITERATIONS\n')
    
    for i in range(10):                                              #I recall the functions created above to initialize the
                                                                     #new centrodis, associate the products to each cluster and
        centroids = define_new_centroids(clusters, vectors)          #calculate the variance of each iteration
        clusters = assign_points_to_clusters(centroids, vectors)
        cluster_var = np.mean(variance(clusters, vectors))
        cluster_variance.append(cluster_var)

        if i == 0:
            print(f'iteration 1: cluster variance: {round(cluster_var,1)}')    

        else:                                                                     #If the difference between the variance of two
            if cluster_variance[i-1] - cluster_variance[i] >= 1.0:                 #itereations is really close, the algorithm has
                print(f'iteration {i+1}: cluster variance: {round(cluster_var,1)}')  #found one of the best ways to cluster the 
            else:                                                                  #products, so we can stop
                break

    return initial_centroids, centroids, clusters, cluster_variance



####################################################################################################
                                           # ANALYSIS OF THE CLUSTERS #
####################################################################################################

## QUESTION 1

def words_cloud(d,k,cluster_list):
    d['cluster']=cluster_list
    for cluster in range(k):
        print('Word cloud for cluster:', cluster+1)
        clus_df = d[d.cluster == (cluster+1)]
        sum_scores = clus_df.sum(axis=0)
        words = dict(zip(d.columns[:-1],sum_scores[:-1]))
        Cloud = wordcloud.WordCloud(background_color="white", max_words=len(words)).generate_from_frequencies(words)
        
        plt.figure(figsize = (12, 8), facecolor = None) 
        plt.imshow(Cloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
        plt.show()
        
        
## QUESTION 2

'''computes the word cloud of only cluster c'''

def one_word_cloud(c,d):
    clus_df = d[d.cluster == c]
    sum_scores = clus_df.sum(axis=0)
    words = dict(zip(d.columns[:-1],sum_scores[:-1]))
    Cloud = wordcloud.WordCloud(background_color="white", max_words=len(words)).generate_from_frequencies(words)
    plt.figure(figsize = (12, 8), facecolor = None) 
    plt.imshow(Cloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()


## QUESTION 3

def review_score_distribution(k,df,data):
    score_means = []

    for i in range(k):

        l = list(df[df.cluster == i+1].ProductID)               #list of ProductId's inside cluster i+1

        d_l = data[data['ProductId'].isin(l)]                   #selecting only the rows of the original dataframe
                                                                 #that conatin the reviews for those particular products
        score_means.append(d_l.Score.mean())
        
        ys = [d_l.ProductId.loc[d_l['Score']== x].count() for x in range(1,6)]
        
        #plot

        ax = sns.histplot(d_l, x = "Score", bins = 5, stat = 'count', kde = 'True')
        plt.title(f'Score distribution of cluster {i+1}')
        plt.show()
    
    return score_means

## QUESTION 4

def number_of_unique_users(k, df, data):

    for i in range(k):

        l = list(df[df.cluster == i+1].ProductID)                   #list of ProductId's inside cluster i+1

        d_l = data[data['ProductId'].isin(l)]                       #selecting only the rows of the original dataframe
                                                                      #that conatin the reviews for those particular products

        number_of_users = len(d_l.UserId.unique())                  #number of unique users in those rows selected 

        print(f'number of unique users writing reviews in cluster {i+1}: {number_of_users}')


