#basic libraries
from math import *
import math
import random
import pandas as pd
import numpy as np
import numpy.matlib
import time
import random
from matplotlib import cm
import multiprocessing.dummy as mp 
from scipy import spatial


# hypeloglog
import hyperloglog


#creation of dictionaries
import collections
from collections import defaultdict
from collections import Counter

#file saving and format
import pickle
import ast

#clustering
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA


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
from mpl_toolkits import mplot3d

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


 '''This function takes in input the dataframe pre-processed and a list of unique products ID and returns a dictionary
    where the keys are the productID's and the value are the list of pre-processed words from all the reviews that refer
    to the same product'''

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

  
 '''Function used to calculate the absolute freuquence of all the words in all the reviews starting from the local
 frequence of the words for each product'''

def frequency(vocabulary,frequency_of_word):
    f = defaultdict()
    for word in vocabulary.keys():
        f[word] = sum(frequency_of_word[i][word] for i in range(len(frequency_of_word)) if word in frequency_of_word[i])
    return f


'''This function takes in input the global frequencies and filter the words by taking only the words with a certian frequence
Output:
- useful_words: words whose frequence is between 20 and 20000
- frequent_words: words whose frequence is higher than 20000 '''

def filter_words(frequencies):
    
    useful_words = []                             #list that contains all the words that we think are useful for clustering
    frequent_words = []                           #list that contains all the words over a certain treshold 

    for key, value in frequencies.items():
        if value > 20 and value < 200000:         #we decided to only consider the words that are inside this range of frequencies 
            useful_words.append(key)
        elif value > 200000:
            frequent_words.append(key)
    
    return(useful_words, frequent_words)


'''Function that maps the useful words into integers'''

def new_dictionary(useful_words):
    
    dictionary = defaultdict()
    
    for i, word in enumerate(useful_words):
        dictionary[word] = i
    
    return dictionary


'''Function that creates a matrix of occurrency where the rows refer to the product ID and the column to a specific
word inside the list of useful_words that we selected. This matrix will be used for the SVD method'''

def get_occurrency_matrix(unique_products,dictionary_filtered,frequency_of_word):
    matrix = np.zeros((len(unique_products),len(dictionary_filtered)))
    for i, product in enumerate(unique_products):            #for every product i  
        for word,j in dictionary_filtered.items():           #for every word with index j in dictionary_filtered
            if word in frequency_of_word[i].keys():          #if the word is inside the dictionary *frequency_of_word*
                matrix[i][j] += frequency_of_word[i][word]   #append that frequence in the correct cell of the matrix
    return matrix


####################################################################################################
                                           # SVD METHOD #
####################################################################################################

'''Function that from the output of TruncatedSVD recognizes the most relevent words for the clustering and returns them in a list'''

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

'''function that calculates the term frequency for the reviews of each product and creates a list of dictionaries where each dictionary 
refers to a single productID. It has for keys all the unique words considered and for values the tf_score = frequency term/total number 
of words in that plot'''

def tf(frequency_of_word,review_per_product):      
 
    tfs = []

    for i, item in enumerate(review_per_product.items()):
        text_filtered = review_per_product[item[0]]

        tf = dict.fromkeys(text_filtered,0)
        tot_number_of_words = len(text_filtered)

        for key,item in frequency_of_word[i].items():
            frequency = item
            tf_score = frequency / tot_number_of_words
            tf[key] = tf_score    
        tfs.append(tf)
        
    return tfs


'''function that calculates the idf score of each token. The output is a dictionary where the keys are the all the different words in 
the list *relevant_words* and the values are given by log(N/n):
- N = total number of unique productID
- n = total number of products in which each token appears'''

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
  
  
'''function that puts together the tf and idf score for each productID and for each token. The output is a list of dictionaries where 
each of them refers to a specific product. The keys are given by the keys in the tf_score dictionary created above and the values are 
given by multiplying the two scores (tf and idf)'''

def tf_idf_score(tf_score,idf_score,reviews_per_product):    

    tf_idf_scores = []
    for i in range(len(reviews_per_product)):
        tf_idf_score = {}
        for key,item in tf_score[i].items():
            tf_idf_score[key] = round(item*idf_score[key],5) 
        tf_idf_scores.append(tf_idf_score)

    return tf_idf_scores


'''Function that trasform the output of the function tf_idf_score into a dataframe to be used in the K-Means algorithm'''

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

'''Function that generates a word-cloud visualization for the words in each cluster. It takes in input the final dataframe d 
that contains the tf.idf.scores of each word for each product, the number of clusters chosen k and the list that tells in which
cluster was assigned each product'''

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

'''function that computes the word cloud of only aspecific cluster c'''

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

''''Function that for each cluster plots the distribution of the scores that the users gave for the producuts in that cluster'''

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

'''Function that computer the number of unique users that wrote reviews for the products in each cluster'''

def number_of_unique_users(k, df, data):

    for i in range(k):

        l = list(df[df.cluster == i+1].ProductID)                   #list of ProductId's inside cluster i+1

        d_l = data[data['ProductId'].isin(l)]                       #selecting only the rows of the original dataframe
                                                                      #that conatin the reviews for those particular products

        number_of_users = len(d_l.UserId.unique())                  #number of unique users in those rows selected 

        print(f'number of unique users writing reviews in cluster {i+1}: {number_of_users}')


####################################################################################################
                                           # DISTANCE MATRIX #
####################################################################################################

''' for each product among the first 1500 rows we calculate the distance between product i and all products '''
def build_distance_matrix(final_dictionary):
    prods = range(len(final_dictionary))[:1500] 
    matrix ={i: [] for i in range(1500)}    

    for i in prods:                           
        p = mp.Pool()    # function for multiprocessing
        p.map(calculate_distance,prods)       
    return matrix


def calculate_distance(j,i,matrix,final_dictionary,all_words):
    prod1 = final_dictionary[i] # we extract the tf-idf vector for product i
    prod2 = final_dictionary[j] # we extract the tf-idf vector for product j
    scores1 = []                # in this lists we will save the scores for calculating the cosine distance
    scores2 = []
    for word in all_words:      # we iterate the sorted list of words (we need to ensure that the indexes are the same)
        try:                      
            scores1.append(prod1[word]) # if that word is in the product we add the score
        except:
            scores1.append(1)           # else we add 1 (max distance)
        try:
            scores2.append(prod2[word])
        except:
            scores2.append(1)
    matrix[i].append(spatial.distance.cosine(scores1,scores2)) # we compute the cosine distance between the two lists (1 - cosine similarity)

    return matrix  

'''Here we build kmeans by assuming we start from a distance matrix'''

def kmeans(k, matrix, centroids):   
    
    # inizializing output
    n_clusters = dict(zip(centroids,range(k)))            # cluster number centroids
    clusters = {i: [] for i in range(k)}                  # empty dictionary for clusters
    WSS = {i: 0 for i in range(k)}                        # will be used to calculate within cluster sum of squares
    
    for point1 in range(len(matrix)):                     # for each point 
        dist = 1                                          # we assume max dist in a range 0-1
        
        # for each centroid (cluster)
        for center in centroids:                          
            cluster_points = clusters[n_clusters[center]] # we consider all points in that cluster
            centroid_dist = matrix[center][point1]        # we calulate distance from centroid
            cum_dist = 0
            for point2 in cluster_points:                 # for each point in the centroid cluster
                cum_dist += matrix[point1][point2]        # we sum all distances
            
            mean_dist = (cum_dist+centroid_dist)/(len(cluster_points)+1) # we get the mean distance 

            if mean_dist < dist:                          # we check if the mean distance between the point and the centroid cluster is smaller than the previous dist
                cluster_id = n_clusters[center]           # assignes (temporarely) the point to cluster is k
                dist = mean_dist                          # update distance from centroid
        
        # point1 is assigned to the best cluster 
        clusters[cluster_id].append(point1)               # update with best cluster
        WSS[cluster_id] += dist                           # updating the chosed cluster with the squared distance from centroid
    
    # calculate mean distance for each cluster (wss)
    WSS = np.mean([WSS[i]/len(clusters[i]) for i in range(k)]) 
    
    return clusters, centroids, n_clusters, WSS

def ElbowMethod(matrix,threshold):
    
    # inizialization
    elbow = []
    ks = range(5,25)
    
    # trying different clusters
    for k in ks:                                              # for each number of clusters
        centroids = random.sample(range(1, len(matrix)), k)   # pick random points
        clusters, centers, n_clusters, WSS = kmeans(k,matrix,centroids)
        elbow.append(WSS)                                     # we append for each iteration the WSS score
    i = 1                                                     
    optimal = 0                                               # we want to best compromise btw low n of clusters and low WSS
    
    # applying threshold
    stop = ((max(elbow)-min(elbow))/100)*threshold+min(elbow)
    while elbow[i] > stop and i<len(elbow)-1:                 # we stop when have reached the threshold% of optimal number of clusters     
        i+=1
        optimal = elbow[i]
    
    return elbow, ks[i]

def silhouette_method(matrix):
    
    #inizialization
    ks = range(5,25)
    # points = {i: {} for i in ks} 
    scores = {}
    
    # trying different clusters 
    for k in ks:                                                      # for each number of clusters
        centroids = random.sample(range(1, len(matrix)), k)           # pick random points
        clusters, centers, n_clusters, WSS = kmeans(k,matrix,centroids)# we apply k-means
        s_score = {}
        points = {}                                                   # add a dictionary where keys are the point and clusters id are the value
        
        # creating points dictionary
        for cluster in clusters.keys():                               # for each cluster in k-means
            points.update(dict(zip(clusters[cluster],[cluster]*len(clusters[cluster])))) 
        
        # calculating mean distance and silhouette score
        for point1 in points.keys():                                  # for each point
            point1_cluster = points[point1]                           # we extract its cluster
            in_distance = 0
            out_distance = 0
            points_in = clusters[point1_cluster]                      # points in same cluster
            points_out = set(points.keys()).difference(set(points_in))# any other point
            
            for point2 in points_in:                                  # for each point in the same cluster
                in_distance+=matrix[point1][point2]                   # we add distance btw point1 and point2 to in_distance
            for point2 in points_out:                                 # for each any other point 
                out_distance+=matrix[point1][point2]                  # we add distance btw point1 and point2 to out_distance
            
            avarage_in = in_distance/len(points_in)                   # we compute avarage distance for both
            avarage_out = out_distance/len(points_out)
            s = (avarage_out-avarage_in)/max(avarage_in,avarage_out)  # we calculate silhouette score
            s_score[point1]=s                                         # we store sil_score for each point
        
        scores[k]=s_score
    
    df=pd.DataFrame.from_dict(scores)
    
    return df, df.columns[df.sum().argmax()]

def kmeans_optimize(k,matrix,iterations):
    
    # inizialization
    plot_data={}
    all_wss = []
    all_centers = []
    
    # simulations
    for i in range(iterations):                               # for each simulation
        centroids = random.sample(range(1, len(matrix)), k)   # pick random points
        itr = []
        clusters, centers, n_clusters, WSS = kmeans(k,matrix,centroids) # a different set of random centroids is generated
        for n in n_clusters.values():                         # for each cluster 
            itr.append(len(clusters[n])/len(matrix))          # we save the percentage of point it contains
        plot_data['iter_'+str(i)]=itr                         # save data to plot for each iteration
        all_wss.append(WSS)                                   # save WSS for each iteration
        all_centers.append(centers)                           # save centroids
    
    # defining best result
    best = all_wss.index(min(all_wss))                       # the best result is that with lowest wss score
    
    return plot_data,all_centers,all_wss,best

def optimize(matrix,iterations):
    
    # optimal number of clusters methods:
    # 1. elbow method
    scores_e, e = ElbowMethod(matrix,30)
    print('Elbow method: Optimal number of clusters is',e)
    # 2. silhouette method
    scores_s, s = silhouette_method(matrix)
    print('Silhouette method: Optimal number of clusters is',scores_s.columns[scores_s.sum().argmax()])    
    # best number of clouds
    k = (e+s)//2
    
    # randomly picking points
    plot_data,all_centers,all_wss,best = kmeans_optimize(k,matrix,iterations)
    print('The best iteration is', best,' whith centroids:', all_centers[best])
    centroids = all_centers[best]
    
    # plots
    plt.figure(figsize=(12,7))
    plt.plot(range(5,25),scores_e,color='#8DC99B')
    plt.title('Elbow method')
    plt.show()
    
    scores_s.sum().plot.barh(figsize=(12,7),color='#8DC99B')
    plt.title('Silhouette method')
    
    colors = sns.color_palette("Pastel1")
    pd.DataFrame.from_dict(plot_data,orient='index').plot.barh(stacked=True,color=colors, figsize=(12,7))
    plt.legend(loc='best',bbox_to_anchor=(0.7,0.5, 0.5, 0.5))
    plt.title('Best centroids')

    
    return centroids, k

####################################################################################################
                                           # PLOTS #
####################################################################################################


def encircle(x,y, ax=None, **kw):
    ''' This function encircle points beloning to the same cluster to make visualization easier'''
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    mean = np.mean(p, axis=0)
    d = p-mean
    r = np.max(np.sqrt(d[:,0]**2+d[:,1]**2 ))
    circ = plt.Circle(mean, radius=0.8*r,**kw)
    ax.add_patch(circ)
