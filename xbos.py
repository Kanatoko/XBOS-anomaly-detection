# Author: Kanatoko(https://twitter.com/kinyuka)
# License: BSD 3 clause

from sklearn.cluster import KMeans
import numpy as np
from pandas import DataFrame
from math import pow
import math

class XBOS:
    
    def __init__(self,n_clusters=15,effectiveness=500,max_iter=2):
        self.n_clusters=n_clusters
        self.effectiveness=effectiveness
        self.max_iter=max_iter
        self.kmeans = {}
        self.cluster_score = {}
        
    def fit(self, data):
        length = len(data)
        for column in data.columns:
            kmeans = KMeans(n_clusters=self.n_clusters,max_iter=self.max_iter)
            self.kmeans[column]=kmeans
            kmeans.fit(data[column].values.reshape(-1,1))
            assign = DataFrame(kmeans.predict(data[column].values.reshape(-1,1)),columns=['cluster'])
            cluster_score=assign.groupby('cluster').apply(len).apply(lambda x:x/length)
            ratio=cluster_score.copy()
        
            sorted_centers = sorted(kmeans.cluster_centers_)
            max_distance = ( sorted_centers[-1] - sorted_centers[0] )[ 0 ]
        
            for i in range(self.n_clusters):
                for k in range(self.n_clusters):
                    if i != k:
                        dist = abs(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[k])/max_distance
                        effect = ratio[k]*(1/pow(self.effectiveness,dist))
                        cluster_score[i] = cluster_score[i]+effect
                        
            self.cluster_score[column] = cluster_score
                    
    def predict(self, data):
        length = len(data)
        score_array = np.zeros(length)
        for column in data.columns:
            kmeans = self.kmeans[ column ]
            cluster_score = self.cluster_score[ column ]
            
            assign = kmeans.predict( data[ column ].values.reshape(-1,1) )
            #print(assign)
            
            for i in range(length):
                score_array[i] = score_array[i] + math.log10( cluster_score[assign[i]] )
            
        return score_array
    
    def fit_predict(self,data):
        self.fit(data)
        return self.predict(data)