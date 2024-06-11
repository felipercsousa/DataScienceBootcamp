import pandas as pd
import numpy as np

def KMeans(data,centroid):
    K = len(centroid)
    array_dist = np.zeros([len(data[0]),K])
    for j in range(K):
        for i in range(len(array_dist)):
                       array_dist[i][j] = euclidean_distance([data[0][i],data[1][i]], centroid[j])
    array_closestCentroid = np.zeros([len(array_dist),len(array_dist[0])])
    for i in range(len(array_dist)):
        array_closestCentroid[i] = array_dist[i]/np.min(array_dist[i])
    array_closestCentroid = np.where(array_closestCentroid>1,0)
    return array_closestCentroid

def euclidean_distance(x,y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)



df = pd.read_csv('3.01. Country clusters.csv')
data = [df['Latitude'].to_numpy(),df['Longitude'].to_numpy()]

centroid = [[1,0],[0,1]]
arr = KMeans(data,centroid)
