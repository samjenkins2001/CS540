import csv
from numpy import linalg as LA
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import sys
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt



def load_data(filepath):
        country_list = []
        with open(filepath) as countries:
            reader = csv.DictReader(countries)
            for row in reader:
                country_list.append(row)
        return country_list

def calc_features(row):
    values = list(row.values())
    features = values[2:]
    if len(features) != 6:
        raise ValueError("Input dictionary must have exactly 6 features")
    feature_vector = np.array(features, dtype=np.float64)
    return feature_vector



def hac(dataset):
    Z = []
    clusters = []
    n = len(dataset)
    for i in range(n):
        clusters.append([i,tuple([i])])
    distances = distance_calc2(dataset)
    for i in range(len(distances)-1):
        next_cluster = calc_cluster(clusters,distances)
        idx_i = next_cluster[0]
        idx_j = next_cluster[1]
        dist = float(next_cluster[2])
        Z, clusters = merge(Z, dataset, clusters, i, idx_i, idx_j, dist)           
    return np.array(Z).astype("float")



def imshow_hac(Z):
    #plt.figure()
    dn = hierarchy.dendrogram(Z)
    plt.show()

    
    
def merge(Z, dataset, clusters, n, idx_i, idx_j, dist):
    for i in clusters:
        idx = i[0]
        if idx == idx_i:
            cluster_i = i
        if idx == idx_j:
            cluster_j = i
    m = len(dataset)
    cluster1 = list(cluster_i[1])
    cluster2 = list(cluster_j[1])
    
    cluster1.extend(cluster2)
    
    minn=0
    maxx=0
    if idx_i <idx_j:
        minn= idx_i
    else:
        minn= idx_j
    
    if idx_i >idx_j:
        maxx= idx_i
    else:
        maxx=idx_j
        
    Z.append([minn, maxx, dist, len(cluster1)]) 
    
    nc = [m+n, tuple(cluster1)]
    clusters.remove(cluster_i)
    clusters.remove(cluster_j)
    clusters.append(nc)
    return Z, clusters



def calc_cluster(clusters,distances):
    answer_i = -1
    answer_j = -1
    mini = float("inf")
    for i in clusters:
        idx_i = i[0]
        cli_i=i[1]
        for j in clusters:
            idx_j = j[0]
            cli_j = j[1]
            if idx_i == idx_j:
                continue             
            maxi = -1
            psuedo_i = -1
            psuedo_j = -1
            for k in cli_i:
                for l in cli_j:
                    if distances[k][l] > maxi:
                        maxi = float(distances[k,l])
                        psuedo_i = k
                        psuedo_j = l 
            if mini > maxi and maxi >= 0:
                mini = maxi
                answer_i = idx_i
                answer_j = idx_j 
    return [answer_i,answer_j,mini]         



def distance_calc2(dataset): #distance matrix
    distances = np.zeros((len(dataset),len(dataset)))
    for i in range(len(dataset)):
        for j in range(len(dataset)):
                dist = np.linalg.norm(dataset[i] - dataset[j])
                distances[i][j] = dist
    return distances


print(hac([calc_features(row) for row in load_data('countries.csv')][:50]))