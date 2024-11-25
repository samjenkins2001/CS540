import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from csv import DictReader


def load_data(filepath):
    country_list = []
    with open(filepath) as countries:
        reader = DictReader(countries)
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

def hac(features):
    n = len(features)
    Z = np.zeros(((n-1), 4))
    cluster_list = list(range(n))
    countries_num = [1] * n

    distance_matrix = get_distance_matrix(features)

    for i in range(n - 1):
        distance = distance_matrix[distance_matrix != np.inf]
        min_dist = np.min(distance)
        min_idx = np.where(distance_matrix == min_dist)
        idx = []
        for l in range(len(min_idx[0])):
            pair = (min(min_idx[0][l], min_idx[1][l]), max(min_idx[0][l], min_idx[1][l]))
            idx.append(pair)
        row, col = idx[0]
        if row > col:
            row, col = col, row 
        z0, z1 = sorted([cluster_list[row], cluster_list[col]])
        z2 = min_dist
        z3 = countries_num[row] + countries_num[col]
        Z[i, 0] = z0
        Z[i, 1] = z1
        Z[i, 2] = z2
        Z[i, 3] = z3
        new_idx = n + i
        cluster_list[row] = new_idx
        countries_num[row] = Z[i, 3]
        cluster_list[col] = -1
        countries_num[col] = 0
        for m in range(n):
            if m != row and cluster_list[m] != -1:
                d = max(distance_matrix[row, m], distance_matrix[col, m])
                distance_matrix[row, m] = d
                distance_matrix[m, row] = d
        distance_matrix[:, col] = np.inf
        distance_matrix[col, :] = np.inf
    return Z

def get_distance_matrix(features):
    n = len(features)
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
                dist = np.linalg.norm(features[i] - features[j])
                distances[i][j] = dist
    np.fill_diagonal(distances, np.inf)
    return distances

def fig_hac(Z, names):
    plot = plt.figure()
    dendrogram(Z, labels = names, leaf_rotation = 90)
    plt.tight_layout()
    return plot

def normalize_features(features):
    features1 = np.array(features)
    mean = np.mean(features1, axis = 0)
    std = np.std(features1, axis = 0)
    normalized_features = (features1 - mean) / std
    normalized_list = list(map(np.array, normalized_features))
    return normalized_list
