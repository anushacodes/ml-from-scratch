# -*- coding: utf-8 -*-
"""kmeans.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13dCtNpsfW-JVj0tytUPhBJna2kSbfa4t
"""

import numpy as np
import pandas as pd

"""
Steps to implement K-means (Lloyd's) algorithm:

params:
  data
  k
  iters
  centroids
  metric

1. calculate distance of every point to centroid
2. assign each data point to a centroid based on previous step
3. update centroid --> average of points assigned to each centroid
4. repeat with new centroids obtained

"""

def euclidean_distance(x, y):
  return np.sqrt(np.sum((x - y) ** 2))

# helper functions to calc distance

def euclidean_distance(self, x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_similarity(self, x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) )

def manhattan_distance(self, x, y):
    return np.sum(np.abs(x - y))

def squared_distance(self, x, y):
    return np.sum((x - y) ** 2)


# calculates distance based on chosen metrics
def distance(metric):
    if metric == 'square':
        return squared_distance
    elif metric == 'cosine':
        return cosine_similarity
    elif metric == 'l2':
        return euclidean_distance
    elif metric == 'l1':
        return manhattan_distance
    else:
      print("wrong metrics.")

import numpy as np
import pandas as pd

# Distance metrics
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def squared_distance(x, y):
    return np.sum((x - y) ** 2)

def cosine_similarity(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

# Choose the metric function
def get_distance_function(metric):

    if metric == 'square':
        return squared_distance

    elif metric == 'cosine':
        return cosine_similarity

    elif metric == 'l2':
        return euclidean_distance

    elif metric == 'l1':
        return manhattan_distance

    else:
      print("wrong metrics.")

# k-means algorithm
def kMeans(data, k=2, iters=5, centroids=None, metric='square'):
    distance_function = get_distance_function(metric)

    # Initialize centroids randomly if not provided
    if centroids is None:
        indices = np.random.choice(len(data), k, replace=False)
        centroids = data[indices]

    for _ in range(iters):
        # Assign points to the nearest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [distance_function(point, centroid) for centroid in centroids]
            closest_idx = np.argmin(distances)
            clusters[closest_idx].append(point)

        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                new_centroids.append(np.random.choice(data))  # Avoid empty cluster issue

        new_centroids = np.array(new_centroids)

        # Stop early if centroids don't change
        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids



data = [
    [2, 1],
    [3, 4],
    [-4, -5],
    [-8, -6],
    [-7, -2],
    [5, 3],
    [-5, -7]
]
k = 2
iters = 5
centroids = np.array([[2, 1], [3, 4]])


model = kMeans(data, k, iters, centroids, metric = 'square')
model

