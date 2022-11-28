# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:44:07 2021

@author: User
"""

import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs
from scipy.spatial import distance_matrix 
from scipy.cluster import hierarchy 


#make dataset
np.random.seed(0)
X, y = make_blobs(n_samples= 200, centers=[[3,2], [6, 4], [10, 5]], cluster_std=0.9)


#k-means
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means.fit(X)


labels = k_means.labels_
cluster_centers = k_means.cluster_centers_

fig = plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], marker='.')

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
ax = fig.add_subplot(1, 1, 1, facecolor = 'black')

for k, col in zip(range(len([[3,2], [6, 4], [10, 5]])), colors):
    my_members = (labels == k)
    cluster_center = cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
    
#set title and tick
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())


#Agglomerative Clustering
agglom = AgglomerativeClustering(n_clusters = 3, linkage = 'average')
agglom.fit(X,y)

x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
X = (X - x_min) / (x_max - x_min)

plt.figure(figsize=(6,4))
cmap = plt.cm.get_cmap("Spectral")
for i in range(X.shape[0]):
    plt.text(X[i, 0], X[i, 1], str(y[i]),
            color=cmap(agglom.labels_[i] / 10.), 
            fontdict={'weight': 'bold', 'size': 9})

#set title and tick
plt.title('Agglomerative Clustering')
plt.xticks([])
plt.yticks([])
plt.axis('off')

#distance matrix
dist_matrix = distance_matrix(X,X) 
print(dist_matrix)

condensed_dist_matrix= hierarchy.distance.pdist(X,'euclidean')
Z = hierarchy.linkage(condensed_dist_matrix, 'complete')

#plot dendogram
plt.figure(figsize=(6,4))
dendro = hierarchy.dendrogram(Z)
