# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:57:40 2021

@author: User
"""
import matplotlib.pyplot as plt


from sklearn.datasets import load_breast_cancer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


####Code start


#load the data
cancer = load_breast_cancer()

#Specify X, y
X = cancer.data
y = cancer.target

#Scale X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA X
pca = PCA(n_components = 2)
Xpca = pca.fit_transform(X_scaled)

print("Original shape: " , X.shape)
print("Reduced shapre: " , Xpca.shape)

#Plot 2D principal
fig, ax = plt.subplots(figsize=(8, 8))


plt.scatter(Xpca[:, 0][cancer.target == 1], Xpca[:, 1][cancer.target == 1], color = 'orange', marker = '^', edgecolors= "black")
plt.scatter(Xpca[:, 0][cancer.target == 0], Xpca[:, 1][cancer.target == 0], color = 'blue', marker = 'o', edgecolors= "black")
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

print("PCA component shape: {}".format(pca.components_.shape))
print("PCA components:\n{}".format(pca.components_))


X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Plot 3D first 3
fignum = 1
fig, ax = plt.subplots(fignum, figsize=(10, 8))
cmap = plt.cm.get_cmap("Spectral")
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=10)
ax.scatter(X_scaled[:, 0],X_scaled[:, 1],X_scaled[:, 2], c=cancer.target, cmap=cmap)
plt.title("first 3 features of scaled X")

#Plot 3D first and second component
fignum = 1
fig, ax = plt.subplots(fignum, figsize=(10, 8))
cmap = plt.cm.get_cmap("Spectral")
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=10)
ax.scatter(Xpca[:, 0],Xpca[:, 1], 0, c=cancer.target, cmap=cmap)
plt.title("first two principal components of X after PCA transformation")


