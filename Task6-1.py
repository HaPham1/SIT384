# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:22:26 2021

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
from matplotlib.colors import ListedColormap

#set up constant.
n_per_class = 50

#read file
df = pd.read_csv("task6_1_dataset.csv")

#set up datas
df1 = df.loc[df['y'] == 0]
x1 = df1['x1']
y1 = df1['x2']

df2 = df.loc[df['y'] == 1]
x2 = df2['x1']
y2 = df2['x2']

df3 = df.loc[df['y'] == 2]
x3 = df3['x1']
y3 = df3['x2']

#first graph
colors = ['green', 'blue', 'magenta']

scale = 75
alpha = 0.6

fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
ax.scatter (x1, y1, alpha = alpha, color = colors[0], s = scale)
ax.scatter (x2, y2, alpha = alpha, color = colors[1], s = scale)
ax.scatter (x3, y3, alpha = alpha, color = colors[2], s = scale)


x = np.r_[x1, x2, x3]
y = np.r_[y1, y2, y3]

X_train = np.c_[x, y]
Y_train = np.r_[0*np.ones(n_per_class), 1*np.ones(n_per_class), 2*np.ones(n_per_class)]

k = 1
knn = neighbors.KNeighborsClassifier(k)
knn.fit(X_train, Y_train)

cmap_bold = ListedColormap(['green', 'blue', 'magenta'])
X_test = [[-4, 8]]
Y_pred = knn.predict(X_test)

ax.scatter(X_test[0][0], X_test[0][1], marker="x", s=scale, lw=2, c = colors[Y_pred.astype(int)[0]])

#graph title and test point label
plt.text(X_test[0][0], X_test[0][1], "({}, {}), test point".format(X_test[0][0], X_test[0][1]), alpha= alpha, color = 'red', size = 10)
plt.title("3-Class classification (k = {})".format(k))
ax.set_title("3-Class classification (k = {})\n the test point is predicted as class {}".format(k, colors[Y_pred.astype(int)[0]]))

######Second part
#second graph
fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

x = np.r_[x1, x2, x3]
y = np.r_[y1, y2, y3]

X_train = np.c_[x, y]
Y_train = np.r_[0*np.ones(n_per_class), 1*np.ones(n_per_class), 2*np.ones(n_per_class)]

k = 15
knn = neighbors.KNeighborsClassifier(k)
knn.fit(X_train, Y_train)

h = 0.05

#set color
cmap_light = ListedColormap(['#AAFFAA', '#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['green', 'blue', 'magenta'])

x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

Z = knn.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

ax.pcolormesh(xx1, xx2, Z, cmap=cmap_light)
ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cmap_bold, alpha=alpha, s=scale)

X_test = [[-2, 5]]
Y_pred = knn.predict(X_test)

ax.scatter(X_test[0][0], X_test[0][1], marker="x", s=scale, lw=2, c = colors[Y_pred.astype(int)[0]])

#graph title
plt.text(X_test[0][0], X_test[0][1], "({}, {}), test point".format(X_test[0][0], X_test[0][1]), alpha= alpha, color = 'red', size = 10)
plt.title("3-Class classification (k = {})".format(k))
ax.set_title("3-Class classification (k = {})\n the test point is predicted as class {}".format(k, colors[Y_pred.astype(int)[0]]))