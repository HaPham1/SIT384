# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:56:17 2021

@author: User
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Read in file
df = pd.read_csv("data_correlation.csv")
a = df['a']
b = df['b']
c = df['c']
d = df['d']

#Calculate
pearson_r1 = np.cov(a, b)[0, 1] / (a.std() * b.std())
pearson_r2 = np.cov(a, c)[0, 1] / (a.std() * c.std())
pearson_r3 = np.cov(a, d)[0, 1] / (a.std() * d.std())
corrcoef1 = np.corrcoef(a, b)
corrcoef2 = np.corrcoef(a, c)
corrcoef3 = np.corrcoef(a, d)

#Print out results
print("a and b pearson_r: ", pearson_r1)
print("a and b corrcoef: ", corrcoef1)
print()
print("a and c pearson_r: ", pearson_r2)
print("a and c corrcoef: ", corrcoef2)
print()
print("a and d pearson_r: ", pearson_r3)
print("a and d corrcoef: ", corrcoef3)

#Graph for a,cb
fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

ax.scatter(a,b, alpha=0.6, edgecolor='none', s=100)
ax.set_xlabel('a')
ax.set_ylabel('b')

line_coef = np.polyfit(a, b, 1)
xx = np.arange(0, 50, 0.1)
yy = line_coef[0]*xx + line_coef[1]
ax.plot(xx, yy, 'r', lw=2)


#Graph for a,c
fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

ax.scatter(a,c, alpha=0.6, edgecolor='none', s=100)
ax.set_xlabel('a')
ax.set_ylabel('c')

line_coef = np.polyfit(a, c, 1)
xx = np.arange(0, 50, 0.1)
yy = line_coef[0]*xx + line_coef[1]
ax.plot(xx, yy, 'g', lw=2)


#Graph for a,d
fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

ax.scatter(a,d, alpha=0.6, edgecolor='none', s=100)
ax.set_xlabel('a')
ax.set_ylabel('d')