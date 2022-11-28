# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:29:51 2021

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

test_size = 0.33

df = pd.read_csv("payment_fraud.csv")
df = pd.get_dummies(df, columns=['paymentMethod'])


#logistic regression
X_axis = df.drop('label', axis=1)
y_axis = df['label']
X_train, X_test, y_train, y_test = train_test_split(X_axis, y_axis, test_size=test_size, shuffle=True)

logreg = LogisticRegression().fit(X_train, y_train)
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
logreg10 = LogisticRegression(C=10).fit(X_train, y_train)
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
logreg0001 = LogisticRegression(C=0.001).fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
print("Logistic regression")

plt.plot(logreg.coef_.T, 'o', label="C=1")
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

plt.plot(logreg100.coef_.T, '^', label="C=100")
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

plt.plot(logreg10.coef_.T, '^', label="C=10")
print("Training set score: {:.3f}".format(logreg10.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg10.score(X_test, y_test)))

plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

plt.plot(logreg0001.coef_.T, 'v', label="C=0.001")
print("Training set score: {:.3f}".format(logreg0001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg0001.score(X_test, y_test)))


plt.xticks(range(X_axis.shape[1]), X_axis.head(), rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-4, 4)
plt.legend()


#decision tree
fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
print()
print()
print("Decision tree")
tree = DecisionTreeClassifier(random_state=0)
tree = tree.fit(X_train, y_train)
print("training set score: {:.3f}".format(tree.score(X_train, y_train)))
print("test set score: {:.3f}".format(tree.score(X_test, y_test)))
print(tree.max_depth)
print(tree.feature_importances_)

n_features = X_axis.shape[1]
plt.barh(np.arange(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_axis.head())
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)


