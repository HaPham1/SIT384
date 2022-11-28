# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:09:41 2021

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import tree

#function
def plot_svc_decision_function_many(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()] + [np.repeat(0, X.ravel().size) for _ in range(55)]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=np.linspace(P.min(), P.max(), 100), alpha= 1.0,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
def plot_svc_decision_function(clf, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([[xi, yj]])
    # plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

def visualize_tree(estimator, X, y, boundaries=True,
                   xlim=None, ylim=None):
    estimator.fit(X, y)

    if xlim is None:
        xlim = (X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    if ylim is None:
        ylim = (X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)

    x_min, x_max = xlim
    y_min, y_max = ylim
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, alpha=0.2, cmap='rainbow')
    plt.clim(y.min(), y.max())

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
    plt.axis('off')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)        
    plt.clim(y.min(), y.max())
    
    # Plot the decision boundaries
    def plot_boundaries(i, xlim, ylim):
        if i < 0:
            return

        tree = estimator.tree_
        
        if tree.feature[i] == 0:
            plt.plot([tree.threshold[i], tree.threshold[i]], ylim, '-k')
            plot_boundaries(tree.children_left[i],
                            [xlim[0], tree.threshold[i]], ylim)
            plot_boundaries(tree.children_right[i],
                            [tree.threshold[i], xlim[1]], ylim)
        
        elif tree.feature[i] == 1:
            plt.plot(xlim, [tree.threshold[i], tree.threshold[i]], '-k')
            plot_boundaries(tree.children_left[i], xlim,
                            [ylim[0], tree.threshold[i]])
            plot_boundaries(tree.children_right[i], xlim,
                            [tree.threshold[i], ylim[1]])
            
    if boundaries:
        plot_boundaries(0, plt.xlim(), plt.ylim())








###########################################
##########################################
#Code Start

test_size = 0.33

#Specify column names with data from spambase.names file
column_names = ["word_freq_make","word_freq_address","word_freq_all","word_freq_3d",
                "word_freq_our","word_freq_over","word_freq_remove","word_freq_internet",
                "word_freq_order","word_freq_mail","word_freq_receive","word_freq_will",
                "word_freq_people","word_freq_report","word_freq_addresses","word_freq_free",
                "word_freq_business","word_freq_email","word_freq_you","word_freq_credit",
                "word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp",
                "word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs",
                "word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85",
                "word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct",
                "word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project","word_freq_re",
                "word_freq_edu","word_freq_table","word_freq_conference","char_freq_;","char_freq_(",
                "char_freq_[","char_freq_!","char_freq_$","char_freq_#","capital_run_length_average",
                "capital_run_length_longest","capital_run_length_total","spam"]

#Read in the data with column names
df = pd.read_csv("spambase.data", names = column_names)
print(df)
#Logistic Regression 
X_axis = df.drop('spam', axis=1)
y_axis = df['spam']
X_train, X_test, y_train, y_test = train_test_split(X_axis, y_axis, test_size=test_size, shuffle=True)

#Log reg and result printing
print("Logistic regression")
logreg = LogisticRegression(solver='lbfgs', max_iter = 5000).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(solver='lbfgs', C=100 , max_iter = 5000).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(solver='lbfgs', C=0.01, max_iter = 5000).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

#Graph for logistic regression
fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
plt.xticks(range(X_axis.shape[1]), X_axis.head(), rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-6, 6)
plt.legend()



#Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print()
print("Random Forest")
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


#Plot trees
fn = column_names
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(forest.estimators_[0],
               feature_names = fn, 
               filled = True);

#Plot forest feature importance
fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
n_features = X_axis.shape[1]
plt.barh(np.arange(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_axis.head())
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)






#Suport Vector Machines using all features
svm = SVC(kernel = 'rbf')
svm.fit(X_train, y_train)
print()
print("Support Vector Machines")
print("Accuracy of SVM classifier on training set: {:.2f}".format(svm.score(X_train, y_train)))
print("Accuracy of SVM classifier on test set: {:.2f}".format(svm.score(X_test, y_test)))

#Plot
support_vectors = svm.support_vectors_

fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], cmap = 'winter')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color = "red")
plot_svc_decision_function_many(svm);

#Preprocessing
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
svc = SVC(kernel = 'linear',C=1000)
svc.fit(X_train_scaled, y_train)

print()
print("Support Vector Machines linear after preprocess")
print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

#Plot
support_vectors = svc.support_vectors_

fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
plt.scatter(X_train_scaled.iloc[:, 0], X_train_scaled.iloc[:, 1], cmap = 'winter')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color = "red")
plot_svc_decision_function_many(svc);


svc = SVC(kernel = 'rbf',C=1000)
svc.fit(X_train_scaled, y_train)




#Support Vector Machines passing in only 2 features

X_train = X_train.iloc[:, :2]
X_test = X_test.iloc[:, :2]


svm = SVC(kernel = 'rbf')
svm.fit(X_train, y_train)
print()
print("Support Vector Machines")
print("Accuracy of SVM classifier on training set: {:.2f}".format(svm.score(X_train, y_train)))
print("Accuracy of SVM classifier on test set: {:.2f}".format(svm.score(X_test, y_test)))

#Plot
support_vectors = svm.support_vectors_

fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
plt.scatter(X_train, X_train, cmap = 'winter')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color = "red")
plot_svc_decision_function(svm);

#Preprocessing
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
svc = SVC(kernel = 'linear',C=1000)
svc.fit(X_train_scaled, y_train)

print()
print("Support Vector Machines linear after preprocess")
print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

#Plot
support_vectors = svc.support_vectors_

fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
plt.scatter(X_train_scaled, X_train_scaled, cmap = 'winter')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color = "red")
plot_svc_decision_function(svc);


svc = SVC(kernel = 'rbf',C=1000)
svc.fit(X_train_scaled, y_train)

print()
print("Support Vector Machines rbf kernel after preprocess")
print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

#Plot random forest
visualize_tree(forest, X_train.iloc[:, :2].values, y_train, boundaries=False);
