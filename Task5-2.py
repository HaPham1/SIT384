# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 22:00:09 2021

@author: User
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

#Read in file
df = pd.read_csv("admission_predict.csv")

#Split
training = df[0:300]
test = df[300:400]

#Preset
TOEFL = training['TOEFL Score']
CGPA = training['CGPA']
Chance = training['Chance of Admit']

TOEFL_Test = test['TOEFL Score']
Chance_Test = test['Chance of Admit']
CGPA_Test = test['CGPA']


Train_X = np.c_[TOEFL.values]
Train_Y = Chance.tolist()
Train_XX = np.c_[CGPA.values]

Test_X = np.c_[TOEFL_Test.values]
Test_Y = Chance_Test.tolist()
Test_XX = np.c_[CGPA_Test.values]



#Create the subplot
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), dpi=100)

#Specify graph
graph1 = ax[0][0]
graph2 = ax[0][1]
graph3 = ax[1][0]
graph4 = ax[1][1]


#Draw the first graph
scales = 20*np.ones(len(Train_Y))
graph1.scatter(Train_X, Train_Y, alpha=0.8, edgecolor='red', s = scales)
graph1.set_title('Linear regression with TOEFL score and chance of admit')
graph1.set_xlabel('X (TOEFL score)')
graph1.set_ylabel('Y (chance of admit)')

#Plot the linear regression line
lr = linear_model.LinearRegression()
lr.fit( Train_X , Train_Y)
y1hat = lr.predict( Train_X )
graph1.plot(Train_X, y1hat)


#Draw second graph
y1hat_test = lr.predict(Test_X)
scales = scales = 30*np.ones(len(Test_X))
graph2.scatter(Test_X, Test_Y, s=scales, color='g', edgecolor='b')
graph2.set_title('TOEFL score VS chance of admit: true value and residual')
graph2.set_xlabel('X (TOEFL score)')
graph2.set_ylabel('Y (chance of admit)')

#Plot predicted points and line
graph2.scatter(Test_X  ,y1hat_test, s = scales, color = 'b', edgecolor = 'r')
graph2.plot(Test_X , y1hat_test , color='b' , linewidth=.2)

#Get all x coordinate
tmp1 = np.reshape(Test_X,[1,len(Test_X)])[0]

# Draw residual
tmp_x1 = []
tmp_y1 = []
for i in range(len(Test_X)):
    tmp_x1 = np.append(tmp_x1,tmp1[i]) 
    tmp_y1 = np.append(tmp_y1,y1hat_test[i]) 
    tmp_x1 = np.append(tmp_x1,tmp1[i]) 
    tmp_y1 = np.append(tmp_y1,Test_Y[i]) 
    graph2.plot(tmp_x1 , tmp_y1, color='red', linewidth=0.5) 
    tmp_x1 = []
    tmp_y1 = []


#Draw the third graph
scales = 20*np.ones(len(Train_Y))
graph3.scatter(Train_XX, Train_Y, color = 'green', alpha=0.8, edgecolor='red', s = scales)
graph3.set_title('Linear regression with CGPA score and chance of admit')
graph3.set_xlabel('X (CGPA score)')
graph3.set_ylabel('Y (chance of admit)')

#Plot the linear regression line
lr = linear_model.LinearRegression()
lr.fit( Train_XX , Train_Y)
y2hat = lr.predict( Train_XX )
graph3.plot(Train_XX, y2hat)


#Draw fourth graph
y2hat_test = lr.predict(Test_XX)
scales = scales = 30*np.ones(len(Test_XX))
graph4.scatter(Test_XX, Test_Y, s=scales, color='yellow', edgecolor='green')
graph4.set_title('CGPA score VS chance of admit: true value and residual')
graph4.set_xlabel('X (CGPA score)')
graph4.set_ylabel('Y (chance of admit)')

#Plot predicted points and line
graph4.scatter(Test_XX  ,y2hat_test, s = scales, color = 'g', edgecolor = 'r')
graph4.plot(Test_XX , y2hat_test , color='g' , linewidth=.2)

#Get all x coordinate
tmp = np.reshape(Test_XX,[1,len(Test_XX)])[0]

# Draw residual
tmp_x = []
tmp_y = []
for i in range(len(Test_XX)):
    tmp_x = np.append(tmp_x,tmp[i]) 
    tmp_y = np.append(tmp_y,y2hat_test[i]) 
    tmp_x = np.append(tmp_x,tmp[i]) 
    tmp_y = np.append(tmp_y,Test_Y[i]) 
    graph4.plot(tmp_x,tmp_y,color='red',linewidth=0.5) 
    tmp_x = []
    tmp_y = []