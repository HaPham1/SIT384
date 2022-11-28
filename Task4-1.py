# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:08:55 2021

@author: User
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Read data from the file using panda
df = pd.read_csv("attack-type-frequency.csv", index_col=0, engine='python' )

#Specify the label and get the frequency for number of attacks per category 
labels = 'DOS', 'U2R', 'R21', 'PROBE'
freq = df.groupby(['category'], sort = False).count()

#create the figure and grid of subplot
fig , ax = plt.subplots(figsize=(7,5), dpi=100)

#Specify the colors 
colors = ['blue', 'red', 'green', 'yellow']
x_pos = np.arange(len(freq['frequency']))

#draw the bar chart using the data above
ax.bar(x_pos, freq['frequency'], color = colors, align = 'center')

#Set the x,y label and the title
ax.set_xlabel("Attack categories")
ax.set_ylabel("Number of attack types in each category")
ax.set_title("Attack categories and num of attack types in Cyber Security")

#Set the ticks for the labels
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)

#create the figure and grid of subplot
fig , ax = plt.subplots(figsize=(10,10))

#draw the pie chart
ax.pie(freq['frequency'], colors=colors, labels = labels, autopct='%1.1f')
