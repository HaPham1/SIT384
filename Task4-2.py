# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:20:59 2021

@author: User
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#read the file
df = pd.read_csv("Malicious_or_criminal_attacks_breakdown-Top_five_industry_sectors_July-Dec-2019.csv", index_col=0, engine='python' )

#Specify color, label, and label for x axis
colors = ['red', 'yellow', 'blue', 'green']

labels = ['Cyber incident', 'Theft of paperwork or data storage device', 'Rogue employee / insider threat', 'Social engineering / impersonation']

labelsx = ['Health service providers', 'Finance', 'Education', 'Legal, accounting & management services', 'Personal services']

n_type, n_sector = df.shape

#Specify xpos and 2 width for 2 charts
x_pos = np.arange(n_sector)
w1 = 0.2
w2 = 0.6

#Create the subplot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), dpi = 100)

#Draw the bar chart
for i in range(n_type):
    ax[0].bar(x_pos + i*w1, df.iloc[i, :], width = w1, align='center', label=labels[i], color=colors[i])
    
#Specify color for bottom stacks
red = df.iloc[0, :]
yellow = df.iloc[1, :]
blue = df.iloc[2, :]

#Draw the stacked bar chart
ax[1].bar(x_pos, df.iloc[0, :], width = w2, align='center', label=labels[0], color=colors[0])
ax[1].bar(x_pos, df.iloc[1, :], bottom = red, width = w2, align='center', label=labels[1], color=colors[1])  
ax[1].bar(x_pos, df.iloc[2, :], bottom = red + yellow, width = w2, align='center', label=labels[2], color=colors[2])  
ax[1].bar(x_pos, df.iloc[3, :], bottom = red + yellow + blue, width = w2, align='center', label=labels[3], color=colors[3])     

#Specify x, y label and title
ax[0].set_xlabel("the top five industry sectors")
ax[0].set_ylabel("Number of attack")
ax[0].set_title("Type of attack by top five industry sectors")

ax[1].set_xlabel("the top five industry sectors")
ax[1].set_ylabel("Number of attack")
ax[1].set_title("Type of attack by top five industry sectors")

#Create tick and tick label
ax[0].set_xticks(x_pos)
ax[0].set_xticklabels(labelsx, rotation = 90)
ax[1].set_xticks(x_pos)
ax[1].set_xticklabels(labelsx, rotation = 90)

#Legend for 2 charts
ax[0].legend()
ax[1].legend()
