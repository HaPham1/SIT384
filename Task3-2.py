# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:18:16 2021

@author: User
"""
import pandas as pd

#read the file
df = pd.read_csv("result_withoutTotal.csv")
#create new column total by calculating the available columns
df['Total'] = (5/100 * (df['Ass1'] + df['Ass3']) + 15/100 * (df['Ass2'] + df['Ass4']) + 50/100 * df['Exam'])


#create Final column by rounding the result we got in Total and put it to type int
df['Final'] = round(df['Total']).astype('int')

#Special condition for people who fail hurdle on Final column
df.loc[(df['Exam'] < 48 ) & (df['Total'] > 44), 'Final'] = 44

#Define Grade by putting conditions on Final column
df.loc[df['Final'] <= 49.45, 'Grade'] = 'N'
df.loc[(df['Final'] <= 59.45) & (df['Final'] > 49.45), 'Grade'] = 'P'
df.loc[(df['Final'] <= 69.45) & (df['Final'] > 59.45), 'Grade'] = 'C'
df.loc[(df['Final'] <= 79.45) & (df['Final'] > 69.45), 'Grade'] = 'D'
df.loc[df['Final'] > 79.45, 'Grade'] = 'HD'

#Save to file result_updated
df.to_csv('result_updated',
          encoding = 'utf-8',
          index = False,
          header = True)

#Save to file failderhurdle
df[df['Exam'] < 48].to_csv('failedhurdle',
          encoding = 'utf-8',
          index = False,
          header = True)

#Print the updated table
print(df)
print( )

#Print the students with exam score < 48
print("students with exam score < 48")
print(df[df['Exam'] < 48])
print()

#Print the students with exam score > 100
print("students with exam score > 100")
print(df[df['Exam'] > 100])