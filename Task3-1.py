# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import and use numpy
import numpy as np
#my function for calculating avg, min, max and printing.
def calcall (n, extra):
    #Generate data from the file
    x = np.genfromtxt("C:/Users/User/Desktop/Deakin Uni/SIT384/3.1P-resources/result_withoutTotal.csv", delimiter = ",")
    #Specify the total number to iterate through
    LEN = 91
    #Start with default value for sum, min, max
    sum = 0
    min = x[1][n]
    max = x[1][n]
    #Go through all the score and update sum, min, max
    for i in range(1,LEN + 1):
        sum += x[i][n]
        if (x[i][n] < min):
            min = x[i][n]
        if (x[i][n] > max):
            max = x[i][n]
    #Print out different output 
    if (n == 1):
        if not(extra):
            print("ass1 average: ", sum / LEN)
            print("ass1 min:", min)
            print("ass1 max:", max)
        else:
            print("student(s) with highest Ass1:")
            print("  ID ", " Ass1 ", "Ass2", " Ass3 ", "Ass4 ", " Exam ")
            for i in range(1,LEN + 1):
                if(x[i][n] == max):
                    print(x[i])
    elif (n == 2):
        if not(extra):
            print("ass2 average: ", sum / LEN)
            print("ass2 min:", min)
            print("ass2 max:", max)
        else:
            print("student(s) with highest Ass2:")
            print("  ID ", " Ass1 ", "Ass2", " Ass3 ", "Ass4 ", " Exam ")
            for i in range(1,LEN + 1):
                if(x[i][n] == max):
                    print(x[i])
    elif (n == 3):
        if not(extra):
            print("ass2 average: ", sum / LEN)
            print("ass2 min:", min)
            print("ass2 max:", max)
        else:
            print("student(s) with highest Ass3:")
            print("  ID ", " Ass1 ", "Ass2", " Ass3 ", "Ass4 ", " Exam ")
            for i in range(1,LEN + 1):
                if(x[i][n] == max):
                    print(x[i])
    elif (n == 4):
        if not(extra):
            print("ass2 average:", sum / LEN)
            print("ass2 min:", min)
            print("ass2 max:", max)
        else:
            print("student(s) with highest Ass4:")
            print("  ID ", " Ass1 ", "Ass2", "Ass3 ", "Ass4 ", " Exam ")
            for i in range(1,LEN + 1):
                if(x[i][n] == max):
                    print(x[i])
    elif (n == 5):
        if not(extra):
            print("exam average:", sum / LEN)
            print("exam min:", min)
            print("exam max:", max)
        else:
            print("student(s) with highest Exam:")
            print("  ID ", " Ass1 ", "Ass2", " Ass3 ", "Ass4 ", " Exam ")
            for i in range(1,LEN + 1):
                if(x[i][n] == max):
                    print(x[i])

    
#Call the functions
calcall(1, False)
print( )
calcall(2, False)
print( )
calcall(3, False)
print( )
calcall(4, False)
print( )
calcall(5, False)
print( )
calcall(1, True)
print( )
calcall(2, True)
print( )
calcall(3, True)
print( )
calcall(4, True)
print( )
calcall(5, True)