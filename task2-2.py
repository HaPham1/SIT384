# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 20:16:34 2021

@author: User
"""

 #fibonacci recursive definition function 
def fibonacci_recursive (a):
    #return value of 1st number
    if a == 1:
        return 0
    #return value of 2nd number
    elif a == 2:
        return 1
    #return value of next number using the previous 2 number's value
    else:
        return fibonacci_recursive (a - 2) + fibonacci_recursive (a - 1)




# Prompt input
num = int(input("How many terms? "))

#Repeat until a valid input
while num <= 0:
    print("Please enter a positive integer")
    num = int(input("How many terms? "))

#use for loop the print all the fibonacci numbers
for i in range(1, num + 1):
    if i == num:
        print(fibonacci_recursive (i), end = " ")
    else:
        print(fibonacci_recursive (i), end = " , ")
    
