# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 20:16:45 2021

@author: User
"""
#import regular expression
import re

#prompt for email input
email = input("Please input your email address: ")

# regular expression to search for input that include words, dot, hyphen on sides and a @ between
match = re.search(r'([\w.-]+)@([\w.-]+)' ,email)

# print out email if valid and error message if not
if match:
    print("email: ", match.group(), " , ", "username: ", match.group(1), " , host: ", match.group(2))
else:
    print("Not a valid email.")
    
