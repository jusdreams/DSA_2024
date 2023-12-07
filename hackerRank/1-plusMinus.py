#!/bin/python3

import math
import os
import random
import re
import sys

# Given an array of integers:
# 1. Calculate the ratios of its elements that are positive, negative, and zero.
# 2. Print the decimal value of each fraction on a new line with 6 places after the decimal.

# Complete the 'plusMinus' function below.
#
# The function accepts INTEGER_ARRAY arr as parameter.

def plusMinus(arr):
    # Write your code here
    print("%.6f" % (len(list(filter(lambda x: (x > 0), arr)))/len(arr)))
    print("%.6f" % (len(list(filter(lambda x: (x < 0), arr)))/len(arr)))
    print("%.6f" % (len(list(filter(lambda x: (x == 0), arr)))/len(arr)))

if __name__ == '__main__':
    print('How many decimal places within your result: ')
    n = int(input().strip())
    print('Array Input: ')
    arr = list(map(int, input().rstrip().split()))
    print('Array:', arr)
    print()

    plusMinus(arr)

'''
    print(): This function prints the specified object to the console.
    "%.6f": This format string specifies that the number should be printed with 6 decimal places.
    len(): This function returns the length of an object.
    list(): This function converts an iterable object to a list.
    filter(): This function creates a new list that contains only the elements of the original list that satisfy the specified condition.
    lambda: This keyword is used to create a small, anonymous function.
    x: This is the name of the variable that will be used to iterate over the elements of the list.
    (x > 0): This is the condition that must be satisfied for an element to be included in the new list.
    arr: This is the name of the list that will be filtered.
'''