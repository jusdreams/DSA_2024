'''
Given five positive integers: 
1. find the minimum and maximum values that can be calculated by summing exactly four of the five integers. 
2. Then print the respective minimum and maximum values as a single line of two space-separated long integers. 
'''

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'miniMaxSum' function below.
#
# The function accepts INTEGER_ARRAY arr as parameter.
#

def miniMaxSum(arr):
    # Sort the array.
    arr.sort()
    # Calculate the minimum sum.
    min_sum = sum(arr[:4])
    # Calculate the maximum sum.
    max_sum = sum(arr[1:])
    
    #Print Results
    print(min_sum, max_sum)
    return min_sum, max_sum

if __name__ == '__main__':

    arr = list(map(int, input().rstrip().split()))

    miniMaxSum(arr)