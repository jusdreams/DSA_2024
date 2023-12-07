'''
There is a collection of input strings and a collection of query strings.
For each query string, determine how many times it occurs in the list of input strings. 
Return an array of the results.  
'''

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'matchingStrings' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. STRING_ARRAY strings
#  2. STRING_ARRAY queries
#

def matchingStrings(strings, queries):
    """Returns a list of the number of times each query
       string appears in the strings list."""

    # Write your code here
    ans = []                            #create an empty list to store the results
    for x in queries:                   #iterate over queires list
        ans.append(strings.count(x));   #append the count (number of times the current query string appears in the strings list) to the resaksults list
    return (ans)                        #return the results list
    # finish your code above

if __name__ == '__main__':
    
    strings_count = int(input().strip())

    strings = []

    for _ in range(strings_count):
        strings_item = input()
        strings.append(strings_item)

    queries_count = int(input().strip())

    queries = []

    for _ in range(queries_count):
        queries_item = input()
        queries.append(queries_item)

    res = matchingStrings(strings, queries)

    print(res)
