#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'lonelyinteger' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY a as parameter.
#

def lonelyinteger(a):
    # Write your code below
    """Returns the lonely integer in the array."""
                                        
    for i in a:                 # Iterate over the array.
        if a.count(i) == 1:     # Add the current element to the set of unique elements.
            return i            # Return the only element in the set.
    # Finish your code above
if __name__ == '__main__':

    n = int(input().strip())

    a = list(map(int, input().rstrip().split()))

    result = lonelyinteger(a)

    print(result)
