#!/bin/python3

import math
import os
import random
import re
import sys

def findMedian(arr):
    sortedArray = sorted(arr)
    middle_index = len(sortedArray) // 2
    median = 0

    if len(sortedArray) % 2 == 1:
        median = sortedArray[middle_index]
        return median
    else:
        median = (sortedArray[middle_index - 1] + sortedArray[middle_index]) / 2
        return median
