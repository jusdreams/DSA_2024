'''
Given a time in 12-hour AM/PM format, convert it to military (24-hour) time.
'''

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'timeConversion' function below.
#
# The function is expected to return a STRING.
# The function accepts STRING s as parameter.
#

def timeConversion(s):
    # Write your code below
    if s[-2:]== 'AM':               # if last two digits are AM then 
        if s[:2]=='12':             # if first two digits are 12 then change to 00
            return '00'+s[2:-2]     # return entire string with 00 start
        else:
            return s[:-2]           # else return the two characters at the end
    else:                           # if last two digits are not AM then
        if s[:2]=='12':             # and if first two digits are 12
            return s[:-2]           # then return the two characters at the end
        else:
            return str(int(s[:2])+12)+s[2:-2] #upon conversion, return (int of first two characters + 12) and rest of the string
    # Finsih your code above
if __name__ == '__main__':
    print('Orginial Time: ')

    s = input()
    result = timeConversion(s)

    print('Military Time: ', result)

'''
The code you provided is a function called timeConversion(). 
It takes a string as input and returns a string as output. 
The input string is a 12-hour time format, and the output string is a 24-hour time format.

The function works by first checking the last two characters of the input string. 
If the last two characters are "AM", then the function checks the first two characters of the input string.
If the first two characters are "12", then the function returns the string "00" followed by the rest of the input string.
Otherwise, the function simply returns the input string.

If the last two characters of the input string are "PM", 
then the function checks the first two characters of the input string. 
If the first two characters are "12", then the function simply returns the input string. 
Otherwise, the function returns the string that is 12 hours greater than the first two characters of the input string, 
followed by the rest of the input string.
'''

