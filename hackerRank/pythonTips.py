'''
# if statements
n = 8
if n > 2:
    n -= 1
elif n == 2:
    n *= 2
else:
    n += 2

print("n = ", n)
'''
# Parentheses needed for multi-line conditiosn.
# and = &&
# or = ||
'''
a,b = 2,2
if ((a>2 and a != b) or a == b):
    a += 1
    print("a = ", a)

#while loops
c = 0
while c < 5:
    print("c =", c)
    c += 1

#for loops
#for (int i=0; i < 5; i++)

for i in range (0, 5, 1):
    print("i =", i)

# Division 
print(5/2) # Division is decimal by default
print(5//2) # double slash rounds down
print(-5//2) # This rounds down instead of toward zero
print(int(-5/2)) # this is the workaround

#Modding
print(5 % 2) 
#negative values don't correlate so
print(-5 % 2) 
# To be consistent with other languages modulo
import math
print(math.fmod(-5,2))

# More math helpers
import math
print(math.floor(3 / 2))
print(math.ceil(3 / 2))
print(math.sqrt(9))
print(math.pow(2,3))
#Max / Min int
float("inf")
float("-inf")
'''
'''
#arrays
arr = [1, 2, 3]
print(arr)

#can  be used as a stack
arr.append(4)
arr.append(5)
print(arr)

arr.pop()
print(arr)

#arr.pop(1) # O(n) linear 
#arr.insert(1, 7) # O(n) linear

arr[1] = 7 # O(1) constant, declare
print(arr)

#intialize array
n = 5
arr2 = [1] * n # 5 integer array full of 1's

print(arr2)
print(len(arr2))

arr2[1] = 2
arr2[2] = 3
arr2[3] = 4
arr2[4] = 5

print(arr2[3:5]) # starts at 0 and does not include end int

# unpacking
a, b, c, d, f = arr2
print(a, b, c, d, f)

#loop thorugh arrays
nums = [1, 2, 3]
#loop thorugh w/ index
for i in range(len(nums)):
    print(nums[i])
#loop through without index
for n in nums:
    print(n)

 #loop through multiple arrays simutaenously
nums1 = [1, 2, 3] 
nums2 = [49, 58, 36]  
for n1, n2 in zip(nums1, nums2):
    print(n1, n2)

    #reverse and sort
nums1.reverse()
print("Reversed: ", nums1)

#sorts ascending order
nums2.sort()
print("Sorted: ", nums2)
#sorts descending order
nums2.sort(reverse=True)
print("Descending sort = ", nums2)

#sorting strings
arr = ["Khyeem", "Chloe", "Justin", "Shacacia"]
arr.sort()
print(arr)

#custom sort (based on length of string (x))
arr.sort(key=lambda x: len(x))  


#list comprehension
arr = [i for i in range(5)]
arr2 = [i+i for i in range(5)]
arr3 = [[1] * 2 for i in range(3)] #2D list of 1 value in sets of 2. (3 total)
print(arr)
print(arr2)
print(arr3)


#ASCII value
print(ord("a"))

# Combine a list of strings with an empty delimitor
strings = ["ab", "cd", "ef"]
print("+".join(strings))


# queues (double ended queues)
from collections import  deque

queue = deque()
queue.append(1)
queue.append(2)

print(queue)

queue.popleft()
print(queue)

queue.appendleft(1)
print(queue)

queue.pop()
print(queue)


#hashSet
mySet = set()
mySet.add(1)
mySet.add(2)
print(mySet)
print(len(mySet))

print(1 in mySet)
print(2 in mySet)
print(3 in mySet)

mySet.remove(2)
print(2 in mySet)



#list to set
print(set([1,2,3]))

#set comprehension
mySet = { i for i in range(5)}
print (mySet)


# HashMap (Dictionaries)
myMap = {}
myMap["alice"] = 88
myMap["bob"] = 77
print(myMap)
print(len(myMap))

myMap["alice"] = 80
print(myMap["alice"])

print("alice" in myMap)
myMap.pop("alice")
print("alice" in myMap)

myMap = { "alice" : 90, "bob" : 70}
print(myMap)

#dict comprehesnion for graph adjacency problems
myMap = {i: 2*i for i in range(3)}
print(myMap)

#looping through maps
for key in myMap:
    print(key, myMap[key]) #iterate through key

for val in myMap.values(): #iterate through value
    print(val)

for key, val in myMap.items(): #unpacking key and value(most concise)
    print( key, val)


    #tuples are immutable ()
tup = (1, 2, 3)
print(tup)

#tuples can also be stored as keys in hashMaps(dictionaries)
#lists cant be used as keys so we use tuples
myMap = {(1,2):3}
print(myMap[(1,2)])
mySet = set()
mySet.add((1,2))
print((1,2) in mySet)


import heapq

#heaps under the hood are arrays but are used for mins and maxs

#build heap from array smallest to largest

arr = [2, 1, 8, 4, 5]
heapq.heapify(arr)
while arr:
    print(heapq.heappop(arr))

#build heap from array largest to smallest
arr2 = [2, 1, 8, 4, 5]
heapq.heapify(arr2)
while arr2:
    print(heapq.heappop(arr2))

    
def outer(a, b):
    c = "c"

    def inner():
        return a + b + c
    return inner()
print(outer("a", "b"))

'''


'''
#class
class MyClass:
    #constructor
    def __init__(self,nums):
        #create member variables
        self.nums = nums
        self.size = len(nums)

    def getLength(self):
        return self.size
    
    def getDoubledLength(self):
        return 2 * self.getLength()
  
#for loops
#for (int i=0; i < 5; i++)
arr = []
for i in range (0, 50, 2):
    arr.append(i)
    #print("i =", i)
print(arr)
'''

for _ in range(int(input())):
        name = input()
        score = float(input())

        students = []

        secondLowestName = students.append(name,score)
        
        print(secondLowestName)