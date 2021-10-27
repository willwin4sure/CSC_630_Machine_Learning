import random
import numpy as np
from numpy.lib.function_base import insert
import time

def is_sorted(list):
    for i in range(len(list)-1):
        if (list[i] > list[i+1]):
            return False
    return True

def bogosort(list):
    while True:
        random.shuffle(list)
        if is_sorted(list):
            return list

def insertionsort(list):
    for i in range(1, len(list)):
        j = i-1
        next = list[i]
        while next < list[j] and j > -1:
            list[j+1] = list[j]
            j -= 1
        list[j+1] = next
    return list

def mergesort(list):
    if len(list) < 20:
        return insertionsort(list)
    else:
        first = list[0:len(list)//2]
        second = list[len(list)//2:len(list)]
        first = mergesort(first)
        second = mergesort(second)
        output = []
        firstpointer = 0
        secondpointer = 0
        while (firstpointer < len(list)//2 and secondpointer < (len(list)+1)//2):
            if (first[firstpointer] > second[secondpointer]):
                output.append(second[secondpointer])
                secondpointer += 1
            else:
                output.append(first[firstpointer])
                firstpointer += 1

        while (firstpointer < len(list)//2):
            output.append(first[firstpointer])
            firstpointer += 1
        
        while (secondpointer < (len(list)+1)//2):
            output.append(second[secondpointer])
            secondpointer += 1

        return output

list = [i for i in range(500000)]
random.shuffle(list)
list2 = list
merge_start = time.time()
mergesort(list)
merge_end = time.time()
print(f'Merge: {merge_end-merge_start}')

sort_start = time.time()
list2.sort()
sort_end = time.time()
print(f'Sort: {sort_end-sort_start}')

