from functools import reduce
import numpy as np

a = [11, 16, 14, 19, 1, 13, 15, 15, 2, 6, 4, 20, 17, 8, 18, 22, 25, 11, 18, -7]
b = [17, 15, 7, 12, 5, 20, 18, 22, 11, 2, 9, 0, 10, 11, 6, 17, 9, 10, 6]
c = [6, 16, 1, 6, 14, 5, 5, 15, 6, 11, 8, 15, 10, 3, 15, 10, 5, 14, 17, 13]
d = a+b+c

def mean(list):
    # sum = reduce(lambda a,b: a+b, list)
    return sum(list) / len(list)

def median(list):
    list.sort()
    return (list[(len(list)-1)//2]+list[(len(list))//2])/2

def single_mode(list):
    return max(list, key=list.count)

def mode(l):
    d = {}
    max = 0
    for i in l:
        if i not in d.keys():
            d[i] = 1
            if max == 0:
                max = 1
        else: 
            d[i] += 1
            if d[i] > max:
                max = d[i]
    return [k for k,v in d.items() if v==max]

def my_stdev(l):
    avg = mean(l)
    return (sum(list(map(lambda x: (x-avg)**2, l)))/len(l))**0.5

print(mean(a))
print(mean(b))
print(mean(c))
print(single_mode(a))
print(single_mode(b))
print(single_mode(c))
print(mode(a))
print(mode(b))
print(mode(c))
print(median(a))
print(median(b))
print(median(c))
print(mean(d))
print(my_stdev(d))
print(np.std(d))