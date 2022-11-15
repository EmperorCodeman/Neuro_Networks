#!/bin/python3

import math
import os
import random
import re
import sys


import numpy as np
#
# Complete the 'getHeaviestPackage' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts INTEGER_ARRAY packageWeights as parameter.
#
forks = []

def recurse(packageWeights):
    packageWeights = np.array(packageWeights)
    
    sub_trys = []
    for i in range(len(packageWeights)-1):
        if packageWeights[i] < packageWeights[i+1]: 
            sub_weights = np.zeros(shape=(len(packageWeights)-1))
            sub_weights[:i] = packageWeights[:i]
            sub_weights[i+1:] = packageWeights[i+2:]
            sub_weights[i] =  np.sum(packageWeights[i:i+2])
            sub_trys.append( sub_weights )
    if len(sub_trys) == 0: 
        return np.max(packageWeights)
    
    for sub in sub_trys:
        forks.append( np.max( recurse(sub) ) )
    
    return 0

def getHeaviestPackage(packageWeights):
    print(packageWeights)
    recurse(packageWeights)
    
    return np.max(forks)
    
    
packageWeights = [20, 13, 8, 9, 70, 5, 40]


print( getHeaviestPackage(packageWeights) )

