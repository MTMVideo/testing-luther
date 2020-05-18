# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:06:57 2020

@author: Luther Ollier

____________________________________________Git, Matrix multiplication_____________________________________________


"""

import numpy as np 

def matmul(A,B):
    """take in entry
    two matrix and return the AB multiplication result
    return an error if the shapes don't match"""
    return np.dot(A,B)
    
A = np.random.randint(10, size=(5,10))
B = np.random.randint(10, size=(10,7))
print("AB={}".format(matmul(A,B)))