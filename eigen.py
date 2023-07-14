"""
Contact - vipin.veetil@gmail.com
Language - Python 2.7.10
"""
from __future__ import division

from datetime import datetime
import snap
import math
import random
from scipy import stats
import numpy as np
import gc
import copy
from array import array
gc.enable()
from collections import Counter
import collections
import sys
from numpy.linalg import eigvals
from scipy.sparse.linalg import eigs

def computeEigen(network, norm): # the "network" is a dictionary of buyer,seller tuple as keys, and the weight between them as value
    """
    network = collections.OrderedDict({})
    with open(network_weights, 'r') as file:
        for line in file:
            line = line.split(",")
            buyer = int(line[0])
            seller = int(line[1])
            w = float(line[2])
            network[(buyer,seller)] = w
    """
    n = len(network.keys())
    A = np.zeros((n,n))

    for pair in network:
        buyer = pair[0]
        seller = pair[1]
        weight = network[pair]
        A[buyer,seller] = weight
    eigen_values = eigs(A=A,return_eigenvectors=False)
    real_eigen_values = []
    if norm:
        for v in eigen_values:
            v = np.absolute(v)
            real_eigen_values.append(v)
    else:
        for v in eigen_values:
            if np.isreal(v):
                v = np.real(v)
                real_eigen_values.append(v)
    real_eigen_values.sort(reverse=True)
    return real_eigen_values



