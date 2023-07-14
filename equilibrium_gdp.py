"""
Contact - vipin.veetil@gmail.com
Language - Python 2.7.10
"""

from __future__ import division
import random
from datetime import datetime
random.seed(datetime.now())
import copy
import gc
import numpy as np
import math
gc.enable()



alpha = 0.25
n = 45000

# y is log GDP, y = influence_vector * episons
influence_vector = None
episons = None

# influence vector 

#W is the structure of the inputout matrix

W = None

# load the network to fill in W
A = np.array([[1, 2, 3], [3, 4, 5],[0, 4, 5]])
print A

Aprim = np.transpose(A)
print Aprim

inverse = np.linalg.inv(A)
print inverse

def create_W():
	network = a
