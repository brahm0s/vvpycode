from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import copy
import random
import time
import math
import collections
import ast
from multiprocessing import Pool
import os
from scipy.sparse import csr_matrix
import os.path
import matlab.engine

def write_dat(num_firms, degree_size):
	child_directory = 'data_realShocks'
	if not os.path.exists(child_directory):
		os.makedirs(child_directory)    
	os.chdir(child_directory)
	if degree_size == 'degree':
		file = "out_distER" + str(num_firms) + ".txt"
		with open(file,'r') as file:
		    for line in file:
		        degree = ast.literal_eval(line)
		data = degree[-1]

	elif degree_size == 'size':
		file = "size_distER" + str(num_firms) + ".txt"
		with open(file,'r') as file:
		    for line in file:
		        sizes = ast.literal_eval(line)
		data = list(sizes[-1])
	

	name = degree_size + ".dat"
	with open(name, 'w') as file:
		data = [str(i) for i in data]		
		data = " ".join(data)					
		file.write('%s ' % data + '\n') 
	os.chdir('..')

def perform_matlab_tests():
	eng = matlab.engine.start_matlab()
	eng.test_dist(nargout=0)
	eng.quit()



def write():
	write_dat(num_firms=10000, degree_size='size')	
	write_dat(num_firms=10000, degree_size='degree')	
	


write()
#perform_matlab_tests()
