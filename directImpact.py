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

def load_lockdown_prop():
	firms_prop = {}
	with open("firms_lockdown_prop.txt", 'r') as file:
		for line in file:
			line = line.split(",")
			firm = int(line[0])
			prop = float(line[-1])
			firms_prop[firm] = prop
	return firms_prop

def load_hh_share():
	hh_share = {}
	with open('firms_hhShare.txt', 'r') as file:
		for line in file:
			line = line.split(",")
			firm = int(line[0])
			share = float(line[-1])
			hh_share[firm] = share
	return hh_share

def compute_direct():
	firms_prop = load_lockdown_prop()
	hh_share = load_hh_share()

	impact = 0 
	for firm in firms_prop:
		prop = firms_prop[firm]
		share = hh_share[firm]
		v = prop * share
		impact += v

	print impact, "max direct imapct on gdp"



	impact = 0 
	for firm in firms_prop:
		prop = firms_prop[firm]
		if prop > 0.9:
			prop = 1
		share = hh_share[firm]
		v = prop * share
		impact += v

	print impact, "min direct imapct on gdp"


compute_direct()