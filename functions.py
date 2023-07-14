"""
Contact - vipin.veetil@gmail.com
Language - Python 2.7.10
"""
from __future__ import division
import random
import itertools
from datetime import datetime
from itertools import cycle
random.seed(datetime.now())


def Cobb_Douglas(quantities_list, exponents_list):
    value = 1
    for q, e in itertools.izip(quantities_list, exponents_list):
        value *= q ** e
    return value


def Cobb_Douglas_simple(q0, q1, e0):
    return q0 ** e0 * q1 ** (1 - e0)

def CES_withB_vals(quantities_dict, B_vals_dict, exponent):
    val = 0
    for i in quantities_dict:
        qty = quantities_dict[i]
        b = B_vals_dict[i]
        val += (b * (qty ** exponent)) 
    return val ** (1.0 / exponent)

def CES(quantities_list, exponent):  # compute the CES function value given quantities and exponent
    return sum([q ** exponent for q in quantities_list]) ** (1.0 / exponent)


def normalized_random_numbers(how_many):
    random_numbers = [random.random() for i in xrange(how_many)]
    s = sum(random_numbers)
    random_numbers = [i/s for i in random_numbers]
    return random_numbers


def write_balanced_network(how_many, degree):
    ids = range(1, how_many + 1)
    with open('%s.txt' % 'network', 'w') as network:
        ids = range(1, how_many + 1)
        pool = cycle(ids)
        next(pool)
        for id in ids:
            ids = range(1, how_many + 1)
            pool = cycle(ids)
            for i in xrange(id):
                next(pool)
            sellers = []
            for i in xrange(degree):
                a = next(pool)
                if a != id:
                    sellers.append(a)
                else:
                    sellers.append(next(pool))
            sellers.insert(0, id)
            sellers = [str(i) for i in sellers]
            new_sellers = ",".join(sellers)
            network.write('%s ' % new_sellers + '\n')



