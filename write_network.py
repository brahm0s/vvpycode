from __future__ import division

from datetime import datetime
import snap
import math
import firm
import household
import random
from scipy import stats
import numpy as np
import gc
import copy
import matplotlib.pyplot as plt
gc.enable()
random.seed(datetime.now())


def createAndWrite(network_type, number_of_firms, mean_degree):
    firms_ID = range(1, number_of_firms + 1)
    firms_IN_links = dict((ID, []) for ID in firms_ID)
    def create_network():
        number_of_links_between_firms = number_of_firms * mean_degree
        if network_type == "scale_free_directional":  # if the network type is scale-free
            firms_network = snap.GenPrefAttach(number_of_firms, mean_degree)  # SNAP produces undirectional  network
        elif network_type == "ER_random_directional":  # if the network is random
            firms_network = snap.GenRndGnm(snap.PNGraph, number_of_firms, number_of_links_between_firms)  # SNAP produces directional random graph
        elif network_type == "circular_directional":  # if the network is a circle
            firms_network = snap.GenCircle(snap.PNGraph, number_of_firms, mean_degree)
        edges = firms_network.Edges()  # get the edges from the network;
        if network_type == "scale_free_directional":
            for edge in edges:
                if random.uniform(0, 1) < 0.5:
                    source = edge.GetSrcNId()
                    destination = edge.GetDstNId()
                else:
                    destination = edge.GetSrcNId()
                    source = edge.GetDstNId()
                seller = firms_ID[source]
                buyer = firms_ID[destination]
                firms_IN_links[buyer].append(seller)
        else:
            for edge in edges:
                source = edge.GetSrcNId()
                destination = edge.GetDstNId()
                seller = firms_ID[source]
                buyer = firms_ID[destination]
                firms_IN_links[buyer].append(seller)

    def write_txt_file():
        with open('network.txt', 'w') as network:
            keys = firms_IN_links.keys()
            for key in keys:
                input_sellers = firms_IN_links[key]
                a = [key] + input_sellers
                a = [str(i) for i in a]
                new_a = ",".join(a)
                network.write('%s ' % new_a + '\n')

    create_network()
    write_txt_file()

def networkInformation(economy,returnOrWrite):
    # using the firms and the household, record the network of their relations and the weights
    # better still just use one file which keeps track or weights and relations
    network = {}
    for firm in economy.firms.itervalues():
        sellers = firm.input_sellers_ID
        for s in sellers:
            weight = firm.input_weights[s]
            pair = (firm.ID, s)
            network[pair] = weight
        labor_weight = firm.labor_weight
        if labor_weight>0:
            pair = (firm.ID, 0)
            network[pair] = labor_weight

    household = economy.households[-1]
    consumer_goods_sellers = household.goods_sellers_ID
    for ID in consumer_goods_sellers:
        if type(household.utility_function_exponents) is float:
            pair = (0, ID)
            network[pair] = household.utility_function_exponents
        else:
            weight = household.utility_function_exponents[ID]
            pair = (0, ID)
            network[pair] = weight
    if returnOrWrite == 'return':
        return network
    elif returnOrWrite == 'write':
        with open('network_weights.txt', 'w') as file:
            for pair in network:
                weight = network[pair]
                info = list(pair) + [weight]
                info = ','.join(map(str, info))
                file.write(info + '\n')
        print "wrote network"


#createAndWrite(network_type='scale_free_directional', number_of_firms=1000, mean_degree=5)