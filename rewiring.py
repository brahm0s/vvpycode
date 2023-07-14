from __future__ import division
import random
import copy


def rewire_degree_preserving(rewiring_factor, file_name, returnDict):
    """rewires a network while preserving the degree of each node"""
    def create_pairs_list_dict():
        """ create a list of tuples with pairs of links between nodes, each tuple represents a unique link """
        connections_pairs_list = []
        connections_dict = {}
        with open('network.txt') as network:
            for line in network:  # for each line in the file
                numbers = line.split(",")  # split numbers which are comma separated strings
                firm_id = int(numbers[0])  # the first number is firm ID
                del numbers[0]  # all but first number are IDs of input sellers
                connections = []
                for n in numbers:
                    pair = (firm_id, int(n))
                    connections_pairs_list.append(pair)
                    connections.append(int(n))
                connections_dict[firm_id] = connections
        return connections_pairs_list, connections_dict

    def rewire(connections_pairs_list, connections_dict):
        """ taking a list of tuples/pair, rewire the pairs, and return a rewired pairs list """
        edges_number = len(pairs_list)
        edges_list = range(edges_number)
        pairs_dict = dict((i, []) for i in edges_list)
        for i in edges_list:
            pairs_dict[i] = connections_pairs_list[i]
        number_of_rewiring = int(rewiring_factor * edges_number)
        attempts = number_of_rewiring * 10
        while number_of_rewiring > 0 and attempts > 0:
            two_sets_keys = random.sample(edges_list, 2)
            k0 = two_sets_keys[0]
            k1 = two_sets_keys[1]
            set0 = pairs_dict[k0]
            set1 = pairs_dict[k1]
            f0 = copy.copy(set0[0])
            f1 = copy.copy(set0[1])
            g0 = copy.copy(set1[0])
            g1 = copy.copy(set1[1])
            attempts -= 1
            if len(set(set0).intersection(set(set1))) == 0:
                if g1 not in connections_dict[f0] and f1 not in connections_dict[g0]:
                    new_set0 = (f0, g1)
                    new_set1 = (g0, f1)
                    pairs_dict[k0] = new_set0
                    pairs_dict[k1] = new_set1
                    number_of_rewiring -= 1
                    connections_dict[f0].remove(f1)
                    connections_dict[f0].append(g1)
                    connections_dict[g0].remove(g1)
                    connections_dict[g0].append(f1)
        return pairs_dict

    def make_connections_dict(pairs_dict):
        """convert pairs dict, to a dictionary of connections"""
        ID = []
        for k in pairs_dict:
            pair = pairs_dict[k]
            for n in pair:
                ID.append(n)
        ID = list(set(ID))
        ID.sort()
        connections_dict = dict((i, []) for i in ID)
        for k in pairs_dict:
            pair = pairs_dict[k]
            f0 = pair[0]
            f1 = pair[1]
            connections_dict[f0].append(f1)
        return connections_dict

    def write_network(connections_dict):
        with open('%s' % file_name, 'w') as network:
            keys = connections_dict.keys()
            for key in keys:
                input_sellers = connections_dict[key]
                input_sellers.sort()
                a = [key] + input_sellers
                a = [str(i) for i in a]
                new_a = ",".join(a)
                network.write('%s ' % new_a + '\n')

    pairs_list_dict = create_pairs_list_dict()
    pairs_list = pairs_list_dict[0]
    pairs_dict = pairs_list_dict[1]
    rewired_dict = rewire(pairs_list, pairs_dict)
    connections_dict = make_connections_dict(rewired_dict)
    write_network(connections_dict)
    if returnDict:
        return connections_dict





