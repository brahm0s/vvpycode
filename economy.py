"""
Contact - vipin.veetil@gmail.com
Language - Python 2.7.10
"""
from __future__ import division

from datetime import datetime
import random
random.seed(datetime.now())
import snap
import math
import firm
import household
import random
from scipy import stats
import numpy as np
import gc
import copy
import os
from array import array
gc.enable()
#random.seed(0)
from collections import Counter
import collections
import sys


class Economy(object):
    def __init__(self, parameters):
        self.parameters = parameters
        #self.networkDict = networkDict
        self.households = {}  # dictionary with household IDs as keys and household objects as values
        self.firms = {}  # dictionary with firm IDs as keys and firm objects as values
        #self.firms = []
        self.retail_firms = []
        self.firms_ID = []
        self.out_of_market_firms_ID = []
        self.households_ID = []
        self.retail_firms_ID = []  # list of IDs of retail firms; retail firms sell goods to households; they may sell goods to other firms as well
        self.retail_suppliers_firms_ID = []
        self.retail_suppliers_firms = []
        self.all_ID = []
        self.inflation_agents_ID = []
        self.number_of_non_retail_firms = 0
        self.stable_prices_firms = {}
        self.PCE_weights = {}
        self.CPI_weights = {}
        self.GDP_shares = {}
        self.lockdown_prop = {}
        self.firms_sectors = {}
        self.sectoral_output_form = {}
        self.degree_distribution_in = []
        self.degree_distribution_out = []
        self.size_distribution = []
        self.cv_firm_size_change = []
        self.pce_sectoral_outputs = {}
        self.sector_hhShare = {}
        if self.parameters.record_variables['economy']['firm_volatility']:
            self.firm_sizes = {}
            self.last_annual_firm_size = {}
        for variable in parameters.record_variables['economy']:
            if parameters.record_variables['economy'][variable]:
                setattr(self, variable, [])

    def set_B_vals_CES(self):
        with open("B_vals_CES.txt",'r') as file:
            for line in file:
                line = line.split(",")
                buyer = int(line[0])
                seller = int(line[1])
                val = float(line[2])

                buyer_firm = self.firms[buyer]                
                buyer_firm.B_vals_CES[seller] = val

    def set_sectors(self):
        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)
        print "setting sectors"
        sectors = []
        #with open("firms_twoD_naics.txt",'r') as file:
        #with open("firms_fourD_naics.txt",'r') as file:
        with open("firms_sectors.txt",'r') as file:
            for line in file:
                line = line.split(",")
                firm = line[0]
                sec = line[1]
                sec = int(sec)
                sec = str(sec)
                self.firms_sectors[int(firm)] = sec
                sectors.append(sec)
                self.pce_sectoral_outputs[sec] = 0

        if self.parameters.files_directory:
            os.chdir('..')
        

        sectors = list(set(sectors))
        sectors = sorted(sectors)
        for s in sectors:
            self.sectoral_output_form[s] = 0

    def loadPCE_weights(self):  

        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)

        if self.parameters.firms_pce_file:
            file_name = "firms_pce_weight.txt"

        elif self.parameters.addFileName:
            file_name = "PCE_weights" + self.parameters.addFileName + '.txt'
        else:
            file_name = "PCE_weights.txt"

        with open(file_name, 'r') as file:
            for line in file:
                if self.parameters.firms_pce_file:
                    line = line.split(" ")
                else:
                    line = line.split(",")
                ID = int(line[0])
                #ID = str(int(line[0]))
                weight = float(line[1])
                self.PCE_weights[ID] = weight

        if self.parameters.files_directory:
            os.chdir('..')

    def load_lockdown_prop(self):
        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)

        #print "loading lockdown prop"
        with open('firms_lockdown_prop.txt','r') as file:
            #print "in file lockdown prop"
            for line in file:
                line = line.split(",")                
                firm = int(line[0])
                prop = float(line[1])
                a = 1 - prop
                a *= self.parameters.lockdownShock_multiple
                prop = 1 - a

                #print prop

                #adj_prop = min(prop,1)



                self.lockdown_prop[firm] = prop
                self.firms[firm].prop_lockdown = prop

                #self.lockdown_prop[firm] = adj_prop
                #self.firms[firm].prop_lockdown = adj_prop

                #print firm,self.firms[firm].prop_lockdown
                #if prop > 1:
                 #   print "greater"

        

        if self.parameters.files_directory:
            os.chdir('..')
            
        for firm in self.firms.itervalues():
            buyers = firm.output_buyers_ID
            firm.lockdown_output_buyers_prop = {}
            for b in buyers:
                p = self.lockdown_prop[b]
                firm.lockdown_output_buyers_prop[b] = p
                #if p > 1:
                 #   print "greater"

    def load_fixedSharesGDP(self):

        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)

            print "changed director"
        
        with open("firms_hhShare.txt", 'r') as file:
            for line in file:
                line = line.split(",")
                firm = int(line[0])
                share = float(line[1])
                self.GDP_shares[firm] = share

        if self.parameters.files_directory:
            os.chdir('..')

    def loadCPI_weights(self):
        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)

        if self.parameters.addFileName:
            file_name = "CPI_weights" + self.parameters.addFileName + '.txt'
        else:
            file_name = "CPI_weights.txt"

        with open(file_name, 'r') as file:
            for line in file:
                line = line.split(",")
                ID = int(line[0])
                weight = float(line[1])
                self.CPI_weights[ID] = weight

        if self.parameters.files_directory:
            os.chdir('..')

    def set_number_of_firms_households(
            self):  # if the network comes from an external file, read the file to count number of firms and set that as the number of firms in the parameter file; use number of firms to compute number of household in parameter file
        # network.txt file must be a txt file in which each line has comma separated IDs of firms; the first ID is ID of the firm, the following IDs are IDs of input sellers of the firm
        # firm IDs must be integers beginning from 1 (not 0)
        # the file must not firm that appear as sellers of inputs, but do not appear as first ID in any line of the file
        # it is ok if the file has firms that have no sellers of inputs, i.e. a line with just one ID
        # the file must not have any repeats, i.e. no line must have repeat an ID
        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)


        if self.parameters.addFileName:
            file_name = "network" + self.parameters.addFileName + '.txt'
        else:
            file_name = "network.txt"

        with open(file_name,'r') as network:
            self.parameters.n = sum(1 for _ in network)  # count the number of firms in the file as the number of lines in the file
            #print self.parameters.n, "nnn firms here"
            if self.parameters.representative_household:
                 self.parameters.number_of_households = 1
            else:                
                self.parameters.number_of_households = self.parameters.hf * self.parameters.n  # set the number of households in the parameters file
        

        if self.parameters.files_directory:
            os.chdir('..')
        #print self.parameters.n, "number of firms", 
        #print self.parameters.number_of_households, "number of households"

    def create_firms_households(self):  # create instances of firm and household objects
        if self.parameters.representative_household:
            self.households = {}
            self.households[-1] = household.Household(-1, self.parameters)
            self.households_ID = [-1]
            self.parameters.number_of_households = 1

        else:
            households_list = [household.Household(-(i + 1), self.parameters) for i in
                               xrange(self.parameters.number_of_households)]  # create a list of household objects
            self.households_ID = range(-self.parameters.number_of_households,
                                       0)  # create a list of household IDs; household IDs are negative
            self.households_ID.reverse()
            self.households = dict(zip(self.households_ID,
                                       households_list))  # create households dictionary with objects as values and IDs as keys
        firms_list = [firm.Firm((i + 1), self.parameters) for i in xrange(self.parameters.n)]  # create a  list of firms
        self.firms_ID = range(1, self.parameters.n + 1)  # create a list of firm IDs; firm IDs begin from 1
        self.firms = dict(
            zip(copy.copy(self.firms_ID), firms_list))  # create firms dictionary with objects as values and IDs as keys
        self.all_ID = self.households_ID + self.firms_ID
        self.firms_list = firms_list

        if self.parameters.record_variables['economy']['firm_volatility']:
            self.firm_volatility = {}
            for ID in self.firms_ID:
                self.firm_sizes[ID] = []
                self.firm_volatility[ID] = []

    def set_household_expenditure_share(self):
        firms_shares = {}
        with open("firms_hhShare.txt", 'r') as file:
            for line in file:
                line = line.split(",")
                firm = int(line[0])
                share = float(line[1])
                firms_shares[firm] = share

        for firm in firms_shares:
            share = firms_shares[firm]
            self.households[-1].utility_function_exponents[firm] = share

    def set_labor_allocation_shares(self):

        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)


        sizes = {}
        with open("sizes.txt",'r') as file:
            for line in file:
                line = line.split(",")
                ID = int(line[0])
                if ID > 0:
                    s = float(line[1])
                    sizes[ID] = s
        total = sum(sizes.values())

        shares = {}

        for ID in sizes:
            s = sizes[ID]
            shar = s/total
            shares[ID] = shar

        self.households[-1].labor_fixed_shares = shares

        if self.parameters.files_directory:
            os.chdir("..")

    def select_inflation_agents(self):
        if type(self.parameters.inflation_agents_ID) is list:
            self.inflation_agents_ID = self.parameters.inflation_agents_ID
        elif self.parameters.inflation_agents_ID == 'file':
            self.inflation_agents_ID = []
            with open('inflationAgents') as inflationIDS:
                for line in inflationIDS:
                    ID = int(line)
                    self.inflation_agents_ID.append(ID)
        else:
            if self.parameters.money_injection_agents == "firms":
                number = max(1, int(self.parameters.n * self.parameters.g))
                self.inflation_agents_ID = random.sample(self.firms_ID, number)
            elif self.parameters.money_injection_agents == "households":
                number = max(1, int(self.parameters.number_of_households * self.parameters.g))
                self.inflation_agents_ID = random.sample(self.households.keys(), number)
            elif self.parameters.money_injection_agents == "all":
                number = max(1, int((self.parameters.number_of_households + self.parameters.n) * self.parameters.g))
                all_ID = self.households.keys() + self.firms_ID
                self.inflation_agents_ID = random.sample(all_ID, number)

    def create_firms_network_snap(self):  # create a network of firms using Stanford Network Analysis Project Library (snaself.parameters.stanford.edu)
        number_of_links_between_firms = self.parameters.n * self.parameters.d

        print self.parameters.network_type, "network type"
        if self.parameters.network_type == "SF":  # if the network type is scale-free
            firms_network = snap.GenPrefAttach(self.parameters.n, self.parameters.d)  # SNAP produces undirectional  network
        elif self.parameters.network_type == "ER":  # if the network is random
            firms_network = snap.GenRndGnm(snap.PNGraph, self.parameters.n,
                                           number_of_links_between_firms)  # SNAP produces directional random graph
        elif self.parameters.network_type == 'B': # if the network is a circle
            firms_network = snap.GenCircle(snap.PNGraph, self.parameters.n, self.parameters.d)
        elif self.parameters.network_type == 'powerlaw': # if the network is a circle
            firms_network = snap.GenRndPowerLaw(self.parameters.n, self.parameters.powerlaw_exponent,True)
            print "created powerlaw network"
        edges = firms_network.Edges()  # get the edges from the network;
        if self.parameters.network_type == "SF" or self.parameters.network_type == "powerlaw":  # if network is scale-free, give it direction; the scale free network produced by SNAP library is undirectional. This part of the code makes the network directional.
            for edge in edges:  # assign firms their buyers and sellers
                if random.uniform(0, 1) < self.parameters.scale_free_network_symmetry:  # the scale_free_network_symmetry parameter determines where in-deg and out-deg distribution have same slope
                    source = edge.GetSrcNId()
                    destination = edge.GetDstNId()
                else:
                    destination = edge.GetSrcNId()
                    source = edge.GetDstNId()
                seller = self.firms[source + 1]
                buyer = self.firms[destination + 1]
                seller.output_buyers_ID.append(destination + 1)
                buyer.input_sellers_ID.append(source + 1)
        elif self.parameters.network_type == "ER" or self.parameters.network_type == "B":  # the Erdos Renyi network produced by SNAP is directional
            for edge in edges:
                source = edge.GetSrcNId()
                destination = edge.GetDstNId()
                seller = self.firms[source + 1]
                buyer = self.firms[destination + 1]
                seller.output_buyers_ID.append(destination + 1)
                buyer.input_sellers_ID.append(source + 1)

    def create_firms_network_txt_file(self):  # create firms text from an external txt file
        self.parameters.number_of_links_between_firms = 0
        input_sellers = {}  # create a dictionary with firm IDs as keys and a list of input sellers IDs as values

        linkages = 0
        numFirms = 0

        if self.parameters.addFileName:
            file_name = "network" + self.parameters.addFileName + '.txt'
        else:
            file_name = "network.txt"

        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)

        with open(file_name, 'r') as network:
            for line in network:  # for each line in the file
                numFirms += 1
                numbers = line.split(",")  # split numbers which are comma separated strings
                firm_ID = copy.copy(int(numbers[0]))  # the first number is firm ID
                del numbers[0]  # all but first number are IDs of input sellers
                sels = [int(ID) for ID in numbers]
                input_sellers[firm_ID] = sels  # assign each firm its input sellers
                linkages += len(input_sellers[firm_ID])


        #print linkages, "number of linkages"
        #print numFirms, "number of firms"
        for firm_ID in self.firms.iterkeys():  # assign each firm IDs of its input-sellers and output buyers
            input_sellers_ID = input_sellers[firm_ID]
            firm = self.firms[firm_ID]
            if len(input_sellers_ID) != len(set(input_sellers_ID)):
                cnt = Counter(input_sellers_ID)
                print [k for k, v in cnt.iteritems() if v > 1]
                print firm_ID

            assert len(input_sellers_ID) == len(set(input_sellers_ID)), "repeat of input seller indices"

            #print len(self.firms), "num firms"
            for seller_ID in input_sellers_ID:
                seller = self.firms[seller_ID]
                firm.input_sellers_ID.append(seller_ID)
                seller.output_buyers_ID.append(firm_ID)
                self.parameters.number_of_links_between_firms += 1

        if self.parameters.files_directory:
            os.chdir("..")

    def set_sector_hhShare(self):
        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)

        with open("sectors_hhShare.txt",'r') as file:
            for line in file:
                line = line.split(",")
                sec = line[0]
                share = float(line[1])
                self.sector_hhShare[sec] = share

        with open("firms_sectors.txt",'r') as file:
            for line in file:
                line = line.split(",")
                firm = int(line[0])
                sec = str(int(line[1]))
                self.firms_sectors[firm] = sec

        if self.parameters.files_directory:
            os.chdir("..")


    def set_networkWeights_txt_file(self):

        if self.parameters.files_directory:
            os.chdir(self.parameters.files_directory)

        hh = self.households[-1]
        hh.utility_function_exponents = collections.OrderedDict({})
        laborBuyers = []
        goodsSellers = []

        count = 0
        highSelfWeight = []
        if self.parameters.addFileName:
            file_name = "weights" + self.parameters.addFileName + '.txt'
        else:
            file_name = "weights.txt"

        with open(file_name, 'r') as file:
            for line in file: # for line in file
                count += 1
                line = line.split(",")
                buyerID = int(line[0]) # buyer is 0
                sellerID = int(line[1]) # seller 1
                w = float(line[2]) # weight is last


                if buyerID != 0 and sellerID != 0: # if the buyer and the seller are both firms
                    buyer = self.firms[buyerID] # get the buyer
                    seller = self.firms[sellerID]
                    buyer.input_weights[sellerID] = w # assign the buyer input weight of seller ID
                    buyer.input_weights_optimal[sellerID] = w # the same for optimal weights
                    buyer.Cobb_Douglas_exponents[sellerID] = w
                    if sellerID not in buyer.input_sellers_ID:
                        buyer.input_sellers_ID.append(sellerID)
                        buyer.input_demands[sellerID] = random.uniform(0,1)
                        buyer.input_quantities[sellerID] = random.uniform(0,1)
                        buyer.input_prices[sellerID] = random.uniform(0,1)
                        buyer.number_of_input_sellers += 1
                    if buyerID not in seller.output_buyers_ID:
                        seller.output_buyers_ID.append(buyerID)
                        seller.output_demands[buyerID] = random.uniform(0,1)
                        seller.output_allocation[buyerID] = random.uniform(0, 1)
                        seller.number_of_output_buyers += 1

                elif buyerID == 0 and sellerID != 0: # if the buyer is the household and the seller is a firm
                    hh.utility_function_exponents[sellerID] = w # add the weight as an exponent of utility function of the household
                    hh.goods_sellers_ID.append(sellerID) # add the firm as a good seller to the household
                    goodsSellers.append(sellerID) # append the list of goods sellers
                elif buyerID != 0 and sellerID == 0: # if buyer is a firm and the seller is a household
                    buyer = self.firms[buyerID] # get the buyer
                    buyer.CD_labor = w  # add the weight as the Cobb Douglas exponent of labor because firm is spending that much on labor
                    buyer.labor_weight = w
                    #hh.labor_buyers_ID.append(buyerID) # add the firm as a labor buyer
                    laborBuyers.append(buyerID) # append labor buyers
                elif buyerID == 0 and sellerID == 0: # if the buyer and the seller are both the household
                    hh.selfWeight = w # add a self weights

        if self.parameters.files_directory:
            os.chdir("..")

        noLabor = []
        for firm in self.firms.itervalues():
            if firm.CD_labor == None:
                noLabor.append(firm.ID)

        if len(noLabor) > 0:
            print noLabor, "no labor"
            sys.exit()

        # for firms who do not sell goods to the household
        # add the firms as a good seller with 0 weight in the household utility function
        # also add the firm as a goods seller

        for firm in self.firms.itervalues():
            ID = firm.ID
            if ID not in goodsSellers:
                hh.utility_function_exponents[ID] = 0  # add the weight as an exponent of utility function of the household
                hh.goods_sellers_ID.append(ID)
        #print len(goodsSellers), "number of goods sellers"

    def set_sizes_txt_file(self):
        sizes = collections.OrderedDict({})
        if self.parameters.addFileName:
            file_name = "sizes" + self.parameters.addFileName + '.txt'
        else:
            file_name = "sizes.txt"

        with open(file_name, 'r') as file:
            for line in file:
                line = line.split(",")
                ID = int(line[0])
                s = float(line[1])
                sizes[ID] = s
        hh = self.households[-1]
        hh.wealth = sizes[0]

        for ID in self.firms:
            s = sizes[ID]
            firm = self.firms[ID]
            firm.wealth = s

    def set_firm_buyer_seller_values(self):
        for firm in self.firms.itervalues():
            firm.number_of_input_sellers = len(firm.input_sellers_ID)
            firm.number_of_output_buyers = len(firm.output_buyers_ID)

    def connect_firms_without_sellers(self):
        for firm in self.firms.itervalues():
            if firm.number_of_input_sellers == 0:
                selected_firm = random.sample(self.firms.values(), 1)[0]
                firm.input_sellers_ID.append(selected_firm.ID)
                firm.number_of_input_sellers = 1
                selected_firm.output_buyers_ID.append(firm.ID)
                selected_firm.number_of_output_buyers += 1

    def connect_firms_without_buyers(self):
        for firm in self.firms.itervalues():
            if firm.number_of_output_buyers == 0:
                selected_firm = random.sample(self.firms.values(), 1)[0]
                firm.output_buyers_ID.append(selected_firm.ID)
                firm.number_of_output_buyers = 1
                selected_firm.input_sellers_ID.append(firm.ID)
                selected_firm.number_of_input_sellers += 1

    def make_retail_firms(self):  # make some firms retail firms, who sell goods to households
        # first convert all firms that do not have others firms as buyers of their output to retail firms
        if self.parameters.all_firms_retail == True:
            for firm in self.firms.values():
                firm.retail = True
            self.retail_firms_ID = copy.copy(self.firms_ID)
        elif self.parameters.all_firms_retail == 'file':
            self.retail_firms_ID = []
            with open('retailFirms') as retailIDS:
                for line in retailIDS:
                    ID = int(line)
                    self.retail_firms_ID.append(ID)
            for ID in self.retail_firms_ID:
                firm = self.firms[ID]
                firm.retail = True
        else:
            count = 0
            non_retail_firms_ID = []
            for firm in self.firms.itervalues():
                if len(firm.output_buyers_ID) == 0:  # if the firm has no firm buyers of its output, make it a retail firm
                    firm.retail = True
                    self.retail_firms_ID.append(firm.ID)
                    count += 1
                else:
                    non_retail_firms_ID.append(firm.ID)
            number_of_retail_firms = int(
                self.parameters.n * self.parameters.proportion_retail_firms)  # compute number of retails firms using the parameter that defines proportion of firms in the economy that are retail firms
            if number_of_retail_firms == 0:
                number_of_retail_firms = 1
            if count < number_of_retail_firms:  # if the number of retails firms created because they had no firm buyers of their input is less than number of retails firm as defined by the parameter, then create more retail firms
                more_retail_firms = number_of_retail_firms - count
                selected_firms_ID = random.sample(non_retail_firms_ID,
                                                  more_retail_firms)  # select some of the non-retail firms to become retails firms
                self.retail_firms_ID += selected_firms_ID

                for ID in selected_firms_ID:  # for each of the selected firms
                    self.firms[ID].retail = True  # make it a retail firm

        self.parameters.number_of_retail_firms_actual = len(self.retail_firms_ID)
        self.number_of_non_retail_firms = self.parameters.n - self.parameters.number_of_retail_firms_actual

        for id in self.retail_firms_ID:
            firm = self.firms[id]
            self.retail_firms.append(firm)

    def create_consumers_firms_network(self):  # create a network between households as consumers and retail firms as producers of consumption goods
        if self.parameters.representative_household and self.parameters.all_firms_retail == True:
            household_ID = self.households.values()[0].ID
            household = self.households.values()[0]
            for firm in self.firms.values():
                firm.consumers_ID.append(household_ID)
                household.goods_sellers_ID.append(firm.ID)
        elif self.parameters.representative_household and self.parameters.all_firms_retail != True:
            household_ID = self.households.values()[0].ID
            household = self.households.values()[0]

            for ID in self.retail_firms_ID:
                firm = self.firms[ID]
                firm.consumers_ID.append(household_ID)
                household.goods_sellers_ID.append(firm.ID)

        elif not self.parameters.representative_household and self.parameters.all_firms_retail != True:
            for household in self.households.values():  # for each household
                number_goods = random.randint(self.parameters.min_consumption_goods,
                                              self.parameters.max_consumption_goods)  # number of consumption goods of a household
                selected_retail_firms_ID = random.sample(self.retail_firms_ID, number_goods)  # select some retail firms
                for i in selected_retail_firms_ID:  # assign each of the selected retail firms the household ID and vice-versa
                    retail_firm = self.firms[i]
                    retail_firm.consumers_ID.append(household.ID)
                    household.goods_sellers_ID.append(retail_firm.ID)

            for j in self.retail_firms_ID:  # if any of the retail firms do not have a consumer, assign a random consumer
                retail_firm = self.firms[j]
                if len(retail_firm.consumers_ID) == 0:
                    household_selected = random.sample(self.households.values(), 1)[0]
                    retail_firm.consumers_ID.append(household_selected.ID)
                    household_selected.goods_sellers_ID.append(retail_firm.ID)

        elif not self.parameters.representative_household and self.parameters.all_firms_retail == True:
            for household in self.households.values():  # for each household
                number_goods = random.randint(self.parameters.min_consumption_goods,
                                              self.parameters.max_consumption_goods)  # number of consumption goods of a household

                selected_firms_ID = random.sample(self.firms_ID, number_goods)  # select some retail firms
                for i in selected_firms_ID:  # assign each of the selected retail firms the household ID and vice-versa
                    firm = self.firms[i]
                    firm.consumers_ID.append(household.ID)
                    household.goods_sellers_ID.append(firm.ID)

            for j in self.firms_ID: # if any of the firms does not have a consumer, assign a random consumer
                firm = self.firms[j]
                if len(firm.consumers_ID) == 0:
                    household_selected = random.sample(self.households.values(), 1)[0]
                    firm.consumers_ID.append(household_selected.ID)
                    household_selected.goods_sellers_ID.append(firm.ID)

    def create_workers_firms_network(self):  # create a network of workers and firms
        if self.parameters.representative_household:
            household_ID = self.households.values()[0].ID
            household = self.households.values()[0]
            for firm in self.firms.values():
                firm.number_of_workers = 1
                firm.workers_ID.append(household_ID)
                household.labor_buyers_ID.append(firm.ID)
        else:
            firms_number_of_workers = {}  # create a dictionary for the number of workers of each firm; the firms have workers in proportion to their in-degrees
            #  the distribution of workers among firms is similar to distribution of in-degree among firms
            factor = self.parameters.number_of_households / (self.parameters.d * self.parameters.n)
            for firm in self.firms.itervalues():
                firms_number_of_workers[firm.ID] = factor * firm.number_of_input_sellers
                if firm.number_of_input_sellers == 0:
                    firms_number_of_workers[firm.ID] = random.randint(1, self.parameters.max_workers_firms_without_input_sellers)
            for i in firms_number_of_workers:  # convert each element in the number of workers dictionary to an integer, some to higher int, some to lower int
                if random.uniform(0, 1) < 0.5:
                    firms_number_of_workers[i] = int(firms_number_of_workers[i])
                else:
                    firms_number_of_workers[i] = int(math.ceil(firms_number_of_workers[i]))
            for i in firms_number_of_workers:  # if a firm has no workers, assign it one worker
                if firms_number_of_workers[i] == 0:
                    firms_number_of_workers[i] = 1

            # use the dictionary of firms IDs and number of workers of each firm to assign worker to each firm
            # workers are assigned in such a way that each worker workers for one firm, a few workers may worker for two firms to match everything up
            count = 1
            for firm in self.firms.itervalues():
                number_of_workers = firms_number_of_workers[firm.ID]  # number of workers of the firm
                end_index = count + number_of_workers
                if end_index > self.parameters.number_of_households:
                    diff = end_index - self.parameters.number_of_households
                    workers_ID = range(count, self.parameters.number_of_households + 1) + range(1, diff + 1)
                    count = diff + 1
                else:
                    workers_ID = range(count, end_index)
                    count += number_of_workers
                workers_ID = [-ID for ID in workers_ID]  # IDs of workers
                firm.workers_ID = workers_ID  # assign the firm these workers
                for ID in workers_ID:  # assign each worker the ID of this firm
                    self.households[ID].labor_buyers_ID.append(firm.ID)
            for firm in self.firms.itervalues():
                firm.number_of_workers = len(firm.workers_ID)

    def fill_retail_suppliers_firms_ID(self):
        for ID in self.retail_firms_ID:
            retail_firm = self.firms[ID]
            self.retail_suppliers_firms_ID += retail_firm.input_sellers_ID
        self.retail_suppliers_firms = [self.firms[ID] for ID in self.retail_suppliers_firms_ID]
        self.retail_suppliers_firms += self.retail_firms


    def record_distributions(self,time_step):
        def record_size_distribution(time_step):
            if self.parameters.record_distribution_time_steps:
                distribution = []
                if time_step in self.parameters.record_distribution_time_steps:
                    for ID in self.firms_ID:
                        firm = self.firms[ID]
                        distribution.append(firm.wealth)
                    self.size_distribution.append(distribution)
            else:
                distribution = []
                for ID in self.firms_ID:
                    firm = self.firms[ID]
                    distribution.append(firm.wealth)
                self.size_distribution.append(distribution)


        def record_degree_distribution_in(time_step):
            if self.parameters.record_distribution_time_steps:
                if time_step in self.parameters.record_distribution_time_steps:
                    distribution = []
                    for ID in self.firms_ID:
                        firm = self.firms[ID]            
                        distribution.append(firm.number_of_input_sellers)
                    self.degree_distribution_in.append(distribution)
            else:
                distribution = []
                for ID in self.firms_ID:
                    firm = self.firms[ID]            
                    distribution.append(firm.number_of_input_sellers)
                self.degree_distribution_in.append(distribution)

        def record_degree_distribution_out(time_step):
            if self.parameters.record_distribution_time_steps:
                if time_step in self.parameters.record_distribution_time_steps:
                    distribution = []
                    for ID in self.firms_ID:
                        firm = self.firms[ID]            
                        distribution.append(firm.number_of_output_buyers)
                    self.degree_distribution_out.append(distribution)
            else:
                distribution = []
                for ID in self.firms_ID:
                    firm = self.firms[ID]            
                    distribution.append(firm.number_of_output_buyers)
                self.degree_distribution_out.append(distribution)

        record_degree_distribution_in(time_step)
        record_degree_distribution_out(time_step)
        record_size_distribution(time_step)


    def record_data(self, time_step, trasient):

        def record_inventory_prop():
            totalOutput = 0
            inventory = 0
            for firm in self.firms.itervalues():
                if firm.ID in self.firms_ID:
                    totalOutput += (firm.total_output * firm.ss_price)
                    inventory += (firm.last_inventory * firm.ss_price)
            prop = inventory/totalOutput
            self.inventory_prop.append(prop)


        def record_intermediate_share_currentOutput():
            totalOutput = 0
            intermediate_allocation = 0
            for firm in self.firms.itervalues():
                if firm.ID in self.firms_ID:
                    totalOutput += (firm.output * firm.ss_price)
                    intermediate_allocation += (sum(firm.output_allocation.values()) * firm.ss_price)
            share = intermediate_allocation / totalOutput
            self.intermediate_share_currentOutput.append(share)

        def record_intermediate_share():
            totalOutput = 0
            intermediate_allocation = 0
            for firm in self.firms.itervalues():
                if firm.ID in self.firms_ID:
                    totalOutput += (firm.total_output * firm.ss_price)
                    intermediate_allocation += (sum(firm.output_allocation.values()) * firm.ss_price)
            share = intermediate_allocation / totalOutput
            self.intermediate_share.append(share)

        def record_intermediate_share_firms():    
            share = []
            for firm in self.firms.itervalues():
                if firm.ID in self.firms_ID:
                    prop = sum(firm.output_allocation.values()) / firm.total_output
                    share.append(prop)
            self.intermediate_share_firms.append(share)


        def record_inverse_size_mean():
            list_inverses = []
            for ID in self.firms_ID:
                firm = self.firms[ID]
                s = 1/firm.wealth
                list_inverses.append(s)
            m = np.mean(list_inverses)
            self.inverse_size_mean.append(m)


        def record_priceLevel():
            if not trasient:
                #if time_step in self.parameters.record_distribution_time_steps:
                sum_price = 0
                count = 0
                for ID in self.firms_ID:
                    count += 1
                    p = self.firms[ID].price
                    sum_price += p
                mean_price = sum_price / count
                self.priceLevel.append(mean_price)

        def record_firm_volatility():
            if not trasient:
                for firm in self.firms.itervalues():
                    if firm.ID in self.firms_ID:
                        self.firm_sizes[firm.ID].append(firm.wealth) 
                    else:
                        self.firm_sizes[firm.ID].append(firm.wealth)

                if time_step in self.parameters.record_distribution_time_steps:
                    olds_sizes = copy.deepcopy(self.last_annual_firm_size)
                    self.last_annual_firm_size = {}
                    for ID in self.firm_sizes:
                        monthly_sizes = self.firm_sizes[ID]
                        annual_size = sum(monthly_sizes)
                        self.last_annual_firm_size[ID] = annual_size

                    for ID in self.firm_sizes:
                        self.firm_sizes[firm.ID] = []

                    if time_step not in self.parameters.record_distribution_time_steps[0:2]:
                        for ID in self.firms.iterkeys():
                            old_size = olds_sizes[ID]
                            new_size = self.last_annual_firm_size[ID]
                            change = (new_size-old_size) / old_size                            
                            self.firm_volatility[ID].append(change)


        def record_log_size_distribution():
            min_size = self.parameters.record_param['log_size_distribution']["min_size"]
            max_size = self.parameters.record_param['log_size_distribution']["max_size"]
            logV = self.parameters.record_param['log_size_distribution']["logV"]
            mul = self.parameters.record_param['log_size_distribution']["mul"]

            list_sizes = []
            for ID in self.firms_ID:
                firm = self.firms[ID]
                wealth = firm.wealth
                list_sizes.append(wealth)
            list_sizes = [i * mul for i in list_sizes]
            sizes = range(min_size,max_size+1)
            sizes_dict = collections.OrderedDict({})
            for s in sizes:
                sizes_dict[s] = 0
            for val in list_sizes:
                l = math.log(val,logV)      
                for s in sizes:
                    if l >= s:
                        sizes_dict[s] += 1
            dist = {}
            for s in sizes_dict:
                v  = sizes_dict[s]
                if v>0:
                    v = math.log(v,logV)
                    dist[s] = v
            self.log_size_distribution.append(dist)

        def record_log_deg_distribution():
            min_size = self.parameters.record_param['log_deg_distribution']["min_deg"]
            max_size = self.parameters.record_param['log_deg_distribution']["max_deg"]
            logV = self.parameters.record_param['log_deg_distribution']["logV"]
            
            list_sizes = []
            for ID in self.firms_ID:
                firm = self.firms[ID]
                #sellers = firm.number_of_input_sellers
                buyers = firm.number_of_output_buyers
                list_sizes.append(buyers)

            sizes = range(min_size,max_size+1)
            sizes_dict = collections.OrderedDict({})
            for s in sizes:
                sizes_dict[s] = 0
            for val in list_sizes:
                for s in sizes:
                    if val >= s:
                        sizes_dict[s] += 1

            dist = {}
            for s in sizes_dict:
                v  = sizes_dict[s]
                if v>0:
                    v = math.log(v,logV)                    
                    s = math.log(s,logV)
                    dist[s] = v
            self.log_deg_distribution.append(dist)


        def record_mean_firm_size_change():
            temp = []
            for ID in self.firms_ID:
                firm = self.firms[ID]
                change = firm.size_change
                temp.append(change)
            self.mean_firm_size_change.append(np.mean(temp))
            self.cv_firm_size_change.append(stats.variation(temp))

        def record_unemployment_rate():
            hh = self.households[-1]
            rate = hh.unemployed
            self.unemployment_rate.append(rate)

        def record_consumer_demand():
            if self.parameters.twoProductFirms:
                demand = 0
                for firm in self.firms.values():
                    demand += firm.consumer_demand
            else:
                demand = 0
                for firm in self.firms.values():
                    demand += firm.consumer_demand

            self.consumer_demand.append(demand)

        def record_intermediate_demand():
            demand = 0
            for firm in self.firms.values():
                temp = firm.total_demand -  firm.consumer_demand
                demand += temp
            self.intermediate_demand.append(demand)


        def record_CPIsansHousing():
            cpi = 0
            for ID in self.CPI_weights:
                if ID not in self.parameters.housingID:
                    firm = self.firms[ID]
                    weight = self.CPI_weights[ID]
                    p = firm.price * weight
                    cpi += p
            self.CPIsansHousing.append(cpi)

        def record_CPI():
            if self.parameters.twoProductFirms:
                cpi = 0
                for ID in self.CPI_weights:
                    firm = self.firms[ID]
                    weight = self.CPI_weights[ID]
                    p = firm.prices['consumer'] * weight
                    cpi += p
            else:
                cpi = 0
                for ID in self.CPI_weights:
                    firm = self.firms[ID]
                    weight = self.CPI_weights[ID]
                    p = firm.price * weight
                    cpi += p
            self.CPI.append(cpi)
        
        def record_weighted_consumption():
            hh = self.households[-1]
            self.weighted_consumption.append(hh.weighted_consumption)
        
        def record_finalOutput():            
            output = 0
            if self.parameters.twoProductFirms:
                for ID in self.PCE_weights:
                    firm = self.firms[ID]
                    weight = self.PCE_weights[ID]
                    out = firm.output['consumer'] * weight
                    output += out
                self.finalOutput.append(output)
            else: 
                temp = copy.deepcopy(self.pce_sectoral_outputs)
                for ID in self.firms_ID:
                    firm = self.firms[ID]
                    sec = self.firms_sectors[firm.ID]
                    if sec in self.PCE_weights:
                        temp[sec] += firm.output
                output = 0
                for sec in temp:
                    if sec in self.PCE_weights:
                        #print "yes in pce"
                        weight = self.PCE_weights[sec]
                    else:
                        weight = 0
                    val = temp[sec]
                    output += (weight * val)
                self.finalOutput.append(output)

                """               
                for ID in self.PCE_weights:
                    firm = self.firms[ID]
                    weight = self.PCE_weights[ID]
                    out = firm.output * weight                       
                    output += out
                #print output, "output"
                """
                
            
            
        def record_finalOutput_equilibrium_prices():
            """
            output = 0
            for ID in self.PCE_weights:
                firm = self.firms[ID]
                weight = self.PCE_weights[ID]
                out = firm.output
                ss_price = firm.ss_price
                v = out * ss_price * weight
                output += v            
            self.finalOutput_equilibrium_prices.append(output)
            """
            temp = copy.deepcopy(self.pce_sectoral_outputs)
            for ID in self.firms_ID:
                firm = self.firms[ID]
                sec = self.firms_sectors[firm.ID]
                if sec in self.PCE_weights:
                    temp[sec] += (firm.output * firm.ss_price)
            output = 0
            for sec in temp:
                if sec in self.PCE_weights:
                    weight = self.PCE_weights[sec]
                else:
                    weight = 0
                val = temp[sec]
                output += (weight * val)
            
            self.finalOutput_equilibrium_prices.append(output)

            


        def record_finalOutput_consumer_equilibrium_prices():
            """
             output = 0
             for ID in self.PCE_weights:
                firm = self.firms[ID]
                weight = self.PCE_weights[ID]
                out = sum(firm.output_consumption_allocation.values())
                ss_price = firm.ss_price
                v = out * ss_price * weight
                output += v
            """
            temp = copy.deepcopy(self.pce_sectoral_outputs)
            for ID in self.firms_ID:
                firm = self.firms[ID]
                sec = self.firms_sectors[firm.ID]
                if sec in self.PCE_weights:
                    out = sum(firm.output_consumption_allocation.values())
                    temp[sec] += (out * firm.ss_price)
            output = 0
            for sec in temp:
                if sec in self.PCE_weights:
                    weight = self.PCE_weights[sec]
                else:
                    weight = 0
                val = temp[sec]
                output += (weight * val)

            self.finalOutput_consumer_equilibrium_prices.append(output)
        
        def record_PCE():
            if self.parameters.twoProductFirms:
                pce = 0
                for ID in self.PCE_weights:
                    firm = self.firms[ID]
                    weight = self.PCE_weights[ID]
                    p = firm.prices['consumer'] * weight
                    pce += p
            else:
                pce = 0
                for ID in self.PCE_weights:
                    firm = self.firms[ID]
                    weight = self.PCE_weights[ID]
                    p = firm.price * weight
                    pce += p
                """
                temp = copy.deepcopy(self.pce_sectoral_outputs)
                for ID in self.firms_ID:
                    firm = self.firms_ID[ID]
                    sec = self.firms_sectors[firm]
                    if sec in self.PCE_weights:
                        price = sum(firm.output_consumption_allocation.values())
                        temp[sec] += (out * ss_price)
                for sec in temp:
                    weight = self.PCE_weights[sec]
                    val = temp[sec]
                    output += (weight * val)
                
                
                """

            self.PCE.append(pce)

        def record_wealthCarry():
            w = 0
            for firm in self.firms.itervalues():
                w += firm.past_reserve_wealth
            for household in self.households.itervalues():
                w += household.past_reserve_wealth
            self.wealthCarry.append(w)


        def record_inventoryCarry():
            inventory = 0
            for firm in self.firms.itervalues():
                inventory += firm.last_inventory
            self.inventoryCarry.append(inventory)

        def record_network_weights_changes():
            if self.parameters.production_function == 'CES' and self.parameters.weights_dynamic == True:
                if not trasient:
                    abs_diff = 0
                    for firm in self.firms.itervalues():
                        if firm.number_of_input_sellers > 1:
                            abs_diff += firm.ss_weights_change
                    abs_diff /= self.parameters.n
                    self.network_weights_changes.append(abs_diff)

        def record_sectoral_output():
            sectoral_output_form = copy.deepcopy(self.sectoral_output_form)
            for ID in self.firms_sectors:
                sec = self.firms_sectors[ID]
                firm = self.firms[int(ID)]
                output = firm.output
                sectoral_output_form[sec] += output
            self.sectoral_output.append(sectoral_output_form)

        def record_mean_price_change():
            price_change = 0
            for firm_ID in self.firms_ID:                
                firm = self.firms[firm_ID]
                price_change += abs(firm.price_change)
                """
                if abs(firm.price_change) > 0.0001:
                    print firm.number_of_input_sellers, "input_sellers", firm.number_of_output_buyers, "output buyers", firm.ID, "ID"
                """
            mean_price_change = price_change / self.parameters.n
            self.mean_price_change.append(mean_price_change)   
            #print len(self.out_of_market_firms_ID), "num firms out of market"
            
        def record_sumOutput():
            output = 0
            for firm_ID in self.firms_ID:
                firm = self.firms[firm_ID]
                out = firm.output
                output += out
            self.sumOutput.append(output)
        
        def record_gdp():
            gdp = 0
            for firm_ID in self.firms_ID:
                firm = self.firms[firm_ID]
                if firm.retail:
                    gdp += sum(firm.output_consumption_allocation.values()) * firm.price
            self.gdp.append(gdp)
        
        def record_gdp_equilibrium_prices():
            gdp = 0
            for firm_ID in self.firms_ID:
                firm = self.firms[firm_ID]
                if firm.retail:
                    finalOutput = sum(firm.output_consumption_allocation.values())
                    equi_value = finalOutput * firm.ss_price
                    gdp += equi_value
            #print gdp, "gdp"
            self.gdp_equilibrium_prices.append(gdp)

        def record_gdp_fixed_share_equilibrium_prices():
            gdp = 0            
            for firmID in self.firms_ID:
                share = self.GDP_shares[firmID]
                firm = self.firms[firmID]
                output = firm.output
                gdp += share * output
            self.gdp_fixed_share_equilibrium_prices.append(gdp)
        
        def record_gdp_fixed_shareSec_equilibrium_prices():
            sectors_contributions = {}
            secs = self.sector_hhShare.keys()
            for s in secs:
                sectors_contributions[s] = 0

            for firmID in self.firms_ID:
                firm = self.firms[firmID]
                #if firm.retail:
                sec = self.firms_sectors[firmID]
                if sec in sectors_contributions:
                    finalOutput = sum(firm.output_consumption_allocation.values())
                    equi_value = finalOutput * firm.ss_price
                    sectors_contributions[sec] += equi_value
                #else:
                 #   print firmID, "not retail"

            all_sec = 0

            for sec in sectors_contributions:
                weight = self.sector_hhShare[sec]
                val = sectors_contributions[sec]
                num = val * weight
                all_sec += num

            #print all_sec, "gdp fixed shares"

            self.gdp_fixed_shareSec_equilibrium_prices.append(all_sec)
        


        def record_sum_output_equilibrium_prices():
            gdp = 0
            for id in self.firms_ID:
                firm = self.firms[id]
                output = firm.output
                equi_value = output * firm.ss_price
                gdp += equi_value
            self.sum_output_equilibrium_prices.append(gdp)
                

        def record_cg():
            goods = 0
            for id in self.retail_firms_ID:
                retail_firm = self.firms[id]
                goods += sum(retail_firm.output_consumption_allocation.values())
            self.cg.append(goods)

        def record_pg():
            output = 0
            for firm in self.firms.itervalues():
                if not firm.retail:
                    output += firm.output
                else:
                    output += firm.output - sum(firm.output_consumption_allocation.values())
            self.pg.append(output)

        def record_cp():
            prices = []
            for ID in self.retail_firms_ID:
                retail_firm = self.firms[ID]
                prices.append(retail_firm.price)
            self.cp.append(sum(prices) / self.parameters.number_of_retail_firms_actual)


        def record_pp():
            if not self.parameters.all_firms_retail:
                prices = []
                for firm in self.firms.itervalues():
                    if not firm.retail:
                        prices.append(firm.price)
                if self.parameters.representative_household:
                    self.pp.append(sum(prices) / self.parameters.n)
                else:
                    self.pp.append(sum(prices) / self.number_of_non_retail_firms)

        def record_wealth():
            wealth = 0
            for firm in self.firms.itervalues():
                wealth += firm.wealth
            for household in self.households.values():
                wealth += household.wealth
            self.wealth.append(wealth)

        def record_cv(): # v is the coefficient of variation of the cross section of normalized prices
            if len(self.stable_prices_firms.keys()) != 0:
                all_prices_ratios = []
                if self.parameters.twoProductFirms:
                    for firm in self.firms.itervalues():
                        price_ratio = firm.prices['firms'] / self.stable_prices_firms[firm.ID]
                        all_prices_ratios.append(price_ratio)
                else:
                    for firm in self.firms.itervalues():
                        price_ratio = firm.price / self.stable_prices_firms[firm.ID]
                        all_prices_ratios.append(price_ratio)
                self.cv.append(stats.variation(all_prices_ratios))

        def record_V(): # coefficient of variation of the cross section of prices
            if self.parameters.twoProductFirms:
                prices = [firm.prices['firms'] for firm in self.firms.values()]
            else:
                prices = [firm.price for firm in self.firms.values()]
            self.V.append(stats.variation(prices))

        def record_price_distribution():
            self.firms_ID.sort()
            price_distribution = []
            if self.parameters.twoProductFirms:
                for ID in self.firms_ID:
                    firm = self.firms[ID]
                    price = firm.prices['firms']
                    price_distribution.append(price)
            else:
                for ID in self.firms_ID:
                    firm = self.firms[ID]
                    price = firm.price
                    price_distribution.append(price)
            self.price_distribution.append(price_distribution)


        def record_out_of_market_firms():
            self.out_of_market_firms.append(len(self.out_of_market_firms_ID))

        def record_eta():
            distance = 0
            for firm in self.firms.itervalues():
                distance += abs(firm.price_change)
            avg_distance = distance / self.parameters.n
            self.eta.append(avg_distance)

        def record_zeta():
            distance_decrease = 0
            for firm in self.firms.itervalues():
                if firm.price_change < 0:
                    distance_decrease += abs(firm.price_change)
            avg_distance_decrease = distance_decrease / self.parameters.n
            self.zeta.append(avg_distance_decrease)

        def record_etaC():
            if not trasient:
                distance = 0
                for firm in self.firms.itervalues():
                    distance += abs(firm.ss_price_change)
                avg_distance = distance / self.parameters.n
                self.etaC.append(avg_distance)

        def record_zetaC():
            if not trasient:
                distance = 0
                for firm in self.firms.itervalues():
                    if firm.price < firm.ss_price:
                        distance += abs(firm.ss_price_change)
                avg_distance = distance / self.parameters.n
                self.zetaC.append(avg_distance)

        def record_rho():
            count = 0
            for firm in self.firms.itervalues():
                if firm.price_change < -self.parameters.rho_omega_threshold:
                    count += 1
            prop = count / self.parameters.n
            self.rho.append(prop)

        def record_omega():
            count = 0
            for firm in self.firms.itervalues():
                if firm.price_change > self.parameters.rho_omega_threshold:
                    count += 1
            prop = count / self.parameters.n
            self.omega.append(prop)

        def record_rhoC():
            if not trasient:
                count = 0
                for firm in self.firms.itervalues():
                    if firm.ss_price_change < -self.parameters.rho_omega_threshold:
                        count += 1
                prop = count / self.parameters.n
                self.rhoC.append(prop)

        def record_omegaC():
            if not trasient:
                count = 0
                for firm in self.firms.itervalues():
                    if firm.ss_price_change > self.parameters.rho_omega_threshold:
                        count += 1
                prop = count / self.parameters.n
                self.omegaC.append(prop)

        def record_welfare_mean():
            count = 0
            for household in self.households.itervalues():
                count += household.utility
            mean_welfare = count / self.parameters.number_of_households
            self.welfare_mean.append(mean_welfare)
            #print mean_welfare, count, 'computing inside', self.parameters.number_of_households, 'self.parameters.number_of_households'
            #print 'representative_households', self.parameters.representative_household

        def record_welfare_cv():
            welfare_list = []
            for household in self.households.itervalues():
                welfare_list.append(household.utility)
            cv = stats.variation(welfare_list)
            self.welfare_cv.append(cv)

        for variable in self.parameters.record_variables['economy']:
            if self.parameters.record_variables['economy'][variable]:
                #print variable, "variable being recorded"
                function_name = 'record_' + variable
                if function_name in locals():
                    #print 'yes in local'
                    locals()[function_name]()
        if self.parameters.record_variables['economy']['degree_distribution_in']:
            self.record_distributions(time_step)

    def assign_initial_values(self):  # assign initial values of variables to households and firms
        for household in self.households.values():
            household.compute_utility_function_exponents()
            household.initial_values()
        for firm in self.firms.itervalues():
            firm.initial_values()

        #if self.parameters.files_directory:
         #   os.chdir(self.parameters.files_directory)

        if self.parameters.sizesFile:            
            with open("sizes.txt",'r') as file:
                for line in file:
                    line = line.split(",")
                    firmID = int(line[0])
                    if firmID > 0:
                        firm = self.firms[firmID]
                        si = float(line[1])
                        firm.wealth = si

        #if self.parameters.files_directory:
         #   os.chdir("..")


    def test_network_consistency(self):
        for firm in self.firms.itervalues():
            for buyer_ID in firm.output_buyers_ID:
                buyer = self.firms[buyer_ID]
                assert firm.ID in buyer.input_sellers_ID, ["buyer ID not in seller ID", "firm ID", firm.ID,
                                                           "buyers seller IDs", buyer.input_sellers_ID]
            for seller_ID in firm.input_sellers_ID:
                seller = self.firms[seller_ID]
                assert firm.ID in seller.output_buyers_ID, ["seller ID not in buyer ID", "firm ID", firm.ID,
                                                            "sellers buyer IDs", seller.input_sellers_ID]

        for firm in self.firms.itervalues():
            assert firm.number_of_input_sellers == len(firm.input_sellers_ID), [
                "number of input sellers not equal to len of input sellers ID list", "firm ID", firm.ID,
                "number of input sellers", firm.number_of_input_sellers, "list of input sellers ID",
                firm.input_sellers_ID]
            assert firm.number_of_output_buyers == len(firm.output_buyers_ID), [
                "number of output buyers not equal to len of output buyers ID list", "firm ID", firm.ID,
                "number of output buyers", firm.number_of_output_buyers, "list of output buyers ID",
                firm.output_buyers_ID]
            assert firm.number_of_workers == len(firm.workers_ID), [
                "number of workers not equal to len of workers ID list", "firm ID", firm.ID, "number of workers",
                firm.number_of_workers, "list of workers ID", firm.workers_ID]
            assert len(firm.output_consumer_demands.keys()) == len(firm.consumers_ID), [
                "consumer demand dictionary keys not equal to number of consumers ID", "firm ID", firm.ID,
                "firm.output_consumer_demands.keys()", firm.output_consumer_demands.keys(), "self.consumers_ID",
                firm.consumers_ID]
            assert set(firm.output_consumer_demands.keys()) == set(firm.consumers_ID), [
                "set of consumer demand dictionary keys not equal to number of consumers ID", "firm ID", firm.ID,
                "firm.output_consumer_demands.keys()", firm.output_consumer_demands.keys(), "self.consumers_ID",
                firm.consumers_ID]
            assert len(firm.output_consumption_allocation.keys()) == len(firm.consumers_ID), [
                "consumption allocation dictionary keys not equal to number of consumers ID", "firm ID", firm.ID,
                "firm.output_consumption_allocation.keys()", firm.output_consumption_allocation.keys(),
                "firm.consumers_ID", firm.consumers_ID]
            assert set(firm.output_consumption_allocation.keys()) == set(firm.consumers_ID), [
                "set of consumption allocation dictionary keys not equal to number of consumers ID", "firm ID", firm.ID,
                "firm.output_consumption_allocation.keys()", firm.output_consumption_allocation.keys(),
                "firm.consumers_ID", firm.consumers_ID]
            assert len(firm.input_demands.keys()) == firm.number_of_input_sellers, [
                "input sellers not equal to input demands keys", "firm ID", firm.ID, "self.input_demands.keys()",
                firm.input_demands.keys(), "firm.number_of_input_sellers", firm.number_of_input_sellers]
            assert len(firm.input_quantities.keys()) == firm.number_of_input_sellers, [
                "input sellers not equal to input quantities keys", "firm ID", firm.ID, "firms.input_quantities.keys()",
                firm.input_quantities.keys(), "firm.number_of_input_sellers", firm.number_of_input_sellers]
            assert len(firm.input_prices.keys()) == firm.number_of_input_sellers, [
                "input sellers not equal to input prices keys", "firm ID", firm.ID, "firms.input_prices.keys()",
                firm.input_prices.keys(), "firm.number_of_input_sellers", firm.number_of_input_sellers]
            assert len(firm.input_weights.keys()) == firm.number_of_input_sellers, [
                "input sellers not equal to input weights keys", "firm ID", firm.ID,
                "number_of_input_sellers", firm.number_of_input_sellers,
                "number of keys", len(firm.input_weights.keys()),
                "firms.input_weights.keys()", firm.input_weights.keys(),
                'input sellers IDS', firm.input_sellers()]
            assert len(firm.input_weights_optimal.keys()) == firm.number_of_input_sellers, [
                "input sellers not equal to input weights optimal keys", "firm ID", firm.ID,
                "firms.input_weights_optimal.keys()", firm.input_weights_optimal.keys(), "firm.number_of_input_sellers",
                firm.number_of_input_sellers]
            assert len(firm.output_demands.keys()) == firm.number_of_output_buyers, [
                "output buyers not equal to output demands keys", "firm ID", firm.ID, "firm.number_of_output_buyers",
                firm.number_of_output_buyers, "firm.output_demands.keys()", firm.output_demands.keys()]
            assert len(firm.output_allocation.keys()) == firm.number_of_output_buyers, [
                "output buyers not equal to output allocation keys", "firm ID", firm.ID, "firm.number_of_output_buyers",
                firm.number_of_output_buyers, "firm.output_allocation.keys()", firm.output_allocation.keys()]
            assert set(firm.output_allocation.keys()) == set(firm.output_demands.keys()), [
                "output allocation set not same as output demands set", "firm ID", firm.ID,
                "firm.output_allocation.keys()", firm.output_allocation.keys(), "firm.output_demands.keys()",
                firm.output_demands.keys()]
            if not firm.labor_input_only:
                assert len(firm.input_sellers_ID) > 0, ["zero input sellers", "firm ID", firm.ID, "labor only", firm.labor_input_only]
            if not firm.retail:
                assert len(firm.output_buyers_ID) > 0, ["zero output buyers", "firm ID", firm.ID,
                                                        "firm.output_buyers_ID", firm.output_buyers_ID]
            assert len(firm.workers_ID) > 0, ["zero workers", "firm ID", firm.ID]
            if not firm.labor_input_only:
                assert len(firm.input_demands.keys()) > 0, ["no input demands", "firm ID", firm.ID]
                assert len(firm.input_quantities.keys()) > 0, ["no input quantities", "firm ID", firm.ID]
                assert len(firm.input_prices.keys()) > 0, ["no input prices", "firm ID", firm.ID]
                assert len(firm.input_weights.keys()) > 0, ["no input weights", "firm ID", firm.ID]
                assert len(firm.input_weights_optimal.keys()) > 0, ["no input weights optimal", "firm ID", firm.ID]
            if firm.retail:
                assert len(firm.consumers_ID) > 0, ["zero consumers of retail firms", "firm ID", firm.ID]

        for firm in self.firms.itervalues():
            for worker_ID in firm.workers_ID:
                worker = self.households[worker_ID]
                assert firm.ID in worker.labor_buyers_ID, ["firm ID not in labor buyers ID", "firm ID", firm.ID,
                                                           "workers ID", worker.ID, "labor buyers ID",
                                                           worker.labor_buyers_ID]
            for consumer_ID in firm.consumers_ID:
                consumer = self.households[consumer_ID]
                assert firm.ID in consumer.goods_sellers_ID, ["firm ID not in goods sellers ID", "firm ID", firm.ID,
                                                              "consumer ID", consumer.ID, "goods sellers ID",
                                                              consumer.goods_sellers_ID]

        for household in self.households.itervalues():
            for firm_ID in household.goods_sellers_ID:
                firm = self.firms[firm_ID]
                assert household.ID in firm.output_consumer_demands.keys(), [
                    "household ID not in firm output consumer demands keys", "firm_ID", firm_ID, "household ID",
                    household.ID, "firm.output_consumer_demands.keys()", firm.output_consumer_demands.keys()]
                assert household.ID in firm.output_consumption_allocation.keys(), [
                    "household ID not in firm output consumption allocation keys", "firm_ID", firm_ID, "household ID",
                    household.ID, "firm.output_consumption_allocation.keys()",
                    firm.output_consumption_allocation.keys()]
                assert firm.ID in household.goods_prices.keys(), ["firm ID not in goods prices keys", "firm_ID",
                                                                  firm_ID, "household ID", household.ID,
                                                                  "household.goods_prices.keys()",
                                                                  household.goods_prices.keys()]
                assert firm.ID in household.goods_quantities.keys(), ["firm ID not in goods quantities keys", "firm_ID",
                                                                      firm_ID, "household ID", household.ID,
                                                                      "household.goods_prices.keys()",
                                                                      household.goods_prices.keys()]
            for firm_ID in household.labor_buyers_ID:
                firm = self.firms[firm_ID]
                assert household.ID in firm.workers_ID, ["household ID n ot in firm workers ID", "household.ID",
                                                         household.ID, "firm_ID", firm_ID, "firm.workers_ID",
                                                         firm.workers_ID]
            assert set(household.labor_demands.keys()) == set(household.labor_allocation.keys()), [
                "set of labor demand keys not equal to set of labor allocation keys", "ID", household.ID,
                "household.labor_demands.keys()", household.labor_demands.keys(), "household.labor_allocation.keys()",
                household.labor_allocation.keys()]
            assert set(household.labor_demands.keys()) == set(household.labor_buyers_ID), [
                "set of labor demand keys not same as set of labor buyers ID", "household.ID", household.ID,
                "household.labor_demands.keys()", household.labor_demands.keys(), "labor_buyers_ID", household.labor_buyers_ID]
            assert set(household.labor_allocation.keys()) == set(household.labor_buyers_ID), [
                "set of labor_allocation keys not same as set of labor buyers ID", "household.ID", household.ID,
                "household.labor_allocation.keys()", household.labor_allocation.keys(), "labor_buyers_ID",
                household.labor_buyers_ID]
            assert len(household.labor_demands.keys()) == len(household.labor_buyers_ID), [
                "length of labor demand keys not same as set of labor buyers ID", "household.ID", household.ID,
                "household.labor_demands.keys()", household.labor_demands.keys(), "labor_buyers_ID", household.labor_buyers_ID]
            assert len(household.labor_allocation.keys()) == len(household.labor_buyers_ID), [
                "length of labor_allocation keys not same as set of labor buyers ID", "household.ID", household.ID,
                "household.labor_allocation.keys()", household.labor_allocation.keys(), "labor_buyers_ID",
                household.labor_buyers_ID]
            assert set(household.goods_prices.keys()) == set(household.goods_quantities.keys()), [
                "goods prices keys not same as goods quantities keys", "household.ID", household.ID,
                "household.goods_prices.keys()", household.goods_prices.keys(), "household.goods_quantities.keys()",
                household.goods_quantities.keys()]
            if not self.parameters.household_preference_homogeneous:
                assert set(household.goods_demand.keys()) == set(household.goods_prices.keys()), [
                    "goods demand set not same as goods prices set", "household.ID", household.ID,
                    "household.goods_demand.keys()", household.goods_demand.keys(), "household.goods_prices.keys()",
                    household.goods_prices.keys()]
            assert len(household.goods_sellers_ID) > 0, ["zero goods sellers", "household.ID", household.ID]
            assert len(household.labor_buyers_ID) > 0, ["zero labor buyers", "household.ID", household.ID]
            assert len(household.labor_demands.keys()) > 0, ["zero labor demand keys", "household.ID", household.ID]
            assert len(household.labor_allocation.keys()) > 0, ["zero labor allocation keys", "household.ID",
                                                                household.ID]
            assert len(household.goods_prices.keys()) > 0, ["zero goods prices keys", "household.ID", household.ID]
            assert len(household.goods_quantities.keys()) > 0, ["zero goods quantities keys", "household.ID",
                                                                household.ID]

    def non_input_seller(self, firm):
        ID = random.choice(self.firms_ID)
        if firm.is_input_seller(ID):
            return self.non_input_seller(firm)
        else:
            return self.firms[ID]

    def firm_exit_process(self, firm):
        if firm.number_of_output_buyers == 0:            
            input_sellers_ID = copy.deepcopy(firm.input_sellers())
            retail = copy.copy(firm.retail)
            firm.exit(copy.copy(firm.ID), self.parameters, retail)
            self.firms_ID.remove(firm.ID)
            if retail:
                self.retail_firms_ID.remove(firm.ID)
            self.out_of_market_firms_ID.append(firm.ID)
            self.households[-1].labor_buyers_ID.remove(firm.ID)

            for ID in input_sellers_ID:
                seller = self.firms[ID]
                seller.remove_output_buyer(firm.ID)
                self.firm_exit_process(seller)

    def firms_change_input_seller(self):
        changes = 0
        #self.test_firm_network_consistency()
        #self.test_repeats()
        for ID in self.firms_ID:
            #print ID
            #print self.firms.keys()
            #print self.out_of_market_firms_ID
            firm = self.firms[ID]
            if firm.attempt_input_seller_change():
                new_seller = self.non_input_seller(firm)
                new_seller_price = new_seller.return_price()
                Boolean_oldID = firm.change_input_seller(new_seller.ID, new_seller_price)
                change = Boolean_oldID[0]
                old_input_seller_ID = Boolean_oldID[1]
                if change:
                    changes += 1
                    old_seller = self.firms[old_input_seller_ID]
                    old_seller.remove_output_buyer(firm.ID)
                    new_seller.add_output_buyer(firm.ID)                    
                    self.firm_exit_process(old_seller)


        #self.test_firm_network_consistency()
        #self.test_repeats()
        self.links_changed.append(changes)

    def firms_entry(self):
        #print "out of market ID", self.out_of_market_firms_ID
        out_of_market_firms_ID = copy.copy(self.out_of_market_firms_ID)
        for ID in out_of_market_firms_ID: # for ID of each firm that is out of the market
            #print "ID entry test", ID
            firm = self.firms[ID] # pick the firm
            #entry_firms = [] # this list should Not be inside the loop!!
            if firm.will_enter():
                #entry_firms.append(firm.ID)
                self.out_of_market_firms_ID.remove(ID)
                self.firms_ID.append(ID)
                if firm.retail:
                    self.retail_firms_ID.append(firm.ID)
                    firm.retail = True

                

                number_of_input_sellers = np.random.binomial(self.parameters.n_binomial,
                                                             self.parameters.p_binomial)

                

                number_of_input_sellers = min(len(self.firms_ID)-1,number_of_input_sellers)
                if self.parameters.connect_firms_without_sellers:
                    number_of_input_sellers = max(1,number_of_input_sellers)
                
                if number_of_input_sellers == 0:
                    firm.labor_input_only = True
                


                number_of_output_buyers = np.random.binomial(self.parameters.n_binomial,
                                                             self.parameters.p_binomial)

                number_of_output_buyers = min(len(self.firms_ID)-1,number_of_output_buyers)


                popu = copy.copy(self.firms_ID)
                popu.remove(firm.ID)

                """
                input_sellers_ID = random.sample(self.firms_ID, number_of_input_sellers)
                output_buyers_ID = random.sample(self.firms_ID, number_of_output_buyers)
                """

                input_sellers_ID = random.sample(popu, number_of_input_sellers)
                output_buyers_ID = random.sample(popu, number_of_output_buyers)


                if firm.ID in input_sellers_ID:
                    input_sellers_ID.remove(firm.ID)

                if firm.ID in output_buyers_ID:
                    output_buyers_ID.remove(firm.ID)

                input_sellers_ID_copy = copy.deepcopy(input_sellers_ID)
                for seller_ID in input_sellers_ID_copy:
                    seller = self.firms[seller_ID]
                    seller_price = seller.return_price()
                    firm.add_input_seller(seller_ID, seller_price)
                    seller.add_output_buyer(ID)

                output_buyers_ID_copy = copy.deepcopy(output_buyers_ID)
                for buyer_ID in output_buyers_ID_copy:
                    buyer = self.firms[buyer_ID]
                    firm_price = firm.return_price()
                    assert firm.ID not in buyer.input_sellers_ID,  "firm already among buyer though it was outside the market"
                    buyer.add_input_seller(firm.ID, firm_price)
                    assert buyer_ID not in firm.output_buyers_ID, "buyer already in firm output buyers thought firm is outside the market"
                    firm.add_output_buyer(buyer_ID)

                    buyer.labor_input_only = False

                """ the code below needs to be changed later to do it parameterically """
                household_ID = self.households.values()[0].ID
                household = self.households.values()[0]
                firm.number_of_workers = 1
                firm.workers_ID.append(household_ID)
                household.labor_buyers_ID.append(firm.ID)
                firm.consumers_ID.append(household_ID)
                firm.initial_values()

                if firm.labor_input_only:
                    if len(firm.input_sellers_ID) > 0:
                        print "LOOK mismatch here when entry occurs"

    def test_firms_without_input_sellers(self):
        without_sellers = []
        for ID in self.firms_ID:
            firm = self.firms[ID]
            if len(firm.input_sellers_ID) == 0:
                without_sellers.append(firm.ID)
        print "firms without sellers in the market", without_sellers

    def test_firm_network_consistency(self):
        for firm_ID in self.firms_ID:
            firm = self.firms[firm_ID]
            for buyer_ID in firm.output_buyers_ID:
                buyer = self.firms[buyer_ID]
                if firm.ID not in buyer.input_sellers_ID:
                    print "firm ID", firm.ID
                    print "firm buyers", firm.output_buyers_ID
                    print "buyer ID", buyer_ID
                    print "buyer input sellers", buyer.input_sellers_ID

    def test_repeats(self):
        for firm_ID in self.firms_ID:
            firm = self.firms[firm_ID]
            if len(set(firm.input_sellers_ID)) !=  len(firm.input_sellers_ID):
                print "repeats in input sellers ID"
                print "firm ID", firm.ID
                print "input sellers ID", firm.input_sellers_ID

            if len(set(firm.output_buyers_ID)) != len(firm.output_buyers_ID):
                print "repeats in output buyers ID"
                print "firm ID", firm.ID
                print "output buyers ID", firm.output_buyers_ID
