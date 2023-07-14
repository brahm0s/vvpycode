"""
Contact - vipin.veetil@gmail.com
Language - Python 2.7.10
"""

from __future__ import division
import functions as f
import random
from datetime import datetime
random.seed(datetime.now())
import copy
import gc
import numpy as np
import math
import collections
#from array import array
gc.enable()


class Firm(object):
    __slots__ = ('parameters', 'ID', 'retail', 'productivity', 'input_sellers_ID', 'output_buyers_ID', 'labor_input_only',
                 'probability_price_change', 'Cobb_Douglas_exponents',
                 'probability_weights_change', 'random_prob', 'number_of_workers', 'number_of_input_sellers',
                 'number_of_output_buyers', 'workers_ID', 'price_change', 'price_change_time_series', 'price_time_series',
                 'consumers_ID', 'wealth', 'reserve_wealth', 'past_reserve_wealth','past_wealth', 'output', 'inventory', 'price', 'market_clearing_price', 'input_demands',
                 'input_quantities', 'input_prices', 'input_weights', 'input_weights_optimal', 'labor_demand',
                 'labor_demand_per_worker', 'labor_quantity', 'labor_weight', 'wage', 'market_clearing_wage',
                 'output_demands', 'output_allocation', 'output_consumer_demands', 'output_consumption_allocation',
                 'CES_output_scaling', 'price_time_series', 'output_time_series', 'wealth_time_series',
                 'output_quantity_change', 'output_quantity_change_time_series', 'ss_price', 'ss_output',
                 'ss_price_change', 'ss_wealth' , 'ss_output_allocation' ,'ss_output_change', 'ss_price_change_time_series', 'ss_output_change_time_series',
                 'ss_weights', 'ss_weights_change', 'rP', 'rW','last_price', 'total_demand', 'total_output',
                 'last_inventory', 'CD_labor', 'total_demand_time_series','total_output_time_series', 'selfLabor',
                 'seeds', 'lastLaborDemand','output_allocation_ratio','output_division', 'prices','consumer_demand', 'CobbDouglas_output_scaling','ss_input_demands','prop_lockdown',
                 'lockdown_money_balance','lockdown_inventory','lockdown_output_buyers_prop','size_change','B_vals_CES','output_sold','output_sold_time_series','ss_intermediate_prop')

    def __init__(self, i, parameters):
        self.parameters = parameters        
        if self.parameters.twoProductFirms:
            self.output_allocation_ratio = None  # the ratio of allocation of output between final consumer and other firms when it produces two variants of the good
            self.output_division =  {'consumer':None,'firms':None}
            self.prices = {'consumer':None,'firms':None}
            self.consumer_demand = None
        self.Cobb_Douglas_exponents = {}
        self.ID = i  # each firm is assigned a unique ID when the economy is created
        self.retail = False
        if self.parameters.firm_productivity_homogeneous:
            self.productivity = 1
        else:
            self.productivity = np.random.lognormal(self.parameters.productivity_shock_mean,self.parameters.productivity_shock_std)

        self.input_sellers_ID = []  # IDs of other firms who supply inputs
        self.output_buyers_ID = []  # IDs of other firms who buy output
        self.labor_input_only = False
        self.lastLaborDemand = random.uniform(0,1)
        self.seeds = 1
        self.probability_price_change = random.random()
        self.probability_weights_change = random.random()
        self.random_prob = random.random()
        self.number_of_workers = 0
        self.number_of_input_sellers = 0
        self.number_of_output_buyers = 0
        self.workers_ID = []
        self.consumers_ID = [-1]
        self.wealth = random.random()  # present wealth of the firm
        self.past_wealth = random.random()
        self.reserve_wealth = 0
        self.past_reserve_wealth = 0
        self.output = random.random()  # present output of the firm
        self.inventory = random.random()  # the inventory of goods carried from past time steps
        self.last_inventory = random.random()
        self.price = random.random()  # present price of the firm's output
        self.price_time_series = [self.price]
        self.price_change = random.random() # the change in price compared to last period
        self.last_price = random.random() # price at the last time step
        self.output_quantity_change = random.random()
        self.market_clearing_price = random.random()
        self.input_demands = {}  # demand for inputs from other firms
        self.input_quantities = {}  # quantity of inputs recieved from other firms
        self.input_prices = {}  # prices of different inputs from other firms
        self.input_weights = {}  # proportion of wealth spend on different inputs from other firms
        self.input_weights_optimal = {}
        self.labor_demand = random.random()  # demand for labor
        self.labor_demand_per_worker = random.random()
        self.labor_quantity = random.random()  # quantity of labor recieved from household
        self.labor_weight = 0  # proportion of wealth spend on labor input
        self.wage = random.random()
        self.output_demands = {}  # demand for the firm's output from other firms
        self.output_allocation = {}  # the allocation of the firm's output among other firms
        self.output_consumer_demands = {}  # demand for the firm's output by the household
        self.output_consumption_allocation = {}  # output sold as consumption good to household
        self.CES_output_scaling = 1
        self.CobbDouglas_output_scaling = 1
        self.ss_output_allocation = None
        self.ss_wealth = None
        self.ss_price = 1
        self.ss_output = None
        self.ss_weights = None
        self.ss_weights_change = None
        self.ss_input_demands = None
        self.rP = random.uniform(self.parameters.psi - self.parameters.gamma, self.parameters.psi + self.parameters.gamma) # smoothening paramter of tthe ifmr
        self.rW = random.uniform(self.parameters.psi - self.parameters.gamma, self.parameters.psi + self.parameters.gamma)  # smoothening paramter of tthe ifmr
        self.total_demand = random.random()
        self.total_demand_time_series = [self.total_demand]
        self.total_output = random.random()
        self.total_output_time_series = [self.total_output]
        self.CD_labor = None
        self.selfLabor = None
        self.prop_lockdown = None
        self.lockdown_money_balance = 0
        self.lockdown_inventory = 0
        self.lockdown_output_buyers_prop = None
        self.B_vals_CES = {}
        self.output_sold = 0
        self.output_sold_time_series = []
        self.ss_intermediate_prop = 0
        if parameters.data_time_series:
            for variable in parameters.record_variables['firm']:
                if parameters.record_variables['firm'][variable]:
                    variable += '_time_series'
                    setattr(self, variable, [])

    def Cobb_Douglas(self):
        if not self.parameters.weightsFile:
            self.CD_labor = self.parameters.alpha

            if self.parameters.production_function == 'CD' or self.parameters.production_function == 'substitutes':
                non_labor_share = 1 - self.CD_labor
                if self.number_of_input_sellers > 0:
                    if self.parameters.Cobb_Douglas_production_function_homogeneous:
                        exponent = non_labor_share / self.number_of_input_sellers
                        exponents = [exponent for i in xrange(self.number_of_input_sellers)]
                    else:
                        proportions = f.normalized_random_numbers(self.number_of_input_sellers)
                        exponents = [i * non_labor_share for i in proportions]
                    self.Cobb_Douglas_exponents = dict((id, None) for id in self.input_sellers_ID)
                    for i in xrange(self.number_of_input_sellers):
                        seller_ID = self.input_sellers_ID[i]
                        exponent = exponents[i]
                        self.Cobb_Douglas_exponents[seller_ID] = exponent
                    
    def set_CobbDouglas_scaling(self):
        if self.number_of_input_sellers > 1:
            self.CobbDouglas_output_scaling = 1/(self.number_of_input_sellers ** (1-self.CD_labor))
            #print self.CobbDouglas_output_scaling, self.number_of_input_sellers
    
    def set_CobbDouglas_exponents_to_one(self):
        for ID in self.Cobb_Douglas_exponents:
            self.Cobb_Douglas_exponents[ID] = 1
        self.CD_labor = 1

    def productivity_growth(self):
        s = np.random.normal(self.parameters.productivity_mu, self.parameters.productivity_sigma)
        s = abs(s)
        self.productivity += s
    
    def weights(self):
        if self.number_of_input_sellers == 0:  # if the firm does not recieve inputs from other firms
            self.labor_input_only = True  # set labor_input_only to be true
            self.labor_weight = 1  # assign it a labor weight of 1
        elif self.number_of_input_sellers == 1:
            self.labor_weight = self.CD_labor
            self.input_weights[self.input_sellers_ID[0]] = 1 - self.CD_labor
            self.input_weights_optimal[self.input_sellers_ID[0]] = 1 - self.CD_labor
            self.input_demands[self.input_sellers_ID[0]] = random.random()
            self.input_quantities[self.input_sellers_ID[0]] = random.random()
            self.input_prices[self.input_sellers_ID[0]] = random.random()
        else:
            self.labor_weight = copy.copy(self.CD_labor)
            if self.parameters.production_function == 'CD':
                for id in self.input_sellers_ID:
                    self.input_weights[id] = self.Cobb_Douglas_exponents[id]
                    self.input_weights_optimal[id] = self.Cobb_Douglas_exponents[id]
                    
            elif self.parameters.production_function == 'substitutes':
                for id in self.input_sellers_ID:
                    self.input_weights[id] = self.Cobb_Douglas_exponents[id]
                    self.input_weights_optimal[id] = self.Cobb_Douglas_exponents[id]
                
            elif self.parameters.production_function == 'CES':
                weights = f.normalized_random_numbers(self.number_of_input_sellers)
                weights_optimal = f.normalized_random_numbers(self.number_of_input_sellers)
                non_labor_share = (1 - self.CD_labor)
                weights = [w * non_labor_share for w in weights]
                weights_optimal = [w * non_labor_share for w in weights_optimal]
                count = 0
                for ID in self.input_sellers_ID:  # assign other weights to other firms
                    self.input_weights[ID] = weights[count]
                    self.input_weights_optimal[ID] = weights_optimal[count]
                    count += 1
                    
    def initial_values(self):  # assign initial values to the firm
        if not self.parameters.weightsFile:
            self.CD_labor = self.CD_labor

        def probability_change():
            if self.parameters.var_prob_price_change:
                self.probability_price_change = random.uniform(self.parameters.probability_price_change-self.parameters.var_prob_price_change,
                    self.parameters.probability_price_change+self.parameters.var_prob_price_change)
            else:
                self.probability_price_change = self.parameters.probability_price_change
            
            self.probability_weights_change = self.parameters.probability_weights_change

        def output():
            for ID in self.output_buyers_ID:  # for each buyer of output
                self.output_demands[ID] = random.random()  # assign random initial demand
                self.output_allocation[ID] = random.random()  # assign random initial output allocation
                #self.output_allocation[ID] = 1  # assign random initial output allocation
                self.output = 1

        def retail():
            if self.retail:  # if firm is a retail firm, assing random household demand and consumption good allocation
                for ID in self.consumers_ID:
                    self.output_consumer_demands[ID] = random.random()
                    self.output_consumption_allocation[ID] = random.random()

        def inputs():
            for ID in self.input_sellers_ID:
                self.input_demands[ID] = random.random()  # assign random initial demand for input sellers
                self.input_quantities[ID] = random.random()  # assign random initial quantity of input
                #self.input_quantities[ID] = 1  # assign random initial quantity of input
                self.input_prices[ID] = random.random()  # assign random initial price of input

        def scaling():
            if self.number_of_input_sellers > 0:
                if self.parameters.CES_output_scaled:
                    if self.parameters.sigma > 0:
                        self.CES_output_scaling = ((1 - self.parameters.sigma) / self.parameters.sigma) * self.number_of_input_sellers ** (
                    (1 - self.parameters.sigma) / self.parameters.sigma)
                    else:
                        self.CES_output_scaling = 1
            else:
                self.CES_output_scaling = 1
           

        probability_change()
        output()
        retail()
        self.Cobb_Douglas()
        if not self.parameters.weightsFile:
            self.weights()
        inputs()
        scaling()

        if self.labor_input_only:
            if len(self.input_sellers_ID) > 0:
                print self.ID, 'Initializing: labor input only but has input sellers'

    def productivity_shock(self):  # productivity shock
        #self.productivity = self.parameters.random.lognormal(self.parameters.productivity_shock_mean,
                                                        #    self.parameters.productivity_shock_std)
        #val = np.random.lognormal(self.parameters.productivity_shock_mean, self.parameters.productivity_shock_std)
        val = np.random.normal(self.parameters.productivity_shock_mean, self.parameters.productivity_shock_std)
        if self.parameters.productivity_multiplicative:
            self.productivity *= val
        else:
            self.productivity = math.e ** val
        #self.productivity = 1
        #sself.productivity = np.random.exponential(self.parameters.productivity_exponential)
    
    def produce_labor_productivity(self):  # produce output
        past_output = copy.copy(self.output)

        if self.labor_input_only:
            self.output = (self.productivity ** self.parameters.alpha) * self.labor_quantity
        else:
            if self.selfLabor:
                self.labor_quantity = 1
                    
            if self.parameters.production_function == 'CES':
                CES_output = f.CES(self.input_quantities.values(),
                                   self.parameters.sigma)  # a part of the production function in which non labor quantities are combined
                #print self.CES_output_scaling, 'self.CES_output_scaling'
                CES_output_scaled = CES_output / self.CES_output_scaling
                self.output = (self.productivity ** self.parameters.alpha) * f.Cobb_Douglas_simple(self.labor_quantity, CES_output_scaled,
                                                                        self.CD_labor)  # the Cobb-Douglas function uses quantity of labor and CES output of non-labor inputs
            elif self.parameters.production_function == 'CD':
                labor_output = self.labor_quantity ** self.CD_labor
                intermediate_output = 1
                for id in self.input_sellers_ID:
                    intermediate_output *= (self.input_quantities[id] ** self.Cobb_Douglas_exponents[id])
                    
                if self.parameters.seedSaving:
                    seeds = copy.copy(self.seeds)
                    exponent = self.parameters.seedExponent
                    output = self.CobbDouglas_output_scaling * (self.productivity ** self.parameters.alpha) * (seeds ** exponent + (labor_output * intermediate_output) ** (1-exponent)) ** (1/exponent)
                    self.seeds = output * self.parameters.seedSaving
                    self.output = output * (1-self.parameters.seedSaving)
                    if self.seeds == 0:
                        print "zero seed"
                        print self.ID, "ID", intermediate_output, "intermediate output"
                else:
                    self.output = self.CobbDouglas_output_scaling * (self.productivity ** self.parameters.alpha) * labor_output * intermediate_output                    
                    #self.output = self.CobbDouglas_output_scaling * (self.productivity ** 0.66) * labor_output * intermediate_output                    
            elif self.parameters.production_function == 'substitutes':
                intermediate_output = 0
                for id in self.input_sellers_ID:
                    intermediate_output += (self.input_quantities[id]) #** self.Cobb_Douglas_exponents[id]
                
                labor_output =  self.labor_quantity #** self.CD_labor
                self.output = (self.productivity ** self.parameters.alpha) * (labor_output + intermediate_output) #/ self.number_of_input_sellers

                

        self.output_quantity_change = (self.output - past_output) / past_output
        if self.ss_output is not None:
            self.ss_output_change = (self.output - self.ss_output) / self.ss_output    
    
    def produce(self):  # produce output
        past_output = copy.copy(self.output)

        if self.labor_input_only:
            self.output = self.productivity * self.labor_quantity
        else:
            if self.selfLabor:
                self.labor_quantity = 1

            if self.parameters.production_function == 'CES_withB_vals':
                CES_output = f.CES_withB_vals(self.input_quantities,self.B_vals_CES,self.parameters.sigma)
                self.output = self.productivity * f.Cobb_Douglas_simple(self.labor_quantity, CES_output, self.CD_labor)
                #print self.ID, self.input_weights

            if self.parameters.production_function == 'CES':
                CES_output = f.CES(self.input_quantities.values(),
                                   self.parameters.sigma)  # a part of the production function in which non labor quantities are combined
                #print self.CES_output_scaling, 'self.CES_output_scaling'
                CES_output_scaled = CES_output / self.CES_output_scaling
                self.output = self.productivity * f.Cobb_Douglas_simple(self.labor_quantity, CES_output_scaled,
                                                                        self.CD_labor)  # the Cobb-Douglas function uses quantity of labor and CES output of non-labor inputs
            elif self.parameters.production_function == 'CD':
                labor_output = self.labor_quantity ** self.CD_labor
                intermediate_output = 1
                for id in self.input_sellers_ID:
                    intermediate_output *= (self.input_quantities[id] ** self.Cobb_Douglas_exponents[id])
                    
                if self.parameters.seedSaving:
                    seeds = copy.copy(self.seeds)
                    exponent = self.parameters.seedExponent
                    output = self.CobbDouglas_output_scaling * self.productivity * (seeds ** exponent + (labor_output * intermediate_output) ** (1-exponent)) ** (1/exponent)
                    self.seeds = output * self.parameters.seedSaving
                    self.output = output * (1-self.parameters.seedSaving)
                    if self.seeds == 0:
                        print "zero seed"
                        print self.ID, "ID", intermediate_output, "intermediate output"
                else:
                    self.output = self.CobbDouglas_output_scaling * self.productivity * labor_output * intermediate_output
                    #print self.ID, self.output
                    if self.output == 0:
                        print "zero output", self.output                         

            elif self.parameters.production_function == 'substitutes':
                intermediate_output = 0
                for id in self.input_sellers_ID:
                    intermediate_output += (self.input_quantities[id]) #** self.Cobb_Douglas_exponents[id]
                
                labor_output =  self.labor_quantity #** self.CD_labor
                self.output = self.productivity * (labor_output + intermediate_output) #/ self.number_of_input_sellers

                

        self.output_quantity_change = (self.output - past_output) / past_output
        if self.ss_output is not None:
            self.ss_output_change = (self.output - self.ss_output) / self.ss_output
            
    def sum_OutputInventory(self,lockdown):
        self.total_output = self.output + self.inventory   
        """
        if self.total_output == 0:
            print "zero total output", self.output     
            print self.ID
            print self.input_sellers_ID, "input sellers ids"
            print self.input_quantities, "input quantity"
            print self.labor_quantity, "labor quantity"
            print self.workers_ID, "workers ID"
            print self.retail, "retail or not"
        """
        #if lockdown:
            #print self.lockdown_inventory
         #   self.total_output += (self.lockdown_inventory * self.parameters.lockdown_inventory_share)
        #self.total_output += self.inventory
        #if self.inventory >0:
         #   print "positive inventory"
        self.last_inventory = copy.copy(self.inventory)
        #self.lockdown_inventory *= (1-self.parameters.lockdown_inventory_share)
        self.inventory = 0      

    def set_lockdown_price(self):  
        old_price = copy.copy(self.price)
        self.last_price = old_price   
        self.price = copy.copy(self.ss_price)
        self.price_change = (self.price - old_price) / old_price
        self.ss_price_change = (self.price - self.ss_price) / self.ss_price
   
    def compute_probability_price_change(self):
        difference = abs(self.price - self.market_clearing_price) / self.price
        prob_component = difference / (1 + difference)

        #self.probability_price_change = self.parameters.price_sensitivity * self.parameters.probability_price_change + (1 - self.parameters.price_sensitivity) * prob_component

        #self.probability_price_change = self.parameters.price_sensitivity * 
        #n = self.parameters.price_sensitivity * difference
        #self.probability_price_change = n / (1 + n)

        self.probability_price_change = prob_component ** self.parameters.price_sensitivity

    def compute_market_clearing_price(self):
        producer_demand = sum(self.output_demands.values())
        self.total_demand = producer_demand
        if self.retail:
            consumer_demand = sum(self.output_consumer_demands.values())
            self.total_demand += consumer_demand
            self.consumer_demand = consumer_demand
        #print self.total_output, "output", self.ID, "ID", self.seeds, "seeds"
        #print 'output', self.total_output, 'ID', self.ID
        self.market_clearing_price = self.total_demand / self.total_output  # price equals total demand divided output
        #print "self.market_clearing_price", self.market_clearing_price, self.ID

        """
        if self.market_clearing_price < 0:
            print "ID", self.ID, self.market_clearing_price
            print "demand", self.total_demand, "consumer demand", consumer_demand
            print "output", self.output
            print "total output", self.total_output
        """
    
    def generate_random_prob(self):
        self.random_prob = random.random()

    def set_price_transient(self):
        old_price = copy.copy(self.price)
        self.last_price = old_price
        self.price = self.market_clearing_price
        self.price_change = (self.price - old_price) / old_price                        
        
    def set_price_probabilistic_stickiness(self):
        old_price = copy.copy(self.price)
        self.last_price = old_price
        if random.random() < self.probability_price_change:
            self.price = self.market_clearing_price

        self.price_change = (self.price - old_price) / old_price
        self.ss_price_change = (self.price - self.ss_price) / self.ss_price
            
    def set_price_flexible(self):
         old_price = copy.copy(self.price)
         self.last_price = old_price
         self.price = self.market_clearing_price
         self.price_change = (self.price - old_price) / old_price
         self.ss_price_change = (self.price - self.ss_price) / self.ss_price
    
    def set_price_linear_stickiness(self):
        old_price = copy.copy(self.price)
        self.last_price = old_price
        self.price = old_price * self.parameters.linear_price_stickiness_old_share + self.market_clearing_price * (1-self.parameters.linear_price_stickiness_old_share)
        self.price_change = (self.price - old_price) / old_price
        self.ss_price_change = (self.price - self.ss_price) / self.ss_price

    def smoothen_price(self):
        if self.price < self.last_price:
            priceRatio = self.price / self.last_price
            d = self.rP * (1 - priceRatio)
            self.total_output = (1-d) * self.total_output
            self.inventory = d * self.total_output
            self.price = (1 + d) * self.price
            self.price_change = (self.price - self.last_price) / self.last_price

    def set_price_sync(self, other_randoms):  # compute price
        if other_randoms == False:
            prob_random = self.random_prob
        else:
            other_randoms.append(self.random_prob)
            if self.parameters.sync_direction == 'sellers':
                prob = sum(other_randoms) / (self.number_of_input_sellers + 1)
            elif self.parameters.sync_direction == 'buyers':
                prob = sum(other_randoms) / (self.number_of_output_buyers + 1)
            prob_random = self.parameters.sync_sensitivity * prob + (1 - self.parameters.sync_sensitivity) * self.random_prob
        if prob_random < self.probability_price_change:
            if not self.parameters.twoProductFirms:
                old_price = copy.copy(self.price)
                self.price = self.market_clearing_price
                self.price_change = (self.price - old_price) / old_price
                self.ss_price_change = (self.price - self.ss_price) / self.ss_price

    def compute_optimal_weights(self, non_labor_share, scaling_price_weights):
        scaled_sum = 0
        for i in self.input_prices:
            if self.input_prices[i] < 0:
                print self.ID, self.input_prices
            scaled_sum += self.input_prices[i] ** scaling_price_weights
        self.input_weights_optimal = {}
        ratio = non_labor_share / scaled_sum        
        
        for i in self.input_prices:
            self.input_weights_optimal[i] = ratio * (self.input_prices[i] ** scaling_price_weights)
            
        if self.ss_weights is not None:
            abs_diff = 0
            for i in self.input_sellers_ID:
                d = abs(self.input_weights[i] - self.ss_weights[i])
                abs_diff += d
            self.ss_weights_change = abs_diff / self.number_of_input_sellers

    def compute_probability_weights_change(self):
        distance = 0
        for i in self.input_sellers_ID:
            distance += abs(1 - self.input_weights_optimal[i] / self.input_weights[i])
        distance /= self.number_of_input_sellers
        n = self.parameters.weights_sensitivity * distance
        self.probability_weights_change = n / (1 + n)

    def set_weights_endogenous_prob(
            self):  # compute optimal proportions in which to buy inputs based on the prices charged by other firms and wage
        if random.uniform(0, 1) < self.probability_weights_change:
            self.input_weights = self.input_weights_optimal

    def compute_weights(self, non_labor_share, scaling_price_weights):
        if random.uniform(0, 1) < self.probability_weights_change:
            self.compute_optimal_weights(non_labor_share, scaling_price_weights)
            self.input_weights = copy.deepcopy(self.input_weights_optimal)

    def compute_sticky_weights(self, non_labor_share, scaling_price_weights):
        self.compute_optimal_weights(non_labor_share, scaling_price_weights)
        old_weights = copy.deepcopy(self.input_weights)

        new_weights = {}

        for ID in old_weights:
            old_w = old_weights[ID]
            optimal_w = self.input_weights_optimal[ID]

            w = self.parameters.linear_weights_stickiness_old_share * old_w + (1-self.parameters.linear_weights_stickiness_old_share) *  optimal_w
            new_weights[ID] = w

        self.input_weights = new_weights

    def compute_input_demands_lockdown(self):
  
        """
        money_diff = 0
        if self.labor_input_only == False:            
            for supplier_ID in self.input_demands:
                demand = self.input_demands[supplier_ID]
                ss = self.ss_input_demands[supplier_ID]
                demand_lockdown = ss * self.prop_lockdown
                lock_demand = min(demand,demand_lockdown)
                self.input_demands[supplier_ID] = lock_demand
                diff = max(demand-demand_lockdown,0)
                money_diff += diff        
        lockdown_wealth = self.ss_wealth * self.prop_lockdown
        if self.wealth > lockdown_wealth:
            self.labor_demand = self.labor_weight * lockdown_wealth
            diff = self.labor_weight * (self.wealth - lockdown_wealth)
            money_diff += diff
        else:
             self.labor_demand = self.labor_weight * self.wealth
        self.labor_demand_per_worker = self.labor_demand / self.number_of_workers
        self.lockdown_money_balance = copy.copy(money_diff)
        """
        
        lockdown_wealth = self.ss_wealth * self.prop_lockdown

        if self.wealth > lockdown_wealth:
            usable_wealth = lockdown_wealth
        else:
            usable_wealth = copy.copy(self.wealth)

        if self.labor_input_only == False:
            for supplier_ID in self.input_sellers_ID:  
                self.input_demands[supplier_ID] = self.input_weights[supplier_ID] * usable_wealth

            self.labor_demand = self.labor_weight * usable_wealth
        else:
            self.labor_demand = usable_wealth

        self.labor_demand_per_worker = self.labor_demand / self.number_of_workers


        self.lockdown_money_balance = max(0,self.wealth - lockdown_wealth)
  
                        
    def compute_input_demands(self):  # compute demand for inputs
        maxLaborFactor = 0.99
        if self.parameters.wageStickiness:
            optimal_laborDemand = self.labor_weight * self.wealth
            sticky_laborDemand = copy.copy(self.lastLaborDemand) * self.parameters.wageStickiness + optimal_laborDemand * (1-self.parameters.wageStickiness)

            self.labor_demand = min(maxLaborFactor*self.wealth, sticky_laborDemand)
            self.lastLaborDemand = copy.copy(self.labor_demand)
            remainingWealth = self.wealth - self.labor_demand
            if self.labor_input_only == False:
                totalWeight = sum(self.input_weights.values())
                for supplier_ID in self.input_sellers_ID:  # demand for other inputs equals wealth multiplied by proportion of wealth spend on them
                    self.input_demands[supplier_ID] = (self.input_weights[supplier_ID]/ totalWeight) * remainingWealth
        else:
            self.labor_demand = self.labor_weight * self.wealth  # labor demand is wealth multiplied by proportion of wealth spend on labor            
            if self.labor_input_only == False:
                for supplier_ID in self.input_sellers_ID:  # demand for other inputs equals wealth multiplied by proportion of wealth spend on them
                    self.input_demands[supplier_ID] = self.input_weights[supplier_ID] * self.wealth
            #print self.ID,self.labor_weight, self.input_weights                    
            """
            if self.labor_demand == 0:
                print self.ID, "ID of zero labor demand"
                print "zero labor demand from firm", self.labor_weight, 'labor weight'
                print self.input_demands, "demands of other inputs"
                print self.wealth, "wealth"
                print self.labor_input_only, "labor input only"
            """
        self.labor_demand_per_worker = self.labor_demand / self.number_of_workers
        
    def compute_output_allocations(self):  # compute allocation of output among buyers
        if self.price > self.market_clearing_price:
            #available_output = copy.copy(self.total_output)



            #print "price higher than market clearing"
            for buyer_ID in self.output_buyers_ID:  # for each buyer in other firms who buy the output
                self.output_allocation[buyer_ID] = self.output_demands[buyer_ID] / self.price  # quantity allocated to a buyer is the buyers demand divided by the price
            
            self.inventory += self.total_output - sum(self.output_allocation.values())
            if self.retail:  # if its a retail firm, compute allocation of output to household
                for consumer_ID in self.consumers_ID:
                    self.output_consumption_allocation[consumer_ID] = self.output_consumer_demands[consumer_ID] / self.price
                self.inventory -= sum(self.output_consumption_allocation.values())
            """
            if self.inventory < 0:
                print "negative inventory", self.ID
            """
                

        elif self.price == self.market_clearing_price:
            #print "market clearing", self.price, self.productivity
            for buyer_ID in self.output_buyers_ID:  # for each buyer in other firms who buy the output
                self.output_allocation[buyer_ID] = self.output_demands[buyer_ID] / self.price  # quantity allocated to a buyer is the buyers demand divided by the price
            if self.retail:  # if its a retail firm, compute allocation of output to household
                for consumer_ID in self.consumers_ID:
                    self.output_consumption_allocation[consumer_ID] = self.output_consumer_demands[consumer_ID] / self.price

            #total_allocation = sum(self.output_allocation.values())
            #if self.retail:
             #   total_allocation += sum(self.output_consumption_allocation.values())

            #if self.output > total_allocation:
             #   print 100*(self.output- total_allocation)/self.output

        else:
            #print " price lss than mkt clearing, divi up demand"
            ratio = self.total_output / self.total_demand
            for buyer_ID in self.output_buyers_ID:  # for each buyer in other firms who buy the output, divide total quantity by proportional demand for input
                self.output_allocation[buyer_ID] = ratio * self.output_demands[buyer_ID]
            if self.retail:
                for ID in self.consumers_ID:
                    self.output_consumption_allocation[ID] = ratio * self.output_consumer_demands[ID]

        #sold = sum(self.output_consumption_allocation.values()) + sum(self.output_allocation.values())
        #self.output_sold = sold
        #self.output_sold_time_series.append(sold)


    def output_allocations_lockdown_adjustment(self):  # compute allocation of output among buyers
        goods = 0
        new_allocation = collections.OrderedDict({})
        for buyer_ID in self.output_buyers_ID:  # for each buyer in other firms who buy the output
            allocation = self.output_allocation[buyer_ID]
            ss = self.ss_output_allocation[buyer_ID]
            prop = self.lockdown_output_buyers_prop[buyer_ID]
            lockdown_allocation = ss * prop
            new = min(allocation,lockdown_allocation)
            new_allocation[buyer_ID] = new
            diff = max(0,allocation-lockdown_allocation)
            goods += diff
        self.output_allocation = copy.deepcopy(new_allocation)
        #self.lockdown_inventory += goods
        self.inventory += goods
        #sold = sum(self.output_consumption_allocation.values()) + sum(self.output_allocation.values())
        #self.output_sold = sold
        #self.output_sold_time_series.append(sold)


    def compute_output_division_twoProducts(self):
        consumer_allocation = self.output_allocation_ratio * self.output
        firms_allocation = self.output - consumer_allocation
        self.output_division = {'consumer': consumer_allocation, 'firms': firms_allocation}

    def compute_prices_twoProducts(self):
        firms_demand =  sum(self.output_demands.values())
        price_firms = firms_demand / self.output_division['firms']
        self.prices['firms'] = price_firms
        self.market_clearing_price = price_firms
        if self.retail:
            consumer_demand = sum(self.output_consumer_demands.values())
            price_consumer = consumer_demand /  self.output_division['consumer']
            self.prices['consumer'] = price_consumer
            self.consumer_demand = consumer_demand

    def compute_allocation_twoProducts(self):
        if self.retail:  # if its a retail firm, compute allocation of output to household
            for consumer_ID in self.consumers_ID:
                self.output_consumption_allocation[consumer_ID] = self.output_consumer_demands[consumer_ID] / self.prices['consumer']

        for buyer_ID in self.output_buyers_ID:  # for each buyer in other firms who buy the output
            self.output_allocation[buyer_ID] = self.output_demands[buyer_ID] / self.prices['firms']  # quantity allocated to a buyer is the buyers demand divided by the price

    def update_wealth(self):  # wealth is the sum of demand for output
        old = copy.copy(self.wealth)
        if self.retail:
            self.wealth = sum(self.output_demands.values()) + sum(self.output_consumer_demands.values())
        else:
            self.wealth = sum(self.output_demands.values())
        self.size_change = abs((old - self.wealth))/old
        
        
        """
        if self.wealth == 0:
            print sum(self.output_demands.values()), "firms demand"
            print sum(self.output_consumer_demands.values()), "consumer demand"
            print self.output_consumer_demands, "output_consumer_demands"
            print self.output_demands, "output demand"
            print self.output_buyers_ID, "output buyers ID"
            print self.number_of_output_buyers, "num of output buyers"
            print self.retail, "retail"
            print self.consumers_ID, "consumers ID", self.ID
        """

    def add_lockdown_money_balance(self):
        self.wealth += copy.copy(self.lockdown_money_balance)
        self.lockdown_money_balance = 0


    def add_reserveWealth_toWealth(self):
        self.wealth += self.reserve_wealth
        self.past_reserve_wealth = copy.copy(self.reserve_wealth)
        self.reserve_wealth = 0

    def adjust_wealth(self):
        if self.wealth > self.past_wealth:
            c = self.rW * (self.wealth - self.past_wealth)
            self.wealth -= c
            self.reserve_wealth = c
        self.past_wealth = copy.copy(self.wealth)
        self.past_reserve_wealth = copy.copy(self.reserve_wealth)

    def record_time_series(self, transient):
        for variable in self.parameters.record_variables['firm']:
            if self.parameters.record_variables['firm'][variable]:
                variable_time_series = variable + '_time_series'
                if variable == 'ss_price_change' or variable == 'ss_output_change':
                    if not transient:
                        getattr(self, variable_time_series).append(getattr(self, variable))
                else:
                    getattr(self, variable_time_series).append(getattr(self, variable))

    def remove_input_seller(self, input_seller_ID):
        self.input_sellers_ID.remove(input_seller_ID)
        self.number_of_input_sellers -= 1
        del self.input_prices[input_seller_ID]
        del self.input_demands[input_seller_ID]
        del self.input_quantities[input_seller_ID]
        del self.input_weights[input_seller_ID]
        assert input_seller_ID in self.input_weights_optimal.keys(), [self.input_weights_optimal, self.input_weights, input_seller_ID]
        del self.input_weights_optimal[input_seller_ID]
        if self.parameters.production_function == 'CD':
            del self.Cobb_Douglas_exponents[input_seller_ID]

    def add_input_seller(self, new_input_seller_ID, new_input_seller_price):
        self.number_of_input_sellers += 1
        self.input_sellers_ID.append(new_input_seller_ID)
        self.input_prices[new_input_seller_ID] = new_input_seller_price
        self.input_demands[new_input_seller_ID] = random.uniform(0,1)
        self.input_quantities[new_input_seller_ID] = random.uniform(0,1)
        self.input_weights[new_input_seller_ID] = 1 - sum(self.input_weights.values())
        self.input_weights_optimal[new_input_seller_ID] = 1- sum(self.input_weights_optimal.values())
        if self.parameters.production_function == 'CD':
            self.Cobb_Douglas()
        self.weights()

    def change_input_seller(self, new_input_seller_ID, new_input_seller_price):
        if not self.labor_input_only:
            input_seller_ID = random.choice(self.input_sellers_ID)
            input_seller_price = self.input_prices[input_seller_ID]
            if new_input_seller_price < input_seller_price:
                self.remove_input_seller(input_seller_ID)
                self.add_input_seller(new_input_seller_ID, new_input_seller_price)
                return True, input_seller_ID
            else:
                return False, False
        else:
            return False, False

    def remove_output_buyer(self, output_buyer_ID):
        self.output_buyers_ID.remove(output_buyer_ID)
        self.number_of_output_buyers -= 1
        del self.output_allocation[output_buyer_ID]
        del self.output_demands[output_buyer_ID]

    def add_output_buyer(self, new_output_buyer_ID):
        self.number_of_output_buyers += 1
        self.output_buyers_ID.append(new_output_buyer_ID)
        self.output_allocation[new_output_buyer_ID] = random.uniform(0,1)
        self.output_demands[new_output_buyer_ID] = random.uniform(0,1)

    def is_input_seller(self, ID):
        if ID in self.input_sellers_ID:
            return True
        else:
            return False

    def return_price(self):
        return self.price

    def input_sellers(self):
        return self.input_sellers_ID

    def attempt_input_seller_change(self):
        if random.uniform(0,1) < self.parameters.probability_firm_input_seller_change:
            return True

    def will_enter(self):
        #print self.ID, self.parameters.probability_firm_entry
        prob = random.uniform(0,1)
        if prob <= self.parameters.probability_firm_entry:
            #print True, "entry"
            return True
        #else:
         #   print False, "no entry", prob

    def exit(self, i, parameters, retail):
        self.__init__(i, parameters)
        self.retail = retail

