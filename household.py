"""
Contact - vipin.veetil@gmail.com
Language - Python 2.7.10
"""
from __future__ import division
import gc
import functions as f
import random
import copy
import numpy as np
from datetime import datetime
import sys
#from array import array
gc.enable()


class Household(object):
    __slots__ = ('parameters', 'ID', 'goods_sellers_ID', 'labor_buyers_ID', 'wage', 'wealth', 'past_wealth', 'reserve_wealth', 'past_reserve_wealth', 'utility',
                 'labor_quantity', 'labor_demands','labor_allocation', 'utility_function_exponents',
                 'goods_prices','goods_quantities', 'goods_demand', 'wage_time_series',
                 'utility_time_series', 'wealth_time_series', 'rW', 'selfWeight', 'saving', 'consumptionExpenditure',
                 'permanent_wealth','saving_time_series','unemployed','last_labor_demand','weighted_consumption',
                 'labor_fixed_shares')

    def __init__(self, i, parameters):
        self.weighted_consumption = 0
        self.saving = random.uniform(0,1)
        self.selfWeight = None
        self.parameters = parameters
        self.consumptionExpenditure = random.uniform(0,1)
        self.ID = i
        self.goods_sellers_ID = [] # IDs of firms who supply goods to the household
        self.labor_buyers_ID = []  # IDs of firms who demand labor from the household
        self.saving_time_series = []
        self.wage = random.random ()
        self.wealth = random.random() # wealth at present time step
        self.past_wealth = random.random() # wealth at last time step
        self.permanent_wealth = random.random()
        self.reserve_wealth = 0
        self.past_reserve_wealth = 0
        self.utility = random.random() # utility at present time step
        self.labor_quantity = 1
        self.labor_demands = {} # demand for labor from different firms
        self.labor_fixed_shares = {} # fixed share of labor to different firms
        self.labor_allocation = {} # the quantity of labor allocated to different firms
        self.utility_function_exponents = None # Cobb-Douglas exponents associated with different goods in the utility function
        # dictionary when preference is heterogeneous, float when preference is homogeneous
        self.goods_prices = {} # prices of goods from different firms
        self.goods_quantities = {} # the quantity of goods bought from different firms
        self.goods_demand = None # demand for goods from different firms;
        self.rW = random.uniform(self.parameters.psi - self.parameters.gamma,
                                  self.parameters.psi + self.parameters.gamma)  # smoothening paramter of tthe ifmr
        self.unemployed = 0
        # dictionary if preference is heterogeneous, float if homogeneous
        # if homogeneous, good_demand is demand for good per firm
        if parameters.data_time_series:
            for variable in parameters.record_variables['household']:
                if parameters.record_variables['household'][variable]:
                    variable += '_time_series'
                    setattr(self, variable,[])

    def initial_values(self):
        for ID in self.labor_buyers_ID: # assign random initial demands and allocations
            self.labor_demands[ID] = random.random()
            self.labor_allocation[ID] = random.random()
        for ID in self.goods_sellers_ID: # assign random initial quantities and demands for goods
            self.goods_quantities[ID] = random.random()
            self.goods_prices[ID] = random.random()
        if self.parameters.household_preference_homogeneous:
            self.goods_demand = random.random()
        else:
            randoms = [random.random() for i in xrange(len(self.goods_sellers_ID))]
            self.goods_demand = dict(zip(self.goods_sellers_ID, randoms))

        if self.parameters.fixed_labor_shares:
            self.labor_allocation = {}
            for ID in self.labor_fixed_shares:
                share = self.labor_fixed_shares[ID]
                l = self.labor_quantity * share
                self.labor_allocation[ID] = l
        

    def compute_utility_function_exponents(self): # compute the exponent of utility function
        if not self.parameters.weightsFile:
            if self.parameters.household_preference_homogeneous: # if household preference is homogeneous
                self.utility_function_exponents = 1 / len(self.goods_sellers_ID)
            else: # if preference is not homogeneous, each consumption good has a different Cobb-Douglas exponent
                exponents = f.normalized_random_numbers(len(self.goods_sellers_ID))
                self.utility_function_exponents = dict(zip(self.goods_sellers_ID, exponents))

    def compute_utility(self):
        if self.parameters.household_preference_homogeneous:
            quantities = self.goods_quantities.values()
            self.utility = sum([q ** self.utility_function_exponents for q in quantities])
        else:
            self.utility = f.Cobb_Douglas(self.goods_quantities.values(), self.utility_function_exponents.values()) # compute utility
            #self.utility = sum(self.goods_quantities.values())

        if self.parameters.household_preference_homogeneous:
            val = 0
            for ind in self.goods_quantities:
                qty = self.goods_quantities[ind]
                weight = self.utility_function_exponents
                v = qty * weight
                val += v
        else:
            val = 0
            for ind in self.goods_quantities:
                qty = self.goods_quantities[ind]
                weight = self.utility_function_exponents[ind]
                v = qty * weight
                val += v

        self.weighted_consumption = val

    def labor_quantity_change(self): # because of limited labor mobility
        demandDecline = 0
        for ID in self.labor_demands:
            present_demand = self.labor_demands[ID]
            last_demand = self.last_labor_demand[ID]
            if present_demand < last_demand:
                demandDecline += (last_demand-present_demand)
        prop_decline = demandDecline / sum(self.labor_demands.values())


        new_unemployed = prop_decline * (1-self.unemployed)
        total_unemployment =  self.unemployed + new_unemployed        
        stay_unemployment = (1-self.parameters.labor_mobility) * total_unemployment 
        total_employed = 1 - stay_unemployment

        self.unemployed = copy.copy(stay_unemployment)
        self.labor_quantity = copy.copy(total_employed)        

        
        """
        employed = 1-self.unemployed 
        new_unemployed = prop_decline * employed
        adjusted_employed = (1-prop_decline) * employed
        
        new_employed = self.unemployed * self.parameters.labor_mobility
        total_employed = adjusted_employed + new_employed            
        self.unemployed -= new_employed
        self.unemployed += new_unemployed
        self.labor_quantity = copy.copy(total_employed)
        #print self.labor_quantity
        """
    
    def compute_wage(self):
        self.last_labor_demand = copy.deepcopy(self.labor_demands)
        demand = sum(self.labor_demands.values())
        self.wage = demand / self.labor_quantity

    def compute_labor_allocation(self):
        if not self.parameters.fixed_labor_shares: 
            for ID in self.labor_demands:
                demand = self.labor_demands[ID]            
                self.labor_allocation[ID] = demand / self.wage

            

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

    def update_wealth(self): # the wealth of the household is sum of labor demand
        if self.selfWeight > 0:
            self.wealth = self.selfWeight * copy.copy(self.wealth)
        else:
            self.wealth = 0
        self.wealth += sum(self.labor_demands.values())

    def compute_goods_demand(self): # demand for each good equals its Cobb-Douglas exponent multiplied by wealth
        if self.parameters.hhSaving:
            wealthDifference = self.wealth - self.permanent_wealth
            if wealthDifference > 0:
                savingNow = wealthDifference * self.parameters.hhSaving
                self.saving += savingNow
                self.consumptionExpenditure = self.wealth - savingNow
            else:
                disSaving = copy.copy(abs(wealthDifference) * self.parameters.hhSaving)
                drawn = min(self.saving, disSaving)
                self.saving -= drawn
                self.consumptionExpenditure = self.wealth + drawn
        else:
            self.consumptionExpenditure = copy.copy(self.wealth)
            
        fac = self.parameters.hhPermanentIncomeFactor
        self.permanent_wealth = fac * self.permanent_wealth + (1 - fac) * self.wealth
        
        
        if self.parameters.household_preference_homogeneous:
            print self.utility_function_exponents
            print self.consumptionExpenditure
            self.goods_demand = self.utility_function_exponents * self.consumptionExpenditure
            #print self.goods_demand, "Good Demand"
        else:
            #print len(self.goods_sellers_ID), "number of goods sellers"
            for ID in self.goods_sellers_ID:
                self.goods_demand[ID] = self.utility_function_exponents[ID] * self.consumptionExpenditure

    def labor_shock(self): # productivity shock
        if self.parameters.stochastic_labor_supply:
            self.labor_quantity *= self.parameters.random.lognormal(self.parameters.productivity_shock_mean, self.parameters.productivity_shock_std)

    def record_time_series(self):
        for variable in self.parameters.record_variables['household']:
            if self.parameters.record_variables['household'][variable]:
                variable_time_series = variable + '_time_series'
                getattr(self, variable_time_series).append(getattr(self, variable))

