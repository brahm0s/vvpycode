"""
Contact - vipin.veetil@gmail.com
Language - Python 2.7.10
"""
from __future__ import division
import random
from datetime import datetime
import copy
from scipy import stats
import numpy as np
import pandas as pd
import sys
import random
import numpy as np



class Samayam(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.economy = None
        self.change_inflation_agents = 0
        self.non_labor_share =  1 - self.parameters.alpha
        self.scaling_price_weights =  - self.parameters.sigma / (1 - self.parameters.sigma)
        self.steadyState = False
        self.roundsDeductions = 0

    def set_initials(self):
        self.change_inflation_agents = int(self.parameters.inflation_change_agents_proportion * len(self.economy.inflation_agents_ID))
        self.non_labor_share = 1 - self.parameters.alpha      
        self.scaling_price_weights = - self.parameters.sigma / (1 - self.parameters.sigma)
    
    def set_CobbDouglas_scaling(self):
        for firm in self.economy.firms.itervalues():
            firm.set_CobbDouglas_scaling()
    
    def set_CobbDouglas_exponents_to_one(self):
        for firm in self.economy.firms.itervalues():
            firm.set_CobbDouglas_exponents_to_one()

    def productivity_growth(self):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            #if firm.active:
            firm.productivity_growth()

    def productivity_shocks(self):  # shock the productivity of each firm
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.productivity_shock()

    def labor_shocks(self):  # shock labor supply of each household
        for household in self.economy.households.itervalues():
            household.labor_shock()

    def interval_productivity_shock(self):
        for firm in self.economy.firms.itervalues():
            rand_val = np.random.lognormal(self.parameters.productivity_shock_mean,self.parameters.productivity_shock_std)
            firm.productivity *=  rand_val

    def one_time_productivity_shock(self):
        for firm in self.economy.firms.itervalues():
            rand_val = np.random.lognormal(self.parameters.one_time_productivity_shock_mean,self.parameters.one_time_productivity_shock_std)
            firm.productivity *=  rand_val
            #rand_val = random.uniform(self.parameters.one_time_productivity_shock_min,self.parameters.one_time_productivity_shock_max)
            #firm.productivity *=  (1-rand_val)

    def begin_lockdown_productivity_shocks(self):
        print "begin lockdown shock"
        for ID in self.economy.lockdown_prop:
            firm = self.economy.firms[ID]
            prop = self.economy.lockdown_prop[ID]            
            firm.productivity *= prop
    
    def end_lockdown_productivity_shocks(self):
        print "end lockdown shock"
        for ID in self.economy.lockdown_prop:
            firm = self.economy.firms[ID]
            prop = self.economy.lockdown_prop[ID]            
            firm.productivity /= prop

    def firms_produce(self):  # firms produce output
        if self.parameters.productivity_labor:
            #print "labor productivity shock"
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]
                firm.produce_labor_productivity()    
        else:
            #print len(self.economy.firms_ID), "num of firms asked to produce"
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]
                firm.produce()

    def firms_addOutputInventory(self,lockdown):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.sum_OutputInventory(lockdown)

    def compute_demands(self,lockdown):  # households and firms compute demands
        for household in self.economy.households.itervalues():
            household.compute_goods_demand()

        if lockdown:
            #print "lockdown demands"
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]            
                firm.compute_input_demands_lockdown()
        else:
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]                        
                firm.compute_input_demands()
        
    def compute_probability_price_change(self):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.compute_probability_price_change()

    def compute_prices_twoProducts(self):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.compute_prices_twoProducts()
            if self.parameters.sync_sensitivity == 0:
                firm.set_price()

        for household_ID in self.economy.households_ID:
            household = self.economy.households[household_ID]
            household.compute_wage()

    def compute_prices_transient(self):
        for firm_ID in self.economy.firms_ID:            
            firm = self.economy.firms[firm_ID]
            firm.compute_market_clearing_price()
            firm.set_price_transient()

            
        #if self.parameters.smoothenPrice:
         #   for firm_ID in self.economy.firms_ID:
          #      firm = self.economy.firms[firm_ID]
           #     firm.smoothen_price()

        for household_ID in self.economy.households_ID:
            household = self.economy.households[household_ID]
            household.compute_wage()

    def set_lockdown_price(self):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.compute_market_clearing_price()
            firm.set_lockdown_price()
    
    def compute_prices_linear_stickiness(self):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.compute_market_clearing_price()
            firm.set_price_linear_stickiness()
            
        if self.parameters.smoothenPrice:
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]
                firm.smoothen_price()

        for household_ID in self.economy.households_ID:
            household = self.economy.households[household_ID]
            household.compute_wage()
    
    def compute_prices_flexible(self):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.compute_market_clearing_price()
            firm.set_price_flexible()
            
        if self.parameters.smoothenPrice:
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]
                firm.smoothen_price()

        for household_ID in self.economy.households_ID:
            household = self.economy.households[household_ID]
            household.compute_wage()
    
    def compute_prices_probabilistic_change(self):
        #print "computing prob prices"
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.compute_market_clearing_price()
            #print "computed market clearing"

        if self.parameters.endogenous_probability_price_change:
            self.compute_probability_price_change()

        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.set_price_probabilistic_stickiness()
        
        #print "completed prob prices"
        if self.parameters.smoothenPrice:
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]
                firm.smoothen_price()

        for household_ID in self.economy.households_ID:
            household = self.economy.households[household_ID]
            household.compute_wage()
        
    def compute_prices_sycn(self):  # firms compute prices and wages, # this needs to rewritten more efficiently
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.compute_market_clearing_price()
            firm.generate_random_prob()
            if self.parameters.sync_direction == "sellers": 
                if firm.number_of_input_sellers != 0:
                    input_sellers = [self.economy.firms[ID] for ID in firm.input_sellers_ID]
                    other_randoms = [seller.random_prob for seller in input_sellers]
                else:
                    other_randoms = False
            elif self.parameters.sync_direction == "buyers":
                if firm.number_of_output_buyers != 0:
                    output_buyers = [self.economy.firms[ID] for ID in firm.output_buyers_ID]
                    other_randoms = [buyer.random_prob for buyer in output_buyers]
                else:
                    other_randoms = False
            firm.set_price_sync(other_randoms)

        if self.parameters.smoothenPrice:
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]
                firm.smoothen_price()

        for household_ID in self.economy.households_ID:
            household = self.economy.households[household_ID]
            if self.parameters.labor_mobility:
                household.labor_quantity_change()                
            household.compute_wage()

    def compute_weights(self):  # the firms compute the proportions in which they want to buy inputs from each other
        if self.parameters.weights_stickiness_type == 'linear':
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]
                if firm.number_of_input_sellers > 1:
                    firm.compute_sticky_weights(self.non_labor_share, self.scaling_price_weights)

        elif self.parameters.weights_stickiness_type == 'probabilistic':
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]
                if firm.number_of_input_sellers > 1:                    
                    firm.compute_optimal_weights(self.non_labor_share, self.scaling_price_weights)
                    firm.compute_probability_weights_change()
                    firm.set_weights_endogenous_prob()
        else:
            for firm_ID in self.economy.firms_ID:
                firm = self.economy.firms[firm_ID]
                if firm.number_of_input_sellers > 1:
                    firm.compute_weights(self.non_labor_share, self.scaling_price_weights)

    def labor_mobility(self):
        for ID in self.economy.households:
            h = self.economy.households[ID]
            h.labor_quantity_change()

    def transfer_price_information_twoProducts(self):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            for worker_ID in firm.workers_ID:
                firm.wage = self.economy.households[worker_ID].wage
            for seller_ID in firm.input_sellers_ID:  # each firm also get information on the price of inputs it buys from other firms
                firm.input_prices[seller_ID] = self.economy.firms[seller_ID].prices['firms']

        for ID in self.economy.retail_firms_ID:
            retail_firm = self.economy.firms[ID]
            for consumer_ID in retail_firm.consumers_ID:
                self.economy.households[consumer_ID].goods_prices[retail_firm.ID] = retail_firm.prices['consumer']

    def transfer_price_information(self):  # firms and households transfer price information to their buyers
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            for worker_ID in firm.workers_ID:
                firm.wage = self.economy.households[worker_ID].wage
            for seller_ID in firm.input_sellers_ID:  # each firm also get information on the price of inputs it buys from other firms
                firm.input_prices[seller_ID] = self.economy.firms[seller_ID].price
        for ID in self.economy.retail_firms_ID:
            retail_firm = self.economy.firms[ID]
            for consumer_ID in retail_firm.consumers_ID:
                self.economy.households[consumer_ID].goods_prices[retail_firm.ID] = retail_firm.price

    def transfer_demand_information(self):  # firms and household transfer demand information
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            for worker_ID in firm.workers_ID:
                self.economy.households[worker_ID].labor_demands[firm.ID] = firm.labor_demand_per_worker

            for buyer_ID in firm.output_buyers_ID:
                # print "firm.output_buyers_ID,", firm.output_buyers_ID
                buyer = self.economy.firms[buyer_ID]
                firm.output_demands[buyer_ID] = buyer.input_demands[firm.ID]

        if self.parameters.household_preference_homogeneous:
            #print len(self.economy.retail_firms_ID), "num retail firms"  

            for ID in self.economy.retail_firms_ID:            
                retail_firm = self.economy.firms[ID]
                for consumer_ID in retail_firm.consumers_ID:
                    consumer = self.economy.households[consumer_ID]
                    retail_firm.output_consumer_demands[consumer_ID] = consumer.goods_demand

            for ID in self.economy.retail_firms_ID:
                f = self.economy.firms[ID]
                if len(f.output_consumer_demands.keys()) == 0:
                    print "zero length of consumer demand"
            """
            for f in self.economy.firms.itervalues():
                if len(f.output_consumer_demands.keys()) == 0:
                    if f.ID in self.economy.out_of_market_firms_ID:
                        print "no consumer demand, out of market too", f.ID
                    else:
                        print "no consumer demand, in market too"
            """

        else:
            for ID in self.economy.retail_firms_ID:
                retail_firm = self.economy.firms[ID]
                for consumer_ID in retail_firm.consumers_ID:
                    consumer = self.economy.households[consumer_ID]
                    consumerDemand = consumer.goods_demand[retail_firm.ID]
                    retail_firm.output_consumer_demands[consumer_ID] = consumerDemand

    def compute_allocations_twoProducts(self):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.compute_output_division_twoProducts()
            firm.compute_allocation_twoProducts()

        for household_ID in self.economy.households_ID:
            household = self.economy.households[household_ID]
            household.compute_labor_allocation()

    def compute_allocations(self,lockdown):  # the household computes allocation of labor, firms compute compute allocations of output among buyers
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.compute_output_allocations()
            if lockdown:
                #print "lockdown output allocation adjustment"
                firm.output_allocations_lockdown_adjustment() 

        for household_ID in self.economy.households_ID:
            household = self.economy.households[household_ID]
            household.compute_labor_allocation()

    def transfer_inputs_labor(self):  # firms and household transfer inputs, labor, and consumption good
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.labor_quantity = 0
            for worker_ID in firm.workers_ID:
                worker = self.economy.households[worker_ID]
                quantity_labor = worker.labor_allocation[firm_ID]
                if quantity_labor == 0:
                    print "zero labor", firm_ID
                firm.labor_quantity += quantity_labor
            for seller_ID in firm.input_sellers_ID:  # each firm receives inputs from its sellers
                seller = self.economy.firms[seller_ID]
                firm.input_quantities[seller_ID] = seller.output_allocation[firm.ID]

    def transfer_consumption_good(self):
        for firm_ID in self.economy.retail_firms_ID:
            firm = self.economy.firms[firm_ID]
            for consumer_ID in firm.consumers_ID:
                consumer = self.economy.households[consumer_ID]
                consumer.goods_quantities[firm_ID] = firm.output_consumption_allocation[consumer_ID]

    def assign_steady_state(self):
        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.ss_output_allocation = firm.output_allocation
            firm.ss_wealth = firm.wealth
            firm.ss_price = firm.price
            firm.ss_output = firm.output
            firm.ss_weights = copy.deepcopy(firm.input_weights)
            firm.ss_input_demands = copy.deepcopy(firm.input_demands)
            #print "assigned steady state values"
            #consumption_allocation = sum(firm.output_consumption_allocation.values())
            firm.ss_intermediate_prop = sum(firm.output_allocation.values())/firm.output

            if self.parameters.twoProductFirms:
                consumer_allocation = sum(firm.output_consumption_allocation.values())
                firms_allocation = sum(firm.output_allocation.values())
                firm.output_allocation_ratio = consumer_allocation / (consumer_allocation + firms_allocation)
                firm.prices['consumer'] =  firm.price
                firm.prices['firms'] = firm.price
                firm.compute_output_division_twoProducts()
                firm.compute_allocation_twoProducts()

    def update_wealth(self,lockdown):  # firms and household update their wealth
        for household in self.economy.households.itervalues():
            household.update_wealth()

        for firm_ID in self.economy.firms_ID:
            firm = self.economy.firms[firm_ID]
            firm.update_wealth()

        if lockdown:
            #print "locdown wealth adjustment"
            for firm in self.economy.firms.itervalues():
                firm.add_lockdown_money_balance()

    def compute_utility(self):
        for household in self.economy.households.itervalues():
            household.compute_utility()

    def adjust_reserves(self):
        for firm in self.economy.firms.itervalues():
            firm.add_reserveWealth_toWealth()
            firm.adjust_wealth()

        for household in self.economy.households.itervalues():
            household.add_reserveWealth_toWealth()
            household.adjust_wealth()

    def rewire_endogenous(self):
        """
        print "testing in samayam"
        count = 0
        for firm in self.economy.firms.itervalues():
            if firm.labor_input_only:
                if len(firm.input_sellers_ID) > 0:
                    print "before labor input only with input sellers", firm.ID 
                    count += 1
        if count == 0:
            print "samayam: no mismatch before"

        """
        #print len(self.economy.out_of_market_firms_ID), "begin num of firms out of market"
        #self.economy.test_firm_network_consistency()
        self.economy.firms_change_input_seller()
        #print len(self.economy.out_of_market_firms_ID), "before num of firms out of market"
        #self.economy.test_firm_network_consistency()
        self.economy.firms_entry()
        #print len(self.economy.out_of_market_firms_ID), "num of firms out of market"
        #print self.economy.links_changed, "links changed"
        #self.economy.test_firm_network_consistency()        
        
        """
        count = 0
        for firm in self.economy.firms.itervalues():
            if firm.labor_input_only:
                if len(firm.input_sellers_ID) > 0:
                    print "before labor input only with input sellers", firm.ID 
                    count += 1
        if count == 0:
            print "samayam: no mismatch after"
        else:
            print "samayam: Mismatch after"
        """

    def inflation_agents_change(self):
        if self.parameters.inflation_change_agents_proportion != 0:
            if self.parameters.money_injection_agents == "firms":
                remove_ID = random.sample(self.economy.inflation_agents_ID, self.change_inflation_agents)
                non_inflation_agents_ID = set(self.economy.firms_ID) - set(self.economy.inflation_agents_ID)
                add_ID = random.sample(non_inflation_agents_ID, self.change_inflation_agents)
            elif self.parameters.money_injection_agents == "households":
                remove_ID = random.sample(self.economy.inflation_agents_ID, self.change_inflation_agents)
                non_inflation_agents_ID = set(self.economy.households_ID) - set(self.economy.inflation_agents_ID)
                add_ID = random.sample(non_inflation_agents_ID, self.change_inflation_agents)
            elif self.parameters.money_injection_agents == "all":
                remove_ID = random.sample(self.economy.inflation_agents_ID, self.change_inflation_agents)
                non_inflation_agents_ID = set(self.economy.all_ID) - set(self.economy.inflation_agents_ID)
                add_ID = random.sample(non_inflation_agents_ID, self.change_inflation_agents)

            self.economy.inflation_agents_ID = set(self.economy.inflation_agents_ID) - set(remove_ID)
            self.economy.inflation_agents_ID.update(add_ID)

    def nominal_wealth_increase_random(self,money):
        weights = np.random.uniform(self.economy.parameters.n)
        weights = [i/sum(weights) for i in weights]
        weights.shuffle()
        for firm in self.economy.firms.itervalues():
            w = weights[firm.ID-1]
            firm.wealth += money *  w
    
    def nominal_wealth_decrease_random(self,money):
        weights = np.random.uniform(self.economy.parameters.n)
        weights = [i/sum(weights) for i in weights]
        weights.shuffle()
        excess = 0
        for firm in self.economy.firms.itervalues():
            w = weights[firm.ID-1]
            deduction = money *  w
            if deduction < firm.wealth:                
                firm.wealth -= deduction
            else:
                excess += deduction
        if excess > 0:
            money = copy.copy(excess)
            return self.nominal_wealth_decrease_random(money)
        else:
            return        
    
    def generate_random_shares(self):
        randomWeights = {}
        num_firms = len(self.economy.firms.values())
        epsilons = []
        for i in xrange(num_firms):
            e = random.uniform(0,1)
            epsilons.append(e)
            
        #epsilons = np.random.uniform(0, 1, num_firms)
    
        sumEp = sum(epsilons)
        epsilons = [i/sumEp for i in epsilons]
        
        for firm in self.economy.firms.itervalues():
            e = epsilons[firm.ID-1]
            randomWeights[firm.ID] = e
        
        #print randomWeights[1], randomWeights[10], 'random weights'
        return randomWeights
    
    def compute_sum_firm_wealth(self):
        sum_firm_wealth = 0
        for firm in self.economy.firms.itervalues():
            sum_firm_wealth += firm.wealth
            
        return sum_firm_wealth
        
    def generate_firms_proportional_shares(self, exponent,sum_firm_wealth):
        
        homogeneousWeights = {}
        
        if exponent:
            for firm in self.economy.firms.itervalues():
                 homogeneousWeights[firm.ID] = (firm.wealth / sum_firm_wealth) ** exponent
        else:
            for firm in self.economy.firms.itervalues():
                homogeneousWeights[firm.ID] = firm.wealth / sum_firm_wealth
                
        totalW = sum(homogeneousWeights.values())
        
        
        for firm in self.economy.firms.itervalues():
            homogeneousWeights[firm.ID] /= totalW
        
        return homogeneousWeights
        
    def combine_shares(self, homogeneousWeights, randomWeights):
        weights = {}
        for firm in self.economy.firms.itervalues():              
            homoW = homogeneousWeights[firm.ID] * (1 - self.parameters.randomWeightShare)
            ranW = randomWeights[firm.ID] * self.parameters.randomWeightShare
            weights[firm.ID] = homoW + ranW 
        return weights
    
    def multiplicativeNoise(self,trueWeights):
        nosiyWeights = {}
        for ID in trueWeights:
            u = random.uniform(self.parameters.a_noiseMul, self.parameters.b_noiseMul)
            w = trueWeights[ID]
            n_w = w * u
            nosiyWeights[ID] = n_w
            
        total = sum(nosiyWeights.values())
        for ID in nosiyWeights:
            w = nosiyWeights[ID]
            w = w/total
            nosiyWeights[ID] = w
            
        return nosiyWeights 

    def nominal_wealth_increase_stochastic(self,money,exponent):
        firm_wealth = self.compute_sum_firm_wealth()
        total_wealth = firm_wealth + self.economy.households[-1].wealth
        
        
        householdShare = self.economy.households[-1].wealth / total_wealth
        householdShare_money = householdShare * money
        self.economy.households[-1].wealth += householdShare_money
        money = (1-householdShare) * money

        homogeneousWeights = self.generate_firms_proportional_shares(exponent,firm_wealth)
        
        if self.parameters.noise_multiplicative:
            weights = self.multiplicativeNoise(homogeneousWeights)
        else:
            if self.parameters.randomWeightShare > 0:            
                randomWeights = self.generate_random_shares()
                weights = self.combine_shares(homogeneousWeights, randomWeights)
                
            else:
                weights = homogeneousWeights


        for firmID in self.economy.firms:
            w = weights[firmID]
            firm = self.economy.firms[firmID]
            firm.wealth += money * w
            
    def nominal_wealth_decrease_stochastic(self,money,exponent):
        #print money, "money"
        #print exponent, "exponent"
        firm_wealth = self.compute_sum_firm_wealth()
        #print firm_wealth, "firm_wealth"
        
        total_wealth = firm_wealth + self.economy.households[-1].wealth
        #print total_wealth, "total_wealth"
        #print "hh share of wealth",  self.economy.households[-1].wealth/total_wealth
        #old = copy.copy(self.economy.households[-1].wealth)
        
        #print len(self.economy.households.values()), 'num of households'
        
        
        
        if self.roundsDeductions == 0:
            householdShare = self.economy.households[-1].wealth / total_wealth
            householdShare_money = householdShare * money
            self.economy.households[-1].wealth -= householdShare_money
            money = (1-householdShare) * money
            #old_firm = copy.copy(firm_wealth)
            #hh_wealth_new = self.economy.households[-1].wealth/old
            #print 'hh wealth, new as share of old', hh_wealth_new
        

        
        homogeneousWeights = self.generate_firms_proportional_shares(exponent, firm_wealth)
        
        if self.parameters.noise_multiplicative:
            weights = self.multiplicativeNoise(homogeneousWeights)
        else:
            if self.parameters.randomWeightShare > 0:            
                randomWeights = self.generate_random_shares()
                weights = self.combine_shares(homogeneousWeights, randomWeights)
                
            else:
                weights = homogeneousWeights
            
        #a = sum(weights.values())
        #print a, "sum weights"
        #print weights[1], 'firm 1 weight', weights[2], 'firm 2 weight'
        extra = 0
        for firmID in self.economy.firms:
            firm = self.economy.firms[firmID]
            w = weights[firmID]
            deduction = money * w
            if firm.wealth > deduction:
                firm.wealth -= deduction
            else:
                adjusted_deducation = firm.wealth * self.parameters.z
                firm.wealth -= adjusted_deducation
                diff = deduction - adjusted_deducation
                extra += diff
            
        
        if extra>0:
            self.roundsDeductions += 1
            #print self.roundsDeductions, 'rounds', extra, "money remaining"
            money = copy.copy(extra)
            return self.nominal_wealth_decrease_stochastic(money,exponent)
        else:
            #new = self.compute_sum_firm_wealth()
            #a = new/old_firm
            #b =  a/hh_wealth_new
            #print b,  "is hh and firm ratio equal"
            return
        
    def nominal_wealth_increaseDistribution(self, money):
        sum_current_wealth = 0
        for firm in self.economy.firms.itervalues():
            sum_current_wealth += firm.wealth
        for household in self.economy.households.itervalues():
            sum_current_wealth += household.wealth

        homogeneousWeights_firms = {}
        homogeneousWeights_households = {}

        for firm in self.economy.firms.itervalues():
            homogeneousWeights_firms[firm.ID] = firm.wealth / sum_current_wealth
        for household in self.economy.households.itervalues():
            homogeneousWeights_households[household.ID] = household.wealth / sum_current_wealth

        number_of_agents = self.economy.parameters.n + self.economy.parameters.number_of_households
        theta = self.parameters.theta
        epsilons = np.random.uniform(0, theta, number_of_agents)
        epsilons = [i / number_of_agents for i in epsilons]
        newWeights_firms = {}
        newWeights_households = {}

        count = 0
        for firmID in homogeneousWeights_firms:
            e = epsilons[count]
            weight = homogeneousWeights_firms[firmID]
            nW = weight + e
            newWeights_firms[firmID] = nW
            count += 1

        for householdID in homogeneousWeights_households:
            e = epsilons[count]
            weight = homogeneousWeights_households[householdID]
            nW = weight + e
            newWeights_households[householdID] = nW
            count += 1

        weights_firms = newWeights_firms.values()
        weights_households = newWeights_households.values()
        weights = weights_firms + weights_households
        total_weight = sum(weights)
        normalizationRatio = 1 / total_weight

        normalizedWeights_firms = {}
        normalizedWeights_households = {}

        for firmID in newWeights_firms:
            w = newWeights_firms[firmID]
            normW = w * normalizationRatio
            normalizedWeights_firms[firmID] = normW

        for householdID in newWeights_households:
            w = newWeights_households[householdID]
            normW = w * normalizationRatio
            normalizedWeights_households[householdID] = normW

        for firmID in self.economy.firms:
            w = normalizedWeights_firms[firmID]
            firm = self.economy.firms[firmID]
            firm.wealth += money * w

        for householdID in self.economy.households:
            w = normalizedWeights_households[householdID]
            household = self.economy.households[householdID]
            household.wealth += money * w

    def nominal_wealth_increase(self, money):
        sum_current_wealth_inflation_agents = 0
        if self.parameters.money_injection_agents == "firms":
            for agent_ID in self.economy.inflation_agents_ID:
                sum_current_wealth_inflation_agents += self.economy.firms[agent_ID].wealth
            ratio = money / sum_current_wealth_inflation_agents
            for agent_ID in self.economy.inflation_agents_ID:
                self.economy.firms[agent_ID].wealth *= (1 + ratio)
        elif self.parameters.money_injection_agents == "households":
            for agent_ID in self.economy.inflation_agents_ID:
                sum_current_wealth_inflation_agents += self.economy.households[agent_ID].wealth
            ratio = money / sum_current_wealth_inflation_agents
            for agent_ID in self.economy.inflation_agents_ID:
                self.economy.households[agent_ID].wealth *= (1 + ratio)
        elif self.parameters.money_injection_agents == "all":
            print "all inflation agents"
            for agent_ID in self.economy.inflation_agents_ID:
                if agent_ID < 0:
                    sum_current_wealth_inflation_agents += self.economy.households[agent_ID].wealth
                else:
                    sum_current_wealth_inflation_agents += self.economy.firms[agent_ID].wealth
            ratio = money / sum_current_wealth_inflation_agents
            print ratio, "ratio"
            for agent_ID in self.economy.inflation_agents_ID:
                if agent_ID < 0:
                    self.economy.households[agent_ID].wealth *= (1 + ratio)
                else:
                    self.economy.firms[agent_ID].wealth *= (1 + ratio)
        elif self.parameters.money_injection_agents == 'single_firm':
            firm = self.economy.firms[self.parameters.money_injection_firm_ID]
            firm.wealth += money

    def nominal_wealthChange_weights(self, money, positiveNegative):
        if positiveNegative == "negative":
            money *= -1
        if money > 0:
            print "positive monetary shock"
        data = pd.read_csv('smallFirmsRatio.csv')
        data = pd.DataFrame(data)
        rows = data.shape[0]
        wealthShare = {}
        for r in xrange(rows):
            ID = data.iloc[r]["index"]
            if ID not in self.parameters.monetaryShock_exclude:
                w = data.iloc[r][self.parameters.monetaryShockWeightsFile_Column]
                ID = int(ID)
                firm = self.economy.firms[ID]
                if self.parameters.monetaryShock_reverseWeight:
                    wealthShare[ID] = (1-w) * firm.wealth
                else:
                    wealthShare[ID] = w * firm.wealth
        totalWealth = sum(wealthShare.values())
        weights = {}
        for ID in wealthShare:
            wealthCont = wealthShare[ID]
            w = wealthCont/totalWealth
            weights[ID] = w

        for ID in weights:
            firm = self.economy.firms[ID]
            w = weights[ID]
            firm.wealth += w * money
            if firm.wealth < 0:
                print "firm wealth negative due to shock"
                sys.exit()

    def nominal_wealth_decrease(self, money):
        count_money = copy.copy(money)
        if self.parameters.money_injection_agents == 'firms':
            all_ID = self.economy.firms_ID
        elif self.parameters.money_injection_agents == 'households':
            all_ID = self.economy.households_ID
        elif self.parameters.money_injection_agents == 'all':
            all_ID = self.economy.households_ID + self.economy.firms_ID

        for ID in all_ID:
            if ID > 0:
                agent = self.economy.firms[ID]
            else:
                agent = self.economy.households[ID]
            if count_money > 0:
                extracted_wealth = agent.wealth * self.parameters.z
                if extracted_wealth < count_money:
                    agent.wealth -= extracted_wealth
                else:
                    agent.wealth -= count_money
                count_money -= extracted_wealth

        if count_money > 0:
            print "code exit: money insufficient for negative shock, consider increasing z or s"
            sys.exit()

    def nominal_wealth_decrease_by_wealth(self, money):
        count_money = copy.copy(money)
        agents = []

        if self.parameters.money_injection_agents == "firms":
            agents = self.economy.firms_list
        elif self.parameters.money_injection_agents == "retail_firms":
            agents = self.economy.retail_firms
        elif self.parameters.money_injection_agents == "suppliers_retail_firms":
            agents = self.economy.retail_suppliers_firms

        """
        agentsID = self.economy.inflation_agents_ID
        agents = [self.economy.firms[i] for i in agentsID]
        """
        print len(agents), "number of inflation agents"
        if self.parameters.negative_shocks_by_wealth == 'Small':
            agents.sort(key=lambda firm: firm.wealth)
        elif self.parameters.negative_shocks_by_wealth == 'Large':
            agents.sort(key=lambda firm: firm.wealth, reverse=True)

        count = 0
        ret = 0
        while count_money > 0 and count < len(agents):
            firm = agents[count]
            if firm.retail:
                ret += 1
            extracted_wealth = firm.wealth * self.parameters.z
            if extracted_wealth < count_money:
                firm.wealth -= extracted_wealth
            else:
                firm.wealth -= count_money
            count_money -= extracted_wealth
            count += 1
        print count, "number of firms shocked"
        print ret, "number of retail firms shocked"

    def nominal_wealth_increase_proportion(self, money):
        scaledWealth = {}
        ratios = {}
        for firm in self.economy.firms.itervalues():
            ID = firm.ID
            wealth = firm.wealth
            scaled = wealth ** self.parameters.monetaryShock_exponent
            scaledWealth[ID] = scaled
            
        total = sum(scaledWealth.values())
        for ID in scaledWealth:
            w = scaledWealth[ID]
            ratios[ID] = w/total
            
            
        for firm in self.economy.firms.itervalues():
            r = ratios[firm.ID]
            m = r * money
            firm.wealth += m               

    def nominal_wealth_decrease_proportion(self, money):
        ratios = {}
        originalWealth = {}
        totalWealth = 0
        if self.parameters.monetaryShock_exponent_fileSizes:
            if self.parameters.addFileName:
                file_name = "sizes" + self.parameters.addFileName + '.txt'
            else:
                file_name = "sizes.txt"
            with open(file_name, 'r') as file:
                for line in file:
                    line = line.split(",")
                    ID = int(line[0])
                    s = float(line[1])
                    totalWealth += s
                    if ID != 0:
                        originalWealth[ID] = copy.copy(s)
                        if self.parameters.monetaryShock_exponent_stochastic:
                            a = self.parameters.monetaryShock_exponent_stochastic[0]
                            b = self.parameters.monetaryShock_exponent_stochastic[1]
                            rand = np.random.uniform(a, b)
                            adjusted_exponent = self.parameters.monetaryShock_exponent * rand
                        else:
                            adjusted_exponent = self.parameters.monetaryShock_exponent
                        #print self.parameters.monetaryShock_exponent, adjusted_exponent
                        s = s ** adjusted_exponent
                        ratios[ID] = s
            total = sum(ratios.values())
            #print totalWealth, "economy wealth"
            #print sum(originalWealth.values()), "firms total wealth"
            for ID in ratios:
                ratios[ID] /= total
            #print sum(ratios.values()), "total ratios"
        else:
            scaledWealth = {}
            for firm in self.economy.firms.itervalues():
                ID = firm.ID
                wealth = firm.wealth
                if self.parameters.monetaryShock_exponent_stochastic:
                    a = self.parameters.monetaryShock_exponent_stochastic[0]
                    b = self.parameters.monetaryShock_exponent_stochastic[1]
                    rand = random.uniform(a,b)
                    adjusted_exponent = self.parameters.monetaryShock_exponent * rand
                    scaled = wealth ** adjusted_exponent
                    #if firm.ID == 2:
                    #   print rand, "rand"
                    #  print adjusted_exponent, "adjusted"
                else:
                    scaled = wealth ** self.parameters.monetaryShock_exponent
                scaledWealth[ID] = scaled
            total = sum(scaledWealth.values())
            for ID in scaledWealth:
                w = scaledWealth[ID]
                ratios[ID] = w/total
            #print sum(ratios.values())
        for firm in self.economy.firms.itervalues():
            r = ratios[firm.ID]
            m = r * money
            #print r, "ratio", m , "money deduction", money, "total deduction" ,firm.wealth, originalWealth[firm.ID]
            #if firm.ID == 2:
            #   print firm.wealth,"before"
            firm.wealth -= m
            #if firm.ID == 2:
            #   print firm.wealth,"after"
            if firm.wealth < 0:
                print "negative wealth after shock, increase exponent"
                sys.exit()

    def record_firm_stable_price(self):
        if self.parameters.twoProductFirms:
            for firm in self.economy.firms.itervalues():
                self.economy.stable_prices_firms[firm.ID] = firm.prices['firms']
        else:
            for firm in self.economy.firms.itervalues():
                self.economy.stable_prices_firms[firm.ID] = firm.price

    def record_time_series(self, time_step, transient):  # record different time series
        if self.parameters.data_time_series['economy']:            
            self.economy.record_data(time_step, transient)
        if self.parameters.data_time_series['firm']:
            for firm in self.economy.firms.itervalues():
                firm.record_time_series(transient)
        if self.parameters.data_time_series['household']:
            for household in self.economy.households.itervalues():
                household.record_time_series()

    def test_network_variables_consistency(self):
        for household in self.economy.households.itervalues():
            assert len(household.goods_sellers_ID) > 0, ["no sellers of goods", "household ID", household.ID]
            assert len(household.labor_buyers_ID) > 0, ["no buyers of labor", "household ID", household.ID]
            assert household.wage > 0, ["wage not positive", "household ID", household.ID, "wage", household.wage]
            assert household.wealth > 0, ["wealth not positive", "household ID", household.ID, "wealth",
                                          household.wealth]
            assert household.utility > 0, ["wealth not positive", "household ID", household.ID, "utility",
                                           household.utility]
            assert household.labor_quantity > 0, ["labor_quantity not positive", "household ID", household.ID,
                                                  "utility", household.labor_quantity]
            for v in household.labor_demands.itervalues():
                assert v > 0, ["labor demand not positive", "household ID", household.ID, "household.labor_demands",
                               household.labor_demands]
            for v in household.labor_allocation.itervalues():
                assert v > 0, ["labor allocation not positive", "household ID", household.ID,
                               "household.labor_allocation", household.labor_allocation]
            for v in household.goods_prices.itervalues():
                assert v > 0, ["goods price not positive", "household ID", household.ID, "household.goods_prices",
                               household.goods_prices]
            for v in household.goods_quantities.itervalues():
                assert v > 0, ["goods quantity not positive", "household ID", household.ID,
                               "household.goods_quantities", household.goods_quantities]
            if self.parameters.household_preference_homogeneous:
                assert household.goods_demand > 0, ["goods demand not positive", "household ID", household.ID,
                                                    "household.goods_demand ", household.goods_demand]
            else:
                for v in household.goods_demand.itervalues():
                    assert v > 0, ["goods demand not positive", "household ID", household.ID, "household.goods_demand ",
                                   household.goods_demand]
            assert household.ID < 0, ["positive household ID", household.ID]
            for seller_ID in household.goods_sellers_ID:
                seller = self.economy.firms[seller_ID]
                price = household.goods_prices[seller_ID]
                assert seller.price == price, ["seller price not equal to household price", "seller.price",
                                               seller.price, "price", price]
                quantity = household.goods_quantities[seller_ID]
                assert seller.output_consumption_allocation[household.ID] == quantity, [
                    "seller quantity not equal to household quantity", "seller.output_allocation[household.ID]",
                    seller.output_allocation[household.ID], "household quantity", quantity]
                if self.parameters.household_preference_homogeneous:
                    assert household.goods_demand == seller.output_consumer_demands[household.ID], [
                        "household demand not equal consumer demand", "household.goods_demand", household.goods_demand,
                        "seller.output_consumer_demands[household.ID]", seller.output_consumer_demands[household.ID]]
                else:
                    assert household.goods_demand[seller_ID] == seller.output_consumer_demands[household.ID], [
                        "household demand not equal consumer demand", "household.goods_demand",
                        household.goods_demand[seller_ID], "seller.output_consumer_demands[household.ID]",
                        seller.output_consumer_demands[household.ID]]
            for labor_buyer_ID in household.labor_buyers_ID:
                labor_buyer = self.economy.firms[labor_buyer_ID]
                demand = household.labor_demands[labor_buyer_ID]
                assert labor_buyer.labor_demand_per_worker == demand, ["labor demand not equal", "firm demand",
                                                                       labor_buyer.labor_demand_per_worker,
                                                                       "household demand", demand]

        for firm in self.economy.firms.itervalues():
            assert firm.price > 0, ["price not positive", "firm ID", firm.ID, "price", firm.price]
            assert firm.wage > 0, ["wage not positive", "firm ID", firm.ID, "wage", firm.wage]
            assert len(firm.input_sellers_ID) > 0, ["no sellers of inputs", "firm ID", firm.ID]
            assert len(firm.output_buyers_ID) > 0, ["no buyers of output", "firm ID", firm.ID]
            assert firm.number_of_workers > 0, ["no workers", "firm ID", firm.ID]
            assert firm.number_of_input_sellers > 0, ["no input sellers", "firm ID", firm.ID]
            assert firm.number_of_output_buyers > 0, ["no output buyers", "firm ID", firm.ID]
            assert len(firm.workers_ID) > 0, ["no workers in workers ID", "firm ID", firm.ID]
            if firm.retail:
                assert len(firm.consumers_ID) > 0, ["no consumers in consumers ID", "firm ID", firm.ID]
            assert firm.wealth > 0, ["wealth not positive", "firm ID", firm.ID, "wealth", firm.wealth]
            assert firm.output > 0, ["output not positive", "firm ID", firm.ID, "output", firm.output]
            assert firm.inventory >= 0, ["inventory negative", "firm ID", firm.ID, "inventory", firm.inventory]
            assert firm.price > 0, ["price not positive", "firm ID", firm.ID, "price", firm.price]
            assert firm.market_clearing_price > 0, ["mkt clearing price not positive", "firm ID", firm.ID,
                                                    "mkt clearing price", firm.market_clearing_price]
            assert firm.labor_demand > 0, ["labor_demand not positive", "firm ID", firm.ID, "labor_demand ",
                                           firm.labor_demand]
            assert firm.labor_demand_per_worker > 0, ["labor_demand_per_worker not positive", "firm ID", firm.ID,
                                                      "labor_demand_per_worker ", firm.labor_demand_per_worker]
            assert firm.labor_quantity > 0, ["labor_quantity not positive", "firm ID", firm.ID, "labor_quantity ",
                                             firm.labor_quantity]
            assert firm.labor_weight > 0, ["labor_weight not positive", "firm ID", firm.ID, "labor_weight ",
                                           firm.labor_weight]
            assert firm.wage > 0, ["wage not positive", "firm ID", firm.ID, "wage ", firm.wage]
            for v in firm.input_demands.itervalues():
                assert v > 0, ["input demand not positive", "firm ID", firm.ID, "input_demands", firm.input_demands]
            for v in firm.input_quantities.itervalues():
                assert v > 0, ["input_quantities not positive", "firm ID", firm.ID, "input_quantities",
                               firm.input_quantities]
            for v in firm.input_prices.itervalues():
                assert v > 0, ["input_prices  not positive", "firm ID", firm.ID, "input_prices ", firm.input_prices]
            for v in firm.input_weights.itervalues():
                assert v > 0, ["input_weights   not positive", "firm ID", firm.ID, "input_weights ", firm.input_weights]
            for v in firm.input_weights_optimal.itervalues():
                assert v > 0, ["input_weights_optimal  not positive", "firm ID", firm.ID, "input_weights_optimal ",
                               firm.input_weights_optimal]
            for v in firm.output_demands.itervalues():
                assert v > 0, ["output_demands  not positive", "firm ID", firm.ID, "output_demands ",
                               firm.output_demands]
            for v in firm.output_allocation.itervalues():
                assert v > 0, ["output_allocation  not positive", "firm ID", firm.ID, "output_allocation ",
                               firm.output_allocation]
            for v in firm.output_consumer_demands.itervalues():
                assert v > 0, ["output_consumer_demands  not positive", "firm ID", firm.ID, "output_consumer_demands ",
                               firm.output_consumer_demands]
            for v in firm.output_consumption_allocation.itervalues():
                assert v > 0, ["output_consumption_allocation  not positive", "firm ID", firm.ID,
                               "output_consumption_allocation ", firm.output_consumption_allocation]

            assert abs(1 - sum(firm.input_weights.values())) < 1.00001 * (1 - self.parameters.alpha), [
                "weights not equal to one", "firm ID", firm.ID, "sum weights", sum(firm.input_weights.values()),
                "firm.input_weights", firm.input_weights]
            for seller_ID in firm.input_sellers_ID:
                seller = self.economy.firms[seller_ID]
                demand = firm.input_demands[seller_ID]
                assert demand == seller.output_demands[firm.ID], ["demand inconsistent", "seller.output_demands",
                                                                  seller.output_demands,
                                                                  "firm.input_demands[seller_ID]",
                                                                  firm.input_demands[seller_ID]]
                quantity = firm.input_quantities[seller_ID]
                assert quantity == seller.output_allocation[firm.ID], ["quantity inconsistent",
                                                                       "seller.output_allocation",
                                                                       seller.output_allocation,
                                                                       "firm.input_quantities[seller_ID]",
                                                                       firm.input_quantities[seller_ID]]
                price = firm.input_prices[seller_ID]
                assert seller.price == price, ["price inconsistent", "seller.price", seller.price, "firm price",
                                               seller.price, price]
            for buyer_ID in firm.output_buyers_ID:
                buyer = self.economy.firms[buyer_ID]
                demand = firm.output_demands[buyer_ID]
                assert demand == buyer.input_demands[firm.ID]
                allocation = firm.output_allocation[buyer_ID]
                assert allocation == buyer.input_quantities[firm.ID]
            if firm.retail:
                for consumer_ID in firm.consumers_ID:
                    consumer = self.economy.households[consumer_ID]
                    demand = firm.output_consumer_demands[consumer_ID]
                    if self.parameters.household_preference_homogeneous:
                        assert consumer.goods_demand == demand, ["preference homogeneous",
                                                                 "firm consumer demand not equal", "consumer demand",
                                                                 consumer.goods_demand, "firm demand", demand]
                    else:
                        assert consumer.goods_demand[firm.ID] == demand, ["firm consumer demand not equal",
                                                                          "consumer demand", consumer.goods_demand,
                                                                          "firm demand", demand]
                    quantity = firm.output_consumption_allocation[consumer_ID]
                    assert consumer.goods_quantities[firm.ID] == quantity, ["firm consumer quantity not equal",
                                                                            "consumer quantity",
                                                                            consumer.goods_quantities[firm.ID],
                                                                            "firm quantity", quantity]
                    assert consumer.goods_prices[firm.ID] == firm.price, ["firm consumer price not equal",
                                                                          "consumer price", consumer.goods_prices,
                                                                          "firm price", firm.price]
                    assert quantity > 0, ["consumption quantity not positive", "firm ID", firm.ID, "quantity", quantity]
