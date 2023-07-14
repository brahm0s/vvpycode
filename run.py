

"""
Contact - vipin.veetil@gmail.com
Language - Python 2.7.10
"""

from __future__ import division
import economy
import numpy as np
import samayam
from datetime import datetime
import time
import gc
import random
import copy
import collections
import write_network
import eigen
import rewiring
import cPickle
import matplotlib.pyplot as plt
import os
random.seed(datetime.now())
gc.enable()


class Run(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.samayam = samayam.Samayam(self.parameters) # create time step economy
        self.utilityHH = False
        self.welfare = False
        self.consumption = False
        self.shockPrintOut = 0
        self.beforeOutput = 0
        self.hhGoodsBefore = 0
        self.interval_shock_time_steps = []

    def create_economy(self):
        #print "began creating economy"
        if self.parameters.interval_productivity_shocks:
            self.create_interval_shock_time_steps()

        if self.parameters.rewire:
            net = rewiring.rewire_degree_preserving(rewiring_factor=10, file_name='network.txt',returnDict=True)
        else:
            net = False
        t0 = time.time()
        
        if self.parameters.rewiring_endogenous:
            if self.parameters.d >= self.parameters.n:
                self.parameters.d = int(self.parameters.n/3)
            self.parameters.n_binomial = copy.copy(self.parameters.n)
            self.parameters.p_binomial = self.parameters.d / self.parameters.n

        self.samayam.economy = economy.Economy(self.parameters)
        if not self.parameters.load_pickle:

            if self.parameters.network_from_file:
                #print 'setting number of firms and HH'
                self.samayam.economy.set_number_of_firms_households()

            self.samayam.economy.create_firms_households()

            #print "created firms and HH"
            if self.parameters.network_from_file:
                self.samayam.economy.create_firms_network_txt_file()
            else:
                self.samayam.economy.create_firms_network_snap() # assign firms buyers and sellers


            self.samayam.economy.set_firm_buyer_seller_values()

            #print "created firms network"
            if self.parameters.connect_firms_without_sellers:
                self.samayam.economy.connect_firms_without_sellers()
            if self.parameters.connect_firms_without_buyers:
                self.samayam.economy.connect_firms_without_buyers()
            self.samayam.economy.make_retail_firms() # make retail firms
            #print "made retail firms"
            self.samayam.economy.create_consumers_firms_network()
            #print "created consumers firms network"
            self.samayam.economy.create_workers_firms_network()
            #print "created workers firm network"
            self.samayam.economy.select_inflation_agents()

            if self.parameters.fixed_labor_shares:
                self.samayam.economy.set_labor_allocation_shares()

            self.samayam.economy.assign_initial_values() # assign initial values to household and firms
            #print "assigned initial values"
            self.samayam.set_initials()
            #self.samayam.set_initials()
            #print "loaded weights"

            if self.parameters.record_variables['economy']['gdp_fixed_shareSec_equilibrium_prices']:
                self.samayam.economy.set_sector_hhShare()
                #print "set sec share in hh"

            if self.parameters.weightsFile:
                self.samayam.economy.set_networkWeights_txt_file()

                #print "set weights"
            
            if self.parameters.CobbDouglas_output_scaled:
                self.samayam.set_CobbDouglas_scaling()
                
            if self.parameters.CobbDouglas_exponent_one:
                self.samayam.set_CobbDouglas_exponents_to_one()

            if self.parameters.sizesFile:
                self.samayam.economy.set_sizes_txt_file()
                #print "set sizes"

            #print "loaded sizes"

            if self.parameters.record_variables['economy']['PCE']:
                self.samayam.economy.loadPCE_weights()

            #print "loaded PCE"

            if self.parameters.record_variables['economy']['CPI']:
                self.samayam.economy.loadCPI_weights()

            self.samayam.economy.fill_retail_suppliers_firms_ID()
            #print "set initials"
            #self.samayam.economy.test_network_consistency()
            #print "check network consistency"

            if self.parameters.record_variables['economy']['gdp_fixed_share_equilibrium_prices']:
                self.samayam.economy.load_fixedSharesGDP()
                self.samayam.economy.set_household_expenditure_share()

            if self.parameters.lockdown or self.parameters.lockdown_as_productivity_shock:
                self.samayam.economy.load_lockdown_prop()
                #print "loading lockdown props"

            if self.parameters.record_variables['economy']['sectoral_output'] or self.parameters.set_sectors:
                self.samayam.economy.set_sectors()

            if self.parameters.record_variables['economy']['eigenValues']:
                network_weights = write_network.networkInformation(economy=self.samayam.economy, returnOrWrite='return')
                eigenValues = eigen.computeEigen(network=network_weights, norm=True)
                self.samayam.economy.eigenValues = eigenValues
                #print "computing Eigen"

        if self.parameters.B_vals_CES:
            self.samayam.economy.set_B_vals_CES()

        count = 0
        countOthers = 0
        for firm in self.samayam.economy.firms.itervalues():
            if firm.retail:
                count += 1
            else:
                countOthers += 1

        #print count, "no of retail firms"
        #print countOthers, "not retail"


        gc.collect()
    
    def create_interval_shock_time_steps(self):
        total_time_steps = self.parameters.num_interval_productivity_shocks * self.parameters.interval_productivity_shocks + 1
        for t in xrange(0,total_time_steps,self.parameters.interval_productivity_shocks):
            self.interval_shock_time_steps.append(t)
        #print self.interval_shock_time_steps, "self.interval_shock_time_steps"

    def shocks(self, time_step):
        if self.parameters.lockdown_as_productivity_shock:
            if time_step == self.parameters.lockdown_productivity_begin:
                self.samayam.begin_lockdown_productivity_shocks()

            if time_step == self.parameters.lockdown_productivity_end:
                #print time_step, "end time step"
                self.samayam.end_lockdown_productivity_shocks()

        if self.parameters.interval_productivity_shocks:
            if time_step in self.interval_shock_time_steps:
                self.samayam.interval_productivity_shock()
        if self.parameters.one_time_productivity_shock:
            if time_step == self.parameters.one_time_productivity_shock_time_step:
                self.samayam.one_time_productivity_shock()                
        """
        if self.parameters.lockdown:
            if time_step in self.parameters.lockdown_timeSteps:
                self.samayam.lockdown_shocks()
        """

        if self.parameters.growing_productivity:
            self.samayam.productivity_growth()
            #print "augmenting productivity"
        if self.parameters.stochastic_productivity:
            self.samayam.productivity_shocks()  # firms recieve a productivity shock
        if self.parameters.stochastic_labor_supply:
            self.samayam.labor_shocks()
        if self.parameters.inflation_change_agents_proportion:
            self.samayam.inflation_agents_change()
        if self.parameters.monetary_shock and time_step == self.parameters.monetary_shock_time_step:
            
            
            money = abs(self.parameters.s) * self.samayam.economy.wealth[-1]
        
            
            if self.parameters.s > 0:
                if self.parameters.monetaryShockWeightsFile:
                    self.samayam.nominal_wealthChange_weights(money, "positive")
                    #print "positive monetary shock: by wealth change weights"
                elif self.parameters.monetaryShock_stochastic:
                    self.samayam.nominal_wealth_increase_stochastic(money,self.parameters.monetaryShock_exponent)

                elif self.parameters.monetaryShock_weightsDistribution:                    
                    #print "positive monetary shock: by weights distribution"
                    self.samayam.nominal_wealth_increaseDistribution(money)
                
                    
                elif self.parameters.monetaryShock_exponent:
                    moneyRandom = self.parameters.randomShare * money
                    moneyExponent = money - moneyRandom
                    
                    self.samayam.nominal_wealth_increase_proportion(moneyExponent)    
                    if moneyRandom>0:
                        self.samayam.nominal_wealth_increase_random(moneyRandom)      

                    #why is this called twice here?

                else:
                    #print "positive monetary shock: general "
                    self.samayam.nominal_wealth_increase(money)
                    
            elif self.parameters.s < 0:
                if self.parameters.monetaryShockWeightsFile:
                    self.samayam.nominal_wealthChange_weights(money, "negative")
                    #print "negative monetary shock:by weights"
                elif self.parameters.monetaryShock_stochastic:
                    self.samayam.nominal_wealth_decrease_stochastic(money,self.parameters.monetaryShock_exponent)
                    
                elif self.parameters.negative_shocks_by_wealth:
                    self.samayam.nominal_wealth_decrease_by_wealth(money)
                    #print "negative monetary shock:by wealth"
                elif self.parameters.monetaryShock_exponent:
                    moneyRandom = self.parameters.randomShare * money                    
                    moneyExponent = money - moneyRandom
                    self.samayam.nominal_wealth_decrease_proportion(moneyExponent)
                    if moneyRandom>0:
                        self.samayam.nominal_wealth_decrease_random(moneyExponent)                    
                else:
                    self.samayam.nominal_wealth_decrease(money)
                    #print "negative monetary shock:general"
            

        if self.parameters.Nominal_GDP_stabilization_policy:
            GDP_change = self.samayam.economy.GDP_series[-1] / self.samayam.economy.GDP_series[-2] - 1
            if abs(GDP_change) < self.parameters.maximum_money_injection_proportion:
                money = self.parameters.NGDP_sensitivity * abs(GDP_change) * self.samayam.economy.nominal_wealth_series[-1]
            else:
                money = self.parameters.NGDP_sensitivity * self.parameters.maximum_money_injection_proportion * self.samayam.economy.nominal_wealth_series[-1]
            if GDP_change > 0:
                self.samayam.nominal_wealth_decrease(money)
            elif GDP_change < 0:
                self.samayam.nominal_wealth_increase(money)
        if self.parameters.inflation_rate != 0:
            if self.parameters.inflation_rate > 0:
                money = self.parameters.inflation_rate * self.samayam.economy.nominal_wealth_series[-1]
                self.samayam.nominal_wealth_increase(money)
            elif self.parameters.inflation_rate < 0:
                money = self.parameters.inflation_rate * self.samayam.economy.nominal_wealth_series[-1]
                self.samayam.nominal_wealth_decrease(money)                              

    def actions(self,transient,timeStep):
        
        lockdown = False
        lockdown_addInventory = False

        #lag_lockdown_time_steps = [i+1 for i in self.parameters.lockdown_timeSteps]
        if self.parameters.lockdown:
            if not transient:
                if timeStep in self.parameters.lockdown_timeSteps:
                    lockdown = True                    
                lockdown_addInventory = True


        self.samayam.firms_produce()  # firms produce output
        #print "produce"

        self.samayam.firms_addOutputInventory(lockdown=lockdown_addInventory)

        self.samayam.compute_demands(lockdown=lockdown)

        #print lockdown, 'lockdown'

        """
        if not transient:
            if self.parameters.lockdown:
                if timeStep in self.parameters.lockdown_timeSteps:
                    self.samayam.lockdown_shocks()
        """
        self.samayam.transfer_demand_information()  # firms and household transfer demand information

        if not transient:
            if self.parameters.labor_mobility:
                self.samayam.labor_mobility()

        """
        if transient:
            self.samayam.compute_prices_transient()
        else:    
            if self.parameters.price_stickiness_type == 'linear':
                self.samayam.compute_prices_linear_stickiness()
            elif self.parameters.price_stickiness_type == 'probabilistic':
                self.samayam.compute_prices_probabilistic_change()
                #print "prob price change"
            else:
                #print "computing flexible price"
                self.samayam.compute_prices_flexible()
            
        """ 
        if transient:
            self.samayam.compute_prices_transient()
        else:
            if timeStep in self.parameters.lockdown_timeSteps:
                self.samayam.set_lockdown_price()
            else:
                if self.parameters.price_stickiness_type == 'linear':
                    self.samayam.compute_prices_linear_stickiness()
                elif self.parameters.price_stickiness_type == 'probabilistic':
                    self.samayam.compute_prices_probabilistic_change()
                    #print "prob price change"
                else:
                    #print "computing flexible price"
                    self.samayam.compute_prices_flexible()
       

            
        self.samayam.transfer_price_information()
       
        """
        if self.parameters.endogenous_probability_price_change:
            self.samayam.compute_probability_price_change()
        """
        
        if self.parameters.production_function == 'CES':
            if self.parameters.weights_dynamic:
                self.samayam.compute_weights()

        if not transient and self.parameters.twoProductFirms:
            self.samayam.compute_allocations_twoProducts()
        else:
            self.samayam.compute_allocations(lockdown=lockdown)  # firms and household compute allocation of output among buyers.

        self.samayam.transfer_inputs_labor()  # firms and household transfers inputs, labor
        self.samayam.transfer_consumption_good()
        self.samayam.update_wealth(lockdown=self.parameters.lockdown)
        self.samayam.compute_utility()
        
        out = 0
        if self.shockPrintOut == 1:   
            for firm in self.samayam.economy.firms.itervalues():
                #print  'id', firm.ID, 'output', firm.output
                out += firm.output
            #print "total output", out
            #print 'output change', 100*(out-self.beforeOutput)/self.beforeOutput
            self.shockPrintOut = 0 
            
        if self.utilityHH:    
            #print self.utilityHH/self.samayam.economy.households[-1].utility, "utility ratio before:after" 
            for firm in self.samayam.economy.firms.itervalues():
                pass
                #print 'after id', firm.ID, 'inputs', firm.input_quantities, 'labor', firm.labor_quantity
            #print "after hh consumer goods", self.samayam.economy.households[-1].goods_quantities
            a = {}
            for i in self.samayam.economy.households[-1].goods_quantities:
                q = self.samayam.economy.households[-1].goods_quantities[i]
                q0 = self.hhGoodsBefore[i]
                change = 100 * (q-q0) / q0
                a[i] = change
            #print "after hh utility", self.samayam.economy.households[-1].utility
            #print "change in hh goods", a
            self.utilityHH = False                
            self.shockPrintOut = 1

        if self.parameters.rewiring_endogenous:
            if transient:
                if self.parameters.rewiring_transient:
                    if timeStep > int(self.parameters.transient_time_steps/4):
                        self.samayam.rewire_endogenous()
            else:                
                if self.parameters.rewiring_timeStep:
                    if timeStep == self.parameters.rewiring_timeStep:
                        self.samayam.rewire_endogenous()
                        #print timeStep
                elif self.parameters.rewiring_stop:
                    if timeStep < self.parameters.rewiring_stop:
                        self.samayam.rewire_endogenous()
                else:
                    self.samayam.rewire_endogenous()

            
        if self.consumption:
            current_consumption =  self.samayam.economy.households[-1].goods_quantities
            diff = []
            for i in current_consumption:
                c1 = current_consumption[i]
                c0 = self.consumption[i]
                d = (c1-c0)/c0
                diff.append(d)
            self.consumption = False
            #print np.mean(diff), "mean change in consumption of goods"
            #print min(diff), max(diff), 'min', 'max'
            count = 0
            for i in diff:
                if i > 0:
                    count += 1
            #print count/len(diff), "prop increase consumption"     
    
    def transient(self):
        if self.parameters.load_pickle:
            pickle_file = "economy.cPickle"
            economy = cPickle.load(open(pickle_file, "rb"))
            parameters = copy.copy(self.parameters)
            economy.parameters = parameters

            for var in self.parameters.record_variables["economy"]:
                if self.parameters.record_variables["economy"][var]:
                    setattr(economy, var, [])                    


            # assign this new parameter to samayam, firm, and household
            for firm in economy.firms.itervalues():
                firm.parameters = parameters
            for household in economy.households.itervalues():
                household.parameters = parameters
            self.samayam.economy = economy

            if self.parameters.record_variables['economy']['PCE']:
                self.samayam.economy.loadPCE_weights()

            if self.parameters.record_variables['economy']['CPI']:
                self.samayam.economy.loadCPI_weights()

            for time_step in xrange(self.parameters.transient_time_steps):                  
                #print time_step, "transient"
                self.actions(transient=True,timeStep=time_step)
                if self.parameters.wealthReserves:
                    self.samayam.adjust_reserves()
                if self.parameters.record_transient_data:
                    self.samayam.record_time_series(time_step=time_step, transient=True)
        else:
            #print "began transient"
            labor_mobility = copy.copy(self.parameters.labor_mobility)
            weights_stickiness_type = copy.copy(self.parameters.weights_stickiness_type)
            prob_price_change = copy.copy(self.parameters.probability_price_change)
            prob_weights_change = copy.copy(self.parameters.probability_weights_change)
            weights_dynamic = copy.copy(self.parameters.weights_dynamic)
            endogenous_probability_weights_change = copy.copy(self.parameters.endogenous_probability_weights_change)
            endogenous_probability_price_change = copy.copy(self.parameters.endogenous_probability_price_change)
            sync_sensitivity = copy.copy(self.parameters.sync_sensitivity)
            stochastic_productivity = copy.copy(self.parameters.stochastic_productivity)
            stochastic_labor_supply = copy.copy(self.parameters.stochastic_labor_supply)
            fixed_labor_shares = copy.copy(self.parameters.fixed_labor_shares)

            self.parameters.probability_price_change = 1
            self.parameters.probability_weights_change = 1
            self.parameters.weights_dynamic = True
            self.parameters.endogenous_probability_weights_change = False
            self.parameters.sync_sensitivity = 0
            self.parameters.stochastic_productivity = False
            self.parameters.stochastic_labor_supply = False
            self.parameters.weights_stickiness_type = None
            self.parameters.labor_mobility = None
            self.parameters.endogenous_probability_price_change = False

            for time_step in xrange(self.parameters.transient_time_steps):
                #print time_step, "transient"
                """
                if time_step % 100 == 0:                    
                    print time_step, "transient"
                """
                #print len(self.samayam.economy.out_of_market_firms_ID), "num firms out of market"
                            
                self.actions(transient=True,timeStep=time_step)
                if self.parameters.wealthReserves:
                    self.samayam.adjust_reserves()
                #if time_step == 1:
                    #self.samayam.test_network_variables_consistency()
                #if random.random() < p.test_value_consistency_probability:
                    #self.samayam.test_network_variables_consistency()
                if self.parameters.record_transient_data:
                    self.samayam.record_time_series(time_step=time_step, transient=True)
            #print "complete transient"

            self.parameters.probability_price_change = prob_price_change
            self.parameters.probability_weights_change = prob_weights_change
            self.parameters.weights_dynamic = weights_dynamic
            self.parameters.endogenous_probability_weights_change = endogenous_probability_weights_change
            self.parameters.sync_sensitivity = sync_sensitivity
            self.parameters.stochastic_productivity = stochastic_productivity
            self.parameters.stochastic_labor_supply = stochastic_labor_supply
            self.parameters.weights_stickiness_type = weights_stickiness_type
            self.parameters.labor_mobility = labor_mobility
            self.parameters.fixed_labor_shares = fixed_labor_shares
            self.parameters.endogenous_probability_price_change = endogenous_probability_price_change

            self.samayam.assign_steady_state()

            economy = self.samayam.economy
            file_name = "economy"
            if self.parameters.write_pickle:
                with open('%s.cPickle' % file_name, 'wb') as econ:
                    cPickle.dump(economy, econ, protocol=cPickle.HIGHEST_PROTOCOL)
                print "wrote pickle economy"

    def time_steps(self):
        print "began time steps"
        
        for time_step in xrange(self.parameters.time_steps): # for each time step in the number of time steps
            #print time_step, "time"
            #print len(self.samayam.economy.out_of_market_firms_ID), "num firms out of market"
            
            """
            if time_step % 100 == 0:
                print time_step, "time step"
            """
            
            
            self.actions(transient=False,timeStep=time_step)
            #print "actions"
            self.shocks(time_step)
            #print "shocks"
            
            if self.parameters.monetary_shock_always and time_step > 0:
                if random.random() < 0.5:
                    s = np.random.normal(self.parameters.monetary_shock_mean, self.parameters.monetary_shock_variance) # need some other random number generator because this is giving the same for all the iterations
                else:
                    s = np.random.normal(self.parameters.monetary_shock_mean, self.parameters.monetary_shock_variance) # need some other random number generator because this is giving the same for all the iterations
                    
                money = abs(s) * self.samayam.economy.wealth[-1]
                #print s, "shock size"
                if s > 0:
                    self.samayam.nominal_wealth_increase_stochastic(money,self.parameters.monetaryShock_exponent)
                #    print "positive"
                else:
                    self.samayam.nominal_wealth_decrease_stochastic(money,self.parameters.monetaryShock_exponent)
                 #   print "negative"
                                                         
            if self.parameters.wealthReserves:
                self.samayam.adjust_reserves()
            if time_step == 1:
                self.samayam.record_firm_stable_price()
                self.samayam.steadyState = True

                #self.samayam.test_network_variables_consistency()
            #if random.random() < p.test_value_consistency_probability:
                #self.samayam.test_network_variables_consistency()
            self.samayam.record_time_series(time_step=time_step, transient=False)
            #print time_step, 'time step'
            #if self.utilityHH:
                #print self.welfare[-1]/self.samayam.economy.welfare_mean[-1], 'weflare ratio before : after'
                #self.utilityHH = False
                #print self.samayam.economy.welfare_mean[-1], self.samayam.economy.households[-1].utility
                #pass
        #print "recording data"
        
        if not self.parameters.data_time_series['economy']:
            self.samayam.economy.record_data(time_step=time_step, transient=False)
        # if time series are not to be recorded, then record data for the economy once all time steps are completed
