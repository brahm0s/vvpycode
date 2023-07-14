"""
Contact - vipin.veetil@gmail.com
Language - Python 2.7.10
"""

from __future__ import division

class Parameters(object):
    def __init__(self):
        self.B_vals_CES = False
        self.files_directory = False
        self.twoProductFirms = False
        self.addFileName = False
        self.write_pickle = False
        self.load_pickle = False
        self.rewire = False
        self.set_household_expenditure_shares = False
        self.hhPermanentIncomeFactor = False
        self.hhSaving = False
        self.seedSaving = False
        self.wageStickiness = False
        self.seedExponent = None
        self.n = 1000 # number of firms
        self.hf = 1 # ratio of number of households to number of firms
        self.number_of_households = self.n * self.hf
        self.representative_household = True
        self.all_firms_retail = True # 'file' or True
        self.d = 2 # mean degree of production network
        self.proportion_retail_firms = 0.05 # the proportion of firms that supply goods to the household
        self.sigma = 0.2 # CES exponent
        self.alpha = 0.25 # Cobb Douglas exponent of production function
        self.household_preference_homogeneous = True # the preference for the different goods; True if exponents of Cobb-Dogulas Utility function is same for all goods; False otherwise
        self.production_function = "CD" # "CD = Cobb Douglas,  CES, substitutes, 'CES_withB_vals'
        self.Cobb_Douglas_production_function_homogeneous = True
        self.smoothenPrice = False # firms attempt to smoothen price changes by building up and drawing down inventories
        self.wealthReserves = False
        self.psi = 0.4 # mean of distribution when r is randomvariable
        self.gamma = 0.2 # determines the draw of r if r is a random variable from a uniform distribution [self.psi - gamma, self.psi+gamma], gamma is in (0,0.5)
        self.firm_productivity_homogeneous = True
        self.CES_output_scaled = True
        self.CobbDouglas_output_scaled = False
        self.CobbDouglas_exponent_one = False
        self.fixed_labor_shares = False
        self.set_sectors = False

        self.lockdown = False
        self.lockdown_timeSteps = []
        self.lockdown_cutOff = 0.9
        self.lockdownShock_multiple = 1
        self.lockdown_inventory_share = 1

        self.lockdown_as_productivity_shock = False
        self.lockdown_productivity_begin = None
        self.lockdown_productivity_end = None

        self.one_time_productivity_shock = False
        self.one_time_productivity_shock_time_step = 10
        self.one_time_productivity_shock_mean = 0
        self.one_time_productivity_shock_std = 0.1
        
        #self.one_time_productivity_shock_min = 0
        #self.one_time_productivity_shock_max = 0.01

        self.weightsFile = False
        self.sizesFile = False
        self.firms_pce_file= False

        self.monetaryShockWeightsFile = False # the weights by which monetary shock is divided among the sectors
        self.monetaryShock_reverseWeight = False # whether the weights by which monetary shock is to be divided is treated as reverse of the file
        self.monetaryShockWeightsFile_Column = False
        self.monetaryShock_exclude = []

        self.monetaryShock_exponent = False # each firms shock is proportion of its capital raised to the exponent
        self.monetaryShock_exponent_fileSizes = False # get the sizes to which to apply the exponent from a file
        self.monetaryShock_exponent_stochastic = False # introduce stochasticity into the exponent used for monetary shock; to make true put [a,b] of uniform variable for shocks
        # as the exponent goes from 1 to 0 small firms are more disporportionately hurt.....
        self.min_consumption_goods = 3
        self.max_consumption_goods = 3
        self.network_from_file = False
        self.rewiring_endogenous = False
        self.transient_time_steps = 100
        self.time_steps = 100

        self.interval_productivity_shocks = False
        self.num_interval_productivity_shocks = 100

        self.max_workers_firms_without_input_sellers = 3

        self.network_type = "SF" #ER erdos renyi random network, SF = scale free network, B = balanced network, powerlaw = "scale free using a powerlaw exponent"
        self.powerlaw_exponent = 1.61
        self.scale_free_network_symmetry = 0.5
        self.number_of_retail_firms_actual = 0

        self.rho_omega_threshold = 10 ** -7

        self.connect_firms_without_buyers = False
        self.connect_firms_without_sellers = False
        
        self.labor_mobility = False
        self.price_stickiness_type = False #  "linear", "probabilistic", or "False" meaning flexible
        self.linear_price_stickiness_old_share = 0.5 # the value denominates the share of last period price

        self.probability_price_change = 0.5     
        self.var_prob_price_change = False    
        
        self.endogenous_probability_price_change = False
        self.price_sensitivity = 1

        self.weights_dynamic = True
        self.weights_stickiness_type = False # linear, probabilistic, or False
        self.linear_weights_stickiness_old_share = 0.5
        self.probability_weights_change = 1
        self.endogenous_probability_weights_change = False
        self.weights_sensitivity = 1

        self.sync_sensitivity = 0
        self.sync_direction = 'sellers'

        
        self.growing_productivity = False
        self.productivity_mu = 0
        self.productivity_sigma = 0.1
        self.productivity_multiplicative = False
        self.productivity_labor = False

        self.stochastic_productivity = False
        self.productivity_shock_mean = 0
        self.productivity_shock_std = 0.1
        self.productivity_exponential = 0.1

        self.stochastic_labor_supply = False
        self.labor_shock_mean = 0
        self.labor_shock_std = 0.1

        self.inflation_rate = 0
        self.inflation_change_agents_proportion = 0
        self.Nominal_GDP_stabilization_policy = False

        self.NGDP_sensitivity = 1.0
        self.maximum_money_injection_proportion = 0.25 # the maximum proportion of money injected or extracted in response to changes in nominal GDP

        self.monetary_shock_always = False
        self.monetary_shock_mean = 0
        self.monetary_shock_variance = 0.01
        
        self.monetary_shock = False
        self.monetaryShock_stochastic = False
        self.randomShare = 0 # share of the monetary shock assinged to the random component
        self.randomWeightShare = 0 # the share of random weight in the implementation of monetary shock
        self.noise_multiplicative = False # multiply the weight of each by some noise
        self.a_noiseMul = 0.3
        self.b_noiseMul = 0.7
        self.s = 0.01 # size of monetary shock
        self.g = 0.1 # proportion of agents who recieve monetary shock
        self.z = 0.25 # upper limit on decrease in wealth with negative monetary shock
        self.monetaryShock_weightsDistribution = False # draw the division or weights of the monetary shocks among firms from a distribution
        self.theta = 0 # parameter of the normal distribution from which weights are draw N(0, theta)
        self.monetary_shock_time_step = 50
        self.money_injection_agents = "firms" # "firms", "households", "all", "single_firm"
        self.inflation_agents_ID = None #'file', 'list', or 'None'
        self.negative_shocks_by_wealth = False
        self.housingID = None
        self.data_time_series = {'economy': True, 'firm': False, 'household': False}
        self.record_distribution_time_steps = False
        self.record_intermediate_share_deviation_time_steps = False
        
        self.record_param = {'log_size_distribution':{'max_size':None,'min_size':None,'logV':None,'mul':None},'log_deg_distribution':{'max_deg':None,'min_deg':None,'logV':None}}
        self.record_variables = {'economy': {'intermediate_share_currentOutput':False,
                                            'intermediate_share':False,
                                            'inventory_prop':False,
                                            'intermediate_share_firms':False,
                                            'inverse_size_mean':False,
                                            'firm_volatility':False,
                                            'priceLevel':False,
                                            'unemployment_rate':False,
                                            'sectoral_output':False,
                                            'consumer_demand':False,
                                            'intermediate_demand':False,
                                            'eigenValues':False,
                                            #'output_consumption':False,
                                            'finalOutput':False,
                                            'finalOutput_equilibrium_prices':False,
                                            'finalOutput_consumer_equilibrium_prices':False,
                                            'sumOutput':False, # large firms greater share,
                                            'CPI':False,
                                             'CPIsansHousing':False,
                                             'PCE':False,
                                            'mean_price_change': False,
                                            'gdp': False,
                                            'gdp_equilibrium_prices':False,
                                            'gdp_fixed_share_equilibrium_prices':False,
                                            'gdp_fixed_shareSec_equilibrium_prices':False,
                                            'sum_output_equilibrium_prices':False,
                                             'cp': False,
                                             'pp': False,
                                             'cg': False,
                                             'pg': False,
                                             'cv': False, # coefficient of variation of cross section of normalized prices
                                             'V': False,
                                             'wealth': False,
                                             'price_distribution': False,
                                             'zeta': False, # norm of price decrease at given time step compared to previous time step
                                             'eta': False, # norm of price changes at given time step compared to previous time step
                                             'zetaC': False, # norm of cumulative price decrease at given time step compared to previous steady state
                                             'etaC': False, # norm of cumulative  price changes at given time step compared to previous steady state
                                             'rho': False, # proportion of prices below previous steady state at a given time step
                                             'omega':False, # proportion of prices above previous steady state at a given time step
                                             'rhoC': False,
                                             'omegaC': False,
                                             'degree_distribution_in': False,
                                             'degree_distribution_out': False,
                                             'size_distribution':False,
                                             'out_of_market_firms': False,
                                             'links_changed': False,
                                             'welfare_mean':False,
                                             'weighted_consumption':False,
                                             'welfare_cv':False,
                                             'network_weights_changes':False,
                                             'inventoryCarry': False,
                                             'wealthCarry': False,
                                             'mean_firm_size_change':False,
                                             'log_size_distribution':False,
                                             'log_deg_distribution':False},
                                 'firm': {'price': False,
                                          'total_output': False,
                                          'total_demand':False,
                                          'wealth': False,
                                          'output_quantity_change':False,
                                          'price_change':False,
                                          'ss_price_change': False,
                                          'ss_output_change': False,
                                          'ss_input_demands':False,
                                          'size_change':False,
                                          'output_sold':False},
                                 'household': {'wage': False,
                                               'utility': False,
                                               'wealth': False,
                                               'saving':False}
                                 }

        self.record_transient_data = False
        self.test_value_consistency_probability = 0.01
        self.price_change_prop_record = 0.01

        """ rewiring parameters """
        self.rewiring_transient = False # if true then rewire during transient
        self.probability_firm_input_seller_change = 0.1
        self.probability_firm_entry = 0.1
        
        self.n_binomial = self.n
        self.p_binomial = self.d / self.n

        self.rewiring_timeStep = False # if numerical value, then rewiring only at this time step
        self.rewiring_stop = False # if numerical value, then stop rewiring after that time step
