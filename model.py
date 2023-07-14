from __future__ import division
import run
import numpy as np
import parameters as p
import math
from multiprocessing import Pool
import gc
import copy
import itertools
gc.enable()


def single_run(arguments):
    parameter_name = arguments[0]
    parameter_value = arguments[1]
    variables_dict = arguments[2]
    other_parameters = arguments[3]
    time_series = arguments[4]
    parameters = p.Parameters()
    parameters.data_time_series = time_series
    if other_parameters:
        for name in other_parameters:
            value = other_parameters[name]
            setattr(parameters, name, value)

    if type(parameter_name) is not tuple:
        setattr(parameters, parameter_name, parameter_value)
    else:
        parameter_value = dict((x,y) for x, y in parameter_value)
        for name in parameter_name:
            val = parameter_value[name]
            setattr(parameters, name, val)

    for k in variables_dict:
        if variables_dict[k] is not None:
            for v in variables_dict[k]:
                parameters.record_variables[k][v] = True


    run_instance = run.Run(parameters)
    #run_instance.set_directory()
    run_instance.create_economy()
    #run_instance.back_directory()
    run_instance.transient()
    run_instance.time_steps()


    data = {}

    for k in variables_dict:
        if variables_dict[k] is not None:
            data[k] = dict((v, None) for v in variables_dict[k])
        else:
            data[k] = None
            
    
    for system_or_agent in variables_dict:
        variable_names = variables_dict[system_or_agent]
        if variable_names is not None:
            if system_or_agent == 'economy':
                for name in variable_names:
                    data['economy'][name] = getattr(run_instance.samayam.economy, name)
            elif system_or_agent == 'firm':
                for name in variable_names:
                    d = dict((ID, None) for ID in run_instance.samayam.economy.firms_ID)
                    for firm in run_instance.samayam.economy.firms.itervalues():
                        if time_series['firm']:
                            name_time_series = name + '_' + 'time_series'
                            d[firm.ID] = getattr(run_instance.samayam.economy.firms[firm.ID], name_time_series)
                        else:
                            d[firm.ID] = [getattr(run_instance.samayam.economy.firms[firm.ID], name)]
                    data['firm'][name] = d
            elif system_or_agent == 'household':
                for name in variable_names:
                    d = dict((ID, None) for ID in run_instance.samayam.economy.households_ID)
                    for household in run_instance.samayam.economy.households.itervalues():
                        if time_series['household']:
                            name_time_series = name + '_' + 'time_series'
                            d[household.ID] = getattr(run_instance.samayam.economy.households[household.ID], name_time_series)
                        else:
                            d[household.ID] = [getattr(run_instance.samayam.economy.households[household.ID], name)]
                    data['household'][name] = d
    return data

def parallel_runs(time_series, parameter_name, parameter_range, parameter_increment, variables_dict, other_parameters,
                  iterations, cores):

    all_data = {'economy': None, 'firm': None, 'household': None}
    if parameter_name is not 'network_type':
        if type(parameter_name) is not tuple:
            parameter_values = []

            if type(parameter_increment) is not list:
                start = parameter_range[0]
                end = parameter_range[1]

            if type(parameter_increment) == int:
                parameter_values = xrange(start, end, parameter_increment)
            elif type(parameter_increment) == float:
                parameter_values = np.arange(start, end, parameter_increment)
            elif parameter_increment is None:
                parameter_values = [True, False]
            elif type(parameter_increment) is list:
                 parameter_values = parameter_increment
        else:
            start = {}
            end = {}
            for name in parameter_name:
                first = parameter_range[name][0]
                last = parameter_range[name][1]
                start[name] = first
                end[name] = last
            #create parameter values dictionary for each parameter
            parameter_values_each = {}
            for name in parameter_name:
                parameter_values_each[name] = []

            for name in parameter_name:
                increment = parameter_increment[name]
                if type(increment) == int:
                    parameter_values_each[name] = xrange(start[name], end[name], increment)
                elif type(increment) == float:
                    parameter_values_each[name] = np.arange(start[name], end[name], increment)
                elif increment is None:
                    parameter_values_each[name] = [True, False]

            parameter_values_ = [dict(zip(parameter_values_each.keys(), a)) for a in itertools.product(*parameter_values_each.values())]
            parameter_values = []
            for v in parameter_values_:
                parameter_values.append(tuple(sorted(v.iteritems())))

            # list of combinations of parameter values
            #param_possibilities = []
            #for name in parameter_values_each:
             #   temp = []
              #  for val in parameter_values_each[name]:
               #     temp.append((name, val))
               # param_possibilities.append(temp)
            #parameter_values = list(itertools.product(*param_possibilities))
    else:
        parameter_values = parameter_range

    
    for k in all_data:
        variables_list = variables_dict[k]         
        if variables_list is not None:
            variable_names_dictionary = dict((name, []) for name in variables_list)
            d = dict((value, copy.deepcopy(variable_names_dictionary)) for value in parameter_values)
            all_data[k] = d

    
    for value in parameter_values:
        print value, parameter_name
        arguments_for_single_run = [parameter_name, value, variables_dict, other_parameters, time_series]
        iterable_list = []
        short_iterable_list = []
        length_short_list = iterations % cores
        for i in xrange(cores):
            iterable_list.append(arguments_for_single_run)
        for i in xrange(length_short_list):
            short_iterable_list.append(arguments_for_single_run)
        loops = int(math.floor(iterations / cores))
        count = {'economy': 0, 'firm': 0, 'household': 0}
        for i in xrange(loops):
            pool = Pool(cores)
            #print i, "iteration"
            gc.collect()
            #print "iterable_list", iterable_list
            #print len(iterable_list), "lenght of iterable list"
            data = pool.map(single_run, iterable_list)
            pool.close()  # I feel this should be added
            pool.join() # I feel this should be added 
            for iteration in data:
                for system_firm in iteration.keys():
                    if variables_dict[system_firm] is not None:
                        for name in variables_dict[system_firm]:
                            d = (count[system_firm], iteration[system_firm][name])
                            all_data[system_firm][value][name].append(d)
                            count[system_firm] += 1

                            
        if len(short_iterable_list):
            pool = Pool(cores)
            data = pool.map(single_run, short_iterable_list)
            pool.close()
            pool.join()
            for iteration in data:
                for system_firm in iteration.keys():
                    for name in variables_dict[system_firm]:
                        d = (count[system_firm], iteration[system_firm][name])
                        all_data[system_firm][value][name].append(d)
                        count[system_firm] += 1
    return all_data

def generate_data(time_series, parameter_names, parameter_ranges, parameter_increments, variables,
                  other_parameters_, iterations_, cores_):
    all_data = dict((name, []) for name in parameter_names)
    for name in parameter_names:
        print name, "parameter name"
        range = parameter_ranges[name]
        increments = parameter_increments[name]
        variables_dict = variables[name]
        other_parameters = other_parameters_[name]
        iterations = iterations_[name]
        cores = cores_[name]

        data = parallel_runs(time_series=time_series, parameter_name=name, parameter_range=range, parameter_increment=increments,
                            variables_dict=variables_dict, other_parameters=other_parameters, iterations=iterations,
                            cores=cores)

        all_data[name] = data

    return other_parameters_, all_data


