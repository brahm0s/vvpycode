from __future__ import division
import pandas as pd
import numpy as np
import ast
import csv
import cPickle
import matplotlib.pyplot as plt
import ast
import os
import collections

def multiParam_mean(csv_file,para_names, variable_name):
    dataDict = collections.OrderedDict({})
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    rows = data.shape[0]
    headers = data.columns.values
    for r in xrange(rows):
        tup = tuple(data.iloc[r][para_names])
        dataDict[tup] = []
    for r in xrange(rows):
        value = data.iloc[r]['mean']
        value = float(value)
        tup = tuple(data.iloc[r][para_names])
        dataDict[tup].append(value)

    mean_dataDict = collections.OrderedDict({})
    for tup in dataDict:
        values = dataDict[tup]
        mean_dataDict[tup] = np.mean(values)

    file_name = 'average_' + variable_name + csv_file
    with open('%s' % file_name, 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        writer_data.writerow(para_names + [variable_name])
        for tup in mean_dataDict:
            val = mean_dataDict[tup]
            writer_data.writerow(list(tup) + [val])






def ratioValues(csv_file,numIndex,denIndex,oneMinus):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    rows = data.shape[0]
    headers = data.columns.values
    parameter_name = headers[0]
    varName = headers[1]
    file_name = 'ratio_' + csv_file
    with open('%s.csv' % file_name, 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        writer_data.writerow([parameter_name] + [varName])
        for r in xrange(rows):
            param = data.iloc[r][0]
            d = data.iloc[r][:]
            d = list(d)
            num = d[numIndex]
            den = d[denIndex]
            if oneMinus:
                ratio = (1-num) / (1-den)
            else:
                ratio = num/den
            writer_data.writerow([param] + [ratio])

def normalize(csv_file, variables, normPosition, transientTime):
    # for given csv file, write normalized values all the variables in separate csv files
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    headers = data.columns.values
    parameter_name = headers[0]
    rows = data.shape[0]
    for var in variables:
        file_name = 'normalized_' + var + '_' + csv_file
        with open('%s' % file_name, 'wb') as data_csv:
            writer_data = csv.writer(data_csv, delimiter=',')
            d = ast.literal_eval(data.iloc[0][var])
            #print d
            #print len(d)

            d = d[transientTime:]
            times = len(d)
            cols = ['t'+ str(i) for i in range(1,times+1)]
            writer_data.writerow([parameter_name] + cols)
            for r in xrange(rows):
                paramValue = data.iloc[r][parameter_name]
                d = ast.literal_eval(data.iloc[r][var])
                d = list(d)
                d = d[transientTime:]
                normValue = d[normPosition]
                d = [float(i)/float(normValue) for i in d]
                writer_data.writerow([paramValue] + d)


def normalize_multiParam(csv_file, parameters, variables, normPosition, transientTime):
    # for given csv file, write normalized values all the variables in separate csv files
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    headers = data.columns.values
    p1 = parameters[0]
    p2 = parameters[1]
    rows = data.shape[0]
    for var in variables:
        file_name = 'normalized_' + var + '_' + csv_file
        with open('%s' % file_name, 'wb') as data_csv:
            writer_data = csv.writer(data_csv, delimiter=',')
            d = ast.literal_eval(data.iloc[0][var])
            d = d[transientTime:]
            times = len(d)
            cols = ['t'+ str(i) for i in range(1,times+1)]
            writer_data.writerow(parameters + cols)
            for r in xrange(rows):
                p1_value = data.iloc[r][p1]
                p2_value = data.iloc[r][p2]
                d = ast.literal_eval(data.iloc[r][var])
                d = list(d)
                d = d[transientTime:]
                normValue = d[normPosition]
                d = [float(i)/float(normValue) for i in d]
                writer_data.writerow([p1_value, p2_value] + d)



def write_pickled_economy(pickle_file, agents_variables, normalize, stable_time_step):
    # takes a pickled economy and writes data int csv file
    # agent_variables is a dictionary with keys as the agent_type and values as a list of variables
    # one csv file is created for each agent_type for each variables, for example firms_prices would be one csv file
    # the variable is normalized if normalize is true; the normalization is based on the variables value at the stable_time_step
    # pickle_file is the name of the pickle file
    economy = cPickle.load(open(pickle_file, "rb"))
    # load the economy
    for agent in agents_variables:
        # for every agent in agents_variables
        file_name = 'pickle_data_' + agent + '_' + pickle_file[:-8]
        # create a csv file name
        agents_data = getattr(economy, agent)
        # get the agents from the economy
        with open('%s.csv' % file_name, 'wb') as data_csv:
            # open a csv file
            writer_data = csv.writer(data_csv, delimiter=',')
            variables = agents_variables[agent]
            writer_data.writerow(['ID'] + variables)
            # the first column is the agents' ID, and following columns are the variables
            for agent in agents_data.itervalues():
                d = dict((var, []) for var in variables)
                d = collections.OrderedDict(d)
                # a dictionary of values of each variable
                for var in variables:
                    v = getattr(agent, var)
                    if normalize:
                        v = [i/v[stable_time_step] for i in v]
                    d[var] = v
                values = d.values()
                writer_data.writerow([agent.ID] + values)

def min_max(csv_file, variable_name, interim_start_time, interim_end_time):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    csv_file = csv_file[:-4]
    headers = data.columns.values
    parameter_name = headers[0]
    new_file_name = csv_file + '_' + 'stats' + '_' + variable_name
    rows = data.shape[0]
    with open('%s.csv' % new_file_name, 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        if csv_file[0:10] == 'multiParam':
            names = csv_file.split('_')
            names = names[1]
            names = list(names)
            writer_data.writerow(names + ['max'] + ['mean'] + ['min'])
        else:
            writer_data.writerow([parameter_name] + ['max'] + ['mean'] + ['min'])
        for r in xrange(rows):
            if str(data.iloc[r][variable_name][1]) != 'n':
                a = True
                for s in str(data.iloc[r][variable_name]):
                    if s == 'n':
                        a = False
                if a:
                    d = ast.literal_eval(data.iloc[r][variable_name])
                    interim_values = d[interim_start_time:interim_end_time]
                    mean_ratio = np.mean(interim_values)
                    max_ratio = max(interim_values)
                    min_ratio = min(interim_values)
                    if csv_file[0:10] == 'multiParam':
                        n = len(names)
                        parameter_value = list(data.iloc[r][0:n])
                        writer_data.writerow(parameter_value + [max_ratio] + [mean_ratio] + [min_ratio])
                    else:
                        parameter_value = data.iloc[r][0]
                        writer_data.writerow([parameter_value] + [max_ratio] + [mean_ratio] + [min_ratio])

def sum_ratio(csv_file, v1, v2, interim_start, interim_end, save_name, threshold):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    headers = data.columns.values
    parameter_name = headers[0]
    rows = data.shape[0]
    file_name = csv_file[:-4] + '_' + 'ratio' + '_' + v1 + v2
    if save_name is not None:
        file_name += '_'
        file_name += save_name
    with open('%s.csv' % file_name, 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        writer_data.writerow([parameter_name] + ['ratio'])
        for r in xrange(rows):
            if str(data.iloc[r][v1][1]) != 'n':
                a = True
                for s in str(data.iloc[r][v1]):
                    if s == 'n':
                        a = False
                for s in str(data.iloc[r][v2]):
                    if s == 'n':
                        a = False
                if a:
                    d1 = ast.literal_eval(data.iloc[r][v1])
                    d2 = ast.literal_eval(data.iloc[r][v2])
                    d1 = sum(d1[interim_start:interim_end])
                    d2 = sum(d2[interim_start:interim_end])
                    if threshold is not None:
                        if d1 > threshold:
                            ratio = d1/d2
                        else:
                            ratio = 0
                    else:
                        ratio = d1 / d2
                    parameter_value = data.iloc[r][0]
                    writer_data.writerow([parameter_value] + [ratio])

def convergence_time1(csv_file, thresholds, variable, save_name, beginStep):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    headers = data.columns.values
    parameter_name = headers[0]
    rows = data.shape[0]
    file_name = csv_file[:-4] + '_' + 'convergence'

    def compute_convergence_time(d, t):
        for value in d:
            if value < 10 ** -t:
                return d.index(value)

    if save_name is not None:
        file_name += '_'
        file_name += save_name
    with open('%s.csv' % file_name, 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        writer_data.writerow([parameter_name] + thresholds)
        for r in xrange(rows):
            if str(data.iloc[r][variable][1]) != 'n':
                a = True
                for s in str(data.iloc[r][variable]):
                    if s == 'n':
                        a = False
                if a:
                    d = ast.literal_eval(data.iloc[r][variable])
                    d = d[beginStep:]
                    convergence_time = []
                    for t in thresholds:
                        convergence_time.append(compute_convergence_time(d, t))
                    parameter_value = data.iloc[r][0]
                    writer_data.writerow([parameter_value] + convergence_time)

def convergence_time(csv_file, thresholds, variable, save_name):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    headers = data.columns.values
    parameter_name = headers[0]
    rows = data.shape[0]
    file_name = csv_file[:-4] + '_' + 'convergence'

    def compute_convergence_time(d, t):
        for value in d:
            if value < 10 ** -t:
                return d.index(value)

    if save_name is not None:
        file_name += '_'
        file_name += save_name
    with open('%s.csv' % file_name, 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        writer_data.writerow([parameter_name] + thresholds)
        for r in xrange(rows):
            if str(data.iloc[r][variable][1]) != 'n':
                a = True
                for s in str(data.iloc[r][variable]):
                    if s == 'n':
                        a = False
                if a:
                    d = ast.literal_eval(data.iloc[r][variable])
                    convergence_time = []
                    for t in thresholds:
                        convergence_time.append(compute_convergence_time(d, t))
                    parameter_value = data.iloc[r][0]
                    writer_data.writerow([parameter_value] + convergence_time)

def convergence_afterShock_time(csv_file, thresholds, variable, save_name, shock_time,otherVariable):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    headers = data.columns.values
    parameter_name = headers[0]
    rows = data.shape[0]
    file_name = 'convergence_afterShock'

    def compute_convergence_time(d, t):
        for value in d:
            if value < 10 ** -t:
                return d.index(value)

    if save_name is not None:
        file_name += '_'
        file_name += save_name
    with open('%s.csv' % file_name, 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        writer_data.writerow([parameter_name] + thresholds + ['One', 'Two', 'Three'])
        for r in xrange(rows):
            if str(data.iloc[r][variable][1]) != 'n':
                a = True
                for s in str(data.iloc[r][variable]):
                    if s == 'n':
                        a = False
                if a:
                    d = ast.literal_eval(data.iloc[r][variable])
                    d = d[shock_time:]
                    other = ast.literal_eval(data.iloc[r][otherVariable])
                    if type(other) is not list:
                        other = [other]
                    convergence_time = []
                    for t in thresholds:
                        convergence_time.append(compute_convergence_time(d, t))
                    parameter_value = data.iloc[r][0]
                    writer_data.writerow([parameter_value] + convergence_time + other)



