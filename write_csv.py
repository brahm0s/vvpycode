from __future__ import division
import datetime
import os
import cPickle
import csv


def write_csv_economy(parameter_name, model_name, set_number, network, variables):
    directory = 'data'+ '_' + model_name + '/' + set_number + '/' + network + '/' + parameter_name
    os.chdir(directory)
    with open('economies.cPickle', 'rb') as data:
         economies = cPickle.load(data)
    column_names = [parameter_name] + variables
    with open('economy.csv', 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        writer_data.writerow(column_names)
        for economy in economies:
            p = getattr(economy.parameters, parameter_name)
            variables_data = []
            for variable in variables:
                v = getattr(economy, variable)
                variables_data.append(v)
            writer_data.writerow([p] + variables_data)
    for sub_directory in xrange(4):
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')


def write_csv_firm(parameter_name, model_name, set_number, network, variable_name, time_series):
    directory = 'data' + '_' + model_name + '/' + set_number + '/' + network + '/' + parameter_name
    os.chdir(directory)
    with open('economies.cPickle', 'rb') as data:
        economies = cPickle.load(data)
    firms_id = economies[0].firms_ID
    column_names = [parameter_name] + firms_id
    file_name = 'firm' + '_' + variable_name

    with open('%s.csv' % file_name, 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        writer_data.writerow(column_names)
        if time_series:
            variable_name = variable_name + '_time_series'
        for economy in economies:
            p = getattr(economy.parameters, parameter_name)
            data = []
            for firm in economy.firms.itervalues():
                v = getattr(firm, variable_name)
                data.append(v)
            writer_data.writerow([p] + data)
        for sub_directory in xrange(4):
            os.chdir('..')
            os.chdir('..')
            os.chdir('..')
            os.chdir('..')


def write_csv_households(parameter_name, model_name, set_number, network, variable_name, time_series):
    directory = 'data' + '_' + model_name + '/' + set_number + '/' + network + '/' + parameter_name
    os.chdir(directory)
    with open('economies.cPickle', 'rb') as data:
        economies = cPickle.load(data)
    household_id = economies[0].household_ID
    column_names = [parameter_name] + household_id
    file_name = 'household' + '_' + variable_name

    with open('%s.csv' % file_name, 'wb') as data_csv:
        writer_data = csv.writer(data_csv, delimiter=',')
        writer_data.writerow(column_names)
        if time_series:
            variable_name = variable_name + '_time_series'
        for economy in economies:
            p = getattr(economy.parameters, parameter_name)
            data = []
            for firm in economy.households.itervalues():
                v = getattr(firm, variable_name)
                data.append(v)
            writer_data.writerow([p] + data)
        for sub_directory in xrange(4):
            os.chdir('..')
            os.chdir('..')
            os.chdir('..')
            os.chdir('..')








"""
write_csv_economy(parameter_name='Cobb_Douglas_exponent_labor',
                  model_name='cantillon',
                  set_number='set_1',
                  network='scale_free_directional',
                  variables=['gdp','consumer_price'])
"""

write_csv_firm(parameter_name='Cobb_Douglas_exponent_labor',
               model_name='cantillon',
               set_number='set_1',
               network='scale_free_directional',
               variable_name='price',
               time_series=True)

