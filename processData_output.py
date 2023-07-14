from __future__ import division
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import os
import processData
import pandas as pd
import csv
import numpy as np

def writeChanges():

    files = ['normalized_finalOutput_equilibrium_prices_linear_price_stickiness_old_share_economy_network_linear_CD_smallTheta_negative.csv',
            'normalized_finalOutput_equilibrium_prices_linear_price_stickiness_old_share_economy_network_linear_CD_smallTheta_positive.csv',
            'normalized_finalOutput_equilibrium_prices_monetaryShock_exponent_economy_network_False_CD_smallTheta_negative.csv',
            'normalized_finalOutput_equilibrium_prices_monetaryShock_exponent_economy_network_False_CD_smallTheta_positive.csv',
            'normalized_finalOutput_equilibrium_prices_s_economy_network_False_CD_largeTheta_negative.csv',
            'normalized_finalOutput_equilibrium_prices_s_economy_network_False_CD_smallTheta_negative.csv',
            'normalized_finalOutput_equilibrium_prices_s_economy_network_linear_CD_smallTheta_negative.csv']


    parameter = {'normalized_finalOutput_equilibrium_prices_linear_price_stickiness_old_share_economy_network_linear_CD_smallTheta_negative.csv':'linear_price_stickiness_old_share',
                'normalized_finalOutput_equilibrium_prices_linear_price_stickiness_old_share_economy_network_linear_CD_smallTheta_positive.csv':'linear_price_stickiness_old_share',
                'normalized_finalOutput_equilibrium_prices_monetaryShock_exponent_economy_network_False_CD_smallTheta_negative.csv':'monetaryShock_exponent',
                'normalized_finalOutput_equilibrium_prices_monetaryShock_exponent_economy_network_False_CD_smallTheta_positive.csv': 'monetaryShock_exponent',
                'normalized_finalOutput_equilibrium_prices_s_economy_network_False_CD_largeTheta_negative.csv':'s',
                'normalized_finalOutput_equilibrium_prices_s_economy_network_False_CD_smallTheta_negative.csv':'s',
                'normalized_finalOutput_equilibrium_prices_s_economy_network_linear_CD_smallTheta_negative.csv':'s'}
     
    for csvfile in files:
        data = pd.read_csv(csvfile)
        data = pd.DataFrame(data)
        rows = data.shape[0]        
        name = csvfile[42:]
        name = 'equiFinOutput_' + name
        with open(name,'w') as file:     
            writer_data = csv.writer(file, delimiter=',')
            param = parameter[csvfile]                      
            writer_data.writerow([param, 'finalOutput_equilibrium_prices'])
            for r in xrange(rows):
                p = data.iloc[r][param]
                output = data.iloc[r]['t4'] - 1
                writer_data.writerow([p, output])

    for csvfile in files:
        data = pd.read_csv(csvfile)
        data = pd.DataFrame(data)
        rows = data.shape[0]        
        name = csvfile[42:]
        name = 'SR_equiFinOutput_' + name
        with open(name,'w') as file:     
            writer_data = csv.writer(file, delimiter=',')
            param = parameter[csvfile]                      
            writer_data.writerow([param, 'finalOutput_equilibrium_prices'])
            for r in xrange(rows):
                p = data.iloc[r][param]
                outputs = []
                ts = ['t4','t5']
                for t in ts:
                    output = data.iloc[r][t] - 1
                    outputs.append(output)
                meanOutput  = np.mean(outputs)
                writer_data.writerow([p, meanOutput])
    


def normalizeOutputIndex_firms(transientTime):
    
    files = ['linear_price_stickiness_old_share_economy_network_linear_CD_smallTheta_negative.csv',
            'linear_price_stickiness_old_share_economy_network_linear_CD_smallTheta_positive.csv',
            'monetaryShock_exponent_economy_network_False_CD_smallTheta_negative.csv',
            'monetaryShock_exponent_economy_network_False_CD_smallTheta_positive.csv',
            's_economy_network_False_CD_largeTheta_negative.csv',
            's_economy_network_False_CD_smallTheta_negative.csv',
            's_economy_network_linear_CD_smallTheta_negative.csv']
    



    for csv_file in files:
        variables = ['finalOutput_equilibrium_prices']
        normPosition = 0
        processData.normalize(csv_file, variables, normPosition, transientTime)


def process():
    directory_name = 'data_output'
    os.chdir(directory_name)
    transientTime = 0
    normalizeOutputIndex_firms(transientTime)
    writeChanges()
    os.chdir('..')

process()
