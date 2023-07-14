from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import random as random

import numpy as np
from math import log
import math
import ast
from itertools import chain
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
#import mpl_toolkits

import collections
import os

def countourSurface(csv_file, x_name, y_name, z_name, x_label, y_label,title,
                    save_name, fontSize_title, fontSize_xlabel, fontSize_ylabel, fontSize_ticks, child_directory,
                    zeroYValue,maxYValue):


    data = pd.read_csv(csv_file)  # read the csv file
    data = pd.DataFrame(data)
    x = list(data.iloc[:][x_name])
    y = list(data.iloc[:][y_name])

    x = list(set(x))
    x.sort()
    y = list(set(y))
    y.sort()

    matrix = np.zeros((len(y), len(x)))

    data_dict = {}

    rows = data.shape[0]
    for r in xrange(rows):
        val = data.iloc[r][z_name]
        s = data.iloc[r][x_name]
        g = data.iloc[r][y_name]
        data_dict[(s,g)] = val

    countY = len(y) -1
    for i in y:
        countX = 0
        for j in x:
            k = (j,i)
            val = data_dict[k]
            matrix[countY,countX] = val
            countX += 1
        countY -= 1

    first_Xvalue=min(x)
    last_Xvalue=max(x)

    if zeroYValue:
        first_Yvalue = 0
    else:
        first_Yvalue=min(y)

    last_Yvalue = max(y)
    if maxYValue:
        last_Yvalue = maxYValue




    plt.imshow(matrix, interpolation='bilinear', aspect='auto', cmap='Blues', extent=[first_Xvalue, last_Xvalue, first_Yvalue, last_Yvalue])
    plt.grid()

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)

    plt.xlabel(x_label, fontsize=fontSize_xlabel)
    plt.ylabel(y_label, fontsize=fontSize_ylabel)
    plt.suptitle(title, fontsize=fontSize_title)
    plt.tick_params(axis='both', labelsize=fontSize_ticks)


    file_name = 'heat_map'  # + '_' + y_name
    file_name += '_'
    file_name += csv_file[:-4]

    if save_name is not None:
        file_name += '_'
        file_name += str(save_name)
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)

    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')


def fourVariable_colorPlot(csv_file, parameterNames, variable_name, save_name):
    data = pd.read_csv(csv_file)  # read the csv file
    data = pd.DataFrame(data)  # convert it to data frame
    x_name = parameterNames[0]
    y_name = parameterNames[1]
    z_name = parameterNames[2]

    x = data.iloc[:][x_name]
    y = data.iloc[:][y_name]
    z = data.iloc[:][z_name]
    c = data.iloc[:][variable_name]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #x = np.random.standard_normal(100)
    #y = np.random.standard_normal(100)
    #z = np.random.standard_normal(100)
    #c = np.random.standard_normal(100)

    ax.scatter(x, y, z, c=c, cmap=plt.hot())
    plt.show()

def multiNetworks_timePlots(networks, names, variables, time_steps, time_shift, parameter_names,
                            parameter_values,x_label,y_labels, titles,
                            fontSize_title, fontSize_xlabel, fontSize_ylabel, fontSize_ticks,
                            child_directory, colors, legendSize):
    for n in parameter_names:
        parameter_value = parameter_values[n]
        for var in variables:
            files = {}
            dataNetworks = {}
            for network in networks:
                dataNetworks[network] = []

                name = n + '_economy_' + network + '_CD_static.csv'
                files[network] = name
                data = pd.read_csv(name)  # read the csv file
                data = pd.DataFrame(data)  # convert it to data frame
                data = data.loc[data[n] <= (parameter_value + 10 ** -10)]
                data = data.loc[data[n] >= (parameter_value - 10 ** -10)]
                if var == 'phi':
                    v1 = data.iloc[:]['zetaC']
                    v1 = list(v1)
                    v1 = [ast.literal_eval(d) for d in v1]
                    v1_byTime = map(list, zip(*v1))

                    v2 = data.iloc[:]['etaC']
                    v2 = list(v2)
                    v2 = [ast.literal_eval(d) for d in v2]
                    v2_byTime = map(list, zip(*v2))

                    data_byTime = []
                    for t in xrange(len(v1_byTime)):
                        d1 = v1_byTime[t]
                        d2 = v2_byTime[t]
                        ratioData = []
                        for k in xrange(len(d1)):
                            a1 = d1[k]
                            a2 = d2[k]
                            if abs(a2) > 10 ** -7:
                                r = a1 / a2
                                ratioData.append(r)
                            else:
                                ratioData.append(0)
                        data_byTime.append(ratioData)
                else:
                    variables_data = data.iloc[:][var]
                    variables_data = list(variables_data)
                    variables_data = [ast.literal_eval(d) for d in variables_data]
                    data_byTime = map(list, zip(*variables_data))

                for time_step in data_byTime:
                    meanValue = np.mean(time_step)
                    dataNetworks[network].append(meanValue)

            for network in networks:
                d = dataNetworks[network]
                timeSteps = time_steps[var]
                t0 = timeSteps[0]
                t1 = timeSteps[1] + 1
                d = d[t0:t1]
                if var == 'cv':
                    xValues = range(0 - time_shift, len(d) - time_shift)
                else:
                    xValues = range(0 - (time_shift + 1), len(d) - (time_shift + 1))

                color = colors[network]
                name = names[network]
                plt.plot(xValues, d, color=color, label=name)
                plt.xlabel(x_label, fontsize=fontSize_xlabel)
                y_label = y_labels[var]
                plt.ylabel(y_label, fontsize=fontSize_ylabel)
                plt.xlim([min(xValues),max(xValues)])
            plt.legend(fontsize=legendSize)
            title = titles[var]
            plt.suptitle(title, fontsize=fontSize_title)
            plt.tick_params(axis='both', labelsize=fontSize_ticks)

            file_name = 'networkCompare_'
            file_name += n
            s = str(parameter_value)
            s = s.replace(".", "")
            file_name += s
            file_name += '_'
            file_name += var

            if not os.path.exists(child_directory):
                os.makedirs(child_directory)
            os.chdir(child_directory)
            plt.savefig('%s.png' % file_name, bbox_inches='tight')
            plt.close()
            os.chdir('..')



def time_series_instances(csv_file, variable_name, variable_name_plot, n_rows, n_columns, time_begin, time_end, time_shift,
                                   xlabel, ylabel, title, fontSize_supTitle, fontSize_title, fontSize_xlabel, fontSize_ylabel,
                                   fontSize_ticks, save_name, color, maxXticks, maxYticks, child_directory):
    # plot time series of instances of an individual variable, for instance the price time series of several different firms
    data = pd.read_csv(csv_file) # read the csv file
    data = pd.DataFrame(data) # convert it to data frame
    rows = data.shape[0] # number of rows in the file
    number_instances = int(n_rows * n_columns)
    selected_rows = random.sample(range(rows), number_instances) # select some rows randomly
    #n = int(number_instances ** 0.5) # the number of rows and columns to create subplots
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_columns, sharex=True, sharey=True, tight_layout=False)
    axs = ax.flatten()
    time_length = time_end - time_begin
    x_axis_time_steps = range(-time_shift, time_length-time_shift) # create x axis values by shifting time
    for i in xrange(number_instances):
        r = selected_rows[i]
        d = ast.literal_eval(data.iloc[r][variable_name])
        axs[i].plot(x_axis_time_steps, d[time_begin:time_end], color=color)
        axs[i].scatter(x_axis_time_steps, d[time_begin:time_end], color=color,s=5)
        index = i + 1
        name = variable_name_plot + ' ' + str(index)
        axs[i].set_title('%s' % name, fontsize=fontSize_title, y=0.7)
        #axs[i].ylim(ymin=0.997,ymax=1.003)
        axs[i].tick_params(axis='both', labelsize=fontSize_ticks)
        yloc = plt.MaxNLocator(maxYticks)
        xloc = plt.MaxNLocator( maxXticks)
        axs[i].yaxis.set_major_locator(yloc)
        axs[i].xaxis.set_major_locator(xloc)
        axs[i].ticklabel_format(useOffset=False, useLocale=True, useMathText=True)
    fig.text(0.5, 0.02, xlabel, ha='center', fontsize=fontSize_xlabel)
    fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=fontSize_ylabel)
    fig.suptitle(title, fontsize=fontSize_supTitle)

    #plt.ylim(ymin=0.9995,ymax=1.0005)
    png_name = 'ts_single' + '_' + variable_name + '_' + csv_file[:-4]
    if save_name is not None:
        png_name += '_'
        png_name += save_name
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    plt.savefig('%s.png' % png_name)
    plt.close()
    os.chdir('..')




def time_series_bunch(csv_file, variable_name, variable_name_plot, numbers_bunch, time_begin, time_end, time_shift,
                      xlabel, ylabel, title, fontSize_supTitle, fontSize_title, fontSize_xlabel, fontSize_ylabel,
                      fontSize_ticks, save_name, maxXticks, maxYticks, child_directory, scatter, shock_time):

    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    rows = data.shape[0]
    if numbers_bunch is not None:
        prices = dict((n, []) for n in numbers_bunch)
        time_diff = time_end - time_begin
        x_values = range(-time_shift, time_diff-time_shift)
        for n in numbers_bunch:
            some_rows = random.sample(range(rows), n)
            some_price_series = []
            for r in some_rows:
                price_series = ast.literal_eval(data.iloc[r][variable_name])
                price_series = price_series[time_begin:time_end]
                some_price_series.append(price_series)
            prices[n] = some_price_series
        n = int(len(numbers_bunch) ** 0.5)
        fig, ax = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, tight_layout=False)
        axs = ax.flatten()
        for i in xrange(len(numbers_bunch)):
            n = numbers_bunch[i]
            d = prices[n]
            for j in d:
                axs[i].plot(x_values, j)
                axs[i].tick_params(axis='both', labelsize=fontSize_ticks)
            #exponent = str(math.log(n, 10))
            #exponent = exponent[:exponent.index('.')]
            #axs[i].set_title('$10^{%s}$ prices' % exponent, fontsize=12)
            minor_title = str(n) + ' ' + variable_name_plot
            axs[i].set_title(minor_title, fontsize=fontSize_title)
            yloc = plt.MaxNLocator(maxYticks)
            xloc = plt.MaxNLocator(maxXticks)
            axs[i].yaxis.set_major_locator(yloc)
            axs[i].xaxis.set_major_locator(xloc)
        fig.text(0.5, 0.02, xlabel, ha='center', fontsize=fontSize_xlabel)
        fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=fontSize_ylabel)
        fig.suptitle(title, fontsize=fontSize_supTitle)
    else:
        prices = []
        a_time = time_begin - shock_time
        b_time = time_end - shock_time
        x = range(a_time, b_time)
        for r in xrange(rows):
            price_series = ast.literal_eval(data.iloc[r][variable_name])
            price_series = price_series[time_begin:time_end]
            prices.append(price_series)
        for series in prices:
            if scatter:
                plt.scatter(x, series, marker='o', alpha=0.25, s=0.5)
            else:
                plt.plot(series)
            plt.xlabel(xlabel, fontsize=fontSize_xlabel)
            plt.ylabel(ylabel, fontsize=fontSize_ylabel)
            plt.suptitle(title, fontsize=fontSize_supTitle)

    png_name = 'ts_bunch' + '_' + variable_name + '_' + csv_file[:-4]
    if save_name is not None:
        png_name += '_' + save_name

    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    plt.savefig('%s.png' % png_name)
    plt.close()
    os.chdir('..')

def histogram_time_steps(csv_file, variable_name, time_steps, xlabel, ylabel, title, shock_time,
                         bins, fontSize_supTitle, fontSize_title, fontSize_xlabel, fontSize_ylabel,
                         fontSize_ticks, save_name, maxXticks, maxYticks, child_directory,
                         xMax, xMin, vline, shareXAxis, shareYAxis):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    rows = data.shape[0]
    data_dictionary = dict((v, []) for v in time_steps)
    for r in xrange(rows):
        price_time_series =  ast.literal_eval(data.iloc[r][variable_name])
        for v in time_steps:
            price = price_time_series[v]
            data_dictionary[v].append(price)
    #n = int(len(time_steps) ** 0.5)
    if len(time_steps) == 4:
        n = 2
        m = 2
    elif len(time_steps) == 6:
        n = 2
        m = 3
    elif len(time_steps) == 9:
        n = 3
        m = 3
    else:
        n = int(len(time_steps) ** 0.5)
        m = int(len(time_steps) ** 0.5)
    fig, ax = plt.subplots(nrows=m, ncols=n, sharex=shareXAxis, sharey=shareYAxis, tight_layout=False)
    #fig, ax = plt.subplots(nrows=n, ncols=n, sharex=True, tight_layout=False)
    axs = ax.flatten()
    for i in xrange(len(time_steps)):
        time = time_steps[i]
        values = data_dictionary[time]
        axs[i].hist(values, bins=bins, weights=np.zeros_like(values) + 1. / len(values), color='black')
        #s = time - shock_time
        s = time - shock_time
        if s != 1:
            axs[i].set_title('%s time steps' % s, fontsize=fontSize_title)
        elif s == 1:
            axs[i].set_title('%s time step' % s, fontsize=fontSize_title)
        yloc = plt.MaxNLocator(maxYticks)
        xloc = plt.MaxNLocator(maxXticks)
        axs[i].yaxis.set_major_locator(yloc)
        axs[i].xaxis.set_major_locator(xloc)
        axs[i].tick_params(axis='both', labelsize=fontSize_ticks)
        axs[i].set_xlim([xMin, xMax])

        axs[i].grid()
        if vline is not None:
            axs[i].axvline(x=vline, color = 'dodgerblue')
    fig.text(0.5, 0.01, xlabel, ha='center', fontsize=fontSize_xlabel)
    fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=fontSize_ylabel)
    fig.suptitle(title, fontsize=fontSize_supTitle)
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    file_name = 'hist' + '_' + variable_name + '_' + csv_file[:-4]
    if save_name is not None:
        file_name += '_'
        file_name += save_name

    plt.savefig('%s.png' % file_name)
    plt.close()
    os.chdir('..')

def time_series_multiple_variables_specific_paramValue(csv_file, variables_names, parameter_name,
                                                       parameter_value, start_time, end_time,
                                                       title, xlabel, ylabel, variables_labels, n, save_name,
                                                       fontSize_title, fontSize_xlabel, fontSize_ylabel,
                                                       fontSize_ticks,child_directory,time_shift, maxXticks, maxYticks, colors,
                                                       tick2Digits,
                                                       normalize):


    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    data = data.loc[data[parameter_name] <= (parameter_value + 10**-10)]
    data = data.loc[data[parameter_name] >= (parameter_value - 10 ** -10)]
    variables_data = []
    for name in variables_names:
        v = data.iloc[:][name]
        variables_data.append(v)
    variables_set = []

    for v in variables_data:
        v_set = random.sample(v, n ** 2)
        v_set = [ast.literal_eval(d) for d in v_set]
        v_set = [d[start_time:end_time] for d in v_set]
        variables_set.append(v_set)
    fig, ax = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, tight_layout=False)
    axs = ax.flatten()
    time_diff = end_time - start_time
    x_values = range(-time_shift, time_diff-time_shift)

    for i in xrange(n ** 2):
        samples = []
        for var in variables_set:
            samples.append(var[i])
        for j in xrange(len(samples)):
            if normalize:
                first = samples[j][0]
                vals = [l/first for l in samples[j]]
            else:
                vals = samples[j]
            axs[i].plot(x_values, vals, color=colors[j], label=variables_labels[j])
            axs[i].tick_params(axis='both', labelsize=fontSize_ticks)
            yloc = plt.MaxNLocator(maxXticks)
            xloc = plt.MaxNLocator(maxYticks)
            axs[i].yaxis.set_major_locator(yloc)
            axs[i].xaxis.set_major_locator(xloc)
            if tick2Digits:
                axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    names = 'n'

    for v in variables_names:
        names += v
    names = names[1:]
    file_name = 'time_series' + '_' + names + '_' + csv_file[:-4]
    if save_name is not None:
        file_name += '_'
        file_name += save_name
    fig.suptitle(title, fontsize=fontSize_title)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=14)
    fig.text(0.5, 0.01, xlabel, ha='center',fontsize=fontSize_xlabel)
    fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=fontSize_ylabel)
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    plt.savefig('%s.png' % file_name)
    plt.close()
    os.chdir('..')

def convergence_mean_error2(csv_file, thresholds,legend, legendSize, fontSize_title, fontSize_xlabel, fontSize_ylabel, fontSize_ticks, child_directory,
               digits_yAxis, maxXticks, maxYticks, colors, title_text, xlabel, ylabel, ymin, ymax, xmin, xmax, save_name,legend_location, cutOff):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)

    rows = data.shape[0]
    parameter_values = []
    for i in xrange(rows):
        v = data.iloc[i,0]
        parameter_values.append(v)

    parameter_values = list(set(parameter_values))
    
    #parameter_values = list(set(data.icol(0)))


    if cutOff is not None:
        maX = cutOff[1]
        miN = cutOff[0]
        parameter_values = [i for i in parameter_values if i <= maX ]
        parameter_values = [i for i in parameter_values if i >= miN]
    parameter_values = sorted(parameter_values)
    rows = data.shape[0]
    fig, ax = plt.subplots()
    count = 0
    for t in thresholds:
        color = colors[count]
        count+=1
        data_dict = collections.OrderedDict({})
        for p in parameter_values:
            data_dict[p] = []

        mean_dict = collections.OrderedDict({})
        errors_dict = collections.OrderedDict({})
        for r in xrange(rows):
            v = data.iloc[r][0]
            if cutOff is not None:
                if v <= cutOff[1] and v >= cutOff[0]:
                    d = data.iloc[r][str(t)]
                    data_dict[v].append(np.mean(d))
            else:
                d = data.iloc[r][str(t)]
                data_dict[v].append(np.mean(d))


        for param in data_dict:
            values = data_dict[param]
            mean_dict[param] = np.mean(values)            
            errors_dict[param] = stats.sem(values)
        y_values = mean_dict.values()
        x_values = mean_dict.keys()

        plt.plot(x_values, y_values,marker='o',color=color, markersize=4,  label=r'$\delta < 10^{-%s}$' % t)
        #ax.errorbar(x_values, y_values, errors_dict.values(), linestyle='None', marker='o',
         #           color=color, markersize=4, capsize=2, label=r'$\delta < 10^{-%s}$' % t)
        ax.ticklabel_format(useOffset=True, useLocale=True, useMathText=True)
        if legend:
            plt.legend(numpoints=1, fontsize=legendSize, loc=legend_location)
        ax.tick_params(axis='both', labelsize=fontSize_ticks)
        ax.xaxis.set_major_locator(plt.MaxNLocator(maxXticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(maxYticks))
        if digits_yAxis:
            a = '%.' + str(digits_yAxis) + 'f'
            ax.yaxis.set_major_formatter(FormatStrFormatter(a))
        plt.suptitle(title_text, fontsize=fontSize_title)
        plt.xlabel(xlabel, fontsize=fontSize_xlabel)
        plt.ylabel(ylabel, fontsize=fontSize_ylabel)
        plt.grid()
        if ymin is not None:
            plt.ylim(ymin=ymin)
        if ymax is not None:
            plt.ylim(ymax=ymax)
        if xmin is not None:
            plt.xlim(xmin=xmin)
        if xmax is not None:
            plt.xlim(xmax=xmax)
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    file_name = 'convergence' + '_' + csv_file[:-4]
    if save_name is not None:
        file_name += '_'
        file_name += save_name
    plt.grid()
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')

def convergence_mean_error3(csv_file, thresholds,legend, legendSize, fontSize_title, fontSize_xlabel, fontSize_ylabel, fontSize_ticks, child_directory,
               digits_yAxis, maxXticks, maxYticks, colors, title_text, xlabel, ylabel, ymin, ymax, xmin, xmax, save_name,legend_location, cutOff,logScale,
               xticksLabel,yticksLabel):
    # same as convergence mean error, but allows for log scale in x, y, or both axis
    
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)

    rows = data.shape[0]
    parameter_values = []
    for i in xrange(rows):
        v = data.iloc[i,0]
        parameter_values.append(v)

    parameter_values = list(set(parameter_values))
    
    #parameter_values = list(set(data.icol(0)))


    if cutOff is not None:
        maX = cutOff[1]
        miN = cutOff[0]
        parameter_values = [i for i in parameter_values if i <= maX ]
        parameter_values = [i for i in parameter_values if i >= miN]
    parameter_values = sorted(parameter_values)
    rows = data.shape[0]
    fig, ax = plt.subplots()
    count = 0
    for t in thresholds:
        color = colors[count]
        count+=1
        data_dict = collections.OrderedDict({})
        mean_dict = collections.OrderedDict({})
        errors_dict = collections.OrderedDict({})

        for v in parameter_values:
            data_dict[v] = []


        #data_dict = dict((v, []) for v in parameter_values)        
        #mean_dict = dict((v, 0) for v in parameter_values)
        #errors_dict = dict((v, 0) for v in parameter_values)

        for r in xrange(rows):
            v = data.iloc[r][0]
            if cutOff is not None:
                if v <= cutOff[1] and v >= cutOff[0]:
                    d = data.iloc[r][str(t)]
                    data_dict[v].append(np.mean(d))
            else:
                d = data.iloc[r][str(t)]
                data_dict[v].append(np.mean(d))

        for param in data_dict:
            values = data_dict[param]
            mean_dict[param] = np.mean(values)
            mean_dict[param] = np.mean(values)
            errors_dict[param] = stats.sem(values)
        y_values = mean_dict.values()
        x_values = mean_dict.keys()

        if logScale == 'x':
            x_values = [math.log(i,10) for i in x_values]
        elif logScale == 'y':
            y_values = [math.log(i,10) for i in y_values]
        elif logScale == 'both':
            x_values = [math.log(i,10) for i in x_values]
            y_values = [math.log(i,10) for i in y_values]


        ax.errorbar(x_values, y_values, errors_dict.values(), linestyle='None', marker='.',
                    color=color, markersize=4, capsize=2, label=r'$\delta < 10^{-%s}$' % t)
        
        plt.plot(x_values, y_values,color=color,lw=0.5)

        ax.ticklabel_format(useOffset=True, useLocale=True, useMathText=True)
        if legend:
            plt.legend(numpoints=1, fontsize=legendSize, loc=legend_location)


        ax.tick_params(axis='both', labelsize=fontSize_ticks)
        ax.xaxis.set_major_locator(plt.MaxNLocator(maxXticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(maxYticks))
        if digits_yAxis:
            a = '%.' + str(digits_yAxis) + 'f'
            ax.yaxis.set_major_formatter(FormatStrFormatter(a))
        plt.suptitle(title_text, fontsize=fontSize_title)
        plt.xlabel(xlabel, fontsize=fontSize_xlabel)
        plt.ylabel(ylabel, fontsize=fontSize_ylabel)
        plt.grid()

        if xticksLabel:
            plt.xticks(xticksLabel.keys(), xticksLabel.values())
        if yticksLabel:
            plt.yticks(yticksLabels.keys(), yticksLabels.values())

        if ymin is not None:
            plt.ylim(ymin=ymin)
        if ymax is not None:
            plt.ylim(ymax=ymax)
        if xmin is not None:
            plt.xlim(xmin=xmin)
        if xmax is not None:
            plt.xlim(xmax=xmax)
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    file_name = 'convergence' + '_' + csv_file[:-4]
    if save_name is not None:
        file_name += '_'
        file_name += save_name
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')


def convergence_mean_error(csv_file, thresholds,legend, legendSize, fontSize_title, fontSize_xlabel, fontSize_ylabel, fontSize_ticks, child_directory,
               digits_yAxis, maxXticks, maxYticks, colors, title_text, xlabel, ylabel, ymin, ymax, xmin, xmax, save_name,legend_location, cutOff):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)

    rows = data.shape[0]
    parameter_values = []
    for i in xrange(rows):
        v = data.iloc[i,0]
        parameter_values.append(v)

    parameter_values = list(set(parameter_values))
    
    #parameter_values = list(set(data.icol(0)))


    if cutOff is not None:
        maX = cutOff[1]
        miN = cutOff[0]
        parameter_values = [i for i in parameter_values if i <= maX ]
        parameter_values = [i for i in parameter_values if i >= miN]
    parameter_values = sorted(parameter_values)
    rows = data.shape[0]
    fig, ax = plt.subplots()
    count = 0
    for t in thresholds:
        color = colors[count]
        count+=1
        data_dict = collections.OrderedDict({})
        mean_dict = collections.OrderedDict({})
        errors_dict = collections.OrderedDict({})

        for v in parameter_values:
            data_dict[v] = []


        #data_dict = dict((v, []) for v in parameter_values)        
        #mean_dict = dict((v, 0) for v in parameter_values)
        #errors_dict = dict((v, 0) for v in parameter_values)

        for r in xrange(rows):
            v = data.iloc[r][0]
            if cutOff is not None:
                if v <= cutOff[1] and v >= cutOff[0]:
                    d = data.iloc[r][str(t)]
                    data_dict[v].append(np.mean(d))
            else:
                d = data.iloc[r][str(t)]
                data_dict[v].append(np.mean(d))

        for param in data_dict:
            values = data_dict[param]
            mean_dict[param] = np.mean(values)
            mean_dict[param] = np.mean(values)
            errors_dict[param] = stats.sem(values)
        y_values = mean_dict.values()
        x_values = mean_dict.keys()
        ax.errorbar(x_values, y_values, errors_dict.values(), linestyle='None', marker='.',
                    color=color, markersize=4, capsize=2, label=r'$\delta < 10^{-%s}$' % t)
        
        plt.plot(x_values, y_values,color=color,lw=0.5)

        ax.ticklabel_format(useOffset=True, useLocale=True, useMathText=True)
        if legend:
            plt.legend(numpoints=1, fontsize=legendSize, loc=legend_location)


        ax.tick_params(axis='both', labelsize=fontSize_ticks)
        ax.xaxis.set_major_locator(plt.MaxNLocator(maxXticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(maxYticks))
        if digits_yAxis:
            a = '%.' + str(digits_yAxis) + 'f'
            ax.yaxis.set_major_formatter(FormatStrFormatter(a))
        plt.suptitle(title_text, fontsize=fontSize_title)
        plt.xlabel(xlabel, fontsize=fontSize_xlabel)
        plt.ylabel(ylabel, fontsize=fontSize_ylabel)
        plt.grid()
        if ymin is not None:
            plt.ylim(ymin=ymin)
        if ymax is not None:
            plt.ylim(ymax=ymax)
        if xmin is not None:
            plt.xlim(xmin=xmin)
        if xmax is not None:
            plt.xlim(xmax=xmax)
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    file_name = 'convergence' + '_' + csv_file[:-4]
    if save_name is not None:
        file_name += '_'
        file_name += save_name
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')



def mean_error(csv_file, statistic, xlabel, ylabel, title_text, color,  save_name, positive, ymin, ymax, xmin, xmax,
               inverse, legend, legendSize, fontSize_title, fontSize_xlabel, fontSize_ylabel, fontSize_ticks, child_directory,
               digits_yAxis, maxXticks, maxYticks, delValuesTreshold, cutOff):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    
    #parameter_values = list(set(data.icol(0)))
    parameter_values = []
    rows = data.shape[0]
    for r in xrange(rows):
        pVal = data.iloc[r][0]
        parameter_values.append(pVal)
    
    print len(parameter_values)
    
    if cutOff is not None:
        parameter_values = [i for i in parameter_values if i >= cutOff[0]]
        parameter_values = [i for i in parameter_values if i <= cutOff[1]]
    parameter_values = sorted(parameter_values)
    rows = data.shape[0]
    data_dict = dict((v, []) for v in parameter_values)
    mean_dict = dict((v, 0) for v in parameter_values)
    errors_dict = dict((v, 0) for v in parameter_values)
    for r in xrange(rows):
        v = data.iloc[r][0]
        if cutOff is not None:
            if v >= cutOff[0] and v <= cutOff[1]:
                d = data.iloc[r][statistic]
                data_dict[v].append(np.mean(d))
        else:
            d = data.iloc[r][statistic]
            data_dict[v].append(np.mean(d))


    for param in data_dict:
        values = data_dict[param]
        mean_dict[param] = np.mean(values)
        mean_dict[param] = np.mean(values)
        errors_dict[param] = stats.sem(values)

    if delValuesTreshold is not None:
        if positive:
            for k in mean_dict.keys():
                if k < delValuesTreshold:
                    del mean_dict[k]
                    del errors_dict[k]
        else:
            for k in mean_dict.keys():
                if k > - delValuesTreshold:
                    del mean_dict[k]
                    del errors_dict[k]

    y_values = mean_dict.values()
    x_values = mean_dict.keys()

    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=True, useLocale=True, useMathText=True)
    ax.tick_params(axis='both', labelsize=fontSize_ticks)
    ax.xaxis.set_major_locator(plt.MaxNLocator(maxXticks))
    ax.yaxis.set_major_locator(plt.MaxNLocator(maxYticks))
    if digits_yAxis:
        a = '%.' + str(digits_yAxis) + 'f'
        ax.yaxis.set_major_formatter(FormatStrFormatter(a))
    plt.scatter(x_values, y_values, marker='.', s=3, color=color, label=legend)
    plt.suptitle(title_text, fontsize=fontSize_title)
    plt.xlabel(xlabel, fontsize=fontSize_xlabel)
    plt.ylabel(ylabel, fontsize=fontSize_ylabel)
    plt.grid()
    if ymin is not None:
        plt.ylim(ymin=ymin)
    if ymax is not None:
        plt.ylim(ymax=ymax)
    if xmin is not None:
        plt.xlim(xmin=xmin)
    if xmax is not None:
        plt.xlim(xmax=xmax)
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    file_name = 'all_points' + statistic + '_' + csv_file[:-4]
    if save_name is not None:
        file_name += '_'
        file_name += save_name
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')

    print max(x_values)


    if inverse == True:
        y_values = [1-i for i in y_values]
    fig, ax = plt.subplots()
    ax.errorbar(x_values, y_values, errors_dict.values(), linestyle='None', marker='.',
                     color=color, markersize=4, capsize=2, label=legend)
    ax.ticklabel_format(useOffset=True, useLocale=True, useMathText=True)
    if legend:
        plt.legend(numpoints = 1, fontsize=legendSize)
    ax.tick_params(axis='both', labelsize=fontSize_ticks)
    ax.xaxis.set_major_locator(plt.MaxNLocator(maxXticks))
    ax.yaxis.set_major_locator(plt.MaxNLocator(maxYticks))
    if digits_yAxis:
        a = '%.' + str(digits_yAxis) + 'f'
        ax.yaxis.set_major_formatter(FormatStrFormatter(a))
    plt.suptitle(title_text, fontsize=fontSize_title)
    plt.xlabel(xlabel, fontsize=fontSize_xlabel)
    plt.ylabel(ylabel, fontsize=fontSize_ylabel)
    plt.grid()
    if ymin is not None:
        plt.ylim(ymin=ymin)
    if ymax is not None:
        plt.ylim(ymax=ymax)
    if xmin is not None:
        plt.xlim(xmin=xmin)
    if xmax is not None:
        plt.xlim(xmax=xmax)
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    file_name = 'mean_error' + statistic + '_' + csv_file[:-4]
    if save_name is not None:
        file_name += '_'
        file_name += save_name
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')



def variations_scatter(csv_file, variable_name, title, xlab, ylab, ymin, ymax,
                       parameter_name, xmax, xmin, save_name, legend, legendSize,
                       fontSize_title, fontSize_xlabel, fontSize_ylabel, fontSize_ticks, child_directory,
                       digits_yAxis, maxXticks, maxYticks, transparency,x_max_value):

    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    y_values = data.iloc[:][variable_name]
    x_values = data.iloc[:][parameter_name]
    new_y_values = []
    new_x_values = []
    if x_max_value is not None:
        for i in xrange(len(x_values)):
            x = x_values[i]
            if x <= x_max_value:
                new_x_values.append(x)
                y = y_values[i]
                new_y_values.append(y)
        y_values = new_y_values
        x_values = new_x_values

    fig, ax = plt.subplots()
    ax.scatter(x_values,y_values, marker='o', alpha=transparency, label=legend)
    plt.xlabel(xlab, fontsize=fontSize_xlabel)
    plt.ylabel(ylab, fontsize=fontSize_ylabel)
    ax.xaxis.set_major_locator(plt.MaxNLocator(maxXticks))
    ax.yaxis.set_major_locator(plt.MaxNLocator(maxYticks))
    if digits_yAxis:
        a = '%.' + str(digits_yAxis) + 'f'
        ax.yaxis.set_major_formatter(FormatStrFormatter(a))
    plt.tick_params(axis='both', labelsize=fontSize_ticks)
    if ymin is not None:
        plt.ylim(ymin=ymin)
    if ymax is not None:
        plt.ylim(ymax=ymax)
    if xmax is not None:
        plt.xlim(xmax=xmax)
    if xmin is not None:
        plt.xlim(xmin=xmin)
    if legend is not None:
        plt.legend(numpoints=1, fontsize=legendSize)
    plt.suptitle(title, fontsize=fontSize_title)
    plt.grid()
    if save_name is not None:
        file_name = 'parameteric_variation_' + variable_name + '_' +  parameter_name + '_' + save_name
    else:
        file_name = 'parameteric_variation_' + variable_name + '_' + parameter_name
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')

"""
def heat_map_simple(csv_file,x_var,y_var, color_var):
    #when data is in three row format
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    x_n = len(set(data.iloc[:][x_var]))
    y_n = len(set(data.iloc[:][y_var]))
    x = list(data.iloc[:][x_var])
    y = list(data.iloc[:][y_var])
    xx, yy = np.meshgrid(x, y)
    Z = [list(data.iloc[:][color_var])]

    

    Z_n = len(list(data.iloc[:][color_var]))
    Z = np.asarray(Z)
    


    print len(x), len(y), len(Z)


    plt.contourf(xx, yy, Z, cmap='RdBu',
                 levels=np.linspace(Z.min(), Z.max(), Z_n)
                 )
    plt.colorbar(label='Z')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
"""

def heat_map_simple(csv_file,x_var,y_var, color_var,
    fontSize_xlabel,fontSize_ylabel,fontSize_title,fontSize_ticks,
    xlabel,ylabel,title,child_directory,yticks_increments,xticks_increments,
    plus_x_tick, plus_y_tick,colorbar):    
    all_x = []
    all_y = []
    dict_info = {}
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    rows = data.shape[0]
    for r in xrange(rows):
        x = data.iloc[r][x_var]
        y = data.iloc[r][y_var]
        v = data.iloc[r][color_var]
        tup = (x,y)
        dict_info[tup] = v
        all_x.append(x)
        all_y.append(y)

    #print dict_info
    all_x = list(set(all_x))
    all_y = list(set(all_y))

    all_x = sorted(all_x)
    all_y = sorted(all_y,reverse=True)


    """
    for y in all_y:
        vals = []
        for x in all_x:
            tup = (x,y)
            v = dict_info[tup]
            vals.append(v)
        plt.plot(vals)

        
    plt.title(ylabel+xlabel)
    plt.show()
    """


    # built the matrix out of this which goes into imhsow
    len_x = len(all_x)
    len_y = len(all_y)
    matrix = np.zeros((len_y,len_x))
    #matrix = np.zeros((len_x,len_y))

    for i in xrange(len_x):
        for j in xrange(len_y):
            x = all_x[i]
            y = all_y[j]
            tup = (x,y)
            v = dict_info[tup]
            matrix[j,i] = v
            #matrix[i,j] = v

    #print all_x

    #plt.imshow(matrix,  interpolation='nearest', aspect='auto', cmap='Blues', extent=[all_x[0],all_x[-1],all_y[0],all_y[-1]])
    plt.imshow(matrix,extent=[all_x[0],all_x[-1],all_y[-1],all_y[0]],aspect='auto',cmap=cm.Blues)
    #plt.imshow(matrix,aspect='auto')
    
    """
    ax = plt.gca()
    if plus_x_tick:
        ax.set_xticks(np.arange(all_x[0], all_x[-1]+xticks_increments, xticks_increments))
    else:
        ax.set_xticks(np.arange(all_x[0], all_x[-1], xticks_increments))
    if plus_y_tick:
        ax.set_yticks(np.arange(all_y[0], all_y[-1]+yticks_increments, yticks_increments))
    else:
        ax.set_yticks(np.arange(all_y[0], all_y[-1], yticks_increments))
    """
    
    if colorbar:
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=20)



    plt.xlabel(xlabel, fontsize=fontSize_xlabel)
    plt.ylabel(ylabel, fontsize=fontSize_ylabel)
    plt.suptitle(title, fontsize=fontSize_title)
    plt.tick_params(axis='both', labelsize=fontSize_ticks)

    file_name = 'color_map'
    file_name += '_'
    file_name += csv_file[:-4]


    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')





def heat_map(csv_file, y_name, z_name, x_label, y_label, title, ratio, ratio_variable_names, positive, positive_threshold,
             save_name, fontSize_title, fontSize_xlabel, fontSize_ylabel, fontSize_ticks, child_directory,
             time_begin, time_end, y_max_value, inverse):
    if ratio:
        name1 = ratio_variable_names[0]
        name2 = ratio_variable_names[1]
    time_range = time_end - time_begin

    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    y_values = data[[y_name]]
    y_values = list(y_values.iloc[:,0])

    if positive_threshold == None:
        positive_threshold = 0
    if positive:
        y_values = [i for i in y_values if i >= positive_threshold]
    else:
        y_values = [i for i in y_values if i <= -positive_threshold]

    if y_max_value is not None:
        y_values = [i for i in y_values if i <= y_max_value]


    z_values_dict = dict((i, []) for i in y_values)
    rows = data.shape[0]
    for r in xrange(rows):
        v = data.iloc[r][y_name]
        if positive:
            if v >= positive_threshold:
                if not ratio:
                    d = ast.literal_eval(data.iloc[r][z_name])
                else:
                    d1 = ast.literal_eval(data.iloc[r][name1])
                    d2 = ast.literal_eval(data.iloc[r][name2])
                    d = []
                    for i in xrange(len(d1)):
                        x = d1[i]
                        y = d2[i]
                        if x > 0 and y > 0:
                            d.append(x / y)
                        else:
                            d.append(0)
                d = d[time_begin:time_end]
                if y_max_value is not None:
                    if v <= y_max_value:
                        z_values_dict[v].append(d)
                else:
                    z_values_dict[v].append(d)
        else:
            if v <= -positive_threshold:
                if not ratio:
                    d = ast.literal_eval(data.iloc[r][z_name])
                else:
                    d1 = ast.literal_eval(data.iloc[r][name1])
                    d2 = ast.literal_eval(data.iloc[r][name2])
                    d = []
                    for i in xrange(len(d1)):
                        x = d1[i]
                        y = d2[i]
                        if x > 0 and y > 0:
                            d.append(x/y)
                        else:
                            d.append(0)
                d = d[time_begin:time_end]
                if y_max_value is not None:
                    if v <= y_max_value:
                        z_values_dict[v].append(d)
                else:
                    z_values_dict[v].append(d)

    z_values_dict_transposed = dict((i, []) for i in y_values)
    for i in z_values_dict:
        v = z_values_dict[i]
        v = map(list, zip(*v))
        z_values_dict_transposed[i] = v
    z_values_mean = dict((i, []) for i in y_values)
    for i in z_values_dict_transposed:
        v = z_values_dict_transposed[i]
        a = []
        for j in v:
            j_mean = np.mean(j)
            a.append(j_mean)
        z_values_mean[i] = a

    z_values_mean = collections.OrderedDict(sorted(z_values_mean.items(), reverse=True))
    number_z_values = len(z_values_mean.keys())
    matrix = np.zeros((number_z_values, time_range))
    count_row = 0
    for i in z_values_mean:
        count_column = 0
        time_step_values = z_values_mean[i]
        for j in time_step_values:
            if inverse:
                matrix[count_row, count_column] = 1 - j
            else:
                matrix[count_row, count_column] = j
            count_column += 1
        count_row += 1
    first_Xvalue = 0
    last_Xvalue = time_end - time_begin
    y = z_values_mean.keys()
    y = sorted(y)
    first_Yvalue = y[0]
    last_Yvalue = y[-1]
    plt.imshow(matrix,  interpolation='nearest', aspect='auto', cmap='Blues', extent=[first_Xvalue, last_Xvalue,first_Yvalue,last_Yvalue])
    ax = plt.gca()
    ax.set_xticks(np.arange(first_Xvalue, last_Xvalue, 1))

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)

    plt.xlabel(x_label, fontsize=fontSize_xlabel)
    plt.ylabel(y_label, fontsize=fontSize_ylabel)
    plt.suptitle(title, fontsize=fontSize_title)
    plt.tick_params(axis='both', labelsize=fontSize_ticks)

    file_name = 'color_map' #+ '_' + y_name
    file_name += '_'
    file_name += csv_file[:-4]

    if not positive:
        file_name += '_negative'
    if save_name is not None:
        file_name += '_'
        file_name += str(save_name)
    if not os.path.exists(child_directory):
        os.makedirs(child_directory)
    os.chdir(child_directory)
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')


"""
def distribution_boxplot(csv_file, variable_name, log_variable, statistic, simple, title, xlab, ylab, ymin, ymax):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    headers = data.columns.values
    parameter_name = headers[0]
    assert variable_name in headers, "variable name not in headers of csv file"
    parameter_values = list(set(data.icol(0)))
    rows = data.shape[0]

    def compute_statistic():
        statistic_dict = dict((v, []) for v in parameter_values)
        for r in xrange(rows):
            v = data.iloc[r][0]
            if simple:
                d = data.iloc[r][variable_name]
            else:
                d = ast.literal_eval(data.iloc[r][variable_name])
                #d = d.values()
                #d = list(chain.from_iterable(d))
                if log_variable:
                    d = [log(i) for i in d]
                if statistic == 'mean':
                    d = np.mean(d)
                elif statistic == 'std':
                    d = np.std(d)
                elif statistic == 'cv':
                    d = stats.variation(d)
            statistic_dict[v].append(d)
        return statistic_dict


    def draw_boxplot(y_values, x_values):
        a = []
        b = []

        for i in xrange(len(x_values)):
            x = x_values[i]
            ys = y_values[i]
            for j in ys:
                a.append(x)
                b.append(j)
        plt.scatter(a,b)
        plt.show()


        width = 0.025
        widths = tuple([width] * len(y_values))
        bp = plt.boxplot(y_values, patch_artist=True, positions = x_values, widths=widths)
        if ymin is not None:
            plt.ylim(ymin=ymin)
        if ymax is not None:
            plt.ylim(ymax=ymax)

        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='royalblue')

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.suptitle(title)
        for patch in bp['boxes']:
            patch.set(facecolor='white')

        v = np.array([box.get_path().vertices for box in bp['boxes']])
        margin = 0.2
        xmin = v[:, :5, 0].min() - (max(x_values) - min(x_values)) * margin
        xmax = v[:, :5, 0].max() + (max(x_values) - min(x_values)) * margin
        plt.xlim(xmin, xmax)

    statistic_dict = compute_statistic()
    y_values = statistic_dict.values()
    x_values = statistic_dict.keys()
    draw_boxplot(y_values, x_values)
    file_name = 'distribution_boxplot_' + parameter_name + '_' + variable_name
    directory_name = 'plots_cantillon'
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    os.chdir(directory_name)
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')

def single_firm(csv_file):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    x_values = data.iloc[:]['In_degree']
    y_values = data.iloc[:]['mean_cv']
    new_x_values = []
    new_y_values = []
    for i in range(len(x_values)):
        x = x_values[i]
        if x > 0:
            new_x_values.append(x)
            y = y_values[i]
            new_y_values.append(y)

    directory_name = 'plots_cantillon'
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    os.chdir(directory_name)
    plt.scatter(new_x_values,new_y_values, marker='.', alpha=0.25)
    plt.xlabel('In Degree')
    plt.ylabel('Price variation')
    plt.title('Price Dispersion with variation in Degree of the Firm Shocked')
    file_name = 'scatter_single_firm'
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')

def network_histograms(csv_file, variable_name, title, xlab, ylab):
    data = pd.read_csv(csv_file)
    data = pd.DataFrame(data)
    sf = data.loc[data['network_type'] == 'scale_free_directional']
    balanced = data.loc[data['network_type'] == 'circular_directional']
    bins = 100
    sf_color = 'royalblue'
    balanced_color = 'green'
    sf_cv = list(sf.iloc[:]['mean'])
    balanced_cv = list(balanced.iloc[:]['mean'])
    plt.hist(sf_cv, bins=bins,  color=sf_color, lw=0.1)
    plt.hist(balanced_cv, bins=bins, color=balanced_color, lw=0.1)
    plt.suptitle(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(xmin=0)
    directory_name = 'plots_cantillon'
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    os.chdir(directory_name)
    file_name = 'hist_network'
    file_name += variable_name
    plt.savefig('%s.png' % file_name, bbox_inches='tight')
    plt.close()
    os.chdir('..')

def all_network_histograms():
    network_histograms(csv_file='network_type_economy_scale_free_directional_Cobb_Douglas_static_statsvariation_normalized_prices_cross_section.csv',
                   variable_name='mean', title='Distribution of Price Dispersion',
                   xlab='Price variation',
                   ylab='Number of simulation runs')
"""