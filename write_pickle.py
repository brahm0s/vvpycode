from __future__ import division
import datetime
import os
import cPickle


def write(all_economies, network, parameter_name, parameter_range, parameter_increment, iterations, file_name):

    def write_txt_file_parameter_sim():
        with open('%s.txt' % parameter_name, 'w') as data_description:
            today = datetime.date.today()
            today_time = datetime.datetime.now().time()
            hr = today_time.hour
            minute = today_time.min
            today_time = str(hr) + '_' + str(minute)
            data_description.write('Date of Simulation: %s \n' % today)
            data_description.write('Time of Simulation: %s \n \n' % today_time)
            data_description.write('Network: %s \n \n' % network)
            data_description.write('Parameter Name: %s \n \n' % parameter_name)
            data_description.write('Parameter Range: %s \n \n' % parameter_range)
            data_description.write('Parameter Increment: %s \n \n' % parameter_increment)
            data_description.write('Iterations Per Parameter Value: %s \n \n' % iterations)

    if not os.path.exists(parameter_name):
        os.makedirs(parameter_name)
    os.chdir(parameter_name)
    if not os.path.exists('%s.txt' % parameter_name):
        write_txt_file_parameter_sim()


    with open('%s.cPickle' % file_name, 'wb') as economies:
        cPickle.dump(all_economies, economies, protocol=cPickle.HIGHEST_PROTOCOL)
    os.chdir('..')
    os.chdir('..')


