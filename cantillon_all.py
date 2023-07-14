from __future__ import division
import main_cantillon
import processData_cantillon
import plots_cantillon

def all_cantillon():
    main_cantillon.simulations()
    main_cantillon.pickle_economy()
    main_cantillon.toy_economy()
    processData_cantillon.process()
    plots_cantillon.all_plots()

all_cantillon()


