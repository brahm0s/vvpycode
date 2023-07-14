from __future__ import division

import main
import rewiring
import os


number_of_rewiring = 10
for i in xrange(number_of_rewiring):
    rewiring.rewire_degree_preserving(rewiring_factor=1, file_name='network.txt')
    paper_name = 'network_rewired' + '_' + str(i)
    main.main(network_name=None, time_series={'economy': False, 'firm': False, 'household': False}, parameter_names=['number_of_firms'],
             parameter_ranges={'number_of_firms': [100, 102]},
             parameter_increments={'number_of_firms': 1},
             variables={'number_of_firms': {'economy': None, 'firm': ['price'], 'household': None}},
             other_parameters_={'number_of_firms': {'network_from_file': True}},
             iterations_={'number_of_firms': 1},
             cores_={'number_of_firms': 1},
         model_name='rewiring',
         model_url='url',
         paper_name=paper_name,
         paper_url=None)
