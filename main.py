from __future__ import division

import model
import write_data



def main(network_name, production_function, time_series, parameter_names, parameter_ranges, parameter_increments,
         variables, other_parameters_, iterations_, cores_, model_name, model_url, paper_name, paper_url):
    
    model_data = model.generate_data(time_series=time_series,
                                     parameter_names=parameter_names,
                                     parameter_ranges=parameter_ranges,
                                     parameter_increments=parameter_increments,
                                     variables=variables,
                                     other_parameters_=other_parameters_,
                                     iterations_=iterations_,
                                     cores_=cores_)

    write_data.write(model_data, network_name=network_name, production_function=production_function, model_name=model_name, code_url=model_url, paper_name=paper_name, paper_url=paper_url)



