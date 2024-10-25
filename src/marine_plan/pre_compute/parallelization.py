import numpy as np
import warnings
warnings.filterwarnings("ignore")





def calc(single_cell, tech_class, found_age=25):

    if np.isnan(single_cell['mean_wind_speed_at_10m']):
        metric_results = {key: np.nan for key in ['capex', 'opex', 'co2+', 'co2-', 'revenue', 'LCOE', 'energy_produced', 'food_produced', 'unit density', 'lifetime']}
        install_decom = {key: np.nan for key in ['foundation_install_cost', 'foundation_decom_cost', 'foundation_install_emissions', 'foundation_decom_emissions', 'turbine_install_cost', 'turbine_decom_cost', 'turbine_install_emissions', 'turbine_decom_emissions']}
        return metric_results, install_decom
    
    else:
        final_metrics = tech_class(single_cell)
        return final_metrics.run(found_age)

def process_tech(tech_name, tech_class, combined_data, found_age=25):
    metrics = {key: [] for key in ['capex', 'opex', 'co2+', 'co2-', 'revenue', 'LCOE', 'energy_produced', 'food_produced', 'unit density', 'lifetime']}
    install_decom_dict = {key: [] for key in ['foundation_install_cost', 'foundation_decom_cost', 'foundation_install_emissions', 'foundation_decom_emissions', 'turbine_install_cost', 'turbine_decom_cost', 'turbine_install_emissions', 'turbine_decom_emissions']}
    
    for i in range(len(combined_data['mean_wind_speed_at_10m'])):
        single_cell = {key: value[i] for key, value in combined_data.items()}
        metric_result, install_decom = calc(single_cell, tech_class, found_age)

        for key, value in metric_result.items():
            metrics[key].append(value)

        for key, value in install_decom.items():
            install_decom_dict[key].append(value)
    
    return tech_name, metrics, install_decom_dict
