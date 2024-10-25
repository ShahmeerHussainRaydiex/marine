import os
import pickle
import shutil
import geopandas as gpd
import pandas as pd
import math
import matplotlib.pyplot as plt

from src.marine_plan.marineplan import MarinePlan
from src.marine_plan.pre_compute.pre_compute import *
from src.metric_util_OLD import shorten_number
import pulp

from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch, Rectangle, PathPatch
from matplotlib.lines import Line2D

from shapely.geometry import MultiPolygon, Polygon

from src.generate_text_outputs import TextOutputGenerator
from src.config.landobject_config import north_sea_ports, custom_north_sea_ports


def remove_old_tech(mod_wind_farms, year, lifetime):
    gdf = gpd.read_file(mod_wind_farms)

    # if the most_common_tech column does not exist, create it
    if 'most_common_tech' not in gdf.columns:
        gdf['most_common_tech'] = 'monopile'
        gdf['real'] = 'yes'

    gdf['YEAR'] = gdf['YEAR'].astype(int)
    
    # Apply filtering based on whether 'real' is 'yes' or 'no' for each row
    gdf = gdf[gdf.apply(lambda row: row['YEAR'] >= (int(year) - 25) if row['real'] == 'yes' 
                        else row['YEAR'] >= (int(year) - lifetime), axis=1)]

    # Save the modified GeoDataFrame
    gdf.to_file(mod_wind_farms, driver='GeoJSON')

    # Get the unique techs
    techs = gdf['most_common_tech'].unique()


def prep_phasing_run(dir, config, mod_wind_farms, year, found_age=25, iterative=False, iteration=1, remove=True):
    raster_folder = os.path.join(dir, 'data', 'rasters', 'international')
    vector_folder = os.path.join(dir, 'data', 'vectors', 'international')

    temp_folder = os.path.join(dir, 'temp')
    temp_pickle_folder = os.path.join(temp_folder, 'pickles')

    # Copy the clean, original rasters into the temp folder
    copied_rasters = [f for f in os.listdir(raster_folder) if os.path.isfile(os.path.join(raster_folder, f))]

    for r in copied_rasters:
        source_path = os.path.join(raster_folder, r)
        destination_path = os.path.join(temp_folder, r)
        shutil.copy(source_path, destination_path)

    # Set the base masks dict and add the existing wind farms mask
    masks = set_base_masks(config, 'international')

    if iteration != 0:
        masks.update({mod_wind_farms: 'mask_outside'})

    # File manip
    original_metrics_file = os.path.join(temp_pickle_folder, 'original_calculated_metrics.pkl')
    metrics_file = os.path.join(temp_pickle_folder, 'calculated_metrics.pkl')
    geo_data_file = os.path.join(dir, 'temp', 'pickles', 'geo_data.pkl')
    land_objects_file = os.path.join(dir, 'temp', 'pickles', 'land_objects.pkl')
    closest_gridconnects_file = os.path.join(dir, 'temp', 'closest_gridconnects.pkl')

    # Make the metrics file the original metrics file
    shutil.copy(original_metrics_file, metrics_file)

    if not iterative and not remove:
        remove_old_tech(mod_wind_farms, year, found_age)

    data_processing(folder=temp_folder, functions=masks, shp_folder=vector_folder, phasing=True)


    # Load geo_data
    with open(geo_data_file, 'rb') as file:
        geo_data = pickle.load(file)

    # Load land_objects
    with open(land_objects_file, 'rb') as file:
        lo = pickle.load(file)

    # Load closest_gridconnects
    with open(closest_gridconnects_file, 'rb') as file:
        closest_gridconnects = pickle.load(file)

    combined = add_geodata(geo_data, lo, 'international', closest_gridconnects, iterative)

    with open (geo_data_file, 'wb') as file:
        pickle.dump(combined, file)


def mod_existing_tech(result_folder, mod_wind_farms, year, iterative=False):
    '''  
    Takes the cluster result from the optimizer and adds the tech into the current mod file
    '''

    if iterative:
        run_name = 'test'
    else:
        run_name = year

    cluster_gdf_path = os.path.join(result_folder, 'runs', run_name, 'SHP', f'{run_name}.geojson')
    cluster_gdf = gpd.read_file(cluster_gdf_path)
    current_farms = gpd.read_file(mod_wind_farms)

    # in the cluster_gdf, remove anything that isnt tech = wind
    # cluster_gdf = cluster_gdf[cluster_gdf['tech'] == 'wind']
    cluster_gdf = cluster_gdf[(cluster_gdf['value'] == 'jacket') | (cluster_gdf['value'] == 'monopile')]
    # in the cluster_gdf, remove all columns except geometry, capacity, tech
    # cluster_gdf = cluster_gdf[['geometry', 'capacity', 'tech', 'country', 'most_common_tech']]
    # multiply the capacity by 1000 and change the column name to POWER_MW
    cluster_gdf['capacity'] = cluster_gdf['capacity'] * 1000
    cluster_gdf.rename(columns={'capacity': 'POWER_MW'}, inplace=True)
    cluster_gdf.rename(columns={'country': 'COUNTRY'}, inplace=True)
    # Add a column YEAR
    cluster_gdf['YEAR'] = year

    # Add the cluster_gdf to the current_farms
    current_farms = pd.concat([current_farms, cluster_gdf], ignore_index=True)

    # Save the current_farms
    current_farms.to_file(mod_wind_farms, driver='GeoJSON')

    # Get the unique techs
    techs = current_farms['value'].unique()
    

def adjust_trackers(P, n, install_decom_tracker, install_decom, foundation_exists, turbine_age, foundation_age, found_age, keys):

    install_decom_tracker[n]['decomissioned_turbines'] = 0
    install_decom_tracker[n]['decomissioned_foundations'] = 0

    turbine_iterations = found_age / 25

    for i in range(len(foundation_exists)):
        
        tech = P.seed[i]
        cell_density = P.data['monopile']['unit density'][P.first_num] * P.resolution

        # Check for decommissioning
        if foundation_exists[i]:
            
            #if turbine_age[i] divides evenly by 25, then decommission:
            if turbine_age[i] % 25 == 0:
                # Decommission turbine
                install_decom_tracker[n]['turbine_decom_cost'] += install_decom['monopile']['turbine_decom_cost'][i] * cell_density / turbine_iterations
                install_decom_tracker[n]['turbine_decom_emissions'] += install_decom['monopile']['turbine_decom_emissions'][i] * P.CO2_VALUE * cell_density
                install_decom_tracker[n]['decomissioned_turbines'] += cell_density
                    
                    # install turbine if you don't have to replace the foundation
                if foundation_age[i] != 0:
                    install_decom_tracker[n]['turbine_install_cost'] += install_decom['monopile']['turbine_install_cost'][i] * cell_density / turbine_iterations
                    install_decom_tracker[n]['turbine_install_emissions'] += install_decom['monopile']['turbine_install_emissions'][i] * P.CO2_VALUE * cell_density
                    install_decom_tracker[n]['new_turbines'] += cell_density
            
            if foundation_age[i] <= 0:
                # Decommission foundation if it is aged out
                install_decom_tracker[n]['foundation_decom_cost'] += install_decom['monopile']['foundation_decom_cost'][i] * cell_density
                install_decom_tracker[n]['foundation_decom_emissions'] += install_decom['monopile']['foundation_decom_emissions'][i] * P.CO2_VALUE * cell_density
                install_decom_tracker[n]['decomissioned_foundations'] += cell_density

                foundation_exists[i] = False

        # Check for new installations
        if keys[tech] in ['monopile', 'jacket']:
            # Install new turbine
            install_decom_tracker[n]['turbine_install_cost'] += install_decom[keys[tech]]['turbine_install_cost'][i] * cell_density / turbine_iterations
            install_decom_tracker[n]['turbine_install_emissions'] += install_decom[keys[tech]]['turbine_install_emissions'][i] * P.CO2_VALUE * cell_density
            install_decom_tracker[n]['new_turbines'] += cell_density
            
            turbine_age[i] = found_age

            if not foundation_exists[i] or foundation_age[i] <= 0:
                # Install new foundation
                install_decom_tracker[n]['foundation_install_cost'] += install_decom[keys[tech]]['foundation_install_cost'][i] * cell_density
                install_decom_tracker[n]['foundation_install_emissions'] += install_decom[keys[tech]]['foundation_install_emissions'][i] * P.CO2_VALUE * cell_density
                install_decom_tracker[n]['new_foundations'] += cell_density
                
                foundation_age[i] = found_age

            foundation_exists[i] = True

    install_decom_tracker['all_time']['turbine_install_cost'] += install_decom_tracker[n]['turbine_install_cost']
    install_decom_tracker['all_time']['foundation_install_cost'] += install_decom_tracker[n]['foundation_install_cost']
    install_decom_tracker['all_time']['turbine_decom_cost'] += install_decom_tracker[n]['turbine_decom_cost']
    install_decom_tracker['all_time']['foundation_decom_cost'] += install_decom_tracker[n]['foundation_decom_cost']
    install_decom_tracker['all_time']['turbine_install_emissions'] += install_decom_tracker[n]['turbine_install_emissions']
    install_decom_tracker['all_time']['foundation_install_emissions'] += install_decom_tracker[n]['foundation_install_emissions']
    install_decom_tracker['all_time']['turbine_decom_emissions'] += install_decom_tracker[n]['turbine_decom_emissions']
    install_decom_tracker['all_time']['foundation_decom_emissions'] += install_decom_tracker[n]['foundation_decom_emissions']
    install_decom_tracker['all_time']['new_turbines'] += install_decom_tracker[n]['new_turbines']
    install_decom_tracker['all_time']['decomissioned_turbines'] += install_decom_tracker[n]['decomissioned_turbines']
    install_decom_tracker['all_time']['new_foundations'] += install_decom_tracker[n]['new_foundations']
    install_decom_tracker['all_time']['decomissioned_foundations'] += install_decom_tracker[n]['decomissioned_foundations']


    return foundation_exists, turbine_age, foundation_age, install_decom_tracker


def calc_max_capacity(current_capacity, turbine_age, foundation_age, foundation_exists, found_age, num_years, start_year, num_boats=25, days_worked=300):

    turbine_iterations = found_age / 25

    construction_times = {
        'decom_turbine': 3.5,
        'decom_foundation': 2.5,
        'decom_inter_array_cables': 1,
        'install_replacement_turbine': 3.5,
        'install_new_turbine': 5,
        'install_new_foundation': 3,
        'install_new_inter_array_cables': 1,
    }

    counters = {
        'decom_turbine': 0,
        'decom_foundation': 0,
        'decom_inter_array_cables': 0,
        'install_replacement_turbine': 0,
        'install_new_turbine': 0,
        'install_new_foundation': 0,
        'install_new_inter_array_cables': 0,
    }

    total_days_available = num_years * days_worked

    for i in range(len(foundation_exists)):
        
        cell_density = 12.5

        # Check for decommissioning
        if foundation_exists[i]:
            
            #if turbine_age[i] divides evenly by 25, then decommission:
            if turbine_age[i] % 25 == 0:
                # Decommission turbine
                counters['decom_turbine'] += cell_density
                counters['decom_inter_array_cables'] += cell_density
                    
                # install turbine if you don't have to replace the foundation
                if foundation_age[i] != 0:
                    counters['install_replacement_turbine'] += cell_density
                    counters['install_new_inter_array_cables'] += cell_density

            if foundation_age[i] <= 0:
                # Decommission foundation if it is aged out
                counters['decom_foundation'] += cell_density

    time_to_replace_turbine = (construction_times['decom_turbine'] + construction_times['decom_inter_array_cables'] + construction_times['install_replacement_turbine'] + construction_times['install_new_inter_array_cables'])
    time_to_fully_decom = (construction_times['decom_turbine'] + construction_times['decom_inter_array_cables'] + construction_times['decom_foundation'])

    days_replacing_turbines = int(time_to_replace_turbine * counters['install_replacement_turbine'] / num_boats)
    days_decommissioning_foundations = int(time_to_fully_decom * counters['decom_foundation'] / num_boats)

    total_days = round(days_replacing_turbines + days_decommissioning_foundations)
    
    time_left = round(total_days_available) - total_days
    
    print(f'days replacing turbines - {days_replacing_turbines}')
    print(f'days decommissioning foundations - {days_decommissioning_foundations}')
    print(f'total days - {total_days}')

    print(f'days left - {time_left}')


    # Calc the possible new installations based on the time left
    fresh_install_time = construction_times['install_new_foundation'] + construction_times['install_new_turbine'] + construction_times['install_new_inter_array_cables']
    possible_new_units = (time_left / fresh_install_time) * num_boats
    possible_added_capacity = ((possible_new_units * 15) / 1000) * 0.75 # 0.75 just to be safe -- can revise or make smarter ;ater

    print(f'Possible new units - {possible_new_units}')
    print(f'Possible added capacity - {possible_added_capacity}')


    def create_capacity_graph():
        daily_capacities = [current_capacity] * total_days_available
        capacity_per_turbine = 15 / 1000  # Capacity of a single turbine in GW
        capacity_per_set = capacity_per_turbine * 5  # Capacity for a set of 5 turbines
        days_per_decom_set = time_to_fully_decom  # Days required to decommission one set of 5 foundations
        days_per_install_set = fresh_install_time  # Days required to install one set of 5 turbines

        for i in range(len(daily_capacities)):
            # Create the portion of the graph where turbines are being replaced in steps
            if i < days_replacing_turbines:
                daily_capacities[i] -= (15 / 1000) * 5 * num_boats  # 15/1000 converts to GW, 5 is because 5 turbines are down while one is replaced

            # Now remove foundations that are being decommissioned in steps
            elif i < days_replacing_turbines + days_decommissioning_foundations:
                days_into_decom = i - days_replacing_turbines
                completed_decom_sets = (days_into_decom // days_per_decom_set) // 5  # Integer division by 5 to get complete sets of 5 foundations
                capacity_decrease = capacity_per_set * completed_decom_sets * num_boats
                daily_capacities[i] = daily_capacities[days_replacing_turbines - 1] - capacity_decrease

            # Now add the new installations in steps of 5 turbines
            else:
                days_into_install = i - days_replacing_turbines - days_decommissioning_foundations
                completed_install_sets = (days_into_install // days_per_install_set) // 5  # Integer division by 5 to get complete sets of 5 turbines
                capacity_increase = capacity_per_set * completed_install_sets * num_boats
                daily_capacities[i] = daily_capacities[days_replacing_turbines + days_decommissioning_foundations - 1] + capacity_increase

            lowest_capacity = min(daily_capacities)

            if daily_capacities[i] > lowest_capacity + possible_added_capacity:
                daily_capacities[i] = lowest_capacity + possible_added_capacity
        
        # lengthen the list to 365 days, filling the rest of the year with the current + added capacity
        if len(daily_capacities) < 365 * 5:
            daily_capacities.extend([lowest_capacity + possible_added_capacity] * (365 * 5 - len(daily_capacities)))

        print(len(daily_capacities))


        # Add the remaining days of the year into the daily capacities

        days_to_end_of_year = 365 - total_days_available

        for i in range(days_to_end_of_year):
            daily_capacities.append(current_capacity + possible_added_capacity)

        # # plot the daily capacities
        # plt.plot(daily_capacities)
        # plt.show()

        return daily_capacities
    
    return possible_added_capacity, create_capacity_graph()


# MARK: ORIGINAL PHASING

def phasing_run(dir, env_name, yearly_goals, sim_params, map_CONFIG, found_age=25, num_format='eu', results_folder=False):
    
    instance_names = list(sim_params.keys())

    # For throwaway testing results
    if results_folder == False:
        out_folder = os.path.join(dir, "test_results", env_name)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    # For any real results
    elif results_folder:
        out_folder = os.path.join(dir, "results", env_name)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    # Set the folders and pathing --- all tech is added into the mod folder with their capacity, and date created
    mod_shp_folder = os.path.join(out_folder, 'mod') # This contains the shp that will have technologies removed from them
    wind_farm_folder = os.path.join(dir, 'data', 'vectors', 'international', 'producing_wind_farms')
    producing_wind_farms_path = os.path.join(wind_farm_folder, 'producing_wind_farms.geojson')
    temp_folder = os.path.join(dir, 'temp')
    pkl_folder = os.path.join(temp_folder, 'pickles')

    # Create the mod folder if it does not exist for the tech to be modified
    if not os.path.exists(mod_shp_folder):
        os.makedirs(mod_shp_folder)
    
    # Modify the current producing wind farms so they will match the incoming tech
    gdf = gpd.read_file(producing_wind_farms_path)
    
    gdf = gdf[['POWER_MW', 'YEAR', 'COUNTRY', 'geometry']]
    gdf['tech'] = 'wind'
    mod_producing_wind_farms_path = os.path.join(mod_shp_folder, 'producing_wind_farms.geojson')
    gdf.to_file(mod_producing_wind_farms_path, driver='GeoJSON')

    # Create the initial temp folder and ocean config, and calc the metrics
    spatial_config = sim_params[instance_names[0]]['sim_env']['config']
    prep_run(dir, spatial_config, hubs={}, add_hubs=False, country='all', found_age=found_age)

    # make a copy and rename the calculated_metrics pkl to original_calculated_metrics
    shutil.copy(os.path.join(pkl_folder, 'calculated_metrics.pkl'), os.path.join(pkl_folder, 'original_calculated_metrics.pkl'))

    # Set the age list
    with open(os.path.join(pkl_folder, 'calculated_metrics.pkl'), 'rb') as file:
        metrics = pickle.load(file)

        list_length = len(metrics['monopile']['capex'])

    with open(os.path.join(pkl_folder, 'install_decom.pkl'), 'rb') as file:
        install_decom = pickle.load(file)
    
    # Convert all nan to 0
    for key in install_decom.keys():
        for sub_key in install_decom[key].keys():
            install_decom[key][sub_key] = [0 if math.isnan(x) else x for x in install_decom[key][sub_key]]

    turbine_age = [0] * list_length
    foundation_age = [0] * list_length
    foundation_exists = [False] * list_length

    install_decom_tracker = {year: {} for year in instance_names}
    install_decom_tracker['all_time'] = {
            'turbine_install_cost': 0,
            'foundation_install_cost': 0,
            'turbine_decom_cost': 0,
            'foundation_decom_cost': 0,
            'turbine_install_emissions': 0,
            'foundation_install_emissions': 0,
            'turbine_decom_emissions': 0,
            'foundation_decom_emissions': 0,
            'new_turbines': 0,
            'decomissioned_turbines': 0,
            'new_foundations': 0,
            'decomissioned_foundations': 0
    } 


    # remove all tifs from the temp folder
    for f in os.listdir(os.path.join(dir, 'temp')):
        if f.endswith('.tif'):
            os.remove(os.path.join(dir, 'temp', f))


    capacity_graphs = {}


    # Start the phasing run
    for i, n in enumerate(instance_names):

        print(n)
        # Prep the run by applying the old tech into the map
        prep_phasing_run(dir, spatial_config, mod_producing_wind_farms_path, n, found_age)

        run_env = sim_params[n]['sim_env']
        run_params = sim_params[n]['sim_params']
        base_single_techs = sim_params[n]['base_single_techs'] 
        run_optimzation_params = sim_params[n]['optimization_params']
        run_env['env_name'] = env_name

        num_boats = run_params['num_boats']
        #days_worked = run_params['days_worked']

        diff = int(n) - int(instance_names[i - 1]) if i != 0 else 0
        turbine_age = [x - diff if x != 0 else x for x in turbine_age]
        foundation_age = [x - diff if x != 0 else x for x in foundation_age]

        gdf = gpd.read_file(mod_producing_wind_farms_path)
        current_capacity = gdf['POWER_MW'].sum() / 1000
        possible_added_capacity, capacity_graphs[n] = calc_max_capacity(current_capacity, turbine_age, foundation_age, foundation_exists, found_age, num_years=5, start_year=n, num_boats=num_boats, days_worked=290)
                
        capacity = current_capacity + possible_added_capacity

        run_params['capacity_needed'] = round(capacity)
        run_env['run_name'] = n

        P = MarinePlan(directory=dir, sim_env=run_env, sim_params=run_params, tech_params=base_single_techs, opt_params=run_optimzation_params, map_CONFIG=map_CONFIG, phasing=True)
        P.prepare_optimization(msg=0, name=env_name)
        P.run_linear_optimization()

        keys = {value: key for key, value in P.uDict.items()}

        install_decom_tracker[n] = {
            'turbine_install_cost': 0,
            'foundation_install_cost': 0,
            'turbine_decom_cost': 0,
            'foundation_decom_cost': 0,
            'turbine_install_emissions': 0,
            'foundation_install_emissions': 0,
            'turbine_decom_emissions': 0,
            'foundation_decom_emissions': 0,
            'new_turbines': 0,
            'decomissioned_turbines': 0,
            'new_foundations': 0,
            'decomissioned_foundations': 0
        }

        foundation_exists, turbine_age, foundation_age, install_decom_tracker = adjust_trackers(P, n, install_decom_tracker, install_decom, foundation_exists, turbine_age, foundation_age, found_age, keys)

        #P.plot_optimal_solution(map_CONFIG=map_CONFIG, install_decom_tracker=install_decom_tracker[n], all_time_tracker=install_decom_tracker['all_time'], num_format=num_format)
        mod_existing_tech(out_folder, mod_producing_wind_farms_path, n)

    def combine_capacity_graphs():
        """
        Combines multiple capacity graphs into one cumulative graph.
        
        :param capacity_graphs: A dictionary where keys are graph names and values are lists of daily capacities
        :return: A list representing the combined cumulative capacity graph
        """
        # Find the total length of the combined graph
        total_length = sum(len(graph) for graph in capacity_graphs.values())
        
        # Initialize the combined graph with zeros
        combined_graph = [0] * total_length
        
        # Current starting index for each graph
        current_index = 0
        
        # Iterate through each graph and add its values to the combined graph
        for graph_name, graph_values in capacity_graphs.items():
            for i, value in enumerate(graph_values):
                combined_graph[current_index + i] += value
            
            # Update the current index for the next graph
            current_index += len(graph_values)

        return combined_graph

    return combine_capacity_graphs()


# MARK: CURSED PHASING

def cursed_phasing(dir, sim_env, sim_params, base_single_tech, optimization_params, gw_step, results_folder=False, goals={}, update_ports=False, custom_ports=False):

    env_name = sim_env['env_name']

    # For throwaway testing results
    if results_folder == False:
        out_folder = os.path.join(dir, "test_results", env_name)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    # For any real results
    elif results_folder:
        out_folder = os.path.join(dir, "results", env_name)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)


    # Set the folders and pathing --- all tech is added into the mod folder with their capacity, and date created
    mod_shp_folder = os.path.join(out_folder, 'mod') # This contains the shp that will have technologies removed from them
    wind_farm_folder = os.path.join(dir, 'data', 'vectors', 'international', 'producing_wind_farms')
    producing_wind_farms_path = os.path.join(wind_farm_folder, 'producing_wind_farms.geojson')
    temp_folder = os.path.join(dir, 'temp')
    pkl_folder = os.path.join(temp_folder, 'pickles')

    # Create the mod folder if it does not exist for the tech to be modified
    if not os.path.exists(mod_shp_folder):
        os.makedirs(mod_shp_folder)
    
    # Modify the current producing wind farms so they will match the incoming tech
    gdf = gpd.read_file(producing_wind_farms_path)
    
    gdf = gdf[['POWER_MW', 'YEAR', 'COUNTRY', 'geometry']]
    gdf['tech'] = 'wind'

    # Remove all rows from the gdf
    gdf = gdf[0:0]

    # Save the modified gdf
    mod_producing_wind_farms_path = os.path.join(mod_shp_folder, 'producing_wind_farms.geojson')
    gdf.to_file(mod_producing_wind_farms_path, driver='GeoJSON')

    # Create the initial temp folder and ocean config, and calc the metrics
    spatial_config = sim_env['config']

    sim_params['capacity_needed'] = gw_step

    # Set trackers
    metric_tracker = {
        'capex': 0,
        'opex': 0,
        'co2+': 0,
        'co2-': 0,
        'revenue': 0,
        'LCOE': 0,
        'energy_produced': 0,
        'food_produced': 0,
        'eco_sensitivity': 0,
    }

    country_capacity_requirements = {
        'FR': 17.0,
        'BE': 8.0,
        'NL': 72.0,
        'DE': 66.0,
        'DK': 35.0,
        'NO': 30.0,
        'UK': 80.0,
    }

    country_capacity_tracker = {
        'FR': 0,
        'BE': 0,
        'NL': 0,
        'DE': 0,
        'DK': 0,
        'NO': 0,
        'UK': 0,
    }
    
    cont = True

    countries_reached = []
    prev_size = len(countries_reached)
    current_size = 0

    original_ports = None
    iteration_tracker = 0
    while cont:

        if (current_size > prev_size) or (iteration_tracker == 0):
            
            if current_size > prev_size:
                prev_size = current_size
                print(f'Countries reached: {countries_reached}, rerunning prep run')

            if iteration_tracker == 0:
                p = prep_run(dir, spatial_config, countries_reached=countries_reached, verbose=False, first_iteration=True, update_ports=update_ports, iterative=True, custom_ports=custom_ports)
                first_ports = copy.deepcopy(p)

            else:
                prep_run(dir, spatial_config, countries_reached=countries_reached, verbose=False, first_iteration=False, update_ports=update_ports, iterative=True, custom_ports=custom_ports)

            shutil.copy(os.path.join(pkl_folder, 'calculated_metrics.pkl'), os.path.join(pkl_folder, 'original_calculated_metrics.pkl'))
        
        prep_phasing_run(dir, spatial_config, mod_producing_wind_farms_path, iteration_tracker, iterative=True, iteration=iteration_tracker)

        P = MarinePlan(directory=dir, sim_env=sim_env, sim_params=sim_params, tech_params=base_single_tech, opt_params=optimization_params)
        P.prepare_optimization(msg=0, name=env_name)
        P.run_linear_optimization()

        metric_tracker['capex'] += pulp.value(P.total_capex)
        metric_tracker['opex'] += pulp.value(P.total_opex)
        metric_tracker['co2+'] += pulp.value(P.total_CO2_emission)
        metric_tracker['co2-'] += pulp.value(P.total_CO2_mitigation)
        metric_tracker['revenue'] += pulp.value(P.total_revenue)
        metric_tracker['LCOE'] += pulp.value(P.total_LCOE)
        metric_tracker['energy_produced'] += pulp.value(P.total_energy_produced)
        metric_tracker['food_produced'] += pulp.value(P.total_food_produced)
        metric_tracker['eco_sensitivity'] += pulp.value(P.total_eco_sensitivity)

        # Save the modified gdf
        mod_existing_tech(out_folder, mod_producing_wind_farms_path, iteration_tracker, iterative=True)
        iteration_tracker += 1

        # Check if the capacity requirements have been met
        for country in country_capacity_requirements.keys():

            if country in countries_reached:
                continue

            country_capacity_tracker[country] += P.country_capacity[country] if country in P.country_capacity.keys() else 0

            if country_capacity_tracker[country] >= country_capacity_requirements[country]:
                countries_reached.append(country)
        
            print(f'Country: {country} -- Capacity: {country_capacity_tracker[country]} -- Requirement: {country_capacity_requirements[country]}')

        # Check to see if any new countries have been reached
        current_size = len(countries_reached)

        map_CONFIG = {
            'scale': 'international',
            'output_type': 'energy_targets',
            'msp': {
                'shipping': (True, "#333333"),
                'military': (True, "#4D4D4D"),
                'sand_extraction': (False, "#F4A460"),
                'nature_reserves': (True, "#4D4D4D"),
                'energy_zones': (False, "#000000"),
                'energy_zones_type': 'whole_zone',
                'wind_farms': (False, {
                    "approved": "#FFD700",
                    "planned": "#EEE8AA",
                    "under_construction": "#FF7F50",
                    "operational": "#008080"
                }),
                'legacy_farms': (False, "#000000"),
                'interconnectors': (False, "#000000"),
                'cables': (False, {
                    "IC_to_grid": "red",
                    "farm_to_IC": "#FFD700",
                    "farm_to_grid": "#808080"
                })
            },
            'colours': {
                'mussel': '#800000',      
                'seaweed': '#006400',    
                'monopile': '#1A5FC1',        
                'jacket': '#00BFFF',           
                'fpv': '#FFA500',           
                'semisub_cat_drag': '#FF6347', 
                'semisub_taut_driv': '#FF6347',
                'semisub_taut_suc': '#FF6347',
                'spar_cat_drag': '#9370DB', 
                'spar_taut_driv': '#9370DB',
                'spar_taut_suc': '#9370DB',
            },
        }


        if len(countries_reached) == len(country_capacity_requirements.keys()):
            cont = False

    for metric, value in metric_tracker.items():
        # Print the value with commas for readability
        print(f'{metric}: {value:,.2f}')

    for countr, capacity in country_capacity_tracker.items():
        print(f'{countr} - {capacity} ----- {country_capacity_requirements[countr]}')
        
    plot_cursed_phasing(P, map_CONFIG=map_CONFIG, metric_tracker=metric_tracker, capacity_tracker=country_capacity_tracker, capacity_targets=country_capacity_requirements, num_format='eu', ports=first_ports)

    return P


def plot_cursed_phasing(mp, map_CONFIG=None, metric_tracker={}, capacity_tracker={}, capacity_targets={}, num_format='eu', ports={}):
    
    # Load the appropriate files
    mp._load_shp(map_CONFIG)

    # Set the formatting for the plot
    custom_cmap = ListedColormap(
        [map_CONFIG["colours"][value] for value in mp.seed_gdf["value"].unique()]
    )  # Set the colourmap

    mp.fig, mp.ax = plt.subplots(figsize=(16, 12))
    mp.fig.patch.set_facecolor("#333333")  # Set the background color of the entire figure
    mp.ax.set_facecolor("#1f1f1f")  # Set the background color of the plot area (the colour that the ocean ends up)

    # Plot the old tech (grey) and the new tech locations (coloured)
    # mp.existing_tech_gdf.plot(ax=mp.ax, column='COUNTRY', cmap=custom_cmap, alpha=0.23) if mp.existing_tech_gdf is not None else None
    # mp.seed_gdf.plot(ax=mp.ax, column="value", cmap=custom_cmap)


    country_colors = {
        'UK': '#E01E3C',  # Bright red
        'NO': '#21B0AA',  # Turquoise
        'DK': '#FFC300',  # Bright yellow
        'DE': 'darkblue', # Dark blue
        'NL': '#F16A2D',  # Orange
        'BE': '#6CACE4',  # Light blue
        'FR': '#78BE20'   # Lime green
    }

    def plot_gdf_with_colors(gdf, ax, column_name, a):
        # Create a color column based on the country column
        gdf['color'] = gdf[column_name].map(country_colors)
        
        try:
            monopiles = gdf[gdf['value'] == 'monopile']
            jackets = gdf[gdf['value'] == 'jacket']
        except:
            monopiles = gdf[gdf['tech'] == 'monipile']
            jackets = gdf[gdf['tech'] == 'jacket']

        monopiles.plot(ax=ax, color=monopiles['color'], alpha=a)
        jackets.plot(ax=ax, color=jackets['color'], alpha=a, hatch='//')

    # Change 'country' to 'COUNTRY' in the seed_gdf
    mp.seed_gdf.rename(columns={'country': 'COUNTRY'}, inplace=True)
    mp.seed_gdf.drop(columns=['capacity'], inplace=True)
    mp.existing_tech_gdf.drop(columns=['POWER_MW', 'YEAR'], inplace=True)

    # combine the seed-gdf and existing_tech_gdf
    mp.existing_tech_gdf = pd.concat([mp.existing_tech_gdf, mp.seed_gdf], ignore_index=True)
    mp.existing_tech_gdf = mp.existing_tech_gdf.dissolve(by=['COUNTRY', 'value'], aggfunc='sum')

    def filter_small_polygons(geom, min_area=134_000_000):
        if geom.geom_type == 'MultiPolygon':
            filtered_polys = [poly for poly in geom.geoms if poly.area >= min_area]
            if len(filtered_polys) > 0:
                return MultiPolygon(filtered_polys)
            else:
                return None
        elif geom.geom_type == 'Polygon':
            return geom if geom.area >= min_area else None
        else:
            return geom

    # Assuming your dissolved GeoDataFrame is called 'dissolved_gdf'
    mp.existing_tech_gdf['geometry'] = mp.existing_tech_gdf['geometry'].apply(filter_small_polygons)
    mp.existing_tech_gdf = mp.existing_tech_gdf.dropna(subset=['geometry'])
    mp.existing_tech_gdf = mp.existing_tech_gdf.reset_index()

    # Plot existing_tech_gdf if it's not None
    if mp.existing_tech_gdf is not None:
        plot_gdf_with_colors(mp.existing_tech_gdf, mp.ax, 'COUNTRY', a=1)

    # Plot cluster_gdf
    # plot_gdf_with_colors(mp.seed_gdf, mp.ax, 'country', a=1)


    mp._plot_files(map_CONFIG, redo_land_objects=True)


    colours = {
    'substation': 'yellow',
    'ins': 'lightgreen',
    'opr': 'orange',
    'both': 'red',
    'port': 'blue',
    'default': 'gray'
    }

    land_objects = []
    for country, locations in ports.items():
        for name, info in locations.items():
            land_objects.append({
                'name': name,
                'designation': info['designation'],
                'country': info.get('country', country),  # Use 'country' from info if available, else use the outer key
                'geometry': Point(info['longitude'], info['latitude'])
            })
    
    # Create GeoDataFrame for ports and substations
    plz = gpd.GeoDataFrame(land_objects, crs="EPSG:4326")
    plz = plz.to_crs("EPSG:3035")

    plz = plz.to_crs("EPSG:3035")
    for designation in plz['designation'].unique():
        subset = plz[plz['designation'] == designation]
        color = colours.get(designation.lower(), colours['default'])
        
        subset.plot(
            ax=mp.ax,
            zorder=20 if designation != 'substation' else 19,
            marker='o' if designation != 'substation' else '*',
            markersize=35 if designation != 'substation' else 200,  # Consistent size for all markers
            color=color,
            alpha=0.75, # Slight transparency
            label=designation,
            linewidths=0.5,
            edgecolors='black'
        )



    # Set the colours of the figure edges
    mp.ax.spines["bottom"].set_color("white")
    mp.ax.spines["top"].set_color("white")
    mp.ax.spines["right"].set_color("white")
    mp.ax.spines["left"].set_color("white")

    # Set the label colours
    mp.ax.xaxis.label.set_color("white")
    mp.ax.yaxis.label.set_color("white")

    # Set the tick colours
    mp.ax.tick_params(axis="x", colors="white", labelfontfamily="monospace", labelsize=12)
    mp.ax.tick_params(axis="y", colors="white", labelfontfamily="monospace", labelsize=12)

    # Set zoom distance
    xlim_min, ylim_min, xlim_max, ylim_max = mp.aoi_gdf.total_bounds
        
    mp.ax.set_xlim(xlim_min, xlim_max)
    mp.ax.set_ylim(ylim_min, ylim_max)
    mp.ax.set_title(mp.run_name, color='white')

    # Format the units so they are % 1000
    mp.ax.xaxis.set_major_formatter(FuncFormatter(mp._utm_formatter))
    mp.ax.yaxis.set_major_formatter(FuncFormatter(mp._utm_formatter))

    def create_combined_legend(config):

        colors = {
            'substation': 'yellow',
            'ins': 'lightgreen',
            'opr': 'orange',
            'both': 'red',
            'port': 'blue',
            'default': 'gray'
        }
            
        existing_legend = [
            plt.Line2D([0], [0], color='white', linewidth=0, label='------- MSP ---------', linestyle='None'),
            Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
            Patch(color=config['msp']['nature_reserves'][1], alpha=0.5, lw=2, hatch='...', label='Nature Reserves'),
            Patch(color=config['msp']['military'][1], alpha=0.5, hatch='xxx', lw=2, label='Military Shooting'),

            plt.Line2D([0], [0], color='white', linewidth=0, label='------- Countries ---------', linestyle='None'),

            Patch(color=country_colors['UK'], alpha=1, lw=2, label='UK'),
            Patch(color=country_colors['NO'], alpha=1, lw=2, label='NO'),
            Patch(color=country_colors['DK'], alpha=1, lw=2, label='DK'),
            Patch(color=country_colors['DE'], alpha=1, lw=2, label='DE'),
            Patch(color=country_colors['NL'], alpha=1, lw=2, label='NL'),
            Patch(color=country_colors['BE'], alpha=1, lw=2, label='BE'),
            Patch(color=country_colors['FR'], alpha=1, lw=2, label='FR'),
    
            plt.Line2D([0], [0], color='white', linewidth=0, label='------- Technologies ---------', linestyle='None'),
            Patch(color='grey', alpha=1, lw=2, label='Monopile'),
            Patch(color='grey', alpha=1, hatch='//', lw=2, label='Jacket'),

        ]

        port_legend = [
            plt.Line2D([0], [0], color='white', linewidth=0, label='------- Ports & Substations ---------', linestyle='None'),
            Line2D([0], [0], color=colors['substation'], marker='*', linestyle='None', markersize=8, label='Substation'),
            Line2D([0], [0], color=colors['ins'], marker='o', linestyle='None', markersize=8, label='Installation'),
            Line2D([0], [0], color=colors['opr'], marker='o', linestyle='None', markersize=8, label='Operational'),
            Line2D([0], [0], color=colors['both'], marker='o', linestyle='None', markersize=8, label='Both'),
        ]
        return existing_legend + port_legend

    # Now you can create the legend when needed:
    legend_handles = create_combined_legend(map_CONFIG)

    # Display the legend
    mp.ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.02, 0.00),  # Adjust these values as needed
        prop={"family": "monospace", "size": 8},  # Reduced font size further
        labelcolor="white",
        facecolor="#1f1f1f",
        frameon=True,
        framealpha=0.7,  # Semi-transparent background
        ncol=3,  # Display in 3 columns
        columnspacing=1.5,  # Adjust spacing between columns
        handlelength=1.5,  # Reduce handle length to save space
        handletextpad=0.5,  # Reduce space between handle and text
    )

    #mp._build_text()
    outputs = TextOutputGenerator(mp)
    text = outputs.generate_map_text('cursed_phasing', metric_tracker=metric_tracker, capacity_tracker=capacity_tracker, capacity_targets=capacity_targets, num_format=num_format)

    props = dict(boxstyle="round", facecolor="#1f1f1f", alpha=0)

    mp.ax.text(
        1.02,
        1.008,
        text,
        transform=mp.ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        fontfamily="monospace",
        color="white",
    )

    mp.ax.set_aspect("equal")

    plt.show()

    plt.close()


# MARK: ROADMAP PHASING

def roadmap_phasing(dir, env_name, yearly_goals, original_sim_params, results_folder=False, year_interval=5):
    

    # For throwaway testing results
    if results_folder == False:
        out_folder = os.path.join(dir, "test_results", env_name)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    # For any real results
    elif results_folder:
        out_folder = os.path.join(dir, "results", env_name)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    # Set the folders and pathing --- all tech is added into the mod folder with their capacity, and date created
    mod_shp_folder = os.path.join(out_folder, 'mod') # This contains the shp that will have technologies removed from them
    wind_farm_folder = os.path.join(dir, 'data', 'vectors', 'international', 'producing_wind_farms')
    producing_wind_farms_path = os.path.join(wind_farm_folder, 'producing_wind_farms.geojson')
    temp_folder = os.path.join(dir, 'temp')
    pkl_folder = os.path.join(temp_folder, 'pickles')
    closest_install_path = os.path.join(temp_folder, 'closest_install_port.pkl')

    # Create the mod folder if it does not exist for the tech to be modified
    if not os.path.exists(mod_shp_folder):
        os.makedirs(mod_shp_folder)
    
    # Modify the current producing wind farms so they will match the incoming tech
    gdf = gpd.read_file(producing_wind_farms_path)
    
    gdf = gdf[['POWER_MW', 'YEAR', 'COUNTRY', 'geometry']]
    gdf['tech'] = 'wind'

    # Remove all rows from the gdf
    gdf = gdf[0:0]

    # Save the modified gdf
    mod_producing_wind_farms_path = os.path.join(mod_shp_folder, 'producing_wind_farms.geojson')
    gdf.to_file(mod_producing_wind_farms_path, driver='GeoJSON')

    # Create the initial temp folder and ocean config, and calc the metrics
    spatial_config = original_sim_params['sim_env']['config']
    run_env = original_sim_params['sim_env']
    run_params = original_sim_params['sim_params']
    base_single_techs = original_sim_params['base_single_techs']
    run_optimzation_params = original_sim_params['optimization_params']
    hubs = original_sim_params['hubs']
    add_hubs = original_sim_params['add_hubs']
    ports_to_modify = original_sim_params['ports_to_modify']
    map_CONFIG = original_sim_params['map_CONFIG']

    update_ports = original_sim_params['update_ports']
    custom_ports = original_sim_params['custom_ports']

    total_capacity = 0
    iteration_tracker = 0

    # Set trackers
    metric_tracker = {
        'capex': 0,
        'opex': 0,
        'co2+': 0,
        'co2-': 0,
        'revenue': 0,
        'LCOE': 0,
        'energy_produced': 0,
        'food_produced': 0,
        'eco_sensitivity': 0,
    }

    country_capacity_requirements = {
        'FR': 17.0,
        'BE': 8.0,
        'NL': 72.0,
        'DE': 66.0,
        'DK': 35.0,
        'NO': 30.0,
        'UK': 80.0,
    }

    country_capacity_tracker = {
        'FR': 0,
        'BE': 0,
        'NL': 0,
        'DE': 0,
        'DK': 0,
        'NO': 0,
        'UK': 0,
    }

    # Start the phasing run
    while total_capacity < 50:

        print(total_capacity)

        i = iteration_tracker * year_interval

        if iteration_tracker == 0:
            p = prep_run(dir, spatial_config, verbose=False, first_iteration=True, update_ports=update_ports, iterative=False, custom_ports=custom_ports, ports_to_modify=ports_to_modify)
            first_ports = copy.deepcopy(p)

            # make a copy and rename the calculated_metrics pkl to original_calculated_metrics
            shutil.copy(os.path.join(pkl_folder, 'calculated_metrics.pkl'), os.path.join(pkl_folder, 'original_calculated_metrics.pkl'))

            # remove all tifs from the temp folder
            for f in os.listdir(os.path.join(dir, 'temp')):
                if f.endswith('.tif'):
                    os.remove(os.path.join(dir, 'temp', f))

        else:
            # recreate the dictionary of port capacities. Honestly dont want to do this properly right now, so just rerun this LandObjects classto do it quickly

            print(f'Added boats: {i}')

            prep_run(dir, spatial_config, verbose=False, first_iteration=False, update_ports=update_ports, iterative=False, custom_ports=custom_ports, roadmap=False, i=i, ports_to_modify=ports_to_modify)
            
            shutil.copy(os.path.join(pkl_folder, 'calculated_metrics.pkl'), os.path.join(pkl_folder, 'original_calculated_metrics.pkl'))

            # remove all tifs from the temp folder
            for f in os.listdir(os.path.join(dir, 'temp')):
                if f.endswith('.tif'):
                    os.remove(os.path.join(dir, 'temp', f))


        prep_phasing_run(dir, spatial_config, mod_producing_wind_farms_path, iteration_tracker, iterative=False, iteration=iteration_tracker, remove=True)

        with open(closest_install_path, 'rb') as file:
            closest_install = pickle.load(file)

        added_turbines = sum([v for v in closest_install.values()])
        added_capacity = added_turbines * 15 / 1000
        total_capacity += added_capacity

        P = MarinePlan(directory=dir, sim_env=run_env, sim_params=run_params, tech_params=base_single_techs, opt_params=run_optimzation_params)
        P.prepare_optimization(msg=0, name=env_name)
        P.run_linear_optimization()

        metric_tracker['capex'] += pulp.value(P.total_capex)
        metric_tracker['opex'] += pulp.value(P.total_opex)
        metric_tracker['co2+'] += pulp.value(P.total_CO2_emission)
        metric_tracker['co2-'] += pulp.value(P.total_CO2_mitigation)
        metric_tracker['revenue'] += pulp.value(P.total_revenue)
        metric_tracker['LCOE'] += pulp.value(P.total_LCOE)
        metric_tracker['energy_produced'] += pulp.value(P.total_energy_produced)
        metric_tracker['food_produced'] += pulp.value(P.total_food_produced)
        metric_tracker['eco_sensitivity'] += pulp.value(P.total_eco_sensitivity)


        #P.plot_optimal_solution(map_CONFIG=map_CONFIG, install_decom_tracker=install_decom_tracker[n], all_time_tracker=install_decom_tracker['all_time'], num_format=num_format)
        mod_existing_tech(out_folder, mod_producing_wind_farms_path, iteration_tracker, iterative=True)
        iteration_tracker += 1

                # Check if the capacity requirements have been met
        for country in country_capacity_requirements.keys():

            country_capacity_tracker[country] += P.country_capacity[country] if country in P.country_capacity.keys() else 0



    plot_roadmap_phasing(P, map_CONFIG=map_CONFIG, metric_tracker=metric_tracker, capacity_tracker=country_capacity_tracker, capacity_targets=country_capacity_requirements, num_format='eu', ports=first_ports)



def plot_roadmap_phasing(mp, map_CONFIG=None, metric_tracker={}, capacity_tracker={}, capacity_targets={}, num_format='eu', ports={}):
    
    # Load the appropriate files
    mp._load_shp(map_CONFIG)

    # Set the formatting for the plot
    custom_cmap = ListedColormap(
        [map_CONFIG["colours"][value] for value in mp.seed_gdf["value"].unique()]
    )  # Set the colourmap

    mp.fig, mp.ax = plt.subplots(figsize=(16, 12))
    mp.fig.patch.set_facecolor("#333333")  # Set the background color of the entire figure
    mp.ax.set_facecolor("#1f1f1f")  # Set the background color of the plot area (the colour that the ocean ends up)

    # Plot the old tech (grey) and the new tech locations (coloured)
    # mp.existing_tech_gdf.plot(ax=mp.ax, column='COUNTRY', cmap=custom_cmap, alpha=0.23) if mp.existing_tech_gdf is not None else None
    # mp.seed_gdf.plot(ax=mp.ax, column="value", cmap=custom_cmap)


    country_colors = {
        'UK': '#E01E3C',  # Bright red
        'NO': '#21B0AA',  # Turquoise
        'DK': '#FFC300',  # Bright yellow
        'DE': 'darkblue', # Dark blue
        'NL': '#F16A2D',  # Orange
        'BE': '#6CACE4',  # Light blue
        'FR': '#78BE20'   # Lime green
    }

    def plot_gdf_with_colors(gdf, ax, column_name, a):
        # Create a color column based on the country column
        gdf['color'] = gdf[column_name].map(country_colors)
        
        try:
            monopiles = gdf[gdf['value'] == 'monopile']
            jackets = gdf[gdf['value'] == 'jacket']
        except:
            monopiles = gdf[gdf['tech'] == 'monipile']
            jackets = gdf[gdf['tech'] == 'jacket']

        monopiles.plot(ax=ax, color=monopiles['color'], alpha=a)
        jackets.plot(ax=ax, color=jackets['color'], alpha=a, hatch='//')

    # Change 'country' to 'COUNTRY' in the seed_gdf
    mp.seed_gdf.rename(columns={'country': 'COUNTRY'}, inplace=True)
    mp.seed_gdf.drop(columns=['capacity'], inplace=True)
    mp.existing_tech_gdf.drop(columns=['POWER_MW', 'YEAR'], inplace=True)

    # combine the seed-gdf and existing_tech_gdf
    mp.existing_tech_gdf = pd.concat([mp.existing_tech_gdf, mp.seed_gdf], ignore_index=True)
    mp.existing_tech_gdf = mp.existing_tech_gdf.dissolve(by=['COUNTRY', 'value'], aggfunc='sum')

    def filter_small_polygons(geom, min_area=134_000_000):
        if geom.geom_type == 'MultiPolygon':
            filtered_polys = [poly for poly in geom.geoms if poly.area >= min_area]
            if len(filtered_polys) > 0:
                return MultiPolygon(filtered_polys)
            else:
                return None
        elif geom.geom_type == 'Polygon':
            return geom if geom.area >= min_area else None
        else:
            return geom

    # Assuming your dissolved GeoDataFrame is called 'dissolved_gdf'
    mp.existing_tech_gdf['geometry'] = mp.existing_tech_gdf['geometry'].apply(filter_small_polygons)
    mp.existing_tech_gdf = mp.existing_tech_gdf.dropna(subset=['geometry'])
    mp.existing_tech_gdf = mp.existing_tech_gdf.reset_index()

    # Plot existing_tech_gdf if it's not None
    if mp.existing_tech_gdf is not None:
        plot_gdf_with_colors(mp.existing_tech_gdf, mp.ax, 'COUNTRY', a=1)

    # Plot cluster_gdf
    # plot_gdf_with_colors(mp.seed_gdf, mp.ax, 'country', a=1)


    mp._plot_files(map_CONFIG, redo_land_objects=True)


    colours = {
    'substation': 'yellow',
    'ins': 'lightgreen',
    'opr': 'orange',
    'both': 'red',
    'port': 'blue',
    'default': 'gray'
    }

    land_objects = []
    for country, locations in ports.items():
        for name, info in locations.items():
            land_objects.append({
                'name': name,
                'designation': info['designation'],
                'country': info.get('country', country),  # Use 'country' from info if available, else use the outer key
                'geometry': Point(info['longitude'], info['latitude'])
            })
    
    # Create GeoDataFrame for ports and substations
    plz = gpd.GeoDataFrame(land_objects, crs="EPSG:4326")
    plz = plz.to_crs("EPSG:3035")

    plz = plz.to_crs("EPSG:3035")
    for designation in plz['designation'].unique():
        subset = plz[plz['designation'] == designation]
        color = colours.get(designation.lower(), colours['default'])
        
        subset.plot(
            ax=mp.ax,
            zorder=20 if designation != 'substation' else 19,
            marker='o' if designation != 'substation' else '*',
            markersize=35 if designation != 'substation' else 200,  # Consistent size for all markers
            color=color,
            alpha=0.75, # Slight transparency
            label=designation,
            linewidths=0.5,
            edgecolors='black'
        )



    # Set the colours of the figure edges
    mp.ax.spines["bottom"].set_color("white")
    mp.ax.spines["top"].set_color("white")
    mp.ax.spines["right"].set_color("white")
    mp.ax.spines["left"].set_color("white")

    # Set the label colours
    mp.ax.xaxis.label.set_color("white")
    mp.ax.yaxis.label.set_color("white")

    # Set the tick colours
    mp.ax.tick_params(axis="x", colors="white", labelfontfamily="monospace", labelsize=12)
    mp.ax.tick_params(axis="y", colors="white", labelfontfamily="monospace", labelsize=12)

    # Set zoom distance
    xlim_min, ylim_min, xlim_max, ylim_max = mp.aoi_gdf.total_bounds
        
    mp.ax.set_xlim(xlim_min, xlim_max)
    mp.ax.set_ylim(ylim_min, ylim_max)
    mp.ax.set_title(mp.run_name, color='white')

    # Format the units so they are % 1000
    mp.ax.xaxis.set_major_formatter(FuncFormatter(mp._utm_formatter))
    mp.ax.yaxis.set_major_formatter(FuncFormatter(mp._utm_formatter))

    def create_combined_legend(config):

        colors = {
            'substation': 'yellow',
            'ins': 'lightgreen',
            'opr': 'orange',
            'both': 'red',
            'port': 'blue',
            'default': 'gray'
        }
            
        existing_legend = [
            plt.Line2D([0], [0], color='white', linewidth=0, label='------- MSP ---------', linestyle='None'),
            Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
            Patch(color=config['msp']['nature_reserves'][1], alpha=0.5, lw=2, hatch='...', label='Nature Reserves'),
            Patch(color=config['msp']['military'][1], alpha=0.5, hatch='xxx', lw=2, label='Military Shooting'),

            plt.Line2D([0], [0], color='white', linewidth=0, label='------- Countries ---------', linestyle='None'),

            Patch(color=country_colors['UK'], alpha=1, lw=2, label='UK'),
            Patch(color=country_colors['NO'], alpha=1, lw=2, label='NO'),
            Patch(color=country_colors['DK'], alpha=1, lw=2, label='DK'),
            Patch(color=country_colors['DE'], alpha=1, lw=2, label='DE'),
            Patch(color=country_colors['NL'], alpha=1, lw=2, label='NL'),
            Patch(color=country_colors['BE'], alpha=1, lw=2, label='BE'),
            Patch(color=country_colors['FR'], alpha=1, lw=2, label='FR'),
    
            plt.Line2D([0], [0], color='white', linewidth=0, label='------- Technologies ---------', linestyle='None'),
            Patch(color='grey', alpha=1, lw=2, label='Monopile'),
            Patch(color='grey', alpha=1, hatch='//', lw=2, label='Jacket'),

        ]

        port_legend = [
            plt.Line2D([0], [0], color='white', linewidth=0, label='------- Ports & Substations ---------', linestyle='None'),
            Line2D([0], [0], color=colors['substation'], marker='*', linestyle='None', markersize=8, label='Substation'),
            Line2D([0], [0], color=colors['ins'], marker='o', linestyle='None', markersize=8, label='Installation'),
            Line2D([0], [0], color=colors['opr'], marker='o', linestyle='None', markersize=8, label='Operational'),
            Line2D([0], [0], color=colors['both'], marker='o', linestyle='None', markersize=8, label='Both'),
        ]
        return existing_legend + port_legend

    # Now you can create the legend when needed:
    legend_handles = create_combined_legend(map_CONFIG)

    # Display the legend
    mp.ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.02, 0.00),  # Adjust these values as needed
        prop={"family": "monospace", "size": 8},  # Reduced font size further
        labelcolor="white",
        facecolor="#1f1f1f",
        frameon=True,
        framealpha=0.7,  # Semi-transparent background
        ncol=3,  # Display in 3 columns
        columnspacing=1.5,  # Adjust spacing between columns
        handlelength=1.5,  # Reduce handle length to save space
        handletextpad=0.5,  # Reduce space between handle and text
    )

    #mp._build_text()
    outputs = TextOutputGenerator(mp)
    text = outputs.generate_map_text('cursed_phasing', metric_tracker=metric_tracker, capacity_tracker=capacity_tracker, capacity_targets=capacity_targets, num_format=num_format)

    props = dict(boxstyle="round", facecolor="#1f1f1f", alpha=0)

    mp.ax.text(
        1.02,
        1.008,
        text,
        transform=mp.ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        fontfamily="monospace",
        color="white",
    )

    mp.ax.set_aspect("equal")

    plt.show()

    plt.close()