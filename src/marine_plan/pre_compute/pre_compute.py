import numpy as np
import pickle
from math import sqrt
import rasterio
import os
import sys
import shutil
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt


from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Imports
import os
import warnings 
import sys


from src.marine_plan.pre_compute.tech_equations.monopile_15MW import monopile_15MW
from src.marine_plan.pre_compute.tech_equations.jacket_15MW import jacket_15MW
from src.marine_plan.pre_compute.tech_equations.floating_PV_solo import floating_PV_solo
from src.marine_plan.pre_compute.tech_equations.semisub_15MW_cat_drag import semisub_15MW_cat_drag
from src.marine_plan.pre_compute.tech_equations.semisub_15MW_taut_driv import semisub_15MW_taut_driv
from src.marine_plan.pre_compute.tech_equations.semisub_15MW_taut_suc import semisub_15MW_taut_suc
from src.marine_plan.pre_compute.tech_equations.spar_15MW_cat_drag import spar_15MW_cat_drag
from src.marine_plan.pre_compute.tech_equations.spar_15MW_taut_driv import spar_15MW_taut_driv
from src.marine_plan.pre_compute.tech_equations.spar_15MW_taut_suc import spar_15MW_taut_suc
from src.marine_plan.pre_compute.tech_equations.mussel_longline import mussel_longline
from src.marine_plan.pre_compute.tech_equations.seaweed_longline import seaweed_longline

from src.marine_plan.pre_compute.parallelization import process_tech
from src.raster_util import *
from src.land_objects import LandObjects


# MARK: Helpers

def equal_dimensions(array):
    '''
    Adds empty rows/columns to an array to create equal dimensions so the array can be reshaped once flattened
    '''
    current_height, current_width = array.shape
    max_dimension = max(current_height, current_width)
    rows_to_add = max_dimension - current_height
    cols_to_add = max_dimension - current_width
    
    return np.pad(array, ((0, rows_to_add), (0, cols_to_add)), mode='constant', constant_values=np.nan)


def full_mask(data_dict):
    """
    Apply a full mask to all arrays in the given data dictionary.

    This function iterates over all arrays in the data dictionary and applies a full mask,
    where all elements that are NaN (Not a Number) are replaced with NaN in all other arrays.

    Parameters:
    - data_dict (dict): A dictionary containing arrays as values.

    Returns:
    - None
    """
    for array in data_dict.values():
        nan_mask = np.isnan(array)
        for other_array in data_dict.values():
            other_array[nan_mask] = np.nan


def create_pickle(folder, scale=None):

    '''
    Creates the dictionary of the geospatial data that is used for the calculations of the metrics

    Parameters:
        folder {directory} : Folder full of the tif files. 
    '''
    
    if scale == None: # Applies if the data is being processed in the temp folder, ie it has been masked
        output_file = os.path.join(folder, 'pickles', 'geo_data.pkl')
    else: # For the initial creation of the dicts, no masks applied to this
        output_file = os.path.join(folder, 'data', 'pickles', scale, 'geo_data.pkl')
        raster_folder = os.path.join(folder, 'data', 'rasters', scale)

    tif_files = [f for f in os.listdir(folder) if f.endswith('.tif')] # Creates list of tif files from folder

    data = {}

    # Loops through tif files and loads them in as arrays
    for tif_file in tif_files:

        tif_path = os.path.join(folder, tif_file)
        
        with rasterio.open(tif_path) as src:
            array = src.read(1)
            array = array.astype(float)
            # If tif file is depth.tif take the absolute value
            if 'depth' in tif_file:
                array = np.abs(array)
            

            if 'eco_sens_aquaculture' in tif_file:
                array[(array <= -5) | (array > 1e16)] = np.nan # Changes extreme values to NaN
            else:
                array[(array <= 0) | (array > 1e16)] = np.nan # Changes extreme values to NaN
            
            array = equal_dimensions(array)
        data[tif_file.split('.')[0]] = array.flatten()

    full_mask(data)

    with open(output_file, 'wb') as file:
        pickle.dump(data, file) 


def add_geodata(geo, lo, scale, closest_gridconnects, iterative=False):

    '''
    Adds the geographic data to each tech

    Adjust as necassary with other data
    '''

    all_metrics = {}
    # Geospatial data
    all_metrics['mean_wind_speed_at_10m'] = geo['wind_speed']
    all_metrics['depth'] = geo['depth']
    all_metrics['dsh'] = geo['d_shore'] / 1_000

    # If the average for dsh is less than 1, * 1000
    if np.nanmean(all_metrics['dsh']) < 1:
        all_metrics['dsh'] = [x * 1_000 for x in all_metrics['dsh']]

    geo_type = {
        1: int(4_000_000),
        2: int(64_000_000),
        3: int(16_000_000),
        4: int(32_000_000),
        5: int(5_000_000),
    } 

    vectorized_func = np.vectorize(lambda x: geo_type.get(x, np.nan), otypes=[float])
    
    # apply the function to the seabed data
    all_metrics['soil_coefficient'] = vectorized_func(geo['seabed_substrate'])
    all_metrics['soil_friction_angle'] = geo['seabed_surface']


    # Country ID
    gc_country = {}

    for name, values in closest_gridconnects.items():
        gc_country[values['index']] = values['country']
    
    no_nan = np.nan_to_num(lo['closest_gridconnect'])

    if iterative:
        all_metrics['country'] = [gc_country[num] for num in no_nan]
    else:
        all_metrics['country'] = geo['country']

    # Distances to the nearest land objects
    shape = sqrt(len(geo['depth']))
    lo['ws'] = geo['wind_speed']
    full_mask(lo) # Gets it to the same shape as the other arrays
    all_metrics['distance_to_OPEX_port'] = lo['dto'] / 1_000
    all_metrics['distance_to_installation_port'] = lo['dti'] / 1_000
    all_metrics['distance_to_onshore_sub'] = lo['dts'] / 1_000
    # all_metrics['distance_to_onshore_sub'] = geo['d_shore'] / 1_000
    all_metrics['closest_gridconnect'] = lo['closest_gridconnect']
    all_metrics['closest_install_port'] = lo['closest_install_port']
    # all_metrics['distance_to_onshore_sub'] = [0] * len(all_metrics['depth'])

    # Fishing intensity
    all_metrics['fishing_hours'] = geo['fishing_hours']
    all_metrics['subsurface_swept_ratio'] = geo['subsurface_swept_ratio']
    all_metrics['surface_swept_ratio'] = geo['surface_swept_ratio']
    
    # eco sensitivity
    all_metrics['eco_sens_wind'] = geo['eco_sens_wind']
    all_metrics['eco_sens_fpv'] = geo['eco_sens_fpv']
    all_metrics['eco_sens_aquaculture'] = geo['eco_sens_aquaculture']            

    return all_metrics


def format_arrays(directory, lo):

    geo_data_path = os.path.join(directory, 'pickles', 'geo_data.pkl')
    lo_path = os.path.join(directory, 'pickles', 'land_objects.pkl')

    with open(geo_data_path, 'rb') as f:
        geo_data = pickle.load(f)

    lo['ws'] = geo_data['wind_speed']
    full_mask(lo)
    del lo['ws']

    with open(lo_path, 'wb') as file:
        pickle.dump(lo, file)


def mask_metrics(temp_folder):

    ''' 
    Uses the geo-data pickle to mask the tech metrics pickle. This ensure that any area that is masked has no values
    '''

    pkl_folder = os.path.join(temp_folder, 'pickles')

    pickle_files = [f for f in os.listdir(pkl_folder) if f.endswith('.pkl')]

    for pkl_file in pickle_files:
        if 'geo_data' in pkl_file:
            with open(os.path.join(pkl_folder, pkl_file), 'rb') as pkl:
                geo = pickle.load(pkl)

        if 'calculated_metrics' in pkl_file:
            with open(os.path.join(pkl_folder, pkl_file), 'rb') as pkl:
                metrics = pickle.load(pkl)

        

    m = np.isnan((geo.get('depth', [])))  # Use get to provide a default empty list if 'depth' is not present

    m = list(m)
    
    for tech, _metrics in metrics.items():
        for metric, values in _metrics.items():
            metrics[tech][metric] = np.where(m, np.nan, values)

    new_metric_file = os.path.join(pkl_folder, 'calculated_metrics.pkl')
    with open(new_metric_file, 'wb') as file:
        pickle.dump(metrics, file)


def data_processing(folder, functions, shp_folder, phasing=False):

    '''
    Applies masking to rasters. Note that this folder ONLY alters data within the temporary folder, so prep_run must be completed first.

    Parameters:
        - directory {path}: THe working directory.
        - functions {dict}: the functions that are to be used on the files. In format:
            - {'mask.shp': 'function'}
                - Where 'mask.shp' is the name of the shapefile to mask, and 'function' can be one of the following:
                    - 'mask_inside' : masks the tifs so that data is only kept if WITHIN the clipped region
                    - 'mask_outside' : masks the tifs so that data is only kept if OUTSIDE the clipped region
                    - 'resample' : resamples all tifs into the designated resolution. NOTE-- does not currently work with 
                                the current method and ordinal seabed data
        - shp_folder {path}: the path to the folder containing the shapefiles to mask
    '''

    if len(functions) > 0:
        

        for var, function in functions.items():

            if function == 'mask_inside':

                shp = os.path.join(shp_folder, var)

                mask_tifs(folder, folder, shp, 1)

            elif function == 'mask_outside':
                
                # if there is no / in the var, it is a single shapefile
                if '/' not in var:
                    shp = os.path.join(shp_folder, var)
                else:

                    # Load the gdf and print the len
                    gdf = gpd.read_file(os.path.join(shp_folder, var))
                    print(len(gdf))

                    shp = var

                mask_tifs(folder, folder, shp, 2)

            elif function == 'resample':

                res = var
                
                resample_rasters(folder, folder, res)

            else:
                print('Enter a correct function')

        # Creates the pickle containing the data, and masks the metrics by them
        create_pickle(folder=folder)

        if phasing:
            mask_metrics(folder)


def set_base_masks(designations, scale):
    ''' 
    Takes in a list of AOI designations from the user and returns a dictionary with all the correct masks merged into one.

    Continue to build as more things are added
    '''
    # Convert single string to list if needed
    if isinstance(designations, str):
        designations = [designations]

    options = {}

    for designation in designations:
        if designation == 'whole area':
            options.update({
                'aoi.geojson': 'mask_inside'
            })
        elif designation == 'full PE zone':
            options.update({
                'aoi.geojson': 'mask_inside',
                'PE_zone.geojson': 'mask_inside'
            })
        elif designation == 'msp':
            options.update({
                'aoi.geojson': 'mask_inside',
                'Shipping.geojson': 'mask_outside',
                # 'legacy_farms.geojson': 'mask_outside',
                'military.geojson': 'mask_outside',
                'nature_reserves.geojson': 'mask_outside',
                #'sand_extraction.geojson': 'mask_outside',
            })
        elif designation == 'renewable energy zones':
            options.update({
                'aoi.geojson': 'mask_inside',
                'energy_zone.geojson': 'mask_inside'
            })
        elif designation == 'exclude shipping':
            options.update({
                'aoi.geojson': 'mask_inside',
                'Shipping.geojson': 'mask_outside'
            })
        elif designation.startswith('PE zone'):
            zone_number = designation.split(' ')[-1]
            options.update({
                f'pe_split//kavel_{zone_number}.geojson': 'mask_inside',
                'gravelbed.geojson': 'mask_outside'
            })

    if scale == 'international':
        options.update({
            #'UK.geojson': 'mask_outside',
            'rivers.geojson': 'mask_outside'
        })

    return options


def get_min_distances(land_objects, scale, key):
    distance_arrays = []
    
    if key in ['installation', 'maintenance']:
        # Include 'port' distances for installation and maintenance
        distance_arrays.extend([item['distances'] for item in land_objects['port'][scale].values()])
    
    # Add distances for the specific key
    distance_arrays.extend([item['distances'] for item in land_objects[key][scale].values()])
    
    stacked_distances = np.stack(distance_arrays)
    return np.nanmin(stacked_distances, axis=0)


# MARK: MAIN

def prep_run(directory, designation, hubs={}, spatial_scale='international', add_hubs=False, country='all', found_age=25, res=5000, update_ports=False, connector_capacity=3, countries_reached=[], verbose=True, first_iteration=False, iterative=False, custom_ports=False, ports_to_modify=[], i=0, roadmap=False):

    ''' 
    Prepares the temporary folder for a run.
    First runs the precompute based on the techs

    Parameters:
        - spatial_scale {str}: The scale at which to process. Can be 'belgium' or 'international'
        - directory {path}: The working directory, this should be the marine-planning github main directory
        - masks {dict}: The base masks that will be applied to the rasters. In format :: {'mask.shp': 'mask_inside | 'mask_outside}.
            See data_processing() below for more information
        - designation {str}: the infrastructure designation
            See aoi_designation.txt for more information
        - tech_params {dict}: The dictionary containing the technology to be run for each year. In format:
            {'tech_type': {'present: T/F'}}
    '''

    # Gathers the paths for all the data folders, which will be copied over to the temp folder
    raster_folder = os.path.join(directory, 'data', 'rasters', spatial_scale)
    vector_folder = os.path.join(directory, 'data', 'vectors', spatial_scale)
    pickle_folder = os.path.join(directory, 'data', 'pickles', spatial_scale)

    temp_folder = os.path.join(directory, 'temp')
    temp_pickle_folder = os.path.join(temp_folder, 'pickles')

    # Create the temp folder
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
        os.makedirs(temp_pickle_folder)
    else:   # Remove an old temp folder and replace it
        shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)
        os.makedirs(temp_pickle_folder)

    # Copy the appropriate rasters into the temp folder
    copied_rasters = [f for f in os.listdir(raster_folder) if os.path.isfile(os.path.join(raster_folder, f))]

    for r in copied_rasters:
        source_path = os.path.join(raster_folder, r)
        destination_path = os.path.join(temp_folder, r)
        shutil.copy(source_path, destination_path)

    # Set the base masks dict
    masks = set_base_masks(designation, spatial_scale)

    if res != 5000:
        masks.update({res: 'resample'})

    # Process the data according to the masks. The result will be a temp folder with the base masks applied
    data_processing(folder=temp_folder, functions=masks, shp_folder=vector_folder)

    # Run the precompute to get the techs
    # Set land objects
    ports = LandObjects(directory, custom_ports=custom_ports)
    ports.initiate_ports(hubs, add_hubs, country, update_ports, connector_capacity, countries_reached, iterative=iterative, ports_to_modify=ports_to_modify, i=i)

    if ports.exit:
        print('Goodbye.')
        return

    lo = {}
    lo['dto'] = ports.results['distance_to_opr']
    lo["dti"] = ports.results['distance_to_ins']
    lo["dts"] = ports.results['distance_to_substation']
    lo['closest_gridconnect'] = ports.results['closest_gridconnects']
    lo['closest_install_port'] = ports.results['closest_install_port']

    format_arrays(temp_folder, lo)

    if not roadmap:
        initial_calc(directory, found_age, verbose, iterative)

    print('Temp folder created and filled')


    pickle_path = os.path.join(temp_pickle_folder, 'updated_ports.pkl')

    with open(pickle_path, 'wb') as file:
        pickle.dump(ports.updated_ports, file)


    if first_iteration:
        return ports.updated_ports


def initial_calc(directory, found_age=25, verbose=True, iterative=False):
    # Set the file paths
    geo_data_file = os.path.join(directory, 'temp', 'pickles', 'geo_data.pkl')
    land_objects_file = os.path.join(directory, 'temp', 'pickles', 'land_objects.pkl')
    output_file = os.path.join(directory, 'temp', 'pickles', f'calculated_metrics.pkl')
    out_install_decom_file = os.path.join(directory, 'temp', 'pickles', 'install_decom.pkl')
    grid_connect_path = os.path.join(directory, 'temp', 'closest_gridconnects.pkl')

    # Load geo_data
    with open(geo_data_file, 'rb') as file:
        geo_data = pickle.load(file)

    # Load land_objects
    with open(land_objects_file, 'rb') as file:
        lo = pickle.load(file)

    # Load closest gridconnects
    with open(grid_connect_path, 'rb') as file:
        closest_gridconnects = pickle.load(file)

    combined = add_geodata(geo_data, lo, 'international', closest_gridconnects, iterative=iterative)

    # Define a dictionary of technology classes
    tech_classes = {
        'monopile': monopile_15MW,
        'jacket': jacket_15MW,
        'fpv': floating_PV_solo,
        'semisub_cat_drag': semisub_15MW_cat_drag,
        'semisub_taut_driv': semisub_15MW_taut_driv,
        'semisub_taut_suc': semisub_15MW_taut_suc,
        'spar_cat_drag': spar_15MW_cat_drag,
        'spar_taut_driv': spar_15MW_taut_driv,
        'spar_taut_suc': spar_15MW_taut_suc,
        'mussel': mussel_longline,
        'seaweed': seaweed_longline
    }

    # Initialize a dictionary to store results for each technology
    all_metrics = {tech: {key: [] for key in ['capex', 'opex', 'co2+', 'co2-', 'revenue', 'LCOE', 'energy_produced', 'food_produced', 'unit density', 'lifetime']} for tech in tech_classes}
    all_install_decom = {tech: {key: [] for key in ['foundation_install_cost', 'foundation_decom_cost', 'foundation_install_emissions', 'foundation_decom_emissions', 'turbine_install_cost', 'turbine_decom_cost', 'turbine_install_emissions', 'turbine_decom_emissions']} for tech in tech_classes}

    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count()

    # Create a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks for each technology
        future_to_tech = {executor.submit(process_tech, tech_name, tech_class, combined, found_age): tech_name 
                            for tech_name, tech_class in tech_classes.items()}

        # Process results as they complete
        for future in as_completed(future_to_tech):
            tech_name = future_to_tech[future]
            try:
                tech_name, metrics, install_decom = future.result()
                all_metrics[tech_name] = metrics
                all_install_decom[tech_name] = install_decom
                print(f"Completed calculations for {tech_name}") if verbose else None
            except Exception as exc:
                print(f"{tech_name} generated an exception: {exc}")

        print("All calculations complete.")

    # Add a tech 'empty' that has all the metrics but as 0
    all_metrics['empty'] = {key: [0] * len(all_metrics['monopile']['capex']) for key in ['capex', 'opex', 'co2+', 'co2-', 'revenue', 'LCOE', 'energy_produced', 'food_produced']}
    all_metrics['empty']['unit density'] = [1] * len(all_metrics['monopile']['capex'])
    all_metrics['empty']['lifetime'] = [1] * len(all_metrics['monopile']['capex'])
    all_install_decom['empty'] = {}

    # Save the results to a pickle file
    with open(output_file, 'wb') as file:
        pickle.dump(all_metrics, file)

    with open(out_install_decom_file, 'wb') as file:
        pickle.dump(all_install_decom, file)

    with open(geo_data_file, 'wb') as file:
        pickle.dump(combined, file)