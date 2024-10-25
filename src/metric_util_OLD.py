import numpy as np
import numpy_financial as np_f
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import math
import copy
import pulp
import os
import shutil 
import pickle
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from math import sqrt

import geopandas as gpd
from shapely.geometry import box, shape, Point, Polygon, LineString
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import fiona
from shapely.ops import unary_union

import networkx as nx
from shapely.ops import nearest_points
from scipy.optimize import minimize

import itertools

def reorganize(data):
    ''' 
    Changes the data structure, make it from tech>cell>metric to tech>metric>cell. Each metric becomes a list
    '''
    new = {}

    for x in data:
        for y in x:
            for metric, value in y.items():
                if metric not in new:
                    new[metric] = []
                new[metric].append(value)

    return new


def show(data, shape, minmax=None, title=None):
    array_to_visualize = np.array(data).reshape(shape)
    float_array = array_to_visualize.astype(float)

    # Convert 0 to NaN
    float_array[float_array == 0] = np.nan
    
    plt.imshow(float_array, cmap='magma')
    plt.colorbar()
    plt.title(title)
    plt.show()


def check_nan(d):

    if not math.isnan(d['dto']):
        return True
    else:
        return False


def convert_mask(d):
    '''
    Fills the masked values with NaN for the calculations
    Parameters:
        dict {dict}: This dict must be full of masked arrays
    '''
    filled = {}
    keys = list(d.keys())
    # Loops through each key
    for key in keys:
        try: # For values that are not equations
            filled[key] = d[key].filled(-1).astype(float) # For some reason I couldn't just fill it with NaN, have to do this weird work around for now
            filled[key][filled[key] == -1] = np.nan
        except: # For equations
            filled[key] = d[key].filled(None).astype(str)
    return filled


def check_float(value):
    try:
        return float(value)
    except ValueError:
        return value
    except TypeError:
        print('typeerror')
    

def calc(original, last):

    '''
    A more complex calc function to account for NaN. Calculates everything that can be calculated within the given dictionary

    Parameters:
        original {dict}: Dictionary containing all of the metrics as their own lists. If the metric is calculated it is a value, if not
                         it is an equation.
        last {int}: The length of the last run through the calculation. To determine if it is getting stuck on an metric, which would
                    indicate that there is an error in the equation.
    
    Returns:
        results {dict}: Dictionary with the newly computed values and any equations to be calculated.
        flag {int}: Determines whether or not to continue the loop. 1 for no, 0 for yes. 1 is only passed if the list of remaining equations
                    is empty or is the same length as the last run.
        len(equations) {int}: The length of the equations list. To tell it to continue or not.
    '''
    
    for key, value in original.items():
        # Convert each item in the value list using check_float
        original[key] = [check_float(item) for item in value]


    equations, numbers = sort_equations(original) # Creates two new dicts, one that has equations and one that has calc'd values
    results = {**equations, **numbers}

    if len(equations) == 0 or len(equations) == last: # If the equations list is empty or is the same length as the last run, return the flag telling the loop to stop
        flag = 1
        
        if len(equations) == last:
            problem = next(iter(equations.keys()), None)
            print(f"Problem with {problem}")

        return numbers, flag, len(equations)
    
    else:

        flag = 0 # Sets flag at 0 to continue loop
        errors = {}
        for metric, equation in equations.items():
            metric_results = [] # Creates a new list for each metric. This list will be populated as the calculations are made 

            error = 0

            for i, eq in enumerate(equation): # Equation is the list from the metric the loop is currently on and begins to loop one by one within that list
                variables = {key: value[i] for key, value in numbers.items()} # Creates a dict containing the values from each metric that has a variable for the index
        
                try: # This try loop is for equations that cannot yet be computed with the current values
                    if check_nan(variables):
                        result = eval(eq.replace('\xa0', ''), {'np':np, 'np_f':np_f, 'exp':math.exp}, variables) # Runs the eval on eq using the dict variables that contains each already calculated metric
                        metric_results.append(result)
                    else: # If the value is NaN apply NaN to the list for this index
                        metric_results.append(np.nan)
                    
                except Exception as e:                     
                    error = 1
                    # add error to the errors dict, add as list if it already exists
                    if metric in errors:
                        errors[metric].append(e)
                    
            
                    

            if error == 1: # Flagged that all parameters are not yet calculated
                results[metric] = original[metric]
            else:
                results[metric] = metric_results
        
        if len(errors) > 0:
            for metric, error in errors.items():
                print(f"Error in metric: {metric}")
                for e in error:
                    print(e)
        
        # print(f"Equations left: {len(equations)}")
        # print(f"{equations.keys()}")

        return results, flag, len(equations)

    
def loop(d):
    '''
    Runs the calculate function until all calculations have been completed

    Returns a dictionary with all the computed values
    '''
    flag = 0
    last_len = 0
    
    while flag == 0:
        d, flag, last_len = calc(d, last_len)

    return d
    

def sort_equations(d):

    '''
    Takes in the dictionary with all of the codes and seperates the equations from calculated values.
    It does this by checking if any operators are in the string, then sorts accordingly.

    Parameters:
        inputs: 
            d {dict}: the dictionary containing all of the codes and their values

        outputs: 
            equations {dict}: the dictionary containing the codes with equations
            numbers {dict}: the dictionary containing the codes with values
    '''

    equations = {}
    values = {}

    for key, value in d.items():
        if all(isinstance(item, (int, float)) for item in value):
            values[key] = value
        elif all(isinstance(item, str) and any(char in item for char in '+-*/') for item in value):
            equations[key] = value

    return equations, values


def sheet_processing(xlxs, sheetname, shape, m):

    '''
    Processes the data from each spreadsheet

    Parameters:
        xlsx {string}: filename to spreadsheet
        sheetname {string}: name of sheet with the tech
        shape {list}: shape of array ([n,n])
        m {mask}: mask created from geo data. I use a list of True/False values in this scenario rather than a masked array
    '''

    sheet = pd.read_excel(xlxs, sheet_name=sheetname, header=0)
    sheet = sheet.iloc[3:].reset_index(drop=True)

    as_dict = pd.Series(sheet.Value.values, index=sheet.Code).to_dict()
    # full_dict = [[as_dict.copy() for i in range(shape[0])] for i in range(shape[1])]

    # dict_lists = reorganize(full_dict)

    # floats = {key: [check_float(item) for item in value] for key, value in dict_lists.items()}

    # masked = {metric: np.ma.masked_array(floats[metric], m) for metric, values in floats.items()}

    # filled = convert_mask(masked)

    return as_dict


def trim_to_square(array):

    # Reshape the array from list to array
    x = math.sqrt(len(array))
    array = np.array(array).reshape(int(x), int(x))

    # Get indices of non-NaN values
    non_nan_indices = np.argwhere(~np.isnan(array))
    
    # Find the bounding box of non-NaN values
    top_left = non_nan_indices.min(axis=0)
    bottom_right = non_nan_indices.max(axis=0) + 1  # +1 to include the last element
    
    # Extract the bounding box
    subarray = array[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    
    # Determine the size of the smallest square
    max_side = max(subarray.shape)
    
    # Create a square array with NaNs
    square_array = np.full((max_side, max_side), np.nan)
    
    # Calculate padding needed to center the subarray in the square array
    pad_top = (max_side - subarray.shape[0]) // 2
    pad_left = (max_side - subarray.shape[1]) // 2
    
    # Place the subarray in the center of the square array
    square_array[pad_top:pad_top + subarray.shape[0], pad_left:pad_left + subarray.shape[1]] = subarray
    
    return square_array.flatten()


def equal_dimensions(array):
    '''
    Adds empty rows/columns to an array to create equal dimensions so the array can be reshaped once flattened
    '''
    current_height, current_width = array.shape
    max_dimension = max(current_height, current_width)
    rows_to_add = max_dimension - current_height
    cols_to_add = max_dimension - current_width
    
    return np.pad(array, ((0, rows_to_add), (0, cols_to_add)), mode='constant', constant_values=np.nan)


def print_counts(uses, index, x):

    counter = {tech: 0 for tech in uses}

    for i in index:
        for tech in uses:
            if pulp.value(x[i][tech]) == 1:
                counter[tech] += 1

    for tech, count in counter.items():
        print(f"{tech}: {count} cells")


def seed_to_shp(template_r, seed, n):
    ''' 
    Converts the seed from the LA to a shp for visualization or storage

    Parameters:
        template_r {string}: path to a template raster to get extent and crs
        seed {list}: the seed as exported from the LA
        n {int}: the number of rows/columns within the original array, to reshape the seed into array format

    Returns:
        dissolved_gdf {gdf}: the geodataframe containing the polygons from the seeds
    '''

    seed_array = np.array(seed).reshape(n, n)

    with rasterio.open(template_r) as template:
        transform = template.transform
        crs = template.crs

    polygons = []
    for row in range(n):
        for col in range(n):

            minx, maxy = rasterio.transform.xy(transform, row-0.5, col-0.5)
            maxx, miny = rasterio.transform.xy(transform, row+0.5, col+0.5)

            value = seed_array[row, col]
            
            if value != 0:  # Skip cells with 0
                polygon = box(minx, miny, maxx, maxy, ccw=True)
                polygons.append((value, polygon))

    gdf = gpd.GeoDataFrame(polygons, columns=['value', 'geometry'], crs=crs)

    dissolved_gdf = gdf.dissolve(by='value')

    return dissolved_gdf


def create_lifespan_dict(names):
    
    result_dict = {}
    lifespans = {'monopile': 25,
                'jacket': 25,
                'solar': 30,
                'mussel': 12,
                'monopile + solar': 25,
                'monopile + mussel': 25
                }
    
    for year in names:
        components_dict = {}
        for component, lifespan in lifespans.items():
            components_dict[component] = int(year) + lifespan

        result_dict[year] = components_dict

    return result_dict


def remove_tech(directory, year, tech):

    shp_path = os.path.join(directory, str(year) + '.geojson') 
    try:
        # Read the shapefile
        gdf = gpd.read_file(shp_path)

        # Filter out polygons with the specified value in the specified column

        filtered_gdf = gdf[~gdf['value'].isin(tech)]

        # Save the filtered shapefile
        if not filtered_gdf.empty:
            filtered_gdf.to_file(shp_path, driver='GeoJSON')
        else:
            # If the DataFrame is empty, delete the shapefile
            if os.path.exists(shp_path):
                os.remove(shp_path)
                print(f"The shapefile {shp_path} was empty and has been deleted.")
            else:
                print(f"The shapefile {shp_path} does not exist.")
    except:
        pass
    

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
                # 'sand_extraction.geojson': 'mask_outside',
                # 'wind_farms.geojson': 'mask_outside'
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


def create_old_tech_raster(directory, scale):

    '''
    Creates the raster of old technologies to be used during a phasing run. 
    The raster is used to determine which cells are already occupied by old technologies and should not be used for new technologies OR
    should be modified to account for the old technology.

    Currently only works for the original old windfarms -- but can be modified to work for any old technology that was created
    and removed from the during the phasing process.
    '''

    for file in os.listdir(os.path.join(directory, "data", "rasters", scale)):
        if file.endswith(".tif"):
            template_r = os.path.join(os.path.join(directory, "data", "rasters", scale), file)
            break

    polygon_folder = os.path.join(directory, 'data', 'vectors', scale, 'WindFarms', 'old')
    output_raster = os.path.join(polygon_folder, 'old_tech_raster.tif') 

    with rasterio.open(template_r) as src:
        raster_array = src.read(1) 
        
        meta = src.meta
        nodata = src.nodata
        
        combined_mask = np.zeros(raster_array.shape, dtype=np.uint8)
        
        for filename in os.listdir(polygon_folder):
            if filename.endswith('.geojson'):
                geojson_file = os.path.join(polygon_folder, filename)
                
                with fiona.open(geojson_file, "r") as shapefile:
                    for feature in shapefile:
                        shapely_polygon = shape(feature['geometry'])
                        
                        mask = geometry_mask([shapely_polygon], out_shape=raster_array.shape,
                                            transform=src.transform, invert=True)
                        
                        combined_mask |= mask.astype(np.uint8)
        
        meta.update({'dtype': 'uint32', 'count': 1, 'nodata': None})
        with rasterio.open(output_raster, 'w', **meta) as dst:
            dst.write(combined_mask.astype('uint32') * 255, 1)


def modify_metrics(directory, scale):

    '''
    Modifies the calculated metrics to account for the old technologies that were removed.

    Currently just divides the cap by 2 for cells that have had tech removed -- return to modify.
    '''

    temp_folder = os.path.join(directory, 'temp')
    main_pkl_file = os.path.join(temp_folder, 'pickles', 'calculated_metrics.pkl')
    temp_pkl_file = os.path.join(temp_folder, 'pickles', 'temp_calculated_metrics.pkl')
    old_wind_farms = os.path.join(directory, 'data', 'vectors', scale, 'WindFarms', 'old', 'old_tech_raster.tif')

    # Set the pkl to be modified
    if not os.path.exists(temp_pkl_file):

        shutil.copyfile(main_pkl_file, temp_pkl_file)

    # If the file has already been created and modified, reset it fresh
    else:
        os.remove(main_pkl_file)
        shutil.copyfile(temp_pkl_file, main_pkl_file)


    # Load in the metrics dictionary
    with open(main_pkl_file, 'rb') as f:
        data = pickle.load(f)

    create_old_tech_raster(directory, scale)

    with rasterio.open(old_wind_farms) as src:
        array = src.read(1)
        array, rows, cols = equal_dimensions(array)
        array = array.flatten()

    toedit = ['cap']

    for tech, metrics in data.items():
        for metric in metrics:
            if metric in toedit:
                data[tech][metric] = [value / 2 if array[i] == 255 else value for i, value in enumerate(data[tech][metric])]


    with open(main_pkl_file, 'wb') as f:
        pickle.dump(data, f)


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

# MARK: Point generation

def _remove_edge_points(points, geom, num_to_remove, buffer_distance=4000):

    '''
    This function removes points that are within a specified buffer distance of the
    exterior of a polygon.

    Parameters
    ----------
    points : list of shapely.geometry.Point
        The list of points to filter.
    geom : shapely.geometry.Polygon
        The polygon to use as the boundary.
    num_to_remove : int
        The number of points to remove.
    buffer_distance : float, optional
        The buffer distance to use around the exterior of the polygon.
        The default is 4000.

    Returns
    ------- 
    list of shapely.geometry.Point
        The filtered list of points.
    '''

    buffered_polygon = geom.exterior.buffer(buffer_distance)
    boundary_points = [point for point in points if buffered_polygon.contains(point)]

    # order the list of boundary points by distance from the geom.exterior
    boundary_points = sorted(boundary_points, key=lambda point: geom.exterior.distance(point))
    to_remove = boundary_points[:num_to_remove]

    return [point for point in points if point not in to_remove]


def create_equidistant_points(polygon, num_points, max_iterations=1000):
    
    ''' 
    This function generates a specified number of points within a polygon,
    such that the points are approximately equidistant from each other.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon within which to generate the points.
    num_points : int
        The number of points to generate.
    max_iterations : int, optional
        The maximum number of iterations to attempt to generate the points.
        The function will attempt to generate the points with an approximate
        grid spacing that is the square root of the area of the polygon divided
        by the number of points. If the function cannot generate the points
        within the polygon after `max_iterations` attempts, a ValueError will be
        raised. The default is 1000.

    Returns
    -------
    list of shapely.geometry.Point
        A list of points within the polygon.
    '''

    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y

    for iteration in range(max_iterations):
        # Calculate the approximate grid spacing
        spacing = (width * height / (num_points * (iteration + 1))) ** 0.5
        
        # Generate a grid of points
        x_coords = np.arange(min_x, max_x, spacing)
        y_coords = np.arange(min_y, max_y, spacing)
        
        grid_points = [Point(x, y) for x in x_coords for y in y_coords]
        
        # Filter points to be within the polygon
        points_within_polygon = [point for point in grid_points if polygon.contains(point)]
        
        # If we have enough points, break the loop
        if len(points_within_polygon) >= num_points:
            num_to_remove = len(points_within_polygon) - num_points
            return _remove_edge_points(points_within_polygon, polygon, num_to_remove)
    
    raise ValueError(f"Could not generate {num_points} points within the polygon after {max_iterations} iterations")


def min_distance(point, gdf):
    ''' 
    Calculates the minimum distance between a point and a gdf with other points
    '''
    return gdf.distance(point).min()


def add_vertices_at_intersections(line_gdf):
    """
    Add vertices to a GeoDataFrame of LineStrings wherever the lines intersect.
    
    Parameters:
    line_gdf (GeoDataFrame): A GeoDataFrame containing LineString geometries.

    Returns:
    GeoDataFrame: A new GeoDataFrame with vertices added at intersections.
    """
    # Ensure we're working with LineStrings
    if not all(line_gdf.geom_type == 'LineString'):
        raise ValueError("All geometries must be LineStrings")

    # Create an empty list to store the updated geometries
    updated_geometries = []

    # Step 1: Find all intersection points by combining all geometries
    union_geom = unary_union(line_gdf.geometry)
    intersection_points = []

    # Iterate over pairs of geometries to find intersections
    for i, line1 in enumerate(line_gdf.geometry):
        for j, line2 in enumerate(line_gdf.geometry):
            if i < j:  # Avoid comparing the same line with itself or repeating pairs
                if line1.intersects(line2):
                    intersection = line1.intersection(line2)
                    if isinstance(intersection, Point):  # Only handle point intersections
                        intersection_points.append(intersection)

    # Step 2: Add intersection points as vertices to the original LineStrings
    for line in line_gdf.geometry:
        # Get all existing points (vertices) of the line
        line_coords = list(line.coords)

        # Add intersection points that intersect this specific line
        for pt in intersection_points:
            if line.intersects(pt):
                line_coords.append((pt.x, pt.y))

        # Sort the points to maintain proper order along the line
        line_coords = sorted(line_coords, key=lambda x: line.project(Point(x)))

        # Create a new LineString with the new vertices
        updated_geometries.append(LineString(line_coords))

    # Step 3: Return a new GeoDataFrame with updated geometries
    updated_gdf = line_gdf.copy()
    updated_gdf['geometry'] = updated_geometries
    return updated_gdf


# MARK: Random stuff for marineplan  

def set_time_divider(scale):
    '''
    Returns 1 if the scale is yearly, 52 if weekly, 365 if daily, and 8760 if hourly
    '''

    if scale.split('_')[0] == 'hourly':
        return 8760
    elif scale.split('_')[0] == 'daily':
        return 365
    elif scale.split('_')[0] == 'weekly':
        return 52
    elif scale == 'yearly':
        return 1
    

def print_substation_costs(P):
    '''
    Print the precalculated and postcalculated costs for the cables, foundation, and total substation cost
    '''

    print(f"precalc cables: {pulp.value(P.monopile_cables)}")
    print(f'precalc foundation: {pulp.value(P.monopile_ssfnc)}')
    print(f'precalc total: {pulp.value(P.monopile_sstc)}')

    print('\npostcalc cables cost: ', P.cables_cost)
    print('postcalc foundation cost: ', P.foundation_cost)
    print('postcalc total cost: ', P.total_substation_cost)

    # Calculate the difference between the precalculated and postcalculated costs as a percentage
    cables_diff = (P.cables_cost - pulp.value(P.monopile_cables)) / pulp.value(P.monopile_cables) * 100
    foundation_diff = (P.foundation_cost - pulp.value(P.monopile_ssfnc)) / pulp.value(P.monopile_ssfnc) * 100
    total_diff = (P.total_substation_cost - pulp.value(P.monopile_sstc)) / pulp.value(P.monopile_sstc) * 100

    print(f'\nCables cost difference: {cables_diff:.2f}%')
    print(f'Foundation cost difference: {foundation_diff:.2f}%')
    print(f'Total cost difference: {total_diff:.2f}%')


def print_production(P, scale):
    '''
    Print the modelled and calculated production for solar, wind, and total energy

    Parameters:
        P {pulp.LpProblem}: The optimization problem
        scale {str}: The scale of the production data
            - 'hourly', 'daily', 'weekly'
    '''
    
    # Print the sum of these two dicts
    calc_solar = sum(P.solar_production[scale].values())
    calc_wind = sum(P.wind_production[scale].values())

    print(f'\nModelled solar production: {pulp.value(P.solar_energy)} GWh/y')
    print(f'Calculated solar production: {calc_solar} GWh/y')
    print(f'\nModelled wind production: {pulp.value(P.monopile_energy)} GWh/y')
    print(f'Calculated wind production: {calc_wind} GWh/y')

    print(f'\nModelled total production: {pulp.value(P.total_energy)} GWh/y')
    print(f'Calculated total production: {calc_solar + calc_wind} GWh/y')

    # Calculate the percent error between the two
    error = (pulp.value(P.total_energy) - (calc_solar + calc_wind)) / (calc_solar + calc_wind) * 100
    print(f'Error: {error:.2f}%')


def get_min_distances(land_objects, scale, key):
    distance_arrays = []
    
    if key in ['installation', 'maintenance']:
        # Include 'port' distances for installation and maintenance
        distance_arrays.extend([item['distances'] for item in land_objects['port'][scale].values()])
    
    # Add distances for the specific key
    distance_arrays.extend([item['distances'] for item in land_objects[key][scale].values()])
    
    stacked_distances = np.stack(distance_arrays)
    return np.nanmin(stacked_distances, axis=0)


def simplify_clusters(gdf):

    # Create a spatial index for efficient querying
    sindex = gdf.sindex

    # Function to find touching polygons
    def find_touching(idx):
        possible_matches_index = list(sindex.intersection(gdf.loc[idx, 'geometry'].bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.touches(gdf.loc[idx, 'geometry'])]
        return precise_matches.index.tolist()

    # Find groups of touching polygons
    groups = []
    processed = set()
    for idx in gdf.index:
        if idx not in processed:
            group = set([idx])
            to_process = set(find_touching(idx))
            while to_process:
                new_idx = to_process.pop()
                group.add(new_idx)
                to_process.update(set(find_touching(new_idx)) - group)
            groups.append(list(group))
            processed.update(group)

    # Create envelopes for groups and keep isolated polygons
    new_geometries = []
    attributes = {'tech': [], 'unit_count': [], 'km2': [], 'capex': [], 'value': [], 'energy': [], 'food': [], 'cluster_num': []}
    count = 0
    for group in groups:
        if len(group) > 1:
            # Create envelope for touching polygons
            group_geom = unary_union(gdf.loc[group, 'geometry'])
            dissolved = unary_union(group_geom)
            exterior = dissolved.convex_hull

            # Simplify the exterior ring
            simplified = exterior.simplify(0.1)
            outline = Polygon(simplified)
            new_geometries.append(outline)

            # Sum attributes
            for attr in attributes.keys():
                
                if attr == 'tech':
                    attributes[attr].append('monopile')
                elif attr == 'cluster_num':
                    attributes[attr].append(count)
                    count += 1
                else:
                    attributes[attr].append(gdf.loc[group, attr].sum())

        else:
            # Convex hull for isolated polygons
            geom = unary_union(gdf.loc[group, 'geometry'])
            exterior = geom.convex_hull
            simplified = exterior.simplify(0.1)
            outline = Polygon(simplified)
            new_geometries.append(outline)

            # Sum attributes
            idx = group[0]
            for attr in attributes.keys():
                if attr == 'tech':
                    attributes[attr].append('monopile')
                elif attr == 'cluster_num':
                    attributes[attr].append(count)
                    count += 1
                else:
                    attributes[attr].append(gdf.loc[idx, attr])

    # Create a new GeoDataFrame with the results
    result_gdf = gpd.GeoDataFrame(attributes, geometry=new_geometries, crs=gdf.crs)
    return result_gdf


def generate_ecoimpact_map(shape, num_clusters=10, cluster_size=20, smoothness=8):
    # Initialize an empty array
    map_array = np.zeros(shape)
    
    # Generate random cluster centers
    for _ in range(num_clusters):
        center = tuple(np.random.randint(0, dim) for dim in shape)
        value = np.random.uniform(2.0, 5.0)  # Float values from 2.0 to 5.0
        
        # Create a small cluster around the center
        for i in range(-cluster_size, cluster_size + 1):
            for j in range(-cluster_size, cluster_size + 1):
                x = (center[0] + i) % shape[0]
                y = (center[1] + j) % shape[1]
                map_array[x, y] = value
    
    # Apply Gaussian filter to smooth the map
    smoothed_map = gaussian_filter(map_array, sigma=smoothness)
    
    # Rescale values to 2-5 range
    smoothed_map = (smoothed_map - smoothed_map.min()) / (smoothed_map.max() - smoothed_map.min()) * 3 + 2
    
    # Round to two decimal places
    return np.round(smoothed_map, decimals=2)


def combine_tech(dict1, dict2, metric_modifiers):
    combined = {}
    
    # Process all keys from dict1
    for key, values in dict1.items():
        if key in dict2:
            # If the key is in both dictionaries, add the values
            # Apply the modifier to dict2 values
            modifier = metric_modifiers.get(key, 1)  # Default to 1 if no modifier specified
            combined[key] = [v1 + (v2 * modifier) for v1, v2 in zip(values, dict2[key])]
        else:
            # If the key is only in dict1, add it unmodified
            combined[key] = values.copy()
    
    # Add any remaining keys from dict2 that weren't in dict1
    for key, values in dict2.items():
        if key not in dict1:
            # Apply the modifier to dict2 values
            modifier = metric_modifiers.get(key, 1)  # Default to 1 if no modifier specified
            combined[key] = [v * modifier for v in values]

    combined['country'] = dict1['country']
    
    return combined


def shorten_number(number, format_type='eu'):
    abs_number = abs(number)

    if format_type.lower() not in ['eu', 'na']:
        raise ValueError("format_type must be either 'eu' or 'na'")

    if abs_number < 1_000_000:
        result = f"{format_number(int(abs_number), format_type)}"
    elif abs_number < 1_000_000_000:
        result = f"{abs_number / 1_000_000:.2f} M"
    elif abs_number < 1_000_000_000_000:
        result = f"{abs_number / 1_000_000_000:.2f} B"
    else:
        result = f"{abs_number / 1_000_000_000_000:.2f} T"

    if format_type.lower() == 'eu':
        result = result.replace(".", ",")
    
    return result if number >= 0 else f"-{result}"


def format_number(number, format_type='eu'):
    if format_type.lower() == 'eu':
        if number == int(number):
            return "{:,.0f}".format(number).replace(",", " ").replace(".", ",").replace(" ", ".")
        else:
            return "{:,.2f}".format(number).replace(",", " ").replace(".", ",").replace(" ", ".")
    elif format_type.lower() == 'na':
        if number == int(number):
            return "{:,}".format(int(number))
        else:
            return "{:,.2f}".format(number)
    else:
        raise ValueError("format_type must be either 'eu' or 'na'")
    

def most_common(x):
    return x.mode().iloc[0] if not x.mode().empty else None


def dissolve_touching_polygons(gdf):
    # Iterate through each geometry in the GeoDataFrame
    for index, row in gdf.iterrows():
        geom = row.geometry

        # If it's a MultiPolygon, process it
        if geom.geom_type == 'MultiPolygon':
            # Merge touching polygons within the MultiPolygon
            merged_geom = unary_union([polygon for polygon in geom.geoms])

            # Update the geometry with the merged polygons
            gdf.at[index, 'geometry'] = merged_geom

    return gdf


def calc_LCOE(CAPEX, OPEX, max_Capacity, lifetime=25, discount_rate=0.075, capacity_factor=0.6, capacity_deg=0):
   
    max_Capacity *= 1000
    
    CAPEX *= lifetime

    Exponent = np.arange(lifetime)   
 
    Energy = np.ones(lifetime)*max_Capacity*365*24*capacity_factor
 
    Energy[:] = Energy[:]*((1-capacity_deg)**(Exponent[:]))
 
    Energy[:] = Energy[:]/((1+discount_rate)**(Exponent[:]+1))
 
    # OPEX = OPEX/lifetime 
 
    OPEX = np.ones(lifetime)*(OPEX)  
   
    OPEX[:] = OPEX[:]/((1+discount_rate)**(Exponent[:]+1))
 
    Cost = (CAPEX + np.sum(OPEX))/np.sum(Energy)
   
    return Cost #this is in €/MWh

# MARK: Interconnector allocation


def allocate_interconnectors(wind_farms, grid_connections, cable_paths_gdf, num_clusters=5):
    def min_distance(point, gdf):
        return gdf.distance(point).min()

    # Prepare wind farms data
    wind_farms['distances'] = wind_farms.apply(
        lambda farm: {
            country: min_distance(farm.geometry, country_grids)
            for country, country_grids in grid_connections.groupby('country')
        },
        axis=1
    )

    wind_farms_dict = {
        index: {
            'country_of_origin': farm['country'],
            'capacity': farm['capacity'],
            'coordinates': (farm.geometry.x, farm.geometry.y),
            'distances': farm['distances'],
            'closest_grid_coords': {
                country: tuple(
                    grid_connections[
                        (grid_connections['country'] == country) & 
                        (grid_connections.distance(farm.geometry) == farm['distances'][country])
                    ]['geometry'].iloc[0].coords[0]
                )
                for country in farm['distances']
            }
        }
        for index, farm in wind_farms.iterrows()
    }

    # Prepare grid connections data
    grid_connections_dict = {
        country: {'needed_capacity': group['capacity_needed'].iloc[0]}
        for country, group in grid_connections.groupby('country')
    }


    def create_interconnectors(wind_farms_dict, grid_connections_dict, num_clusters, linestrings_gdf):
        class DynamicGraph:
            def __init__(self, linestrings_gdf):
                self.G = nx.Graph()
                self.linestrings_gdf = linestrings_gdf
                self._build_initial_graph()

            def _build_initial_graph(self):
                for _, line in self.linestrings_gdf.iterrows():
                    coords = list(line.geometry.coords)
                    for i in range(len(coords) - 1):
                        self.G.add_edge(coords[i], coords[i+1], weight=Point(coords[i]).distance(Point(coords[i+1])))

            def add_interconnector_point(self, interconnector_point, k=3):
                nearest_points = self._find_nearest_points_on_linestrings(interconnector_point, k)
                for near_point in nearest_points:
                    self.G.add_edge(tuple(interconnector_point.coords[0]), tuple(near_point.coords[0]),
                                    weight=interconnector_point.distance(near_point))

            def _find_nearest_points_on_linestrings(self, point, k=3):
                all_points = list(self.G.nodes())
                distances = [Point(p).distance(point) for p in all_points]
                sorted_points = [Point(p) for _, p in sorted(zip(distances, all_points))]
                return sorted_points[:k]

            def shortest_path(self, start_point, end_point):
                start_on_graph = min(self.G.nodes, key=lambda n: Point(n).distance(start_point))
                end_on_graph = min(self.G.nodes, key=lambda n: Point(n).distance(end_point))

                try:
                    path = nx.shortest_path(self.G, start_on_graph, end_on_graph, weight='weight')
                except nx.NetworkXNoPath:
                    reachable_nodes = nx.node_connected_component(self.G, start_on_graph)
                    closest_reachable = min(reachable_nodes, key=lambda n: Point(n).distance(Point(end_on_graph)))
                    path = nx.shortest_path(self.G, start_on_graph, closest_reachable, weight='weight')
                    path.append(end_on_graph)

                if len(path) < 2:
                    path = [start_point, end_point]

                return LineString(path)

        def find_nearest_point_on_linestrings(point):
            nearest_points_series = linestrings_gdf.geometry.apply(lambda line: nearest_points(point, line)[1])
            distances = linestrings_gdf.geometry.distance(point)
            return nearest_points_series.iloc[distances.idxmin()]

        dynamic_graph = DynamicGraph(linestrings_gdf)

        interconnectors = []
        new_connections = []

        available_farms = [(index, farm) for index, farm in wind_farms_dict.items() if farm['capacity'] > 0]
        if not available_farms:
            return gpd.GeoDataFrame(interconnectors, crs='EPSG:3035'), gpd.GeoDataFrame(new_connections, crs='EPSG:3035')

        farm_coords = np.array([farm['coordinates'] for _, farm in available_farms])
        n_clusters = min(num_clusters, len(available_farms))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(farm_coords)

        # Create interconnectors
        for label in range(n_clusters):
            cluster_farms = [farm for farm, label_i in zip(available_farms, kmeans.labels_) if label_i == label]
            if not cluster_farms:
                continue

            avg_lat = np.mean([farm['coordinates'][0] for _, farm in cluster_farms])
            avg_lon = np.mean([farm['coordinates'][1] for _, farm in cluster_farms])
            initial_interconnector_point = Point(avg_lat, avg_lon)

            interconnector_point = find_nearest_point_on_linestrings(initial_interconnector_point)
            dynamic_graph.add_interconnector_point(interconnector_point)

            interconnector = {
                'geometry': interconnector_point,
                'connected_farms': [index for index, _ in cluster_farms],
                'grid_connections': {}
            }
            interconnectors.append(interconnector)

        # Connect farms to interconnectors
        for farm_index, farm in available_farms:
            closest_interconnector = min(interconnectors, key=lambda ic: ic['geometry'].distance(Point(farm['coordinates'])))
            farm_to_ic_path = dynamic_graph.shortest_path(Point(farm['coordinates']), closest_interconnector['geometry'])
            new_connections.append({
                'geometry': farm_to_ic_path,
                'farm_country': farm['country_of_origin'],
                'farm_index': farm_index,
                'grid_country': f'Interconnector_{interconnectors.index(closest_interconnector)}',
                'allocated_capacity': farm['capacity'],
                'is_interconnector': True,
                'connection_type': 'farm_to_IC'
            })
            closest_interconnector['connected_farms'].append(farm_index)

        # Connect interconnectors to grid points
        for country, info in grid_connections_dict.items():
            needed_capacity = info['needed_capacity']
            if needed_capacity <= 0:
                continue

            # Sort interconnectors by distance to the country's grid points
            sorted_interconnectors = sorted(
                interconnectors,
                key=lambda ic: min(
                    Point(grid_point.geometry.coords[0]).distance(ic['geometry'])
                    for _, grid_point in grid_connections[grid_connections['country'] == country].iterrows()
                )
            )

            for interconnector in sorted_interconnectors:
                if needed_capacity <= 0:
                    break

                available_capacity = sum(wind_farms_dict[farm_index]['capacity'] for farm_index in interconnector['connected_farms'])
                allocated_capacity = min(needed_capacity, available_capacity)

                if allocated_capacity > 0:
                    closest_grid_point = min(
                        grid_connections[grid_connections['country'] == country].itertuples(),
                        key=lambda gp: Point(gp.geometry.coords[0]).distance(interconnector['geometry'])
                    )

                    ic_to_grid_path = dynamic_graph.shortest_path(interconnector['geometry'], Point(closest_grid_point.geometry.coords[0]))
                    new_connections.append({
                        'geometry': ic_to_grid_path,
                        'farm_country': f'Interconnector_{interconnectors.index(interconnector)}',
                        'farm_index': None,
                        'grid_country': country,
                        'allocated_capacity': allocated_capacity,
                        'is_interconnector': True,
                        'connection_type': 'IC_to_grid'
                    })

                    interconnector['grid_connections'][country] = allocated_capacity
                    needed_capacity -= allocated_capacity

                    # Update wind farm capacities
                    for farm_index in interconnector['connected_farms']:
                        farm = wind_farms_dict[farm_index]
                        farm_allocation = (farm['capacity'] / available_capacity) * allocated_capacity
                        farm['capacity'] -= farm_allocation
                        if 'allocated_capacity' not in farm:
                            farm['allocated_capacity'] = {}
                        if country not in farm['allocated_capacity']:
                            farm['allocated_capacity'][country] = 0
                        farm['allocated_capacity'][country] += farm_allocation

        return gpd.GeoDataFrame(interconnectors, crs='EPSG:3035'), gpd.GeoDataFrame(new_connections, crs='EPSG:3035')


    def allocate_capacity_and_create_lines():
        connection_lines = []

        for country, grid_info in grid_connections_dict.items():
            needed_capacity = grid_info['needed_capacity']
            sorted_farms = sorted(
                [(index, farm) for index, farm in wind_farms_dict.items() if farm['country_of_origin'] == country],
                key=lambda x: x[1]['distances'][country]
            )

            for farm_index, farm in sorted_farms:
                if needed_capacity <= 0:
                    break

                available_capacity = farm['capacity']
                used_capacity = min(available_capacity, needed_capacity)
                needed_capacity -= used_capacity
                wind_farms_dict[farm_index]['capacity'] -= used_capacity

                if 'allocated_capacity' not in wind_farms_dict[farm_index]:
                    wind_farms_dict[farm_index]['allocated_capacity'] = {}
                wind_farms_dict[farm_index]['allocated_capacity'][country] = used_capacity

                farm_coords = farm['coordinates']
                grid_coords = farm['closest_grid_coords'][country]
                line = LineString([farm_coords, grid_coords])

                connection_lines.append({
                    'geometry': line,
                    'farm_country': farm['country_of_origin'],
                    'farm_index': farm_index,
                    'grid_country': country,
                    'allocated_capacity': used_capacity,
                    'connection_type': 'farm_to_grid'
                })

            grid_connections_dict[country]['needed_capacity'] = needed_capacity

        return gpd.GeoDataFrame(connection_lines, crs='EPSG:3035')

    def add_cable_costs(connections_gdf):
        cable_types = {
            'HVDC_320kV': {'max_capacity': 1216, 'cost_per_km': 500000},
            'HVDC_400kV': {'max_capacity': 1520, 'cost_per_km': 620000},
            'HVDC_525kV': {'max_capacity': 2000.25, 'cost_per_km': 825000},
            'XLPE_300kV': {'max_capacity': 874.8, 'cost_per_km': 835000}
        }

        def calculate_cable_combination(capacity_mw):
            combinations = []
            remaining_capacity = capacity_mw
            sorted_cables = sorted(cable_types.items(), key=lambda x: x[1]['max_capacity'], reverse=True)

            for cable_name, cable_info in sorted_cables:
                num_cables = int(remaining_capacity // cable_info['max_capacity'])
                if num_cables > 0:
                    combinations.append((cable_name, num_cables))
                    remaining_capacity -= num_cables * cable_info['max_capacity']

            if remaining_capacity > 0:
                combinations.append((sorted_cables[-1][0], 1))

            return combinations

        def calculate_cost(row):
            length_km = row['geometry'].length / 1000
            capacity_mw = row['allocated_capacity'] * 1000

            if row['is_interconnector']:
                if row['farm_country'] == 'Interconnector':
                    cable_combination = calculate_cable_combination(capacity_mw)
                    return sum(cable_types[cable]['cost_per_km'] * num * length_km for cable, num in cable_combination)
                else:
                    return length_km * 3000000
            else:
                return length_km * 3000000

        connections_gdf['length_km'] = connections_gdf['geometry'].length / 1000
        connections_gdf['cable_cost'] = connections_gdf.apply(calculate_cost, axis=1)
        connections_gdf['cable_combination'] = connections_gdf.apply(
            lambda row: calculate_cable_combination(row['allocated_capacity'] * 1000), axis=1)

        return connections_gdf

    def add_foundation_costs(interconnectors_gdf, connections_gdf):
        for interconnector in interconnectors_gdf.itertuples():
            name = f'Interconnector_{interconnector.Index}'
            total_capacity = sum(interconnector.grid_connections.values())
            total_connected_capacity_mw = total_capacity * 1000

            hvdc_converter_stations = (142.61 * total_connected_capacity_mw * 1000) * 2
            substation_topside_cost = 12000 * 14500
            design_cost_substation = 4_500_000
            diesel_generator_backup = 1_000_000
            workshop_accomodation_fire_protection = 2_000_000
            ancillary_cost = 3_000_000

            cables = connections_gdf[connections_gdf['farm_country'] == name]
            cable_combinations = cables['cable_combination']

            total_voltage = sum(sum(int(cable[0].split('_')[1].replace('kV', '')) * cable[1] for cable in combo) for combo in cable_combinations)
            total_number_cables = sum(sum(cable[1] for cable in combo) for combo in cable_combinations)

            onshore_substation_base = 6533.1 * (total_voltage / 1000) * total_number_cables

            total_cost = (hvdc_converter_stations + substation_topside_cost + design_cost_substation +
                          diesel_generator_backup + workshop_accomodation_fire_protection +
                          ancillary_cost + onshore_substation_base)

            interconnectors_gdf.at[interconnector.Index, 'foundation_cost'] = total_cost

        return interconnectors_gdf

    def summarize_allocations(interconnectors_gdf, connections_gdf):
        summary = {}

        for idx, interconnector in interconnectors_gdf.iterrows():
            interconnector_id = f"Interconnector_{idx}"
            connected_farms = interconnector['connected_farms']

            summary[interconnector_id] = {
                'incoming_capacity': {},
                'outgoing_capacity': interconnector['grid_connections'],
                'total_capacity': sum(interconnector['grid_connections'].values()),
                'foundation_cost': interconnector['foundation_cost']
            }

            incoming_connections = connections_gdf[
                (connections_gdf['is_interconnector'] == True) & 
                (connections_gdf['farm_index'].isin(connected_farms)) &
                (connections_gdf['grid_country'] == 'Interconnector')
            ]

            for _, connection in incoming_connections.iterrows():
                origin_country = wind_farms_dict[connection['farm_index']]['country_of_origin']
                capacity = connection['allocated_capacity']

                if origin_country not in summary[interconnector_id]['incoming_capacity']:
                    summary[interconnector_id]['incoming_capacity'][origin_country] = 0
                summary[interconnector_id]['incoming_capacity'][origin_country] += capacity

            cable_cost = connections_gdf[
                (connections_gdf['farm_index'].isin(connected_farms)) & 
                (connections_gdf['is_interconnector'] == True)
            ]['cable_cost'].sum()

            summary[interconnector_id]['cable_cost'] = cable_cost

            # Round the capacities for readability
            for country in summary[interconnector_id]['incoming_capacity']:
                summary[interconnector_id]['incoming_capacity'][country] = round(
                    summary[interconnector_id]['incoming_capacity'][country], 2
                )
            for country in summary[interconnector_id]['outgoing_capacity']:
                summary[interconnector_id]['outgoing_capacity'][country] = round(
                    summary[interconnector_id]['outgoing_capacity'][country], 2
                )
            summary[interconnector_id]['total_capacity'] = round(
                summary[interconnector_id]['total_capacity'], 2
            )

        return summary

    # Main execution
    connection_lines_gdf = allocate_capacity_and_create_lines()
    interconnectors_gdf, new_connections_gdf = create_interconnectors(wind_farms_dict, grid_connections_dict, num_clusters, linestrings_gdf=cable_paths_gdf)
    all_connections_gdf = pd.concat([connection_lines_gdf, new_connections_gdf], ignore_index=True)
    all_connections_gdf = add_cable_costs(all_connections_gdf)
    interconnectors_gdf = add_foundation_costs(interconnectors_gdf, all_connections_gdf)
    summary = summarize_allocations(interconnectors_gdf, all_connections_gdf)

    return interconnectors_gdf, all_connections_gdf, summary


def optimize_interconnector_configuration(wind_farms, grid_connections, cable_paths_gdf, max_interconnectors=5):
    def calculate_total_cost(summary):
        total_foundation_cost = sum(ic['foundation_cost'] for ic in summary.values())
        total_cable_cost = sum(ic['cable_cost'] for ic in summary.values())
        return total_foundation_cost + total_cable_cost

    best_config = None
    best_cost = float('inf')
    results = {}

    for num_interconnectors in range(1, max_interconnectors + 1):
        interconnectors_gdf, connections_gdf, summary = allocate_interconnectors(
            wind_farms, grid_connections, cable_paths_gdf, num_clusters=num_interconnectors
        )
        
        total_cost = calculate_total_cost(summary)
        total_capacity = sum(ic['total_capacity'] for ic in summary.values())
        
        results[num_interconnectors] = {
            'interconnectors_gdf': interconnectors_gdf,
            'connections_gdf': connections_gdf,
            'summary': summary,
            'total_cost': total_cost,
            'total_capacity': total_capacity
        }
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_config = num_interconnectors

    return results, best_config





def generate_all_hub_configurations(hubs, distance_km=20):
    # Convert 20 km to degrees (approximate)
    distance_deg = distance_km / 111  # 1 degree is approximately 111 km

    # Generate the 5 possible locations for each hub
    hub_locations = {}
    for hub_name, hub_data in hubs.items():
        base_lat, base_lon = hub_data['latitude'], hub_data['longitude']
        hub_locations[hub_name] = [
            (base_lat, base_lon),  # Current
            (base_lat + distance_deg, base_lon),  # North
            (base_lat, base_lon + distance_deg),  # East
            (base_lat, base_lon - distance_deg),  # West
            (base_lat - distance_deg, base_lon),  # South
        ]

    # Generate all possible combinations
    hub_names = list(hubs.keys())
    location_combinations = itertools.product(range(5), repeat=len(hub_names))

    for combo in location_combinations:
        new_hubs = copy.deepcopy(hubs)
        for hub_name, location_index in zip(hub_names, combo):
            new_lat, new_lon = hub_locations[hub_name][location_index]
            new_hubs[hub_name]['latitude'] = new_lat
            new_hubs[hub_name]['longitude'] = new_lon
        yield new_hubs



def distance(coord1, coord2):
    return np.sqrt(np.sum((np.array(coord1) - np.array(coord2))**2))

def total_distance(coordinates, data):
    total = 0
    for key, value in data.items():
        coord = coordinates[2*list(data.keys()).index(key):2*list(data.keys()).index(key)+2]
        connecting_coords = value['connecting_coordinates']
        for i, connecting_coord in enumerate(connecting_coords):
            # Give double weight to the final coordinate
            weight = 3 if i == len(connecting_coords) - 1 else 1
            total += weight * distance(coord, connecting_coord)
    return total

def optimize_coordinates(data):
    initial_coordinates = [coord for value in data.values() for coord in value['coordinates']]

    result = minimize(
        lambda x: total_distance(x, data),
        initial_coordinates,
        method='BFGS'
    )
    
    optimized_coordinates = result.x
    
    optimized_data = {}
    for i, (key, value) in enumerate(data.items()):
        optimized_data[key] = {
            'coordinates': list(optimized_coordinates[2*i:2*i+2]),
            'connecting_coordinates': value['connecting_coordinates']
        }
    
    return optimized_data

def convert_to_epsg3035(lat, lon):
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:3035").to_crs("EPSG:3035")
    return [point.x.iloc[0], point.y.iloc[0]]

def get_coordinates(location, hubs, north_sea_ports):
    if location in hubs:
        return convert_to_epsg3035(hubs[location]['latitude'], hubs[location]['longitude'])
    for country_ports in north_sea_ports.values():
        if location in country_ports:
            return convert_to_epsg3035(country_ports[location]['latitude'], country_ports[location]['longitude'])
    return None

def generate_hubs_with_connections(hubs, set_cables, north_sea_ports, hub_cluster_df, cluster_gdf):
    hubs_with_connections = {}
    
    # Ensure cluster_gdf is in EPSG:3035
    cluster_gdf = cluster_gdf.to_crs("EPSG:3035")
    
    # Create a dictionary mapping hubs to their cluster centroids
    hub_to_cluster_centroid = {}
    for _, row in hub_cluster_df.iterrows():
        hub = row['to']
        cluster_id = row['from']
        cluster_geometry = cluster_gdf[cluster_gdf['cluster_id'] == cluster_id].geometry.iloc[0]
        centroid = cluster_geometry.centroid
        hub_to_cluster_centroid[hub] = [centroid.x, centroid.y]
    
    for hub_name, hub_info in hubs.items():
        connecting_coordinates = []
        
        # Add connections from set_cables
        for cable_info in set_cables.values():
            if cable_info['from'] == hub_name:
                coords = get_coordinates(cable_info['to'], hubs, north_sea_ports)
                if coords:
                    connecting_coordinates.append(coords)
            elif cable_info['to'] == hub_name:
                coords = get_coordinates(cable_info['from'], hubs, north_sea_ports)
                if coords:
                    connecting_coordinates.append(coords)
        
        # Add cluster centroid coordinate
        if hub_name in hub_to_cluster_centroid:
            connecting_coordinates.append(hub_to_cluster_centroid[hub_name])
        
        # Convert hub's own coordinates to EPSG:3035
        hub_coords_3035 = convert_to_epsg3035(hub_info['latitude'], hub_info['longitude'])
        hubs_with_connections[hub_name] = {
            "coordinates": hub_coords_3035,
            "connecting_coordinates": connecting_coordinates
        }
    
    return hubs_with_connections