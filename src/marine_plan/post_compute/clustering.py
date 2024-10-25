import rasterio
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from skimage.measure import label
from shapely.ops import unary_union
import os
import pickle
from scipy.ndimage import binary_dilation
import copy


def get_unit_amounts(km2, tech):
    '''
    Returns the number of units for a given area and technology
    '''

    ud = {
        'wind': 0.5,
        'jacket': 0.5,
        'monopile': 0.5,
        'semisub_cat_drag': 0.33,
        'semisub_taut_driv': 0.33,
        'semisub_taut_suc': 0.33,
        'spar_cat_drag': 0.33,
        'spar_taut_driv': 0.33,
        'spar_taut_suc': 0.33,
        
        'seaweed': 36,
        'mussel': 36,
        'fpv': 1150
        }
    
    return round(km2 * ud[tech])


def reshape_metrics(mp):
    '''
    Reshapes the metrics to a 2D array for use in the clustering algorithm
    '''

    # Reshape each list to a 2D array
    mp.reshaped_metrics = {}
    mp.reshaped_geo_data = {}

    for tech, metrics in mp.data.items():
        mp.reshaped_metrics[tech] = {}
        for metric, value in metrics.items():
            mp.reshaped_metrics[tech][metric] = np.array(value).reshape(mp.seed_array.shape)

    for data in mp.geo_data:
        mp.reshaped_geo_data[data] = np.array(mp.geo_data[data]).reshape(mp.seed_array.shape)


def get_metrics(mp, row, col, tech, no_country=False):
        ''' 
        Returns the metrics for a given row and column in the original metrics array
        '''

        metrics = {}

        if tech == 'wind':
            for metric, array in mp.reshaped_metrics['monopile'].items():
                metrics[metric] = array[row, col]

        else:
            for metric, array in mp.reshaped_metrics[tech].items():
                metrics[metric] = array[row, col]

        metrics['country'] = mp.reshaped_geo_data['country'][row, col] if not no_country else None
        metrics['closest_gridconnect'] = mp.reshaped_geo_data['closest_gridconnect'][row, col]

        # check if 'aep' and 'afp' are in the metrics, if not, add an array of zeros
        if 'energy_produced' not in metrics:
            metrics['energy_produced'] = 0
        if 'food_produced' not in metrics:
            metrics['food_produced'] = 0

        #multiply all metrics by the resolution
        for metric, value in metrics.items():
            if metric == 'country' or metric == 'closest_gridconnect':
                continue
            metrics[metric] = value * mp.resolution


        return metrics


def add_gridconnect_to_cluster(cluster_gdf, combined_gdf, locations):
    # Create a dictionary mapping location names to their geometries
    point_geometries = dict(zip(combined_gdf['name'], combined_gdf['geometry']))

    # Create a reverse lookup dictionary for locations
    index_to_name = {v['index']: k for k, v in locations.items()}

    # Function to get gridconnect info
    def get_gridconnect_info(closest_gridconnect):
        if pd.notna(closest_gridconnect):
            try:
                gridconnect_id = int(closest_gridconnect)
                point_name = index_to_name.get(gridconnect_id)
                if point_name and point_name in point_geometries:
                    point = point_geometries[point_name]
                    return point_name, point, (point.x, point.y)
            except ValueError:
                print(f"Unable to convert {closest_gridconnect} to integer")
        return None, None, None

    # Apply the function to each row in cluster_gdf
    result = cluster_gdf['closest_gridconnect'].apply(get_gridconnect_info)
    cluster_gdf['gridconnect_name'] = result.apply(lambda x: x[0] if x is not None else None)
    #cluster_gdf['gridconnect_geometry'] = result.apply(lambda x: x[1] if x is not None else None)
    cluster_gdf['gridconnect_coords'] = result.apply(lambda x: x[2] if x is not None else None)

    return cluster_gdf


def clusters_to_gdf(mp):
    with rasterio.open(mp.template_r) as template:
        transform, crs = template.transform, template.crs

    all_polygons = []
    wind_polygons = []

    for tech, _clusters in mp.clusters.items():
        if tech == 'empty':
            continue

        for cluster_id, cluster in _clusters.items():
            rows, cols = np.where(cluster['a'] == 1)
            km2 = round(cluster['count'] * mp.resolution, 2)
            unit_count = get_unit_amounts(km2, tech)

            cluster_polygons = []
            for row, col in zip(rows, cols):
                minx, maxy = rasterio.transform.xy(transform, row - 0.5, col - 0.5)
                maxx, miny = rasterio.transform.xy(transform, row + 0.5, col + 0.5)
                polygon = box(minx, miny, maxx, maxy, ccw=True)
                cluster_polygons.append(polygon)

            if cluster_polygons:
                merged_polygon = unary_union(cluster_polygons)
                cluster_data = {
                    "tech": tech,
                    "cluster_id": cluster_id,
                    "km2": km2,
                    "unit_count": unit_count,
                    "capex": sum(get_metrics(mp, row, col, tech)['capex'] for row, col in zip(rows, cols)),
                    "opex": sum(get_metrics(mp, row, col, tech)['opex'] for row, col in zip(rows, cols)),
                    "revenue": sum(get_metrics(mp, row, col, tech)['revenue'] for row, col in zip(rows, cols)),
                    "average_LCOE": sum(get_metrics(mp, row, col, tech)['LCOE'] for row, col in zip(rows, cols)) / unit_count * 0.5,
                    "energy": sum(get_metrics(mp, row, col, tech)['energy_produced'] for row, col in zip(rows, cols)),
                    "food": sum(get_metrics(mp, row, col, tech)['food_produced'] for row, col in zip(rows, cols)),
                    "country": max(set(get_metrics(mp, row, col, tech)['country'] for row, col in zip(rows, cols)), key=list(get_metrics(mp, row, col, tech)['country'] for row, col in zip(rows, cols)).count),
                    "closest_gridconnect": min(get_metrics(mp, row, col, tech)['closest_gridconnect'] for row, col in zip(rows, cols)),
                    "geometry": merged_polygon
                }
                
                if tech in ['monopile', 'jacket']:
                    wind_polygons.append(cluster_data)
                else:
                    all_polygons.append(cluster_data)

    # Process wind polygons
    wind_gdf = gpd.GeoDataFrame(wind_polygons, crs=crs)
    
    # Find touching wind clusters
    def find_touching_clusters(gdf):
        touched = []
        for idx, geom in gdf.geometry.items():
            touching = gdf[gdf.geometry.touches(geom)].index.tolist()
            if touching:
                touched.append(set([idx] + touching))
        return touched

    touching_groups = find_touching_clusters(wind_gdf)

    # Merge touching wind clusters
    for group in touching_groups:
        # Filter out indices that are not in the DataFrame
        valid_indices = [idx for idx in group if idx in wind_gdf.index]
        if not valid_indices:
            continue  # Skip this group if no valid indices
        
        merged_cluster = wind_gdf.loc[valid_indices]

        merged_cluster['units_monopile'] = merged_cluster[merged_cluster['tech'] == 'monopile']['unit_count'].sum()
        merged_cluster['units_jacket'] = merged_cluster[merged_cluster['tech'] == 'jacket']['unit_count'].sum()

        new_cluster_data = {
            "tech": "wind",
            "cluster_id": f"wind-{len(all_polygons)}",
            "km2": merged_cluster['km2'].sum(),
            "unit_count": merged_cluster['unit_count'].sum(),
            "capex": merged_cluster['capex'].sum(),
            "opex": merged_cluster['opex'].sum(),
            "revenue": merged_cluster['revenue'].sum(),
            "average_LCOE": (merged_cluster['average_LCOE'] * merged_cluster['unit_count']).sum() / merged_cluster['unit_count'].sum(),
            "energy": merged_cluster['energy'].sum(),
            "food": merged_cluster['food'].sum(),
            "country": merged_cluster['country'].mode().iloc[0],
            "closest_gridconnect": merged_cluster['closest_gridconnect'].mode().iloc[0],
            "geometry": merged_cluster.geometry.unary_union,
            "units_monopile": merged_cluster[merged_cluster['tech'] == 'monopile']['unit_count'].sum(),
            "units_jacket": merged_cluster[merged_cluster['tech'] == 'jacket']['unit_count'].sum(),
            "most_common_tech": "monopile" if merged_cluster['units_monopile'].sum() > merged_cluster['units_jacket'].sum() else "jacket"
        }
        all_polygons.append(new_cluster_data)
        wind_gdf = wind_gdf.drop(valid_indices)

    # Add remaining non-touching wind clusters
    for _, row in wind_gdf.iterrows():
        cluster_data = row.to_dict()
        cluster_data['units_monopile'] = cluster_data['unit_count'] if cluster_data['tech'] == 'monopile' else 0
        cluster_data['units_jacket'] = cluster_data['unit_count'] if cluster_data['tech'] == 'jacket' else 0
        cluster_data['tech'] = 'wind'
        cluster_data['most_common_tech'] = 'monopile' if cluster_data['units_monopile'] > cluster_data['units_jacket'] else 'jacket'
        all_polygons.append(cluster_data)

    cluster_gdf = gpd.GeoDataFrame(all_polygons, crs=crs)
    cluster_gdf['capacity'] = cluster_gdf['unit_count'] * 15 / 1000

    # if the tech is mussel or seaweed, the capacity is 0
    cluster_gdf.loc[cluster_gdf['tech'].isin(['mussel', 'seaweed']), 'capacity'] = 0

    substations = mp.land_object_gdf[mp.land_object_gdf['designation'] == 'substation'][['name', 'geometry']]


    with open(os.path.join(mp.directory, "temp", "closest_gridconnects.pkl"), 'rb') as f:
        locations = pickle.load(f)

    if mp.hubs_included:
        combined_gdf = gpd.GeoDataFrame(pd.concat([substations, mp.hubs_gdf], ignore_index=True))
        return add_gridconnect_to_cluster(cluster_gdf, combined_gdf, locations)
    else:
        return add_gridconnect_to_cluster(cluster_gdf, substations, locations)


def create_clusters(mp, min_cluster_size=2):

    mp.clusters = {}
    mp._set_resolution()
    reshape_metrics(mp) 

    for value in range(len(mp.uDict) - 1):
        tech = next(t for t, v in mp.uDict.items() if v == value)

        array = (mp.seed_array == value).astype(int)
        cluster_array = label(array)

        _clusters = {
            f"cluster-{code}": {'a': (cluster_array == code).astype(int), 'count': count}
            for code, count in enumerate(np.bincount(cluster_array.ravel())[1:], 1)
            if count >= min_cluster_size
        }

        if _clusters:
            mp.clusters[tech] = _clusters

    return clusters_to_gdf(mp)



# MARK: Min cluster culling

def get_too_small_clusters(mp, min_cluster_size=6):
    """
    Identify clusters that are smaller than the specified minimum size.

    Args:
        mp (instance): The NCN optimized run.
        min_cluster_size (int): The minimum allowed cluster size. Defaults to 6.

    Returns:
        dict: A dictionary of technologies and their small clusters.
    """
    clusters = {}
    for value in range(len(mp.uDict) - 1):
        tech = next(t for t, v in mp.uDict.items() if v == value)

        if tech not in ['monopile', 'jacket', 'semisub_cat_drag', 'semisub_taut_driv', 'semisub_taut_suc', 'spar_cat_drag', 'spar_taut_driv', 'spar_taut_suc']:
            continue

        array = (mp.seed_array == value).astype(int)
        cluster_array = label(array)

        _clusters = {
            f"cluster-{code}": {'a': (cluster_array == code).astype(int), 'count': count}
            for code, count in enumerate(np.bincount(cluster_array.ravel())[1:], 1)
            if count < min_cluster_size
        }

        if _clusters:
            clusters[tech] = _clusters

    return clusters


def remove_metrics(mp, small_clusters):
    """
    Remove metrics for small clusters and track the changes.

    Args:
        mp (instance): The NCN optimized run.
        small_clusters (dict): Dictionary of small clusters to remove.

    Returns:
        tuple: A tuple containing the total metric tracker and removed indices.
    """
    total_metric_tracker = {tech: {
        'removed': {metric: 0 for metric in ['capex', 'opex', 'revenue', 'energy_produced', 'food_produced', 'co2+', 'co2-', 'eco', 'LCOE']},
        'added': {metric: 0 for metric in ['capex', 'opex', 'revenue', 'energy_produced', 'food_produced', 'co2+', 'co2-', 'eco', 'LCOE']},
        'num_removed': 0
    } for tech in small_clusters.keys()}

    removed_indices = []

    for tech, clusters in small_clusters.items():
        for cluster in clusters.values():
            rows, cols = np.where(cluster['a'] == 1)
            cluster_metrics = [get_metrics(mp, row, col, tech) for row, col in zip(rows, cols)]

            for metric in total_metric_tracker[tech]['removed']:
                total_metric_tracker[tech]['removed'][metric] += sum(c_metric[metric] for c_metric in cluster_metrics) * mp.resolution * mp.data[tech]["unit density"][mp.first_num]

            total_metric_tracker[tech]['num_removed'] += cluster['count']
            removed_indices.extend(list(zip(rows, cols)))

    return total_metric_tracker, removed_indices


def calculate_objective_value(metrics, objective_function):
    """
    Calculate the objective value based on the given metrics and objective function.

    Args:
        metrics (dict): Dictionary of metric values.
        objective_function (dict): Dictionary defining the objective function.

    Returns:
        float: The calculated objective value.
    """
    value = 0
    
    for key, weight in objective_function['positives'].items():
        metric_key = key.replace('total_', '')
        metric_value = metrics['co2+'] - metrics['co2-'] if metric_key == 'CO2_net' else metrics.get(metric_key, 0)
        value += metric_value * weight
    
    for key, weight in objective_function['negatives'].items():
        metric_key = key.replace('total_', '')
        value -= metrics.get(metric_key, 0) * weight
    
    return -value if objective_function['direction'] == 'minimize' else value


def create_suitability_maps(mp, seed_array, removed_indices, objective_function):
    """
    Create suitability maps for each technology based on the current seed array and removed indices.

    Args:
        mp (instance): The NCN optimized run.
        seed_array (numpy.ndarray): The current seed array.
        removed_indices (list): List of indices to be removed.
        objective_function (dict): Dictionary defining the objective function.

    Returns:
        tuple: A tuple containing suitability maps and new metrics for each technology.
    """
    rows, cols = seed_array.shape
    
    for row, col in removed_indices:
        seed_array[row, col] = mp.uDict['empty']
    
    suitability_maps = {}
    new_metrics = {tech: {} for tech in mp.uDict if tech != 'empty'}

    struct = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=bool)    
    empty_mask = (seed_array == mp.uDict['empty'])

    for tech, value in mp.uDict.items():
        if tech == 'empty':
            continue
        
        tech_mask = (seed_array == value)
        dilated = binary_dilation(tech_mask, structure=struct)
        suitability_map = dilated & empty_mask

        suitable_metrics = {metric: np.zeros((rows, cols)) for metric in ['capex', 'opex', 'revenue', 'energy_produced', 'food_produced', 'co2+', 'co2-', 'eco', 'LCOE', 'unit density', 'lifetime', 'country', 'closest_gridconnect', 'objective']}

        for row, col in zip(*np.where(suitability_map)):
            metrics = get_metrics(mp, row, col, tech, no_country=True)
            if metrics['capex'] == 0:
                continue

            metrics['objective'] = calculate_objective_value(metrics, objective_function)

            for metric, value in metrics.items():
                suitable_metrics[metric][row, col] = value

        new_metrics[tech] = suitable_metrics
        suitability_maps[tech] = suitability_map
    
    return suitability_maps, new_metrics


def rank_objectives(objectives_2d, direction, ignore_zeros=True):
    """
    Rank the objectives based on the given direction.

    Args:
        objectives_2d (numpy.ndarray): 2D array of objective values.
        direction (str): Either 'minimize' or 'maximize'.
        ignore_zeros (bool): Whether to ignore zero values. Defaults to True.

    Returns:
        list: Ranked positions as (col, row) tuples.
    """
    if direction not in ['minimize', 'maximize']:
        raise ValueError("Direction must be either 'minimize' or 'maximize'")
    
    indexed_objectives = [(row, col, value) for row, row_values in enumerate(objectives_2d) for col, value in enumerate(row_values)]
    
    key_func = lambda x: (x[2] == 0, x[2] if direction == 'minimize' else -x[2])
    sorted_objectives = sorted(indexed_objectives, key=key_func)
    
    return [(col, row) for row, col, val in sorted_objectives if val != 0 or not ignore_zeros]


def modify_seed(mp, seed_array, removed_indices, rankings, total_metrics):
    """
    Modify the seed array based on the rankings and update the total metrics.

    Args:
        mp (instance): The NCN optimized run.
        seed_array (numpy.ndarray): The current seed array.
        removed_indices (list): List of indices to be removed.
        rankings (dict): Dictionary of rankings for each technology.
        total_metrics (dict): Dictionary of total metrics for each technology.

    Returns:
        tuple: Updated total metrics and modified seed array.
    """
    for row, col in removed_indices:
        seed_array[row, col] = mp.uDict['empty']

    for tech, indices in rankings.items():

        if len(indices) == 0:
            continue

        total_metrics = {tech: {
            'removed': {metric: 0 for metric in ['capex', 'opex', 'revenue', 'energy_produced', 'food_produced', 'co2+', 'co2-', 'eco', 'LCOE']},
            'added': {metric: 0 for metric in ['capex', 'opex', 'revenue', 'energy_produced', 'food_produced', 'co2+', 'co2-', 'eco', 'LCOE']},
            'num_removed': 0
        }}

        num_to_add = total_metrics[tech]['num_removed']

        for col, row in indices[:num_to_add]:
            seed_array[row, col] = mp.uDict[tech]
    
            metrics = get_metrics(mp, row, col, tech)
            for metric, value in metrics.items():
                if metric in total_metrics[tech]['added']:
                    total_metrics[tech]['added'][metric] += value * mp.resolution * mp.data[tech]["unit density"][mp.first_num]

    return total_metrics, seed_array


def cull_small_clusters(mp):
    """
    Identify and remove small clusters, then replace them with suitable alternatives.

    Args:
        mp (instance): The NCN optimized run.

    Returns:
        tuple: Updated total metrics and modified seed array.
    """
    mp._set_resolution()
    reshape_metrics(mp) 

    seed_array = copy.deepcopy(mp.seed_array)
    objective_function = mp.optimization_params

    small_clusters = get_too_small_clusters(mp, min_cluster_size=6)
    total_metrics, removed_indices = remove_metrics(mp, small_clusters)
    suitability_maps, new_metrics = create_suitability_maps(mp, seed_array, removed_indices, objective_function)

    rankings = {tech: rank_objectives(metrics['objective'], objective_function['direction'], ignore_zeros=True)
                for tech, metrics in new_metrics.items()}

    return modify_seed(mp, seed_array, removed_indices, rankings, total_metrics)