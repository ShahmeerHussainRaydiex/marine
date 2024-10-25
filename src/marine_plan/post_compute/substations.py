import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import distance_matrix
from collections import Counter
import rasterio
import pulp
import math
from src.config.substation_config import sub_CONFIG

TURBINE_DENSITY = 0.5  # turbines per square kilometer
SPACING = math.sqrt(1 / TURBINE_DENSITY) * 1000  # convert km to meters



def optimize_cluster(cluster_id, total_units):
    # First solve: Minimize the number of substations
    model1 = pulp.LpProblem(f"Substation_Optimization_Step1_{cluster_id}", pulp.LpMinimize)
    x1 = {config: pulp.LpVariable(f"x1_{config}", lowBound=0, cat='Integer') for config in sub_CONFIG}
    
    # Objective: Minimize the number of substations
    model1 += pulp.lpSum(x1[config] for config in sub_CONFIG)
    
    # Constraint: Meet or exceed the required number of turbines
    model1 += pulp.lpSum(sub_CONFIG[config]['max_turbines'] * x1[config] for config in sub_CONFIG) >= total_units

    # Solve the model
    solver = pulp.HiGHS(msg=0)
    status1 = model1.solve(solver)

    if status1 != pulp.LpStatusOptimal:
        print(f"Warning: Non-optimal solution for cluster {cluster_id} in Step 1. Status: {pulp.LpStatus[status1]}")

    # Get the minimum number of substations
    min_substations = sum(int(x1[config].value()) for config in sub_CONFIG)

    # Second solve: Minimize the difference in turbines while keeping the number of substations fixed
    model2 = pulp.LpProblem(f"Substation_Optimization_Step2_{cluster_id}", pulp.LpMinimize)
    x2 = {config: pulp.LpVariable(f"x2_{config}", lowBound=0, cat='Integer') for config in sub_CONFIG}
    difference = pulp.LpVariable("difference", lowBound=0)

    # Objective: Minimize the difference
    model2 += difference

    # Constraints
    model2 += pulp.lpSum(x2[config] for config in sub_CONFIG) == min_substations
    model2 += pulp.lpSum(sub_CONFIG[config]['max_turbines'] * x2[config] for config in sub_CONFIG) >= total_units
    model2 += difference >= pulp.lpSum(sub_CONFIG[config]['max_turbines'] * x2[config] for config in sub_CONFIG) - total_units

    # Solve the second optimization
    status2 = model2.solve(solver)

    if status2 != pulp.LpStatusOptimal:
        print(f"Warning: Non-optimal solution for cluster {cluster_id} in Step 2. Status: {pulp.LpStatus[status2]}")

    # Extract final results
    results = {config: int(x2[config].value()) for config in sub_CONFIG}
    assigned_turbines = sum(sub_CONFIG[config]['max_turbines'] * results[config] for config in sub_CONFIG)
    num_substations = sum(results.values())
    turbine_difference = assigned_turbines - total_units

    return results, assigned_turbines, num_substations, turbine_difference


def create_grid(bounds, spacing=SPACING):
    minx, miny, maxx, maxy = bounds
    x_coords = np.arange(minx, maxx + spacing, spacing)
    y_coords = np.arange(miny, maxy + spacing, spacing)
    grid = [Point(x, y) for y in y_coords for x in x_coords]
    return grid


def remove_edge_points(points, polygon, num_points):
    points_in_poly = [p for p in points if polygon.contains(p)]
    if len(points_in_poly) <= num_points:
        return points_in_poly
    centroid = polygon.centroid
    sorted_points = sorted(points_in_poly, key=lambda p: p.distance(centroid))
    return sorted_points[:num_points]


def place_turbines_with_substations(row, crs, optimization_results):
    grid = create_grid(row['geometry'].bounds)
    selected_points = remove_edge_points(grid, row['geometry'], row['unit_count'])
    
    cluster_id = row['cluster_id']
    substation_config = optimization_results[cluster_id]['substation_config']
    
    turbine_data = []
    turbine_count = 0
    substation_count = {config: 0 for config in substation_config}
    current_substation = None
    current_substation_capacity = 0
    
    for point in selected_points:
        if turbine_count < row['unit_count']:
            if current_substation is None or current_substation_capacity == 0:
                # Assign a new substation
                for config, count in substation_config.items():
                    if substation_count[config] < count:
                        current_substation = config
                        current_substation_capacity = sub_CONFIG[config]['max_turbines']
                        substation_count[config] += 1
                        break
            
            turbine_data.append({
                'cluster_id': cluster_id,
                'geometry': point,
                'substation_type': current_substation,
                'substation_number': substation_count[current_substation]
            })
            turbine_count += 1
            current_substation_capacity -= 1

    return gpd.GeoDataFrame(turbine_data, crs=crs)


def get_geo_data(directory, point):

    depth_path = os.path.join(directory, 'temp', 'depth.tif')
    seabed_path = os.path.join(directory, 'temp', 'seabed_substrate.tif')

    # Open the raster file
    with rasterio.open(depth_path) as src:
        # Extract the coordinates from the point
        point_coords = (point.x, point.y)
        # Get the row and column of the raster for the point
        row, col = src.index(*point_coords)
        # Read the value at the raster cell
        depth = src.read(1)[row, col]

    if depth < 0:
        depth = 1

    table = {
    1: int(4_000_000),
    2: int(64_000_000),
    3: int(16_000_000),
    4: int(32_000_000),
    5: int(1),
    } 

    with rasterio.open(seabed_path) as src:
        # Extract the coordinates from the point
        point_coords = (point.x, point.y)
        # Get the row and column of the raster for the point
        row, col = src.index(*point_coords)
        # Read the value at the raster cell
        seabed_val = src.read(1)[row, col]
        
        if seabed_val not in table:
            seabed_val = 2

    return depth, table[seabed_val]


def calc_ss_foundation_cost(directory, point, wattage):

    depth, seabed = get_geo_data(directory, point)

    tton = 4.5 * wattage + 285
    cpmw = (201 * (depth ** 2) + 612.93 * depth + 171464) * 0.92
    beta = (-0.1203 * (depth ** (-0.272)))
    alfa = cpmw / (32000000 ** beta)
    cpmwds = alfa * (seabed ** beta)
    tmtm = tton/100
    fnc = cpmwds*tmtm*0.4

    return fnc


def create_substations_from_turbines(turbines_gdf):
    substations = []
    for (cluster_id, substation_type, substation_number), group in turbines_gdf.groupby(['cluster_id', 'substation_type', 'substation_number']):
        centroid = Point(group.geometry.x.mean(), group.geometry.y.mean())
        substations.append({
            'cluster_id': cluster_id,
            'substation_id': f"{cluster_id}_{substation_type}_{substation_number}",
            'substation_type': substation_type,
            'num_turbines': len(group),
            'geometry': centroid
        })
    
    return gpd.GeoDataFrame(substations, crs=turbines_gdf.crs)


def create_cable_connections(turbines_gdf, substations_gdf):
    cables = []
    for _, substation in substations_gdf.iterrows():
        substation_turbines = turbines_gdf[
            (turbines_gdf['cluster_id'] == substation['cluster_id']) & 
            (turbines_gdf['substation_type'] == substation['substation_type']) & 
            (turbines_gdf['substation_number'] == int(substation['substation_id'].split('_')[-1]))
        ]
        
        for i in range(0, len(substation_turbines), 5):
            string = substation_turbines.iloc[i:i+5]
            for j in range(len(string) - 1):
                cables.append({
                    'type': 'x185' if j < 3 else 'x630',
                    'geometry': LineString([string.iloc[j].geometry, string.iloc[j+1].geometry])
                })
            cables.append({
                'type': 'x630',
                'geometry': LineString([string.iloc[-1].geometry, substation.geometry])
            })
    
    return gpd.GeoDataFrame(cables, crs=turbines_gdf.crs)


def connect_substations_to_grid(substations_gdf, grid_connection_points):
    cables = []
    for _, substation in substations_gdf.iterrows():
        nearest_grid_point = grid_connection_points.iloc[
            grid_connection_points.distance(substation.geometry).argmin()
        ]
        cables.append({
            'type': substation['substation_type'],
            'geometry': LineString([substation.geometry, nearest_grid_point.geometry])
        })
    return gpd.GeoDataFrame(cables, crs=substations_gdf.crs)


def calculate_total_cost(inter_array_cables, substation_cables, substations_gdf, directory):
    cost = 0
    for _, cable in inter_array_cables.iterrows():
        length_km = min(cable.geometry.length / 1000, 5)  # Cap at 40 km
        cost += sub_CONFIG[cable['type']]['cost_per_km'] * length_km

    for _, cable in substation_cables.iterrows():
        length_km = min(cable.geometry.length / 1000, 50)  # Cap at 40 km
        cost += sub_CONFIG[cable['type']]['cost_per_km'] * length_km
    for _, substation in substations_gdf.iterrows():
        sub_type = substation['substation_type']
        cost += sub_CONFIG[sub_type]['dev_cost'] + sub_CONFIG[sub_type]['insurance'] + (sub_CONFIG[sub_type]['install_uninstall'] * 1.6)
        cost += calc_ss_foundation_cost(directory, substation.geometry, (substation['num_turbines'] * 15))

    return cost


def simplified_wind_farm_layout(turbines_gdf, grid_connection_points, directory):
    if grid_connection_points.crs != turbines_gdf.crs:
        grid_connection_points = grid_connection_points.to_crs(turbines_gdf.crs)
    
    substations_gdf = create_substations_from_turbines(turbines_gdf)
    inter_array_cables = create_cable_connections(turbines_gdf, substations_gdf)
    substation_cables = connect_substations_to_grid(substations_gdf, grid_connection_points)
    total_cost = calculate_total_cost(inter_array_cables, substation_cables, substations_gdf, directory)
    
    return substations_gdf, inter_array_cables, substation_cables, total_cost


def compute_substation_cabling(mp):
    # Optimize for each cluster
    optimization_results = {}
    for index, row in mp.cluster_gdf.iterrows():
        cluster_id = row['cluster_id']
        total_units = row['unit_count']
        results, assigned_turbines, num_substations, turbine_difference = optimize_cluster(cluster_id, total_units)
        optimization_results[cluster_id] = {
            'total_units': total_units,
            'assigned_turbines': assigned_turbines,
            'num_substations': num_substations,
            'turbine_difference': turbine_difference,
            'substation_config': results
        }

    # Main execution
    turbine_gdfs = []

    for idx, row in mp.cluster_gdf.iterrows():
        turbine_gdf = place_turbines_with_substations(row, mp.cluster_gdf.crs, optimization_results)
        turbine_gdfs.append(turbine_gdf)

    turbines_gdf = gpd.GeoDataFrame(pd.concat(turbine_gdfs, ignore_index=True))
    turbines_gdf.set_crs(mp.cluster_gdf.crs, inplace=True)
    turbines_gdf['turbine_id'] = range(1, len(turbines_gdf) + 1)

    grid_connects = mp.land_object_gdf[mp.land_object_gdf['designation'] == 'substation'][['name', 'geometry']]
    combined_gdf = gpd.GeoDataFrame(pd.concat([grid_connects, mp.hubs_gdf], ignore_index=True))

    # Usage
    substations, inter_array_cables, substation_cables, total_cost = simplified_wind_farm_layout(turbines_gdf, combined_gdf, mp.directory)

    return substations, inter_array_cables, substation_cables, total_cost