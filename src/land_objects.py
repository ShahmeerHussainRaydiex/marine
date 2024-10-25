import os
import shutil
import pyproj
from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree
from pyproj import Transformer
from typing import Union, Tuple, Dict, List

from src.raster_util import *
from src.metric_util_OLD import *
from src.config.landobject_config import north_sea_ports, custom_north_sea_ports


class LandObjects:

    def __init__(self, directory, custom_ports=False):
        self.directory = directory
        self.original_data = north_sea_ports if not custom_ports else custom_north_sea_ports
        self.exit = False
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        self.iterative = False
        self.results = {}
        self.updated_ports = {}


    def _set_pathing(self):
        for file in os.listdir(os.path.join(self.directory, 'temp')):
            if file.endswith(".tif") and 'country' in file:
                self.template_r = os.path.join(self.directory, 'temp', file)
                break

        if not hasattr(self, 'template_r'):
            for file in os.listdir(os.path.join(self.directory, 'data', 'rasters', 'international')):
                if file.endswith(".tif") and 'country' in file:
                    self.template_r = os.path.join(self.directory, 'data', 'rasters', 'international', file)
                    break

        self.temp_folder = os.path.join(self.directory, 'temp')
        self.working_folder = os.path.join(self.temp_folder, 'processing')


    def _get_closest_features(self, used_north_sea_ports, hubs, add_hubs, template_raster_path, output_folder, designation):

        if designation == 'substation':
            output_raster_path = os.path.join(output_folder, 'closest_gridconnects.tif')
            output_dict_path = os.path.join(output_folder, 'closest_gridconnects.pkl')

            designations = ['substation']
        
        elif designation == 'install_port':
            output_raster_path = os.path.join(output_folder, 'closest_install_port.tif')
            output_dict_path = os.path.join(output_folder, 'closest_install_port.pkl')

            designations = ['both', 'ins']

        locations, location_names, location_types, location_country, capacities, num_boats = [], [], [], [], [], []

        for country, country_locations in used_north_sea_ports.items():
            for name, info in country_locations.items():
                if info['designation'] in designations:
                    x, y = self.transformer.transform(info['longitude'], info['latitude'])
                    locations.append((x, y))
                    location_names.append(name)
                    location_types.append(designations[0])
                    location_country.append(country)
                    capacities.append(None)
                    num_boats.append(info['num_boats'] if designation == 'install_port' else None)

        if add_hubs and designation == 'substation':
            for name, info in hubs.items():
                x, y = self.transformer.transform(info['longitude'], info['latitude'])
                locations.append((x, y))
                location_names.append(name)
                location_types.append('hub')
                location_country.append('international')
                capacities.append(None)

        tree = cKDTree(locations)

        with rasterio.open(template_raster_path) as src:
            template_raster = src.read(1)
            transform = src.transform
            crs = src.crs

        rows, cols = template_raster.shape
        xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
        x, y = rasterio.transform.xy(transform, ys, xs)
        geo_coords = np.stack((x, y), axis=-1)

        _, indices = tree.query(geo_coords.reshape(-1, 2))
        nearest_location_index = indices.reshape(rows, cols)

        with rasterio.open(
            output_raster_path, 'w',
            driver='GTiff', height=rows, width=cols, count=1,
            dtype=np.int32, crs=crs, transform=transform,
        ) as dst:
            dst.write(nearest_location_index, 1)

        location_info_dict = {
            name: {
                'index': i,
                'type': location_types[i],
                'capacity': capacities[i],
                'country': location_country[i],
                'num_boats': num_boats[i]
            } for i, name in enumerate(location_names)
        }


        if designation == 'install_port':
            turbine_dict = {v['index']: v['num_boats'] * 32 for k, v in location_info_dict.items()} # 32 is the number of turbines that can be installed per boat

            with open(output_dict_path, 'wb') as f:
                pickle.dump(turbine_dict, f)
        
        else:
            with open(output_dict_path, 'wb') as f:
                pickle.dump(location_info_dict, f)
 

    def _update_north_sea_ports(self, used_north_sea_ports, connector_capacity, countries_reached, ports_to_modify):

        shoreline_path = '/Users/loucas/Documents/ORG/github/marine-planning/data/vectors/international/simple_shoreline.geojson'
        north_sea_path = '/Users/loucas/Documents/ORG/github/marine-planning/data/vectors/international/aoi.geojson'

        north_sea = gpd.read_file(north_sea_path)
        shoreline = gpd.read_file(shoreline_path)
        
        #clip shoreline to North Sea buffered
        north_sea = north_sea['geometry']
        shoreline = gpd.clip(shoreline, north_sea)

        north_sea_offshore_wind = {
            'FR': {2020: 0.0, 2030: 2.1, 2050: 17.0},
            'BE': {2020: 2.3, 2030: 6, 2050: 8.0},
            'NL': {2020: 2.5, 2030: 21.0, 2050: 72.0},
            'DE': {2020: 7.7, 2030: 26.4, 2050: 66.0},
            'DK': {2020: 2.3, 2030: 5.3, 2050: 35.0},
            'NO': {2020: 0.0, 2030: 3, 2050: 30.0},
            'UK': {2020: 11.0, 2030: 50.0, 2050: 80.0},
        }

        countries = ['FR', 'BE', 'NL', 'DE', 'DK', 'NO', 'UK']

        # Remove countries reached
        for country in countries_reached:
            if country in countries:
                countries.remove(country)

        substations = {country: 0 for country in countries}

        # Count existing substations for each country
        for locations in used_north_sea_ports.values():
            for location, info in locations.items():
                if info['designation'] == 'substation':
                    country = info['country']
                    if country in substations:
                        substations[country] += 1
   
        

        point_gdfs = []

        def add_points_along_line(gdf, num_points):
            def interpolate_points(geom, num):
                distances = [float(i) / (num + 1) for i in range(1, num + 1)]
                return [geom.interpolate(d, normalized=True) for d in distances]
            
            new_points = gdf.geometry.apply(lambda geom: interpolate_points(geom, num_points))
            
            new_gdf = gdf.loc[new_points.index.repeat(num_points)].reset_index(drop=True)
            new_gdf['geometry'] = [point for points in new_points for point in points]
            
            new_gdf['point_type'] = 'interpolated'
            
            return new_gdf

        # Change iso_3166_1_alpha_2_codes 'GB' to 'UK'
        shoreline['iso_3166_1_alpha_2_codes'] = shoreline['iso_3166_1_alpha_2_codes'].replace('GB', 'UK')

        for country in countries:
            total_capacity = north_sea_offshore_wind[country][2050]
            required_connects = math.ceil(total_capacity / connector_capacity)
            
            num_connects = substations[country]
            print(f'{country}: {num_connects} existing substations, {required_connects} required connects')

            new_connects = max(0, required_connects - num_connects)

            if new_connects > 0:
                country_shoreline = shoreline[shoreline['iso_3166_1_alpha_2_codes'] == country]
                points = add_points_along_line(country_shoreline, new_connects)
                point_gdfs.append(points)

        if point_gdfs:
            new_points = gpd.GeoDataFrame(pd.concat(point_gdfs, ignore_index=True))
            new_points = new_points.to_crs('EPSG:4326')

            # Process each row in the new data
            for _, row in new_points.iterrows():
                country = row['iso_3166_1_alpha_2_codes']
                lat = row['geometry'].y
                lon = row['geometry'].x
                
                # Create a new substation entry
                new_substation = {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "designation": "substation",
                    "country": country
                }
                
                # Find the next available substation number for this country
                substation_numbers = [int(key.split('_')[-1]) for key in used_north_sea_ports[country].keys() if key.startswith(f"{country.lower()}_") and key.split('_')[-1].isdigit()]
                next_number = max(substation_numbers or [0]) + 1
                
                # Add the new substation
                used_north_sea_ports[country][f"{country.lower()}_{next_number}"] = new_substation

        return used_north_sea_ports
    

    def create_distance_raster(self, objects_gdf: gpd.GeoDataFrame, template_raster_path: str, output_raster_path: str, country_column: str = 'country', country_code_map: Dict[str, int] = {
            'NO': 4, 'DK': 7, 'DE': 6, 'NL': 9, 'BE': 5, 'FR': 1, 'UK': 2
        }) -> None:
        
        with rasterio.open(template_raster_path) as src:
            template_raster = src.read(1)
            transform = src.transform
            crs = src.crs

        country_kdtrees = {
            country_code_map[country]: cKDTree(np.array([(geom.x, geom.y) for geom in group.geometry]))
            for country, group in objects_gdf.groupby(country_column)
            if country in country_code_map
        }

        distance_raster = np.full_like(template_raster, np.finfo(np.float32).max, dtype=np.float32)

        rows, cols = template_raster.shape
        for row in range(rows):
            for col in range(cols):
                country_code = template_raster[row, col]
                if country_code in country_kdtrees:
                    x, y = rasterio.transform.xy(transform, row, col, offset='center')
                    distance, _ = country_kdtrees[country_code].query((x, y))
                    distance_raster[row, col] = distance

        with rasterio.open(
            output_raster_path, 'w',
            driver='GTiff', height=rows, width=cols, count=1,
            dtype=np.float32, crs=crs, transform=transform,
        ) as dst:
            dst.write(distance_raster, 1)


    def _calc_nearest_object(self, used_north_sea_ports: Dict, hubs: Dict, add_hubs: bool, object_type: str, template_raster_path: str, output_folder: str) -> None:
        output_path = os.path.join(output_folder, f'distance_to_{object_type}.tif')

        if object_type == 'substation' and not self.iterative:
            land_objects = [
                {
                    'name': name,
                    'designation': info['designation'],
                    'country': info.get('country', country),
                    'geometry': Point(info['longitude'], info['latitude'])
                }
                for country, locations in used_north_sea_ports.items()
                for name, info in locations.items()
            ]

            land_object_gdf = gpd.GeoDataFrame(land_objects, crs="EPSG:4326").to_crs("EPSG:3035")
            land_object_gdf = land_object_gdf[land_object_gdf['designation'] == 'substation']

            self.create_distance_raster(
                objects_gdf=land_object_gdf,
                template_raster_path=template_raster_path,
                output_raster_path=output_path
            )
        else:
            objects = self._get_objects(used_north_sea_ports, object_type, add_hubs, hubs)
            self._create_distance_raster_generic(objects, template_raster_path, output_path, object_type)


    def _get_objects(self, used_north_sea_ports: Dict, object_type: str, add_hubs: bool, hubs: Dict) -> List[Tuple[float, float]]:
        objects = [
            self.transformer.transform(info['longitude'], info['latitude'])
            for country, locations in used_north_sea_ports.items()
            for name, info in locations.items()
            if info['designation'] == object_type or (info['designation'] == 'both' and object_type in ['ins', 'opr'])
        ]

        if add_hubs and object_type == 'substation':
            objects.extend([
                self.transformer.transform(info['longitude'], info['latitude'])
                for info in hubs.values()
            ])

        return objects


    def _create_distance_raster_generic(self, objects: List[Tuple[float, float]], template_raster_path: str, output_path: str, object_type: str) -> None:
        tree = cKDTree(objects)

        with rasterio.open(template_raster_path) as src:
            template_raster = src.read(1)
            transform = src.transform
            crs = src.crs

        rows, cols = template_raster.shape
        xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
        x, y = rasterio.transform.xy(transform, ys, xs)
        geo_coords = np.stack((x, y), axis=-1)

        distances, _ = tree.query(geo_coords.reshape(-1, 2))
        distance_raster = distances.reshape(rows, cols)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff', height=rows, width=cols, count=1,
            dtype=np.float32, crs=crs, transform=transform,
        ) as dst:
            dst.write(distance_raster.astype(np.float32), 1)
            dst.set_band_description(1, f"Distance to nearest {object_type}")


    def initiate_ports(self, hubs, add_hubs, country, update_ports=False, connector_capacity=3, countries_reached=[], iterative=False, ports_to_modify=[], i=0, just_boats=False):

        self.iterative = iterative

        if country == 'all':
            used_north_sea_ports = copy.deepcopy(self.original_data)
            
            if countries_reached:
                for reached_country in countries_reached:
                    if reached_country in used_north_sea_ports:
                        used_north_sea_ports[reached_country] = {
                            k: v for k, v in used_north_sea_ports[reached_country].items()
                            if v['designation'] != 'substation'
                        }
        else:
            used_north_sea_ports = {country: self.original_data[country]}

        if update_ports:
            used_north_sea_ports = self._update_north_sea_ports(used_north_sea_ports, connector_capacity, countries_reached, ports_to_modify)
        else:
            for locations in used_north_sea_ports.values():
                for location, info in locations.items():
                    if location in ports_to_modify:
                        info['num_boats'] += i

        self._set_pathing()

        for object_type in ['substation', 'ins', 'opr']:
            self._calc_nearest_object(used_north_sea_ports, hubs, add_hubs, object_type, self.template_r, self.temp_folder)

        self._get_closest_features(used_north_sea_ports, hubs, add_hubs, self.template_r, self.temp_folder, 'substation')
        self._get_closest_features(used_north_sea_ports, hubs, add_hubs, self.template_r, self.temp_folder, 'install_port')

        if not just_boats:

            tif_files = [f for f in os.listdir(self.temp_folder) if f.endswith('.tif')]

            for tif_file in tif_files:
                if tif_file in ['distance_to_substation.tif', 'distance_to_ins.tif', 'distance_to_opr.tif', 'closest_gridconnects.tif', 'closest_install_port.tif']:
                    tif_path = os.path.join(self.temp_folder, tif_file)
                    
                    with rasterio.open(tif_path) as src:
                        array = src.read(1)
                        array = array.astype(float)
                        array = equal_dimensions(array)

                    self.results[tif_file.split('.')[0]] = array.flatten()

            self.updated_ports = used_north_sea_ports