# IMport the necessary libraries
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba, LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.colors as mcolors

import pickle
import numpy as np
import rasterio
from shapely.geometry import box
from scipy.stats import linregress
from scipy.ndimage import label

from shapelysmooth import taubin_smooth


import math
4
import pandas as pd 


class PlotParallelRuns:
    """
    A class for plotting parallel runs.

    Parameters:
    - name (str): The name of the parallel run.
    - directory (str): The directory where the parallel run data is stored.
    - scale (str): The scale of the data.
    - result_folder (str): The folder where the results are stored. Default is 'test'.

    Attributes:
    - name (str): The name of the parallel run.
    - directory (str): The directory where the parallel run data is stored.
    - scale (str): The scale of the data.
    - run_folder (str): The folder where the results are stored.
    - out_jsons (str): The path to the JSONS folder.
    - template_r (str): The path to the template raster file.
    - metric_df (pd.DataFrame): The metric dataframe.
    - prob_dict (dict): The probability dictionary.

    Methods:
    - _set_pathing(): Set the pathing for the parallel run.
    - _create_metric_df(): Create the metric dataframe.
    - _create_probability_dict(): Create the probability dictionary.
    - _add_probable_tech(): Add the most probable technology to the probability dictionary.
    - _assign_to_bin(value, bins): Assign a value to a bin.
    - _calc_conflict(row, col): Calculate the conflict value for a given row and column.
    - _convert_prob_arrays(): Convert the probability arrays to polygons.
    """

    def __init__(self, name, directory, scale, result_folder='test'):
        self.name = name
        self.scale = scale
        self.directory = directory

        if result_folder == 'test':
            self.run_folder = os.path.join(self.directory, 'test_results', name) 
        elif result_folder == 'real':
            self.run_folder = os.path.join(self.directory, 'results', name) 


    def _set_pathing(self):
        """
        Set the pathing for the parallel run.
        """
        self.out_jsons = os.path.join(self.run_folder, 'JSONS')
        os.makedirs(self.out_jsons, exist_ok=True)

        for file in os.listdir(os.path.join(self.directory, 'data', 'rasters', self.scale)):     
            if file.endswith('.tif'):
                self.template_r = os.path.join(self.directory, 'temp', file)
                break

        self.original_metrics_path = os.path.join(self.directory, 'data', 'pickles', self.scale, 'calculated_metrics.pkl')


    def _create_metric_df(self):
        """
        Create the metric dataframe.
        """
        self.metric_df = None
        folders = sorted([folder for folder in os.listdir(os.path.join(self.run_folder, 'runs'))])

        for folder in folders:
            file_path = os.path.join(self.run_folder, 'runs', folder, f'{folder}.pkl') 

            try:
                with open(file_path, 'rb') as file:
                    run_dict = pickle.load(file)

                if self.metric_df is None:
                    self.metric_df = pd.DataFrame([{'folder': folder, **run_dict[folder]['criteria'], **run_dict[folder]['cell_amounts']}])
                else:
                    new_data = pd.DataFrame([{'folder': folder, **run_dict[folder]['criteria'], **run_dict[folder]['cell_amounts']}])
                    self.metric_df = pd.concat([self.metric_df, new_data], ignore_index=True)
            except:
                print(f'Error with {folder}')

    def _add_all_wind_cells(self):

        prob_dict_copy = self.prob_dict.copy()

        for run, arrays in prob_dict_copy.items():
            first_array = list(self.prob_dict.values())[0]
            first_array_dimensions = np.shape(next(iter(first_array.values())))
            cell_use = np.zeros(first_array_dimensions, dtype=float)

            for tech, array in arrays.items():
                if tech == 'monopile' or tech == 'jacket':
                    # add 1 to the cell_use array where the tech is used
                    cell_use += array
            self.prob_dict[run]['wind'] = cell_use


    def _create_probability_dict(self):
        """
        Create the probability dictionary.
        """
        folders = sorted([folder for folder in os.listdir(os.path.join(self.run_folder, 'runs'))])
        self.prob_dict = {}

        for folder in folders:
            file_path = os.path.join(self.run_folder, 'runs', folder, f'{folder}.pkl') 
            try:
                with open(file_path, 'rb') as file:
                    run_dict = pickle.load(file)
            except:
                print(f'Error with {folder}')
                continue
            r = run_dict[folder]['result_names']
            t = r.flatten()
            u = set(t)
            u = list(u)

            run_results = {}

            for tech in u:
                if tech != 'empty':
                    new_array = np.where(r == tech, 1, 0)
                    run_results[tech] = new_array
                
            self.prob_dict[folder] = run_results

        # Add the wind cells (combine monopile and jacket) to the probability dictionary
        self._add_all_wind_cells()

        # Below calculates the cell proportions
        first_array = list(self.prob_dict.values())[0]
        first_array_dimensions = np.shape(next(iter(first_array.values())))
        cell_use = np.zeros(first_array_dimensions, dtype=float)

        possible_tech = ['monopile', 'mussel', 'solar', 'jacket', 'wind']
        tech_use = {tech: np.zeros_like(cell_use, dtype=float) for tech in possible_tech}

        num_runs = len(self.prob_dict)

        for run_name, tech_dict in self.prob_dict.items():
            used_cells = np.zeros_like(cell_use, dtype=float)
            
            for tech_name, array in tech_dict.items():
                if tech_name != '0':
                    used_cells += array
                    tech_use[tech_name] += array
                
            cell_use += np.where(used_cells > 0, 1, 0)

        # Calculate proportions
        cell_use /= num_runs
        for tech_name in tech_use:

            if tech_name != '0':

                tech_use[tech_name] /= num_runs
                tech_use[tech_name] = np.where(tech_use[tech_name] == 0, np.nan, tech_use[tech_name])

        self.prob_dict['cell_use'] = np.where(cell_use == 0, np.nan, cell_use)
        self.prob_dict['tech_use'] = tech_use


    def _add_probable_tech(self):
        # Initialize dictionary to store the tech with the highest proportion for each cell
        self.prob_dict['most_prob_tech'] = {}

        # Iterate over each cell in the arrays
        for i in range(self.prob_dict['cell_use'].shape[0]):
            for j in range(self.prob_dict['cell_use'].shape[1]):
                # Initialize variables to track max tech and proportion
                max_tech = None
                max_proportion = 0.0
                
                # Iterate over tech_use_dict to find the max tech and its proportion for this cell
                for tech_name, proportion_array in self.prob_dict['tech_use'].items():
                    proportion = proportion_array[i, j]
                    if proportion > max_proportion:
                        max_tech = tech_name
                        max_proportion = proportion
                
                # Store the max tech and its proportion for this cell as a tuple in max_tech_dict
                self.prob_dict['most_prob_tech'][(i, j)] = (max_tech, max_proportion)


    def _assign_to_bin(self, value, bins):
        for key, bin_value in bins.items():
            # Extract the lower and upper bounds of the range
            lower, upper = map(int, key.split('-'))
            
            # Convert nan to 0
            if math.isnan(value):
                return 0

            if value > 100:
                return 1

            # Check if the value falls within the range
            if lower <= value < upper:
                return bin_value
        
        # If the value doesn't fall within any range, return None or raise an exception
        return 0


    def _calc_conflict(self, row, col):
        """
        Calculate the conflict value for a given row and column.

        Parameters:
        - row (int): The row index.
        - col (int): The column index.

        Returns:
        - float: The conflict value normalized by the most probable technology value.


        *** COME BACK TO THIS FUNCTION TO MAKE IT MORE ROBUST WITH INCLUSION OF WIND ***

        """
        
        if self.prob_dict['most_prob_tech'][(row, col)][0] != None:
            mu = self.prob_dict['most_prob_tech'][(row, col)][0]
            mu_val = self.prob_dict['most_prob_tech'][(row, col)][1]
            cu = self.prob_dict['cell_use'][row, col]

            t = 0
            counter = 1 if mu_val != 0 else 0
            for tech in self.prob_dict['tech_use']:
                if tech != mu and tech != 'wind':
                    val = self.prob_dict['tech_use'][tech][row, col]

                    t += val if not math.isnan(val) else 0
                    counter += 1 if not math.isnan(val) else 0
                    
            avg_t = t / len(self.prob_dict['tech_use'])
            conflict = mu_val - avg_t

            if counter == 1 or (counter == 2 and cu < 0.6):
                if cu < 0.6:
                    colour = 'grey' # Some interest by one or two techs
                else:
                    colour = 'white' # High interest by one tech
            elif counter >= 2 and cu >= 0.6:
                if cu > 0.9:
                    colour = 'red' # Highly desired by multiple techs
                else:
                    colour = 'pink' # Interest by multiple techs
            else:
                colour = 'pink' # No interest by any tech
            #return conflict / mu_val
            return colour


    def _convert_prob_arrays(self):
            
            with rasterio.open(self.template_r) as template:
                transform = template.transform
                crs = template.crs

            n = self.prob_dict['cell_use'].shape

            # Create the polygons for each cell use no matter the tech
            polygons_cu = []
            polygons_mpt = []

            # Create alpha bins
            bins = {
                '0-30': 0.2,
                '30-50': 0.4,
                '50-70': 0.5,
                '70-90': 0.7,
                '90-99': 0.9,
                '100-101': 1,
            }

            bins = {
                '0-25': 0.2,
                '25-50': 0.3,
                '50-75': 0.4,
                '75-95': 0.5,
                '95-101': 1,
            }
            
            for row in range(n[0]):
                for col in range(n[0]):
                    
                    # Get the extent from the original rasters --- Something is janky and the scaled polygons are not correct
                    minx, maxy = rasterio.transform.xy(transform, row-0.5, col-0.5)
                    maxx, miny = rasterio.transform.xy(transform, row+0.5, col+0.5)

                    value_cell_use = self.prob_dict['cell_use'][row, col]
                    alpha_val = self._assign_to_bin(value_cell_use * 100, bins)

                    most_prob_tech = self.prob_dict['most_prob_tech'][row, col][0]
                    value_most_prob_tech = self.prob_dict['most_prob_tech'][row, col][1]


                    if not np.isnan(value_cell_use):
                        conflict = self._calc_conflict(row, col)
                        
                        polygon = box(minx, miny, maxx, maxy, ccw=True)
                        polygons_cu.append((value_cell_use, alpha_val, most_prob_tech, value_most_prob_tech, conflict, polygon))
                        

            gdf = gpd.GeoDataFrame(polygons_cu, columns=['value', 'alpha_val', 'most_prob_tech', 'val_most_prob', 'conflict', 'geometry'], crs=crs)

            gdf.reset_index() # Resets the index so that the values are a column

            gdf.to_file(os.path.join(self.out_jsons, 'cell_use.geojson'), driver='GeoJSON')


            # Loop through the individual tech dicts to make their probabilities
            for tech, array in self.prob_dict['tech_use'].items():
                polygons = []
                for row in range(n[0]):
                    for col in range(n[0]):
                        
                        # Get the extent from the original rasters --- Something is janky and the scaled polygons are not correct
                        minx, maxy = rasterio.transform.xy(transform, row-0.5, col-0.5)
                        maxx, miny = rasterio.transform.xy(transform, row+0.5, col+0.5)

                        value = array[row, col]
                        alpha_val = self._assign_to_bin(value * 100, bins)
                        
                        if not np.isnan(value):
                            polygon = box(minx, miny, maxx, maxy, ccw=True)
                            polygons.append((value, alpha_val, polygon))

                if len(polygons) != 0:
                    gdf = gpd.GeoDataFrame(polygons, columns=['value', 'alpha_val', 'geometry'], crs=crs)
                    gdf.reset_index() # Resets the index so that the values are a column
                    gdf.to_file(os.path.join(self.out_jsons, f'{tech}_prob.geojson'), driver='GeoJSON')

                    self.g = gdf


    def prepare_statistics(self):
        self._create_metric_df()
        self._create_probability_dict()
        self._set_pathing()
        self._add_probable_tech()
        self._convert_prob_arrays()


# Visual plotting of layouts -- aide functions
            
    def _utm_formatter(self, x, pos):
        ''' 
        NOTE: I haven't been able to implement this without using this function.

        This converts the coordinates beside along the map into km rather than m
        '''
        return f"{int(x/1000)}"


    def _load_files(self, mapping):
        """
        Load files based on the provided mapping.

        Args:
            mapping (dict): A dictionary containing mapping information.

        Returns:
            None
        """
        # Load background files
        self.shipping_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'Shipping.geojson')) if mapping['msp']['shipping'] else None
        self.military_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'military.geojson')) if mapping['msp']['military'] else None
        self.sand_extraction_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'sand_extraction.geojson')) if mapping['msp']['sand_extraction'] else None
        self.nature_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'nature_reserves.geojson')) if mapping['msp']['nature_reserves'] else None
        self.legacy_farms_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'legacy_farms.geojson')) if mapping['msp']['legacy_farms'] else None

        if mapping['msp']['energy_zones']:
            if mapping['msp']['energy_zones_type'] == 'whole_zone':
                self.energy_zone_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'energy_zone.geojson'))
            elif mapping['msp']['energy_zones_type'] == 'split_zone':
                self.pe1 = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'pe_split/kavel_1.geojson'))
                self.pe2 = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'pe_split/kavel_2.geojson'))
                self.pe3 = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'pe_split/kavel_3.geojson'))
                self.gravel = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'gravelbed.geojson'))

        self.aoi_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'aoi.geojson'))
        self.cities_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'cities.geojson'))
        self.eez_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'eez.geojson'))
        self._eez_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'eez.geojson'))
        self.shoreline_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'Shoreline.geojson'))
        self.boundaries_gdf = gpd.read_file(os.path.join(self.directory, 'data', 'vectors', self.scale, 'boundaries.geojson'))


    def _plot_files(self, mapping):
        """
        Plot various geographic files based on the provided mapping.

        Parameters:
        - mapping (dict): A dictionary containing mapping information for different file types.

        Returns:
        - None
        """

        # MSP specific files
        self.military_gdf.plot(ax=self.ax, color='#6B8EAD', alpha=0.5, hatch='x', edgecolor='black', linewidth=1) if mapping['msp']['military'] else None
        self.sand_extraction_gdf.plot(ax=self.ax, color='#F5F5DC', hatch='//', alpha=0.5, edgecolor='black', linewidth=1) if mapping['msp']['sand_extraction'] else None
        self.nature_gdf.plot(ax=self.ax, color='lawngreen', hatch='/', alpha=0.5, edgecolor='lawngreen', linewidth=2) if mapping['msp']['nature_reserves'] else None

        self.shipping_gdf.plot(ax=self.ax, color='#333333', alpha=0.8) if mapping['msp']['shipping'] else None

        if mapping['msp']['energy_zones']:
            if mapping['msp']['energy_zones_type'] == 'whole_zone':
                self.energy_zone_gdf.plot(ax=self.ax, facecolor='none', edgecolor='red', linewidth=1.3, alpha=1)
            elif mapping['msp']['energy_zones_type'] == 'split_zone':
                self.pe1.plot(ax=self.ax, facecolor='none', edgecolor='red', linewidth=1.3, alpha=1)
                self.pe2.plot(ax=self.ax, facecolor='none', edgecolor='red', linewidth=1.3, alpha=1)
                self.pe3.plot(ax=self.ax, facecolor='none', edgecolor='red', linewidth=1.3, alpha=1)
                self.gravel.plot(ax=self.ax, facecolor='#F5F5DC', hatch='//', alpha=0.7)

        # Iterate through each of the current windfarm shapefiles and plot
        if mapping['msp']['wind_farms']:
            shapefiles = [os.path.join(self.wind_farms_path, file) for file in os.listdir(self.wind_farms_path) if file.endswith('.geojson')]
            for shapefile in shapefiles:
                self.temp_gdf = gpd.read_file(shapefile)
                self.temp_gdf.plot(ax=self.ax, hatch='|', color='chocolate', alpha=0.5)

        self.legacy_farms_gdf.plot(ax=self.ax, hatch='|', color='chocolate', alpha=0.5) if mapping['msp']['legacy_farms'] else None

        # Geographic things
        self.shoreline_gdf.plot(ax=self.ax, color='#696969', alpha=0.8)
        self.boundaries_gdf.plot(ax=self.ax, color='black', linewidth=1.5, linestyle='--')

        # self.cities_gdf.plot(ax=self.ax, marker='s', color='white', markersize=25, label='Points')
        # for x, y, label in zip(self.cities_gdf.geometry.x, self.cities_gdf.geometry.y, self.cities_gdf['name']):
        #     self.ax.text(x + 5000, y - 1000, label, fontsize=14, ha='left', color='white', fontfamily='monospace')

        self._eez_gdf.plot(ax=self.ax, color='#1f1f1f', linewidth=2) # black line behind the eez dashed line to give it some more pop
        self.eez_gdf.plot(ax=self.ax, color='white', linewidth=1, linestyle='--')

        plt.subplots_adjust(left=0, right=0.75, top=0.99, bottom=0.03)


    def _set_backgrounds(self):
        """
        Set the background styles for the plot.

        This method sets the color and style of the plot's spines, labels, ticks,
        and sets the x and y axis formatters. It also sets the x and y limits of
        the plot based on the total bounds of the area of interest (aoi_gdf).

        Parameters:
        None

        Returns:
        None
        """
        self.ax.spines['bottom'].set_color('white')  
        self.ax.spines['top'].set_color('white')  
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')

        # Set the label colours
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')

        # Set the tick colours
        self.ax.tick_params(axis='x', colors='white', labelfontfamily='monospace', labelsize=12)
        self.ax.tick_params(axis='y', colors='white', labelfontfamily='monospace', labelsize=12)

        self.ax.xaxis.set_major_formatter(FuncFormatter(self._utm_formatter))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self._utm_formatter))

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Set zoom distance
        xlim_min, ylim_min, xlim_max, ylim_max = self.aoi_gdf.total_bounds
        self.ax.set_xlim(xlim_min, xlim_max)
        self.ax.set_ylim(ylim_min, ylim_max)
        self.ax.set_title(self.name)


    def _add_prob_legend(self, map_type=''):

        """
        Add a legend to the plot based on the map type.

        Parameters:
        - map_type (str): The type of map to display the legend for. Can be 'all' or any other value.

        Returns:
        - None
        """

        # Create the legend handles
        if map_type == 'all' or map_type == 'single_tech':
            legend_handles = [
                Patch(color='green', alpha=1, label='100% use'),
                Patch(color='white', alpha=0.9, label='90 - 99% use'),
                Patch(color='white', alpha=0.7, label='70 - 90% use'),
                Patch(color='white', alpha=0.5, label='50 - 70% use'),
                Patch(color='white', alpha=0.3, label='30 - 50% use'),
                Patch(color='white', alpha=0.2, label='0 - 30% use'),    
            ]  

        else:
            legend_handles = [
                Patch(color='red', label='Highly desired by multiple techs'),
                Patch(color='pink', label='Interest by multiple techs'),
                Patch(color='white', label='Highly desired by one tech'),
                Patch(color='grey', label='Some interest by one or two techs'),
                Patch(color='black', label='No interest by any tech'),
            ]

        # Display the legend
        if map_type == 'all':
            self.ax.legend(
                handles=legend_handles,
                loc="center",
                prop={"family": "monospace", "size": 22},
                labelcolor="white",
                facecolor="#1f1f1f",
                frameon=False,
            )

        else:
            if self.scale == "belgium":
                self.ax.legend(
                    handles=legend_handles,
                    loc="upper right",
                    bbox_to_anchor=(0.998, 0.998),
                    prop={"family": "monospace", "size": 14},
                    labelcolor="white",
                    facecolor="#1f1f1f",
                    frameon=False,
                )

            elif self.scale == "international":
                self.ax.legend(
                    handles=legend_handles,
                    loc="upper right",
                    bbox_to_anchor=(0.3, 0.998),
                    prop={"family": "monospace", "size": 14},
                    labelcolor="white",
                    facecolor="#1f1f1f",
                    frameon=False,
                )


    def _get_color(self, alpha):
        if alpha == 1:
            return 'green'
        else:
            return 'white'


# Visual plotting of layouts -- main functions

    def plot_all_runs(self, mapping):

        option_folders = sorted([folder for folder in os.listdir(os.path.join(self.run_folder, 'runs'))])

        num_files = len(option_folders)
        num_rows = (num_files - 1) // 5 + 1

        fig, axes = plt.subplots(num_rows, 6, figsize=(20, num_rows * 4))


        self._load_files(mapping)

        for i, option_folder in enumerate(option_folders):
            shp_folder_path = os.path.join(self.run_folder, 'runs', option_folder, 'SHP')
            
            if os.path.exists(shp_folder_path):

                for file in os.listdir(shp_folder_path):
                    if file.endswith('.geojson'):
                        geojson_file = os.path.join(shp_folder_path, file)
                        
                gdf = gpd.read_file(geojson_file)
                
                self.ax = axes[i // 5, i % 5]


                unique_values = gdf['value'].unique()
                custom_cmap = ListedColormap([mapping['colours'][value] for value in unique_values])
                fig.patch.set_facecolor('#333333') # Set the background color of the entire figure
                self.ax.set_facecolor('#1f1f1f') # Set the background color of the plot area (the colour that the ocean ends up)

                self._set_backgrounds()

                gdf.plot(ax=self.ax, column='value', cmap=custom_cmap)
                self._plot_files(mapping)
                
                self.ax.set_title(option_folder)

            if (i + 1) % 5 == 0:
                axes[i // 5, -1].axis('off')

        for i in range(num_files, num_rows * 5):
            fig.delaxes(axes.flatten()[i])

        legend_patches = [Rectangle((0, 0), 1, 1, color=color, alpha=1, label=label.capitalize()) for label, color in mapping['colours'].items()]
        all_legend_handles = mapping['legend'] + legend_patches

        plt.legend(handles=all_legend_handles, loc='upper right', bbox_to_anchor=(1, 2.1), ncol=1, prop={'family': 'monospace', 'size': 11}, labelcolor='white', facecolor='#1f1f1f', frameon=False)

        plt.show()


    def plot_cell_probability(self, mapping):
        path = os.path.join(self.out_jsons, 'cell_use.geojson')
        cell_use_gdf = gpd.read_file(path)

        self._load_files(mapping)        

        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.fig.patch.set_facecolor('#333333') # Set the background color of the entire figure
        self.ax.set_facecolor('#1f1f1f') # Set the background color of the plot area (the colour that the ocean ends up)

        self._set_backgrounds()

        self._plot_files(mapping)

        # Define the colors for the colormap
        colors = [(1, 0, 0), (1, 1, 1)]  # Red to white

        # Create the custom colormap
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

        # Plot using the custom colormap, assuming 'conflict' is the column with values ranging from 0 to 1
        #cell_use_gdf.plot(ax=self.ax, cmap=custom_cmap, column='conflict', alpha=cell_use_gdf['alpha_val'], vmin=0, vmax=1)

        cell_use_gdf.plot(ax=self.ax, color=cell_use_gdf['conflict'])

        self._add_prob_legend(map_type='conflict')
        # Add a colorbar to show the mapping of values to colors
        # plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap), ax=self.ax, shrink=0.5)
        plt.show()


    def plot_all_tech(self, mapping, to_plot=['wind', 'mussel']):

        jsons = os.listdir(self.out_jsons)
        jsons.remove('cell_use.geojson')

        to_remove = []

        for json in jsons:
            tech = json.split('_')[0]
            if tech not in to_plot:
                to_remove.append(json)

        for json in to_remove:
            jsons.remove(json)
        
        self._load_files(mapping)        

        geojson_files = [file for file in jsons if file.endswith('.geojson')]

        num_plots = len(geojson_files) + 1  # Add 1 for legend plot
        num_cols = len(jsons) + 1  # Add 1 for legend plot
        num_rows = -(-num_plots // num_cols)

        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 6*num_rows))

        # Flatten axs if there's only one row or column
        if num_rows == 1 or num_cols == 1:
            axs = axs.flatten()

        for i, f in enumerate(jsons):

            if i == 0:  # For the first plot, set axs[num_cols-1] for legend plot
                self.ax = axs[i]
            else:
                self.ax = axs[i]

            fig.patch.set_facecolor('#333333') # Set the background color of the entire figure
            self.ax.set_facecolor('#1f1f1f') # Set the background color of the plot area (the colour that the ocean ends up)

            self._set_backgrounds()
            

            if i < len(jsons):  # Only for actual data plots, not for the legend
                tech = f.split('_')[0]

                gdf = gpd.read_file(os.path.join(self.out_jsons, f))
                
                gdf.plot(ax=self.ax, color=gdf['alpha_val'].apply(self._get_color), alpha=gdf['alpha_val'])
                #gdf.plot(ax=self.ax, color='white', alpha=gdf['alpha_val'])

                self.ax.set_title(tech.capitalize(), color='white', size=22)  # Capitalize the tech name
                self._plot_files(mapping)

        # Add legend plot
        self.ax = axs[num_cols-1]
        self.ax.axis('off')  # Turn off axis for legend plot
        self._add_prob_legend(map_type='all')
        plt.tight_layout()  # Adjust layout
        plt.show()


    def plot_single_tech(self, mapping, tech):

        jsons = os.listdir(self.out_jsons)
        jsons.remove('cell_use.geojson')

        # Check that the tech exists before any plotting happens
        t_list = []
        for f in jsons:
            t = f.split('_')[0]
            t_list.append(t)
        if tech not in t_list:
            print(f'The tech entered ({tech}) does not exist within this run')
            return    

        self._load_files(mapping)        

        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.fig.patch.set_facecolor('#333333') # Set the background color of the entire figure
        self.ax.set_facecolor('#1f1f1f') # Set the background color of the plot area (the colour that the ocean ends up)

        self._set_backgrounds()
        self._plot_files(mapping)

        for f in jsons:
            t = f.split('_')[0]

            if t == tech:
                gdf = gpd.read_file(os.path.join(self.out_jsons, f))
                gdf.plot(ax=self.ax, color=gdf['alpha_val'].apply(self._get_color), alpha=gdf['alpha_val'])
        self._add_prob_legend(map_type='single_tech')

        # Add a title to the plot
        self.ax.set_title(tech.capitalize(), color='white', size=22)  # Capitalize the tech name

        
# Statistical plotting

    def cell_histogram(self):
        columns_to_plot = ['solar', 'mussel', 'monopile', 'jacket']

        min_value = self.metric_df[columns_to_plot].min().min()
        max_value = self.metric_df[columns_to_plot].max().max()

        tick_positions = np.arange(min_value, max_value + 250, 250)

        # Create a single histogram with all tech
        plt.figure(figsize=(10, 6))
        for column in columns_to_plot:
            plt.hist(self.metric_df[column], bins=20, alpha=0.5, label=column)

        plt.xticks(tick_positions)
        plt.xlabel('Cell Amount')
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.legend()
        plt.show()


    def metric_histogram(self, metric):

        # min_value = self.metric_df[metric].min().min()
        # max_value = self.metric_df[metric].max().max()

        #tick_positions = np.arange(min_value, max_value + 250, 250)

        # Create a single histogram with all tech
        plt.figure(figsize=(10, 6))
        plt.hist(self.metric_df[metric], bins=20, alpha=0.5, label=metric)

        #plt.xticks(tick_positions)
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.legend()
        plt.show()


    def bar_graph(self, column):
        # Plotting
        plt.figure(figsize=(10, 6))

        # Extracting data
        folders = self.metric_df['folder']
        capex_values = self.metric_df[column]

        # Plotting bars
        plt.bar(folders, capex_values, color='skyblue')

        plt.xticks(rotation=90)
        plt.xlabel('Folder')
        plt.ylabel(column)

        plt.show()


    def scatterplot(self, x, y):

        plt.scatter(self.metric_df[x], self.metric_df[y])
        slope, intercept, r_value, p_value, std_err = linregress(self.metric_df[x], self.metric_df[y])
        x_vals = np.array(self.metric_df[x])
        y_vals = slope * x_vals + intercept

        # Plot the regression line
        plt.plot(x_vals, y_vals, color='red')

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'{x} vs {y}')
        plt.grid(True)
        plt.show()

# Random plotting
        
    def _format_func(self, value, tick_number):
        return f'{value * 100:.0f}%'  # Multiply by 100 and add '%' sign


    def simple_show_celluse(self, filter=1):

        # if filter is above zero, remove all values below the filter

        if filter > 0:
            cell_use = np.where(self.prob_dict['cell_use'] < filter, np.nan, self.prob_dict['cell_use'])
        else:
            cell_use = self.prob_dict['cell_use']



        sequence = np.arange(0.0, 1.01, 0.01)

        colors = [(0, 0, 0, val) for val in sequence]


        cmap = ListedColormap(colors)


        plt.imshow(cell_use, cmap=cmap, vmin=0, vmax=1)  # You can choose a different colormap if desired

        # Add colorbar for reference
        cbar = plt.colorbar()
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(self._format_func))

        plt.title(f'Cell Use - {self.name}')
        plt.show()


    def simple_show_techuse(self, tech):

        sequence = np.arange(0.0, 1.01, 0.01)

        colors = [(0, 0, 0, val) for val in sequence]

        cmap = ListedColormap(colors)

        plt.imshow(self.prob_dict['tech_use'][tech], cmap='viridis')  # You can choose a different colormap if desired

        # Add colorbar for reference
        cbar = plt.colorbar()
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(self._format_func))

        plt.title(f'tech use - {self.name} - {tech}')
        plt.show()


    def simple_show_prob_tech(self):

        tech_colours = {
            'mussel': 'red',
            'monopile': 'blue',
            'jacket': 'teal',
            'solar': 'orange',
            # 'monopile + solar': 'purple',
            # 'monopile + mussel': 'pink',
            #'hydrogen': 'black'
        }

        rows, cols = self.prob_dict['cell_use'].shape
        rgba_values = np.zeros((rows, cols, 4))

        for i in range(rows):
            for j in range(cols):
                max_tech, proportion = self.prob_dict['most_prob_tech'][(i, j)]
                color = tech_colours.get(max_tech, 'black')  # Get color from tech_colours, default to black

                if max_tech is not None:
                    rgba = to_rgba(color)
                    rgba_values[i, j] = (*rgba[:3], proportion * rgba[3]) # Set colour and opacity
                else:
                    rgba_values[i, j] = (0, 0, 0, 0)  # Set transparent for cells with no max tech

        plt.imshow(rgba_values)
        plt.title(f'Max Tech Use - {self.name}')
        plt.show()

# MARK: Clustering

# Helper functions

    def _add_colours(self):
        tech_colours = {
            'mussel': 'red',
            'monopile': 'blue',
            'jacket': 'teal',
            'solar': 'orange',
            'wind': 'green',
            # 'monopile + solar': 'purple',
            # 'monopile + mussel': 'pink',
            #'hydrogen': 'black'
        }

        categories = ['mussel', 'monopile', 'jacket', 'solar', 'wind']

        # Create a list of colors based on the order of categories
        colors = [tech_colours[tech] for tech in categories]

        # Create a colormap
        self.tech_cmap = mcolors.ListedColormap(colors)

        self.cluster_gdf['colour'] = self.cluster_gdf['tech'].apply(lambda x: tech_colours.get(x, 'black'))


    def _set_resolution(self):
        '''
        sets the resolution of the raster, to be used for calculating the metrics
        '''
        # Get the resolution from the template raster, to adjust the results accordingly
        with rasterio.open(self.template_r) as src:
            transform = src.transform

            self.x_resolution = transform[0]
            self.y_resolution = -transform[4]

            # NOTE this is how to get the real resolution of the raster, belgium has to be hacked because the rasters are wrong
            self.resolution = (self.x_resolution / 1000) * (self.y_resolution / 1000)

        if self.scale == "belgium":
            self.resolution = 0.81  # The current resolution of the belgium data, not what the rasters say


    def _get_unit_amounts(self, km2, tech):

        ud = {
            'wind': 0.5,
            'jacket': 0.5,
            'monopile': 0.5,
            'mussel': 35,
            'solar': 1150
            }
        
        return round(km2 * ud[tech])


    def _load_metrics(self):

        with open(self.original_metrics_path, "rb") as file:
            self.original_metrics = pickle.load(file)

        # Reshape each list to a 2D array
        for tech, metrics in self.original_metrics.items():
            for metric, value in metrics.items():
                self.original_metrics[tech][metric] = np.array(value).reshape(self.prob_dict['cell_use'].shape)


    def _get_metrics(self, row, col, tech):
            ''' 
            
            Returns the metrics for a given row and column in the original metrics array

            '''

            metrics = {}
    
            if tech == 'wind':
                for metric, array in self.original_metrics['monopile'].items():
                    metrics[metric] = array[row, col]                
            else:
                for metric, array in self.original_metrics[tech].items():
                    metrics[metric] = array[row, col]
            
            # check if 'aep' and 'afp' are in the metrics, if not, add an array of zeros
            if 'aep' not in metrics:
                metrics['aep'] = 0
            if 'afp' not in metrics:
                metrics['afp'] = 0
    
            return metrics


# Create clusters

    def _smooth_geometry(self, geometry):
        smoothed_geometry = taubin_smooth(geometry, 0.5, -0.5, 5)
        return smoothed_geometry


    def _clusters_to_gdf(self, simplified):
    
        with rasterio.open(self.template_r) as template:
            transform = template.transform
            crs = template.crs

        # Create an empty gdf that will be used to store the polygons
        self.cluster_gdf = gpd.GeoDataFrame(columns=["tech_val", "tech", "km2", "unit_count", "capex", "value", "energy", "food", "geometry"])


        for tech, _clusters in self.clusters.items():
             
            # Get the shape of the first array in the _clusters
            n = list(_clusters.values())[0]['a'].shape[0]

            for cluster in _clusters.values():
                km2 = round(cluster['count'] * self.resolution, 2)
                unit_count = self._get_unit_amounts(km2, tech)
                # Create the polygons for each tech
                polygons = []
                for row in range(n):
                    for col in range(n):

                        # Get the extent from the original rasters --- Something is janky and the scaled polygons are not correct
                        minx, maxy = rasterio.transform.xy(transform, row - 0.5, col - 0.5)
                        maxx, miny = rasterio.transform.xy(transform, row + 0.5, col + 0.5)

                        tech_value = cluster['a'][row, col]
                        metrics = self._get_metrics(row, col, tech)

                        # Applies the value of each cell to the polygons
                        if tech_value == 1:  # Skip cells designated as empty
                            polygon = box(minx, miny, maxx, maxy, ccw=True)
                            polygons.append((1, tech, km2, unit_count, metrics['cap'], metrics['val'], metrics['aep'], metrics['afp'], polygon))

                gdf = gpd.GeoDataFrame(polygons, columns=["tech_val", "tech", "km2", "unit_count", "capex", "value", "energy", "food", "geometry"], crs=crs)

                aggfunc = {
                    "tech_val": "first",
                    "tech": "first",
                    "km2": "first",
                    "unit_count": "first",
                    "capex": "sum",
                    "value": "sum",
                    "energy": "sum",
                    "food": "sum",
                }


                tech_gdf = gdf.dissolve(by="tech_val", aggfunc=aggfunc)
                #tech_gdf = tech_gdf.reset_index()  # Resets the index so that the values are a column
                    
                tech_gdf['geometry'] = tech_gdf['geometry'].apply(self._smooth_geometry)

                # Concatenate the gdfs
                self.cluster_gdf = pd.concat([self.cluster_gdf, tech_gdf], ignore_index=True)


    def create_clusters(self, threshold=0.95, min_cluster_size=10, simplified=False):

        self.clusters = {}
        self._set_resolution()
        self._load_metrics()

        for tech, use in self.prob_dict['tech_use'].items():

            # create deep copy of the array
            array = use.copy()

            array[array < threshold] = np.nan
            array[array >= threshold] = 1

            # nan values are set to 0
            array = np.nan_to_num(array)

            cluster_array, cluster_count = label(array)

            _clusters = {}

            for cluster_code in range(cluster_count+1):
                if cluster_code == 0:
                    continue
                # print the unique value of the cluster
                unique, counts = np.unique(cluster_array, return_counts=True)
                
                _clusters[f"cluster-{cluster_code}"] = {}
                            
                if counts[cluster_code] > min_cluster_size:
                    _clusters[f"cluster-{cluster_code}"]['a'] = (cluster_array == cluster_code).astype(int)
                    _clusters[f"cluster-{cluster_code}"]['count'] = counts[cluster_code]
                
                if len(_clusters[f"cluster-{cluster_code}"]) == 0:
                    _clusters.pop(f"cluster-{cluster_code}")

            if len(_clusters) > 0:
                self.clusters[tech] = _clusters

        self._clusters_to_gdf(simplified=simplified)
        self._add_colours()

