import pickle
import math
import numpy as np
import pandas as pd
import os
import scipy
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import pulp
import json
import re
import copy
from scipy .ndimage import label, binary_dilation, generate_binary_structure
from scipy.spatial import distance_matrix
from pathlib import Path
from collections import Counter
import time
from pyproj import Transformer
import random

from shapely import simplify
from shapely.geometry import box, MultiPolygon, Point, LineString, Polygon
from shapely.ops import unary_union
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch, Rectangle, PathPatch
from matplotlib.lines import Line2D

from matplotlib.markers import MarkerStyle
import matplotlib.path as mpath



from shapelysmooth import taubin_smooth

from src.metric_util_OLD import *
from src.config.substation_config import sub_CONFIG
from src.config.viz_config import set_map_config
from src.land_objects import LandObjects
from src.config.landobject_config import north_sea_ports
from src.generate_text_outputs import TextOutputGenerator
from src.config.logistics import Logistics
from src.marine_plan.post_compute.clustering import *
from src.marine_plan.post_compute.substations import *
from src.marine_plan.pre_compute.tech_equations.hybrid_caisson import hybrid_caisson
from src.marine_plan.pre_compute.tech_equations.interconnector import interconnector


class MarinePlan:
    """
    A MarinePlan class represents the following basic and minimum requirements for specifiying a marine use plan:
    1. Area of interest to optimize for (the area available for new technologies)
    2. A set of goals and contraints to optimize for and within.
    3. A set of available technologies and their formulas to calculate metrics.
    4. A method for optimization implemented.
    5. A resulting solution including spatial layout and metric results.

    Additional information that might be useful in the future:
    1. The estimated percent error of our results. This might change depending on technologies used.
    2. Any additional considerations of the results that can be computed automatically.
    3. The time and supply chain capacity required to build.
    """

    def __init__(self, directory=None, sim_env=None, sim_params=None, tech_params=None, opt_params=None, map_CONFIG=None, phasing=False):

        # Base attributes
        self.directory = directory
        self.name = sim_env["env_name"]
        self.run_name = sim_env["run_name"]
        self.map_CONFIG = map_CONFIG
        self.phasing = phasing
        # Sim env params
        try:
            self.scale = 'international'
            self.aoi = "aoi.geojson"
            self.goals = sim_env["goals"]
            self.result_folder = sim_env["result_folder"]
            self.config = sim_env["config"]
            self.energy_targets = sim_params["energy_targets"]
            self.optimizing_country = sim_params["country"]
            self.capacity_needed = sim_params["capacity_needed"]
            self.coop = sim_params["coop"]
            self.synergies = sim_params["synergies"]
            self.calc_interconnectors = sim_params["calc_interconnectors"]
            self.hub_constraints = sim_params["hub_constraints"]
            self.eco_limit = sim_params["eco_limit"]
            self.hubs = sim_env["hubs"]
            self.hubs_included = sim_env["hubs_included"]
            self.set_cables = sim_env["set_cables"]
            self.iterative = sim_env["iterative"]
            self.optimize_hubs = sim_env["optimize_hubs"]
            self.port_limits = sim_env["port_limits"]
        except:
            # Throw error and exit
            print("Error: sim_env parameters not set correctly")
            exit()

        # Tech params (these are a bit wonky implementations... but we can clean up later)
        self.technologies = tech_params
        self.technologies['empty'] = {'present': True, 'metric_mods': {}}
        

        self.all_uses = [tech for tech, info in self.technologies.items() if info['present']]
        self.final_uses = self.all_uses  # hacky.. but I guess it works

        # Optimization params
        self.optimization_params = opt_params

        # Parameters set after __init__
        self.out_folder = None
        self.criteria_names = None
        self.has_targets = False

        # Marine plan metrics
        self.penalty = 1
        # Marine plan representation -- tbd

        self.cables_cost = 0
        self.foundation_cost = 0

        self.clusters = {}

    # MARK:Prep methods

    def _set_paths(self, name=""):
        """
        Generic function for setting/creating paths/folders needed to run a sim.
        
        Also sets the output folder for the simulation, depending on the result_folder attribute.

        Parameters:
            name {str}: The name of the simulation. Defaults to an empty string.
            self.run_name {str}: The name of the simulation. Defaults to the name of the simulation.
            self.result_folder {str}: The folder where the results will be stored. Defaults to the result_folder attribute.
                - "test": The results will be stored in the test_results folder.
                - "real": The results will be stored in the results folder.
                - Any other string: The results will be stored in the folder specified by the string.
        """

        # Set output folder for the named sim

        if '/' in self.run_name:
            out_name = self.run_name.split('/')[0]
        else:
            out_name = self.run_name


        # For throwaway testing results
        if self.result_folder == "test":

            if name == "":
                self.out_folder = os.path.join(self.directory, "test_results", out_name)
                self.existing_tech_path = ''
            else:
                self.out_folder = os.path.join(self.directory, "test_results", name, "runs", out_name)
                self.existing_tech_path = os.path.join(self.directory, "test_results", name, "mod", "producing_wind_farms.geojson")

            if not os.path.exists(self.out_folder):
                os.makedirs(self.out_folder)

        # For any real results
        elif self.result_folder == "real":

            if name == "":
                self.out_folder = os.path.join(self.directory, "results", out_name)
            else:
                self.out_folder = os.path.join(self.directory, "results", name, "runs", out_name)

            if not os.path.exists(self.out_folder):
                os.makedirs(self.out_folder)

        # For any user defined paths
        else:
            self.out_folder = os.path.join(self.result_folder, out_name)

            if not os.path.exists(self.result_folder):
                print(f"This is not a valid path: {self.result_folder}")
                print("Please enter an output folder that exists and re-initiate the instance")

        self.wind_farms_path = os.path.join(self.directory, "data", "vectors", self.scale, "wind_farms.geojson")
        

    def _load_metrics(self):
        """
        Loads in the metrics from the file path.
        These metrics are pre-processed and stored in a pickle file.
        Converts the data to a more efficient structure using NumPy arrays.
        """
        metric_path = os.path.join(self.directory, "temp", "pickles", "calculated_metrics.pkl")
        geo_path = os.path.join(self.directory, "temp", "pickles", "geo_data.pkl")
        
        with open(metric_path, "rb") as file:
            self.data = pickle.load(file)

        with open(geo_path, "rb") as file:
            self.geo_data = pickle.load(file)

        for tech in self.data.keys():
            if 'LCOE' in self.data[tech].keys():
                self.data[tech]['LCOE'] = [1000 if x > 1000 else x for x in self.data[tech]['LCOE']]
            if 'co2+' in self.data[tech].keys():
                self.data[tech]['co2+'] = [30_000_000 if x > 30_000_000 else x for x in self.data[tech]['co2+']]
            if 'capex' in self.data[tech].keys():
                 # anything above the 100_000_000 is set to 100_000_000
                self.data[tech]['capex'] = [100_000_000 if x > 100_000_000 else x for x in self.data[tech]['capex']]

        # Replace NaN values with 0
        for key, value in self.geo_data.items():
            if key != 'country':
                self.geo_data[key] = np.nan_to_num(value)

        for tech, metrics in self.data.items():
            for key, value in metrics.items():
                self.data[tech][key] = np.nan_to_num(value)


        if not self.iterative:
            country_map = {
                1: "FR", 2: "UK", 3: "IE", 4: "NO", 5: "BE",
                6: "DE", 7: "DK", 8: "LX", 9: "NL"
            }

            # Convert country IDs to country names
            country_array = np.array([country_map.get(country, np.nan) for country in self.geo_data['country']], dtype=object)
            self.geo_data['country'] = country_array

        # Add the eco_sens from the geodata to the metrics
        wind_tech = ['monopile', 'jacket', 'semisub_cat_drag', 'semisub_taut_driv', 'semisub_taut_suc', 'spar_cat_drag', 'spar_taut_driv', 'spar_taut_suc']
        aquaculture = ['mussel', 'seaweed']

        for tech in self.all_uses:
            
            if tech == 'fpv':
                self.data[tech]['eco'] = self.geo_data['eco_sens_fpv']
            elif tech in aquaculture:
                self.data[tech]['eco'] = self.geo_data['eco_sens_aquaculture']
            elif tech == 'empty':
                self.data[tech]['eco'] = [0] * len(self.geo_data['eco_sens_aquaculture'])
            else:
                if tech in wind_tech:
                    self.data[tech]['eco'] = self.geo_data['eco_sens_wind']
               

    def _set_raster_template(self):
        """Sets the template raster, to be used for formatting and aligning the outputs."""

        # for file in os.listdir(os.path.join(self.directory, "temp")):
        #     if file.endswith(".tif"):
        #         self.template_r = os.path.join(self.directory, 'temp', file)
        #         break

        self.template_r = os.path.join(self.directory, "temp", "depth.tif")


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


    def prepare_optimization(self, msg=1, name=""):
        """All work needed to prepare for a optimization to run. Includes checks and data processing."""

        if name != "": # Just incase there is a name passed in for the output folder
            self._set_paths(name=name)
        else:
            self._set_paths()

        start_time = time.time()
        self._load_metrics()
        end_time = time.time()
        print(f"Time taken for _load_metrics: {end_time - start_time} seconds")
        self._set_raster_template()
        self._set_resolution()

        # Sets the message level for the solver -- this determines how much output is given
        self.message = msg

    # MARK:Optimization helper methods

    def _unit_conversions(self):
        """
        Converts the units given in the goals dictionary into the units required for the model
        """

        self.goals_copy = copy.deepcopy(self.goals)

        # The conversions to get from the units in the goals to the units in the model (kg)
        conversions = {"T": 1000, "MT": 1_000_000, "GT": 1_000_000_000}

        for criteria, info in self.goals_copy.items():
            unit = info["unit"]

            # Check if the unit starts with any key in the conversions dictionary
            for key in conversions.keys():
                if unit.startswith(key):
                    conversion_key = key
                    break
            else:
                continue  # Skip if no match found

            goal_type = next(iter(info))

            # Convert the value based on the conversion factor
            info[goal_type] *= conversions[conversion_key]

            # Update the unit based on CO2 presence
            if "CO2" in unit:
                info["unit"] = "kg CO2/y"
            else:
                info["unit"] = "kg/y"

    # GEOSPATIAL CONSTRAINTS

    def _set_geo_constraints(self):
        """
        Sets the geospatial constraints for each technology.
        """

        constraints = []

        for i in self.index:
            if self._is_masked(i):
                constraints.append(self._create_constraint(i, "empty", 1))
            
            
            constraints.extend(self._apply_solar_constraints(i))

            if 'mussel' in self.all_uses or 'seaweed' in self.all_uses:
                constraints.extend(self._apply_aquaculture_constraints(i))

            constraints.extend(self._apply_wind_constraints(i))
            #constraints.extend(self._apply_combined_constraints(i))

        if self.hub_constraints:
            constraints.extend(self._apply_hub_constraints())

        if self.port_limits:
            constraints.extend(self._set_port_limits())

        return constraints


    def _set_port_limits(self):
        print("Setting port limits")
        with open(os.path.join(self.directory, "temp", "closest_install_port.pkl"), 'rb') as f:
            ports = pickle.load(f)
            self.ports = ports

        constraints = []
        
        self.possible_turbines = 0

        for port_index, turbine_amount in ports.items():
            print(port_index, turbine_amount)
            # Create constraints for this hub
            self.possible_turbines += turbine_amount

            self.prob += (
                pulp.lpSum(
                    self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                    for i in self.index 
                    for tech in ['monopile', 'jacket']
                    if self.geo_data['closest_install_port'][i] == port_index
                ) >= turbine_amount * 0.9,
                f"Hub_{port_index}_Capacity_LowerBound"
            )

            self.prob += (
                pulp.lpSum(
                    self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                    for i in self.index 
                    for tech in ['monopile', 'jacket']
                    if self.geo_data['closest_install_port'][i] == port_index
                ) <= turbine_amount * 1.1,
                f"Hub_{port_index}_Capacity_UpperBound"
            )
        
        return constraints


    def _apply_hub_constraints(self):

        with open(os.path.join(self.directory, "temp", "closest_gridconnects.pkl"), 'rb') as f:
            locations = pickle.load(f)

        constraints = []
        
        for hub_name, hub_info in self.hubs.items():
            hub_index = None
            for location, loc_info in locations.items():
                if loc_info['type'] == 'hub' and location == hub_name:
                    hub_index = loc_info['index']
                    break
            
            if hub_index is None:
                continue  # Skip if hub not found in locations
            
            # Calculate max turbines for this hub
            max_turbines = int((hub_info['capacity'] * 1000) / 15)

            # Create constraints for this hub
            self.prob += (
                pulp.lpSum(
                    self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                    for i in self.index 
                    for tech in ['monopile', 'jacket', 'semisub_cat_drag', 'semisub_taut_driv', 'semisub_taut_suc', 'spar_cat_drag', 'spar_taut_driv', 'spar_taut_suc']
                    if self.geo_data['closest_gridconnect'][i] == hub_index
                ) >= max_turbines * 0.99,
                f"Hub_{hub_name}_Capacity_LowerBound"
            )

            self.prob += (
                pulp.lpSum(
                    self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                    for i in self.index 
                    for tech in ['monopile', 'jacket', 'semisub_cat_drag', 'semisub_taut_driv', 'semisub_taut_suc', 'spar_cat_drag', 'spar_taut_driv', 'spar_taut_suc']
                    if self.geo_data['closest_gridconnect'][i] == hub_index
                ) <= max_turbines,
                f"Hub_{hub_name}_Capacity_UpperBound"
            )
        
        return constraints


    def _apply_combined_constraints(self, i):
        constraints = []
        
        # Constraints for monopile_solar
        if "monopile_solar" in self.technologies and self.technologies["monopile_solar"]["present"]:
            if (self.geo_data["depth"][i] >= 60 or 
                self.data["monopile"]["soil_coefficient"][i] == 1 or 
                self.data["monopile"]["dsh"][i] <= 30 or 
                #self.data["monopile"]["eco"][i] >= 4 or
                #self.data["solar"]["d"][i] <= 2 or 
                self.data["solar"]["mean_wind_speed_at_10m"][i] >= 40): 
                #self.data["solar"]["dsh"][i] >= 10 or 
                #self.data["solar"]["eco"][i] >= 4):
                constraints.append(self._create_constraint(i, "monopile_solar", 0))
        
        # Constraints for monopile_mussel
        if "monopile_mussel" in self.technologies and self.technologies["monopile_mussel"]["present"]:
            if (self.data["monopile"]["depth"][i] >= 60 or 
                self.data["monopile"]["soil_coefficient"][i] == 1 or 
                #self.data["monopile"]["dsh"][i] <= 30 or 
                #self.data["monopile"]["eco"][i] >= 4 or
                self.data["mussel"]["d"][i] <= 10):
                #self.data["mussel"]["eco"][i] >= 4):
                constraints.append(self._create_constraint(i, "monopile_mussel", 0))
        
        return constraints


    def _is_masked(self, i):
        return self.data[self.all_uses[0]]["capex"][i] == 0
 

    def _create_constraint(self, i, tech, rhs):
        return pulp.LpConstraint(self.x[i][tech], sense=pulp.LpConstraintEQ, rhs=rhs)


    def _apply_solar_constraints(self, i):
        constraints = []

        if 'fpv' in self.all_uses:
            if (self.geo_data["depth"][i] >= 30 or 
                self.geo_data["mean_wind_speed_at_10m"][i] >= 40 or
                self.geo_data["dsh"][i] >= 10 or
                self.geo_data["eco_sens_fpv"][i] >= self.eco_limit):
                constraints.append(self._create_constraint(i, 'fpv', 0))

        return constraints


    def _apply_aquaculture_constraints(self, i):

        if 'mussel' in self.all_uses or 'seaweed' in self.all_uses:
            constraints = []
            for tech in ['mussel', 'seaweed']:
                if (self.geo_data["depth"][i] >= 10 or
                    self.data[tech]["eco"][i] >= self.eco_limit):
                    constraints.append(self._create_constraint(i, tech, 0))
            return constraints


    def _apply_wind_constraints(self, i):
        constraints = []

        # wind depth constraints
        floating_wind = ['semisub', 'spar']
        all_wind = ['monopile', 'jacket', 'semisub', 'spar']

        for tech in self.all_uses:
            for floating in floating_wind:
                if floating in tech:

                    if self.geo_data["depth"][i] <= 80:
                        constraints.append(self._create_constraint(i, tech, 0))

                    if 'drag' in tech:
                        if self.geo_data["soil_coefficient"][i] == 5_000_000:
                            constraints.append(self._create_constraint(i, tech, 0))


            for all in all_wind:
                if all in tech:
                    if (self.geo_data["dsh"][i] <= 40 or
                        self.geo_data["eco_sens_wind"][i] >= self.eco_limit):
                        constraints.append(self._create_constraint(i, tech, 0))

        if self.geo_data["depth"][i] >= 60:
            constraints.append(self._create_constraint(i, "monopile", 0))

        if self.geo_data["depth"][i] >= 100:
            constraints.append(self._create_constraint(i, "jacket", 0))
        
        if self.geo_data["depth"][i] <= 70:
            constraints.append(self._create_constraint(i, "jacket", 0))


        # RETURN TO ADD GROUND TYPE CONSTRAINTS
        
        return constraints

    # CRITERIA AND GOALS

    def _set_criteria_metrics(self):
        """
        Sets the total and tech metrics for each country and overall.
        Applies any modifications and resolution adjustments.
        Also creates country_criteria and country_tech_criteria variables.
        Converts CO2 metrics to monetary value (euros).
        """
        start_time = time.time()
        # Find the first non-zero value in the unit density column
        self.first_num = next(i for i, value in enumerate(self.data[list(self.technologies.keys())[0]]["capex"]) if value != 0)

        # Initialize total metrics
        total_metrics = {metric: 0 for metric in ['capex', 'opex', 'CO2_emission', 'CO2_mitigation', 'CO2_net', 'revenue', 'LCOE', 'energy_produced', 'food_produced', 'eco_sensitivity']}

        self.countries = set(self.geo_data['country'])
        # remove nan
        self.countries = [country for country in self.countries if country is not np.nan]
        # Initialize country_criteria and country_tech_criteria
        self.country_criteria = {country: {metric: 0 for metric in total_metrics} for country in self.countries}
        self.country_tech_criteria = {country: {} for country in self.countries}

        self.CO2_VALUE = 0.072  # euros per kg CO2

        for country in self.countries:
            for t, info in ((tech, info) for tech, info in self.technologies.items() if info['present']):
                # Calculate base metrics
                base_metrics = {
                    'CO2_emission': self.data[t]["co2+"] * self.data[t]['unit density'][self.first_num] / self.data[t]['lifetime'][self.first_num],
                    'revenue': self.data[t]["revenue"] * self.data[t]['unit density'][self.first_num],
                    'opex': self.data[t]["opex"] * self.data[t]['unit density'][self.first_num] / self.data[t]['lifetime'][self.first_num], # Divide by lifetime
                    'capex': self.data[t]["capex"] * self.data[t]['unit density'][self.first_num] / self.data[t]['lifetime'][self.first_num], # Divide by lifetime
                    'CO2_mitigation': self.data[t]["co2-"] * self.data[t]['unit density'][self.first_num],
                    'LCOE': self.data[t]["LCOE"] * self.data[t]['unit density'][self.first_num],
                    'energy_produced': self.data[t]['energy_produced'] * self.data[t]['unit density'][self.first_num],
                    'food_produced': self.data[t]['food_produced'] * self.data[t]['unit density'][self.first_num],
                    'eco_sensitivity': self.data[t]['eco'], 
                }

                # Initialize country_tech_criteria for this country and tech
                self.country_tech_criteria[country][t] = {}

                # Calculate and set metrics for each country and technology
                for metric, values in base_metrics.items():
                    metric_sum = pulp.lpSum(values[i] * self.x[i][t] 
                                            for i in self.index 
                                            if self.geo_data['country'][i] == country)

                    if metric != 'eco_sensitivity':
                        metric_value = metric_sum * self.resolution
                    else:
                        metric_value = metric_sum

                    # Convert CO2 metrics to monetary value
                    if metric in ['CO2_emission', 'CO2_mitigation']:
                        metric_value *= self.CO2_VALUE

                    setattr(self, f"{country}_{t}_{metric}", metric_value)

                    # Set country_tech_criteria
                    self.country_tech_criteria[country][t][metric] = metric_value

                    # Apply metric modifications
                    if metric in info['metric_mods']:
                        metric_value *= info['metric_mods'][metric]
                        setattr(self, f"{country}_{t}_{metric}", metric_value)
                        self.country_tech_criteria[country][t][metric] = metric_value

                    # Add to country_criteria
                    self.country_criteria[country][metric] += metric_value

                    # Add to total metrics
                    total_metrics[metric] += metric_value

            #Country net CO2
            setattr(self, f"{country}_CO2_net", self.country_criteria[country]['CO2_mitigation'] - self.country_criteria[country]['CO2_emission'])
            self.country_criteria[country]['CO2_net'] = getattr(self, f"{country}_CO2_net")
            total_metrics['CO2_net'] += getattr(self, f"{country}_CO2_net")

        # Set total metrics
        for metric, value in total_metrics.items():
            setattr(self, f"total_{metric}", value)

        # Calculate total CO2 (now in euros)
        self.total_CO2 = self.total_CO2_emission - self.total_CO2_mitigation

        end_time = time.time()
        print(f"Time taken for _set_criteria_metrics: {end_time - start_time} seconds")
        

    def _set_energy_targets(self):
        print("Setting energy targets")
        start_time = time.time()

        # Load wind farms data if not already loaded
        # if not hasattr(self, 'wind_farms_gdf'):
        #     self.wind_farms_gdf = gpd.read_file(self.wind_farms_path)
        #     country_mapping = {
        #         'Belgium': 'BE', 'Germany': 'DE', 'Denmark': 'DK', 'France': 'FR',
        #         'Ireland': 'IE', 'Lithuania': 'LT', 'Netherlands': 'NL', 'Norway': 'NO',
        #         'Sweden': 'SE', 'United Kingdom': 'UK'
        #     }
        #     self.wind_farms_gdf['COUNTRY'] = self.wind_farms_gdf['COUNTRY'].map(country_mapping).fillna(self.wind_farms_gdf['COUNTRY'])
            
        # Set offshore wind targets -- Currently not used as it uses the capacity needed entered by the user, keeping here for reference
        # This is also used for calculation the interconnectors since it needs to know the countries needs        

        self.north_sea_offshore_wind = {
            #'IE': {2020: 0.025, 2030: 7.0, 2050: 1.0},
            'FR': {2020: 0.0, 2030: 2.1, 2050: 17.0}, # All from the GNSBI NSEC report
            'BE': {2020: 2.3, 2030: 6, 2050: 8.0},
            'NL': {2020: 2.5, 2030: 21.0, 2050: 72.0},
            'DE': {2020: 7.7, 2030: 26.4, 2050: 66.0},
            'DK': {2020: 2.3, 2030: 5.3, 2050: 35.0},
            'NO': {2020: 0.0, 2030: 3, 2050: 30.0}, # 1.5 GW floating in 2030
            'UK': {2020: 11.0, 2030: 50.0, 2050: 80.0},# 5 GW floating | 2050 From an unconfirmed table
        }

        total_new_turbines_needed = 0   

        wind_tech = ['monopile', 'jacket', 'semisub_cat_drag', 'semisub_taut_driv', 'semisub_taut_suc', 'spar_cat_drag', 'spar_taut_driv', 'spar_taut_suc']
        used_wind_tech = [tech for tech in wind_tech if tech in self.all_uses]


        if self.phasing == False:
            if self.optimizing_country == "all":  # If optimizing all countries at once
                if self.coop:  # If coop is allowed
                    # Calc total 2050 target energy
                    
                    if self.iterative:
                        energy_needed = self.capacity_needed * 1000
                    else:
                        energy_needed = sum([self.north_sea_offshore_wind[country][2050] for country in self.north_sea_offshore_wind]) * 1000

                    total_new_turbines_needed = int(energy_needed / 15)
                    print(total_new_turbines_needed)
                    # Set constraint for total number of turbines
                    self.prob += (
                        pulp.lpSum(
                            self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                            for i in self.index 
                            for tech in used_wind_tech
                        ) >= total_new_turbines_needed - 10,
                        "Minimum total turbines constraint (lower bound)"
                    )
                    self.prob += (
                        pulp.lpSum(
                            self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                            for i in self.index 
                            for tech in used_wind_tech
                        ) <= total_new_turbines_needed + 10,
                        "Maximum total turbines constraint (upper bound)"
                    )
                else:  # If coop is not allowed
                    for country in self.countries:
                        if country in self.north_sea_offshore_wind:
                            country_target_energy = self.north_sea_offshore_wind[country][2050] * 1000
                            country_target_turbines = int(country_target_energy / 15)
                            total_new_turbines_needed += country_target_turbines

                            # Set constraint for each country
                            self.prob += (
                                pulp.lpSum(
                                    self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                                    for i in self.index 
                                    for tech in used_wind_tech
                                    if self.geo_data['country'][i] == country
                                ) >= country_target_turbines - 10,
                                f"Minimum turbines constraint for {country} (lower bound)"
                            )
                            self.prob += (
                                pulp.lpSum(
                                    self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                                    for i in self.index 
                                    for tech in used_wind_tech
                                    if self.geo_data['country'][i] == country
                                ) <= country_target_turbines + 10,
                                f"Maximum turbines constraint for {country} (upper bound)"
                            )

            else:  # when optimizing a single country
                energy_needed = self.capacity_needed * 1000  # Using this so the user can enter it
                total_new_turbines_needed = int(energy_needed / 15)

                if self.coop:
                    self.prob += (
                        pulp.lpSum(
                            self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                            for i in self.index 
                            for tech in used_wind_tech
                        ) >= total_new_turbines_needed * 0.99,
                        f"Minimum turbines constraint for {self.optimizing_country} (lower bound)"
                    )
                    self.prob += (
                        pulp.lpSum(
                            self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                            for i in self.index 
                            for tech in used_wind_tech
                        ) <= total_new_turbines_needed * 1.01,
                        f"Maximum turbines constraint for {self.optimizing_country} (upper bound)"
                    )
                else:
                    self.prob += (
                        pulp.lpSum(
                            self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                            for i in self.index 
                            for tech in used_wind_tech
                            if self.geo_data['country'][i] == self.optimizing_country
                        ) >= total_new_turbines_needed * 0.99,
                        f"Minimum turbines constraint for {self.optimizing_country} (lower bound)"
                    )
                    self.prob += (
                        pulp.lpSum(
                            self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                            for i in self.index 
                            for tech in used_wind_tech
                            if self.geo_data['country'][i] == self.optimizing_country
                        ) <= total_new_turbines_needed * 1.01,
                        f"Maximum turbines constraint for {self.optimizing_country} (upper bound)"
                    )
        else: # If phasing is enabled
            energy_needed = self.capacity_needed * 1000  # Using this so the user can enter it
            total_turbines_needed = int(energy_needed / 15)
            print(f"Total turbines needed: {total_turbines_needed}")

            self.existing_tech_gdf = gpd.read_file(self.existing_tech_path) if os.path.exists(self.existing_tech_path) else None
            total_existing_MW = self.existing_tech_gdf['POWER_MW'].sum()
            total_existing_turbines = int(total_existing_MW / 15)
            print(f"Total existing turbines: {total_existing_turbines}")

            total_new_turbines_needed = total_turbines_needed - total_existing_turbines
            print(f"Total new turbines needed: {total_new_turbines_needed}")

            self.prob += (
                pulp.lpSum(
                    self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                    for i in self.index 
                    for tech in used_wind_tech
                ) >= total_new_turbines_needed * 0.99,
                f"Minimum turbines constraint for {self.optimizing_country} (lower bound)"
            )
            self.prob += (
                pulp.lpSum(
                    self.x[i][tech] * (self.resolution * self.data[tech]["unit density"][self.first_num])
                    for i in self.index 
                    for tech in used_wind_tech
                ) <= total_new_turbines_needed * 1.01,
                f"Maximum turbines constraint for {self.optimizing_country} (upper bound)"
            )


        end_time = time.time()
        self.needed_turbines = total_new_turbines_needed
        print(f"Time taken for _set_energy_targets: {end_time - start_time} seconds")


    def _set_goals(self):
        """
        Sets the goals or constraints. That is any minimum or maximum values for any of the metrics, total or by tech.
        Review self.criteria_names for a list of all the possible metrics and their 'full' name.
        """
        
        # Set any energy targets and cooperation
        if self.energy_targets:
            self._set_energy_targets()

        else:
            self.needed_turbines = self.possible_turbines
        
        # Set any criteria or goals for the optimization
        if not self.goals:
            return

        self._unit_conversions()
        self.z = []
        self.pen_list = []
        self.has_targets = False
        synergies = ['monopile_solar', 'monopile_mussel']

        def extract_tech(criteria):
            parts = criteria.split('_')
            if criteria in synergies:
                return criteria
            elif len(parts) > 2 and '_'.join(parts[:2]) in synergies:
                return '_'.join(parts[:2])
            else:
                return parts[0]

        for criteria, info in self.goals_copy.items():
            tech = extract_tech(criteria)

            if tech == 'semisub':
                tech = 'semisub_cat_drag'
            elif tech == 'spar':
                tech = 'spar_cat_drag'
            

            goal_type = next(iter(info))
            target = info[goal_type]
            unit = info["unit"]
            severity = info["penalty"]

            if "_units" in criteria:
                self._set_unit_goal(tech, goal_type, target)
            else:
                self._set_metric_goal(criteria, goal_type, target, unit, severity)


    def _set_unit_goal(self, tech, goal_type, target):
        num_cells = self._calculate_num_cells(tech, target)
        constraint = pulp.lpSum(self.x[i][tech] for i in self.index)

        if goal_type == "max":
            self.prob += constraint <= num_cells, f"Max {tech} units"
        elif goal_type == "min":
            self.prob += constraint >= num_cells, f"Min {tech} units"
        elif goal_type == "exact":
            self.prob += constraint == num_cells, f"Exact {tech} units"
        elif goal_type == "range":
            lower, upper = self._calculate_num_cells(tech, target[0]), self._calculate_num_cells(tech, target[1])
            self.prob += constraint >= lower, f"Lower bound {target} {tech} units"
            self.prob += constraint <= upper, f"Upper bound {target} {tech} units"


    def _set_metric_goal(self, criteria, goal_type, target, unit, severity):
        metric = getattr(self, criteria)

        if goal_type in ["max", "min", "exact"]:
            operator = {"max": "<=", "min": ">=", "exact": "=="}[goal_type]
            self.prob += eval(f"metric {operator} target"), f"{goal_type.capitalize()} {criteria} - {target} {unit}"
        elif goal_type == "range":
            self.prob += metric >= target[0], f"Lower bound {criteria} - {target[0]} {unit}"
            self.prob += metric <= target[1], f"Upper bound {criteria} - {target[1]} {unit}"
        elif goal_type == "target":
            self._set_target_goal(criteria, target, severity)


    def _set_target_goal(self, criteria, target, severity):
        self.has_targets = True
        z_var = pulp.LpVariable(f'deviation_{criteria}', lowBound=0)
        metric = getattr(self, criteria)

        self.prob += z_var >= metric - target
        self.prob += z_var >= -(metric - target)

        penalty = {"light": 100, "medium": 1000, "harsh": 100_000_000}[severity]
        self.z.append(z_var)
        self.pen_list.append(penalty)


    def _calculate_num_cells(self, tech, target):

        if tech in ['semisub', 'spar']:
            return target / (self.resolution * 0.33)

        return target / (self.resolution * self.data[tech]["unit density"][self.first_num])

    # OUTPUTS AND RESULTS GENERATIO

    def _seed_to_gdf(self, to_file=False, filter=True):
        """
        Converts the seed from the LA to a shp for visualization or storage

        Parameters:
            to_file {bool}: Whether or not to convert the gdf to a shapefile. Defaults to False. The output is in the output folder within the wd.
            filter {bool}: Whether or not to run a majority filter over the seed array. Defaults to True.
        """

        # Converts the seed into an array
        self.seed_array = np.array(self.seed).reshape(self.n, self.n)

        # Removes small clusters from the seed array
        # self.metric_adjustments, self.seed_array = cull_small_clusters(self)

        with rasterio.open(self.template_r) as template:
            transform = template.transform
            crs = template.crs

        # Create the polygons for each tech
        polygons = []
        for row in range(self.n):
            for col in range(self.n):

                # Get the extent from the original rasters --- Something is janky and the scaled polygons are not correct
                minx, maxy = rasterio.transform.xy(transform, row - 0.5, col - 0.5)
                maxx, miny = rasterio.transform.xy(transform, row + 0.5, col + 0.5)

                value = self.seed_array[row, col]

                # Applies the value of each cell to the polygons
                if value != self.uDict["empty"]:  # Skip cells designated as empty
                    key = [k for k, v in self.uDict.items() if v == value][0]
                    polygon = box(minx, miny, maxx, maxy, ccw=True)
                    country = self.geo_data['country'][row * self.n + col]
                    capacity = self.data[key]["unit density"][self.first_num] * self.resolution * 15 / 1000
                    polygons.append((key, country, capacity, polygon))

        # Create GeoDataFrame with appropriate columns and CRS
        gdf = gpd.GeoDataFrame(polygons, columns=["value", "country", "capacity", "geometry"], crs=crs)

        # Use dissolve with sum aggregation for capacity
        self.seed_gdf = gdf.dissolve(by=["value", "country"], aggfunc={"capacity": "sum"})
        self.seed_gdf = self.seed_gdf.reset_index()  # Resets the index so that the values are a column
        # self.seed_gdf = dissolve_touching_polygons(self.seed_gdf)
        
        # Saves the gdf to a shp
        if to_file:

            if '/' in self.run_name:
                json_name = self.run_name.split('/')[1]
            else:
                json_name = self.run_name

            os.makedirs(os.path.join(self.out_folder, "SHP"), exist_ok=True)
            self.seed_gdf.to_file(os.path.join(self.out_folder, "SHP", f"{json_name}.geojson"), driver="GeoJSON")


    def _generate_seed(self):
        """
        Creates the list that from the result of the optimizer.
        """

        # self.seed = [0] * (self.n * self.n)
        # self.seed_names = [0] * (self.n * self.n)
        # self.uDict = {key: tech for tech, key in enumerate(self.final_uses)}

        # for i in self.index:
        #     for k in self.final_uses:
        #         if pulp.value(self.x[i][k]) == 1:
        #             self.seed[i] = self.uDict[k]
        #             self.seed_names[i] = k

        self.seed = [0] * (self.n * self.n)
        self.seed_names = [0] * (self.n * self.n)
        self.uDict = {key: tech for tech, key in enumerate(self.final_uses)}
        self.counter = {country: {tech: 0 for tech in self.all_uses} for country in self.countries}
        self.port_counter = {port: {tech: 0 for tech in self.all_uses} for port in self.ports} if self.port_limits else None

        for i in self.index:
            for tech in self.final_uses:
                if pulp.value(self.x[i][tech]) == 1:
                    if tech != 'empty':
                        country = self.geo_data['country'][i]
                        self.counter[country][tech] += 1

                        if self.port_limits:
                            port = self.geo_data['closest_install_port'][i]
                            self.port_counter[port][tech] += 1
                    
                    self.seed[i] = self.uDict[tech]
                    self.seed_names[i] = tech


    def _out_seed(self):
        """
        Writes the result to a dictionary that is pickled into the results folder

        For phasing and sensitivity runs, each instance is added into the created pickle
        """
        self._generate_seed()

        inst_dict = {}

        inst_dict["modifications"] = self.technologies
        inst_dict["technologies"] = list(self.technologies.keys())

        inst_dict["cell_amounts"] = {}
        counter = {tech: 0 for tech in self.all_uses}

        # Calculate the impact on fishing while we're here 
        self.fishing_intensities = {'hours': {}, 'surface': {}, 'subsurface': {}}


        for tech in self.all_uses:
            if tech == 'empty':
                continue

            hours = 0
            surface = 0
            subsurface = 0
            
            for i in self.index:
                
                if pulp.value(self.x[i][tech]) == 1 and tech == 'monopile':

                    hours += self.geo_data["fishing_hours"][i]
                    surface += self.geo_data["surface_swept_ratio"][i]
                    subsurface += self.geo_data["subsurface_swept_ratio"][i]
                    
                    counter[tech] += 1

            try:
                self.fishing_intensities['hours'][tech] = hours / counter[tech]
                self.fishing_intensities['surface'][tech] = surface / counter[tech]
                self.fishing_intensities['subsurface'][tech] = subsurface / counter[tech]
            except ZeroDivisionError:
                self.fishing_intensities['hours'][tech] = 0
                self.fishing_intensities['surface'][tech] = 0
                self.fishing_intensities['subsurface'][tech] = 0

        for tech, count in counter.items():
            inst_dict["cell_amounts"][tech] = count

        inst_dict["criteria"] = {}

        inst_dict["criteria"]["emission"] = pulp.value(self.total_CO2_emission)
        inst_dict["criteria"]["mitigation"] = pulp.value(self.total_CO2_mitigation)
        inst_dict["criteria"]["opex"] = pulp.value(self.total_opex)
        inst_dict["criteria"]["capex"] = pulp.value(self.total_capex)
        inst_dict["criteria"]["revenue"] = pulp.value(self.total_revenue)
        inst_dict["criteria"]["energy"] = pulp.value(self.total_energy_produced)
        inst_dict["criteria"]["food"] = pulp.value(self.total_food_produced)
        inst_dict["criteria"]["eco_ben"] = pulp.value(self.total_eco_sensitivity)

        inst_dict["objective"] = self.objective

        inst_dict["result"] = np.array(self.seed).reshape(self.n, self.n)
        inst_dict["result_names"] = np.array(self.seed_names).reshape(self.n, self.n)

        if '/' in self.run_name:
            pkl_name = self.run_name.split('/')[0]
        else:
            pkl_name = self.run_name

        try:
            with open(os.path.join(self.out_folder, f"{pkl_name}.pkl"), "rb") as pkl:
                out_pkl = pickle.load(pkl)

            out_pkl[self.name] = inst_dict

            with open(os.path.join(self.out_folder, f"{pkl_name}.pkl"), "wb") as pkl:
                pickle.dump(out_pkl, pkl)

        except FileNotFoundError:

            out_pkl = {}
            out_pkl = {self.name: inst_dict}

            with open(os.path.join(self.out_folder, f"{pkl_name}.pkl"), "wb") as pkl:
                pickle.dump(out_pkl, pkl)

        except Exception as e:
            print(f"Error: {e}")


    def build_optimization_problem(self):
        # Determine the optimization direction
        if self.optimization_params['direction'] == 'maximize':
            prob = pulp.LpProblem("Optimization Problem", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("Optimization Problem", pulp.LpMinimize)

        # Build the objective function
        objective = pulp.LpAffineExpression()

        # Process negative parameters
        for param, multiplier in self.optimization_params.get('negatives', {}).items():
            if hasattr(self, param):
                objective -= getattr(self, param) * multiplier

        # Process positive parameters
        for param, multiplier in self.optimization_params.get('positives', {}).items():
            if hasattr(self, param):
                objective += getattr(self, param) * multiplier

        # Set the objective
        prob += objective

        return prob


    def _calculate_total_capacity(self):
        total_capacity = {}
        for country, values in self.counter.items():
            monopile_capacity = values.get('monopile', 0) * self.resolution * 0.5 * 15 / 1000  # converting to GW
            jacket_capacity = values.get('jacket', 0) * self.resolution * 0.5 * 15 / 1000     # converting to GW
            total_capacity[country] = monopile_capacity + jacket_capacity
        
        self.country_capacity = total_capacity


    # MARK: Optimization implementation methods -- add more as we build them

    def run_linear_optimization(self, run_optimize=True):
        """
        The parameters for the Linear Algorithm.

        It first prepares the data so that it is in a format for the solver to calculate. Then loads in the criteria metrics and the obejective functions, and lastly applies the constraints*

        It then runs the solver and saves the results to the instance
        """

        # Set simulation parameters
        # print(f'Setting optimization parameters for {self.name}')

        self.first = next(iter(self.data.values()))
        self.n = int(math.sqrt(len(self.first["capex"])))
        self.total_cells = sum(1 for val in self.first["capex"] if not np.isnan(val) or val == 0)
        self.shape = [self.n, self.n]
        self.num = self.n * self.n


        self.index = range(self.num)
        self.x = pulp.LpVariable.dicts("cell", (self.index, self.final_uses), cat="Binary")

        # Criteria Metrics
        self._set_criteria_metrics()

        # Defines the optimization problem:
        self.prob = self.build_optimization_problem()

        # One tech per cell
        for i in self.index:
            self.prob += pulp.lpSum([self.x[i][k] for k in self.final_uses]) == 1

        # Physiographic constraints
        self.prob.extend(self._set_geo_constraints())

        # Constraints on goals
        self._set_goals()

        if run_optimize:
            # print(f'\nBeginning optimization for {self.name}')

            start_time = time.time()

            if self.optimization_params['solver'] == 'CBC':
                self.prob.solve(pulp.PULP_CBC_CMD(timeLimit=10, msg=self.message))

            elif self.optimization_params['solver'] == 'HiGH':
                op = os.path.join(self.directory, '/HiGHSstatic.v1.7.1.aarch64-apple-darwin/bin/highs')
                solver = pulp.HiGHS(path=op, msg=self.message)
                self.prob.solve(solver)
            else:
                print(f"Solver not recognized, please use 'CBC' or 'HiGH'")

            self.status = pulp.LpStatus[self.prob.status]
            self.objective = pulp.value(self.prob.objective)

            if self.status == "Optimal":
                print(f"Optimal solution found for {self.name}:", self.objective)
                
            else:
                print(f"Optimization did not find an optimal solution for {self.name} - {self.run_name} -- {self.status}.")

            end_time = time.time()
            print(f"Time taken for optimization: {end_time - start_time} seconds")

            self._out_seed()
            if self.status != 'infeasible' or self.status != 'Not Solved':
                self._seed_to_gdf(to_file=True, filter=False)

                self._load_land_objects()

                #remove gridconnect_coords column
                self.cluster_gdf.drop(columns=['gridconnect_coords'], inplace=True)
                self.cluster_gdf.to_file(os.path.join(self.out_folder, "SHP", f"{self.run_name}_clusters.geojson"), driver="GeoJSON")

                self._calculate_total_capacity() # Get the per country capacity

                # self._set_metrics()



            # if self.map_CONFIG:
            #     self.plot_optimal_solution(map_CONFIG=self.map_CONFIG)


    # MARK: Viz and output methods -- need to be universal to all optimization methods

    def _add_unit(self, criteria):

        if criteria == "co2" or criteria == "mit" or criteria == "emi":
            return " kg CO2/y\n"
        elif criteria == "cap" or criteria == "cost":
            return " €\n"
        elif criteria == "food":
            return " kg/y\n"
        elif criteria == "opr" or criteria == "val":
            return " €/y\n"
        elif "energy" in criteria:
            return " GWh/y\n"


    def _utm_formatter(self, x, pos):
        """
        NOTE: I haven't been able to implement this without using this function.

        This converts the coordinates beside along the map into km rather than m
        """
        return f"{int(x/1000)}"


    def get_2050_target(self, country):

        if not hasattr(self,  'north_sea_offshore_wind'):

            # Set offshore wind targets
            self.north_sea_offshore_wind = {
                'IE': {2020: 0.025, 2030: 7.0, 2050: 1.0},
                'FR': {2020: 0.0, 2030: 5.0, 2050: 20.0},
                'BE': {2020: 2.3, 2030: 5.4, 2050: 6.0},
                'NL': {2020: 2.5, 2030: 21.0, 2050: 60.0},
                'DE': {2020: 7.7, 2030: 19.8, 2050: 22.0},
                'DK': {2020: 2.3, 2030: 3.65, 2050: 1.0},
                'NO': {2020: 0.0, 2030: 4.5, 2050: 1.0},
                'UK': {2020: 11.0, 2030: 50.0, 2050: 80.0},
                'total': {2020: 25.825, 2030: 116.35, 2050: 190.0}
            }

        if not hasattr(self, 'wind_farms_gdf'):
            self.wind_farms_gdf = gpd.read_file(self.wind_farms_path)
            country_mapping = {
                'Belgium': 'BE', 'Germany': 'DE', 'Denmark': 'DK', 'France': 'FR',
                'Ireland': 'IE', 'Lithuania': 'LT', 'Netherlands': 'NL', 'Norway': 'NO',
                'Sweden': 'SE', 'United Kingdom': 'UK'
            }
            self.wind_farms_gdf['COUNTRY'] = self.wind_farms_gdf['COUNTRY'].map(country_mapping).fillna(self.wind_farms_gdf['COUNTRY'])
            


        target = self.north_sea_offshore_wind.get(country, {}).get(2050, 0)
        currently_planned = self.wind_farms_gdf[self.wind_farms_gdf['COUNTRY'] == country]['POWER_MW'].sum() / 1000

        return target - currently_planned


    def _create_cable_lines_gdf(self):
        cable_lines = []
        COST_PER_KM = 825000  # Cost per kilometer in the specified currency

        for cable_name, cable_info in self.set_cables.items():
            from_point, from_location = self._get_point(cable_info['from'], cable_info['to'])
            to_point, to_location = self._get_point(cable_info['to'], cable_info['from'])
            designation = cable_info['designation']
            capacity = cable_info['capacity']

            if from_point and to_point:
                line = LineString([from_point, to_point])
                length_km = line.length / 1000  # Convert meters to kilometers

                data = {
                    'cable_distance': length_km,
                    'HVDC_capacity': capacity * 1000,
                    'interconnector_function': designation
                }

                cable = interconnector(data)

                results = cable.run()

                cable_lines.append({
                    'name': cable_name,
                    'capacity': cable_info['capacity'],
                    'from': from_location,
                    'to': to_location,
                    'geometry': line,
                    'length_km': length_km,
                    'capex': results['capex'],
                    'opex': results['opex'],
                    'co2+': results['co2+'],
                    }
                )
                

        return gpd.GeoDataFrame(cable_lines, crs="EPSG:3035")


    def _get_point(self, location, other_location):
        # Check if location is in hubs
        hub = self.hubs_gdf[self.hubs_gdf['name'] == location]
        if not hub.empty:
            return hub.geometry.iloc[0], location

        grid_connect = self.land_object_gdf[self.land_object_gdf['name'] == location]
        if not grid_connect.empty:
            return grid_connect.geometry.iloc[0], location
        
        print(f"Warning: Location '{location}' not found in hubs or as a valid country code.")
        return None, None


    def _get_other_end_point(self, location):
        hub = self.hubs_gdf[self.hubs_gdf['name'] == location]
        if not hub.empty:
            return hub.geometry.iloc[0]
        # If it's a country code, we can't determine a specific point, so we return None
        return None        


    def _add_foundation_costs(self):
        def calculate_cost(row):
            capacity = row['connected_capacity']
            geom = row['geometry']
            x, y = geom.x, geom.y

            with rasterio.open(self.template_r) as src:
                depth = next(src.sample([(x, y)]))[0]

                if depth < -100:
                    depth = random.randint(-70, -30)

            info = {
                'depth': depth * -1,
                'distance_to_onshore_sub': 0,
                'distance_to_hydrogen_sub': 0,
                'HVAC_capacity': 0,
                'HVDC_capacity': capacity * 1000,
                'H2_capacity': 0,
            }

            hub = hybrid_caisson(info)
            
            results = hub.run()
            
            if results is None:
                # Handle the case where _calculations() returns None
                results = {}
                print(f"Warning: hub._calculations() returned None for row with capacity {capacity}")
            
            # Add the original info to the results
            results.update(info)
            
            return pd.Series(results)

        # Apply the function and expand the result into new columns
        new_columns = self.hubs_gdf.apply(calculate_cost, axis=1, result_type='expand')
        
        # Join the new columns to the original DataFrame
        self.hubs_gdf = self.hubs_gdf.join(new_columns)


    def _calc_connected_hub_capacity(self):

        # with open(os.path.join(self.directory, "temp", "closest_gridconnects.pkl"), 'rb') as f:
        #     self.locations = pickle.load(f)

        # hub_index = {value['index']: key for key, value in self.locations.items() if value['type'] == 'hub'}
        
        # hub_counter = {hub: 0 for hub in hub_index.keys()}


        # for i in self.index:
        #     for tech in ['monopile', 'jacket']: # 'semisub_cat_drag', 'semisub_taut_driv', 'semisub_taut_suc', 'spar_cat_drag', 'spar_taut_driv', 'spar_taut_suc']
        #         if pulp.value(self.x[i][tech]) == 1:
        #             hub_num = self.geo_data['closest_gridconnect'][i]
        #             if hub_num in hub_index.keys():
        #                 hub_counter[hub_num] += (self.resolution * self.data[tech]['unit density'][self.first_num]) * 15 / 1000

        # hub_counter_names = {hub_index[hub_num]: count for hub_num, count in hub_counter.items()}

        # self.hubs_gdf['connected_capacity'] = self.hubs_gdf['name'].map(hub_counter_names)

        # make the connected_capacity = capacity
        self.hubs_gdf['connected_capacity'] = self.hubs_gdf['capacity']


    def _connect_clusters_to_hubs(self):
        connections = []
        
        for idx, hub in self.hubs_gdf.iterrows():
            hub_point = hub.geometry
            hub_capacity = hub['capacity']
            hub_country = hub['country']
            
            country_clusters = self.cluster_gdf[self.cluster_gdf['country'] == hub_country]

            # Calculate distances from the hub to all clusters
            country_clusters['distance'] = country_clusters.geometry.distance(hub_point)
            
            # Sort clusters by distance and iterate through them
            for _, cluster in country_clusters.sort_values('distance').iterrows():
                if cluster['capacity'] >= hub_capacity:
                    # Create a LineString connection
                    connection = LineString([hub_point, cluster.geometry.centroid])
                    
                    # Calculate distance in kilometers
                    distance_km = connection.length / 1000  # Assuming the CRS is in meters
                    
                    # Calculate number of cables needed
                    cables_needed = math.ceil(hub_capacity / 0.855)
                    
                    # Calculate total cost
                    cost = cables_needed * distance_km * 3_400_000  # 3.4 million per km per cable
                    
                    connections.append({
                        'to': hub['name'],
                        'from': cluster['cluster_id'],
                        'geometry': connection,
                        'capacity': hub_capacity,
                        'length_km': distance_km,
                        'cables_needed': cables_needed,
                        'capex': cost
                    })
                    break  # Move to the next hub
        
        # Create a GeoDataFrame from the connections
        self.farm_to_hub = gpd.GeoDataFrame(connections, crs=self.hubs_gdf.crs)


    def _convert_coords_to_epsg3035(self):

        hubs = copy.deepcopy(self.hubs)

        for hub, data in hubs.items():
            transformer = Transformer.from_crs("epsg:4326", "epsg:3035", always_xy=True)
            # Extract the original lat and lon
            lon, lat = data["longitude"], data["latitude"]
            
            # Transform the coordinates
            x, y = transformer.transform(lon, lat)
            # Replace the latitude and longitude with x and y
            self.hubs[hub]["longitude"], self.hubs[hub]["latitude"] = x, y

        print(self.hubs)
    
    
    def _load_land_objects(self):
        """
        Convert north_sea_ports and hubs dictionaries to GeoDataFrames.
        
        Parameters:
        north_sea_ports (dict): Dictionary containing north sea ports and substations.
        hubs (dict): Dictionary containing hub information.
        
        Returns:
        tuple: A tuple containing two GeoDataFrames:
            - ports_gdf: GeoDataFrame of ports and substations
            - hubs_gdf: GeoDataFrame of hubs
        """
        # Process north_sea_ports

        pickle_path = os.path.join(self.directory, "temp", 'pickles', "updated_ports.pkl")

        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                self.updated_ports = pickle.load(f)

        land_objects = []
        for country, locations in self.updated_ports.items():
            for name, info in locations.items():
                land_objects.append({
                    'name': name,
                    'designation': info['designation'],
                    'country': info.get('country', country),  # Use 'country' from info if available, else use the outer key
                    'geometry': Point(info['longitude'], info['latitude'])
                })
        
        # Create GeoDataFrame for ports and substations
        self.land_object_gdf = gpd.GeoDataFrame(land_objects, crs="EPSG:4326")
        
        if len(self.hubs) > 0:
            # Process hubs
            hubs_data = []
            for name, info in self.hubs.items():
                hubs_data.append({
                    'name': name,
                    'capacity': info['capacity'],
                    'geometry': Point(info['longitude'], info['latitude']),
                    'country': info['country']
                })
            
            # Create GeoDataFrame for hubs
            self.hubs_gdf = gpd.GeoDataFrame(hubs_data, crs="EPSG:4326")
            
            # Convert both GeoDataFrames to EPSG:3035
            self.land_object_gdf = self.land_object_gdf.to_crs("EPSG:3035")
            self.hubs_gdf = self.hubs_gdf.to_crs("EPSG:3035")

            self.cluster_gdf = create_clusters(self)           

            if self.hubs_included:
                self.cable_lines_gdf = self._create_cable_lines_gdf()
                self._calc_connected_hub_capacity()
                self._add_foundation_costs()
                self._connect_clusters_to_hubs()

                if self.optimize_hubs:
                    self._convert_coords_to_epsg3035()

                    new_hubs_dict = generate_hubs_with_connections(self.hubs, self.set_cables, self.updated_ports, self.farm_to_hub, self.cluster_gdf)
                    
                    optimized_data = optimize_coordinates(new_hubs_dict)
                
                    for hub, coords in self.hubs.items():
                        if hub == 'PE zone':
                            continue
                        self.hubs[hub]['latitude'] = optimized_data[hub]['coordinates'][1]
                        self.hubs[hub]['longitude'] = optimized_data[hub]['coordinates'][0]

                    if len(self.hubs) > 0:
                        # Process hubs
                        hubs_data = []
                        for name, info in self.hubs.items():
                            hubs_data.append({
                                'name': name,
                                'capacity': info['capacity'],
                                'geometry': Point(info['longitude'], info['latitude']),
                                'country': info['country']
                            })
                        
                        # Create GeoDataFrame for hubs
                        self.hubs_gdf = gpd.GeoDataFrame(hubs_data, crs="EPSG:3035")

                        self.cable_lines_gdf = self._create_cable_lines_gdf()
                        self._calc_connected_hub_capacity()
                        self._add_foundation_costs()
                        self._connect_clusters_to_hubs()

        else:
            with open(os.path.join(self.directory, "temp", "closest_gridconnects.pkl"), 'rb') as f:
                self.locations = pickle.load(f)
            self.hubs_included = False

            self.cluster_gdf = create_clusters(self) 


    def _load_shp(self, mapping):

        # Load background files
        self.shipping_gdf = (
            gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "Shipping.geojson"))
            if mapping["msp"]["shipping"]
            else None
        )
        self.military_gdf = (
            gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "military.geojson"))
            if mapping["msp"]["military"]
            else None
        )
        self.sand_extraction_gdf = (
            gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "sand_extraction.geojson"))
            if mapping["msp"]["sand_extraction"]
            else None
        )
        self.nature_gdf = (
            gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "nature_reserves.geojson"))
            if mapping["msp"]["nature_reserves"]
            else None
        )

        self.wind_farms_gdf = gpd.read_file(self.wind_farms_path)
        country_mapping = {
            'Belgium': 'BE', 'Germany': 'DE', 'Denmark': 'DK', 'France': 'FR',
            'Ireland': 'IE', 'Lithuania': 'LT', 'Netherlands': 'NL', 'Norway': 'NO',
            'Sweden': 'SE', 'United Kingdom': 'UK'
        }
        self.wind_farms_gdf['COUNTRY'] = self.wind_farms_gdf['COUNTRY'].map(country_mapping).fillna(self.wind_farms_gdf['COUNTRY'])
        

        #self.substations = gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "substations.geojson"))
        self.aoi_gdf = gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, self.aoi))
        self.cities_gdf = gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "cities.geojson"))
        self.eez_gdf = gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "eez.geojson"))
        self._eez_gdf = gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "eez.geojson"))
        self.shoreline_gdf = gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "Shoreline.geojson"))
        # self.boundaries_gdf = gpd.read_file(
        #     os.path.join(self.directory, "data", "vectors", self.scale, "boundaries.geojson")
        # )

        # Load the old shapefiles with the locations of previous tech
        self.existing_tech_gdf = gpd.read_file(self.existing_tech_path) if os.path.exists(self.existing_tech_path) else None

        if mapping['msp']['cables']:
            self.current_cables = gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "cables.geojson"))            


    def _plot_files(self, mapping, redo_land_objects=False):
        # Shipping
        if mapping["msp"]["shipping"][0]:
            self.shipping_gdf.plot(ax=self.ax, color=mapping["msp"]["shipping"][1], alpha=0.8)

        # Military
        if mapping["msp"]["military"][0]:
            self.military_gdf.plot(ax=self.ax, color=mapping["msp"]["military"][1], alpha=0.5, hatch="xxx", edgecolor="black", linewidth=0.5)

        # Sand extraction
        if mapping["msp"]["sand_extraction"][0]:
            self.sand_extraction_gdf.plot(ax=self.ax, color=mapping["msp"]["sand_extraction"][1], hatch="//", alpha=0.5, edgecolor="black", linewidth=1)

        # Nature reserves
        if mapping["msp"]["nature_reserves"][0]:
            self.nature_gdf.plot(ax=self.ax, color=mapping["msp"]["nature_reserves"][1], hatch="...", alpha=0.5, edgecolor=mapping["msp"]["nature_reserves"][1], linewidth=0.5)

        # Cables
        if mapping["msp"]["cables"][0]:
            if self.calc_interconnectors:
                cable_colors = {
                    'IC_to_grid': mapping["msp"]["cables"][1]["IC_to_grid"],
                    'farm_to_IC': mapping["msp"]["cables"][1]["farm_to_IC"],
                    'farm_to_grid': mapping["msp"]["cables"][1]["farm_to_grid"]
                }


                # Filter cables by type
                IC_to_grid_gdf = self.all_connections_gdf[self.all_connections_gdf['connection_type'] == 'IC_to_grid']
                farm_to_IC_gdf = self.all_connections_gdf[self.all_connections_gdf['connection_type'] == 'farm_to_IC']
                farm_to_grid_gdf = self.all_connections_gdf[self.all_connections_gdf['connection_type'] == 'farm_to_grid']

                # Plot cables with different colors based on type
                for cable_type, gdf in [
                    ('IC_to_grid', IC_to_grid_gdf),
                    ('farm_to_IC', farm_to_IC_gdf),
                    ('farm_to_grid', farm_to_grid_gdf),
                ]:
                    gdf.plot(
                        ax=self.ax,
                        color=cable_colors[cable_type],
                        linewidth=0.8 if cable_type == 'IC_to_grid' else 0.5,
                        alpha=0.8 if cable_type == 'IC_to_grid' else 0.6,
                        zorder=6 if cable_type == 'IC_to_grid' else 5
                    )

            self.current_cables.plot(ax=self.ax, color="yellow", linewidth=0.4, alpha=1, zorder=4)
        # Enhance the interconnectors
        if self.calc_interconnectors:
            self.interconnectors_gdf = self.interconnectors_gdf[self.interconnectors_gdf['grid_connections'].map(len) > 0]
            # Plot the interconnector markers
            self.interconnectors_gdf.plot(
                ax=self.ax, 
                color="red", 
                markersize=150,  # Increased size to fit numbers
                marker='o',
                edgecolor='white',
                linewidth=2,
                zorder=10
            )
            
            for idx, row in self.interconnectors_gdf.iterrows():
                self.ax.text(
                    row.geometry.x, 
                    row.geometry.y - 0.002,  # Slight downward adjustment
                    str(idx),
                    color='white',
                    fontsize=8,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    zorder=11
                )

        # Wind farms
    # Wind farms
        if mapping["msp"]["wind_farms"][0]:
            approved = self.wind_farms_gdf[self.wind_farms_gdf["STATUS"] == "Approved"]
            planned = self.wind_farms_gdf[self.wind_farms_gdf["STATUS"] == "Planned"]
            under_construction = self.wind_farms_gdf[self.wind_farms_gdf["STATUS"] == "Construction"]
            operational = self.wind_farms_gdf[self.wind_farms_gdf["STATUS"] == "Production"]

            approved.plot(ax=self.ax, color=mapping["msp"]["wind_farms"][1]["approved"], alpha=0.5, hatch='|', edgecolor="black", linewidth=1, zorder=-1)
            planned.plot(ax=self.ax, color=mapping["msp"]["wind_farms"][1]["planned"], alpha=0.5, hatch='|', edgecolor="black", linewidth=1, zorder=-1)
            under_construction.plot(ax=self.ax, color=mapping["msp"]["wind_farms"][1]["under_construction"], alpha=0.5, hatch='|', edgecolor="black", linewidth=1, zorder=-1)
            operational.plot(ax=self.ax, color=mapping["msp"]["wind_farms"][1]["operational"], alpha=0.5, hatch='|', edgecolor="black", linewidth=1, zorder=-1)

            # approved.plot(ax=self.ax, color="#FFD700", alpha=0.5, hatch='|', edgecolor="black", linewidth=1, zorder=-1)
            # planned.plot(ax=self.ax, color="#EEE8AA", alpha=0.5, hatch='|', edgecolor="black", linewidth=1, zorder=-1)
            # under_construction.plot(ax=self.ax, color="#FF7F50", alpha=0.5, hatch='|', edgecolor="black", linewidth=1, zorder=-1)
            # operational.plot(ax=self.ax, color="#008080", alpha=0.5, hatch='|', edgecolor="black", linewidth=1, zorder=-1)

        # Geographic things
        self.shoreline_gdf.plot(ax=self.ax, color="#696969", alpha=0.8, linewidth=25)
        #self.boundaries_gdf.plot(ax=self.ax, color="black", linewidth=1.5, linestyle="--")

        self.cities_gdf.plot(ax=self.ax, marker="s", color="white", markersize=25, label="Points")
        for x, y, label in zip(self.cities_gdf.geometry.x, self.cities_gdf.geometry.y, self.cities_gdf["name"]):
            self.ax.text(x + 5000, y - 1000, label, fontsize=14, ha="left", color="white", fontfamily="monospace")


        if self.coop:
            self._eez_gdf.plot(
                ax=self.ax, color="black", linewidth=0.8
            )  # black line behind the eez dashed line to give it some more pop
            self.eez_gdf.plot(ax=self.ax, color="white", linewidth=0.5, linestyle="--")
        else:
            self._eez_gdf.plot(
                ax=self.ax, color="black", linewidth=1.5
            )  # black line behind the eez dashed line to give it some more pop
            self.eez_gdf.plot(ax=self.ax, color="white", linewidth=1, linestyle="--")

        # # Add pop dens
        # with rasterio.open(os.path.join(self.directory, "data", "vectors", self.scale, 'pop_dens.tif')) as src:
        #     population_data = src.read(1)  # Read the first band
        #     population_data = np.ma.masked_where(population_data == src.nodata, population_data)  # Mask no data values

        #     # Get the extent of the TIF file
        #     bounds = src.bounds
        #     extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        #     # Create custom colormap with brighter colors
        #     colors = ['#404040', '#696969', '#FFFF00']  # Darker Grey, Bright Yellow, Bright Orange-Red
        #     n_bins = 100  # Number of color gradations
        #     custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

        #     # Use logarithmic normalization
        #     vmin = max(0.1, population_data.min())  # Avoid log(0)
        #     vmax = population_data.max()
        #     norm = LogNorm(vmin=vmin, vmax=vmax)

        #     # Plot the TIF data with higher alpha value and logarithmic normalization
        #     im = self.ax.imshow(population_data, 
        #                         extent=extent, 
        #                         cmap=custom_cmap,
        #                         norm=norm,
        #                         alpha=0.7,
        #                         zorder=1)


        ## FOR PORTS AND SUBSTATIONS

        def create_color_scheme():
            return {
                'substation': 'yellow',
                'ins': 'lightgreen',
                'opr': 'orange',
                'both': 'red',
                'port': 'blue',
                'default': 'gray'
            }

        # In your plotting function:
        colors = create_color_scheme()

        self.land_object_gdf = self.land_object_gdf.to_crs("EPSG:3035")
        for designation in self.land_object_gdf['designation'].unique():
            subset = self.land_object_gdf[self.land_object_gdf['designation'] == designation]
            color = colors.get(designation.lower(), colors['default'])
            
            subset.plot(
                ax=self.ax,
                zorder=20 if designation != 'substation' else 19,
                marker='o' if designation != 'substation' else '*',
                markersize=35 if designation != 'substation' else 200,  # Consistent size for all markers
                color=color,
                alpha=0.75, # Slight transparency
                label=designation,
                linewidths=0.5,
                edgecolors='black'
            )


        if self.hubs_included:
            # Plot hubs
            self.hubs_gdf.plot(
                ax=self.ax,
                marker='x',
                markersize=50,
                color='red',
                label='hub',
                zorder=30
            )

            # Add annotations for capacity and connected_capacity
            # for idx, row in self.hubs_gdf.iterrows():
            #     self.ax.annotate(
            #         f"Capacity: {round(row['capacity'], 2)} GW\nConnected: {round(row['connected_capacity'], 2)} GW",
            #         (row.geometry.x, row.geometry.y),
            #         xytext=(0, 10),  # 10 points vertical offset
            #         textcoords="offset points",
            #         ha='center',
            #         va='bottom',
            #         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            #         fontsize=8,
            #         zorder=10
            #     )

            # Plot cable lines
            self.cable_lines_gdf.plot(ax=self.ax, color='yellow', linewidth=0.3, zorder=5)
            self.farm_to_hub.plot(ax=self.ax, color='green', linewidth=0.3, zorder=5)


        plt.subplots_adjust(left=0, right=0.75, top=0.99, bottom=0.03)


    def plot_optimal_solution(self, belgium=False, map_CONFIG=None, install_decom_tracker={}, all_time_tracker={}, num_format='eu'):
        
        # set mapping
        if map_CONFIG:
            mapping = map_CONFIG
        else:
            mapping = set_map_config(self.config[0], self.scale)

        # Load the appropriate files
        self._load_shp(mapping)

        # Set the formatting for the plot
        custom_cmap = ListedColormap(
            [mapping["colours"][value] for value in self.seed_gdf["value"].unique()]
        )  # Set the colourmap

        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.fig.patch.set_facecolor("#333333")  # Set the background color of the entire figure
        self.ax.set_facecolor("#1f1f1f")  # Set the background color of the plot area (the colour that the ocean ends up)

        if self.iterative == False:
            # Plot the old tech (grey) and the new tech locations (coloured)
            self.existing_tech_gdf.plot(ax=self.ax, column='COUNTRY', cmap=custom_cmap, alpha=0.23) if self.existing_tech_gdf is not None else None
            self.seed_gdf.plot(ax=self.ax, column="value", cmap=custom_cmap)

        else:
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
                
                # Plot using the 'color' column

                try:
                    monopiles = gdf[gdf['value'] == 'monopile']
                    jackets = gdf[gdf['value'] == 'jacket']
                except:
                    monopiles = gdf[gdf['tech'] == 1]
                    jackets = gdf[gdf['tech'] == 2]

                monopiles.plot(ax=ax, color=gdf['color'], alpha=a)
                jackets.plot(ax=ax, color=gdf['color'], alpha=a, hatch='//')
                
            # Plot existing_tech_gdf if it's not None
            if self.existing_tech_gdf is not None:
                plot_gdf_with_colors(self.existing_tech_gdf, self.ax, 'COUNTRY', a=0.23)

            # Plot cluster_gdf
            plot_gdf_with_colors(self.seed_gdf, self.ax, 'country', a=1)


        self._plot_files(mapping)

        # Set the colours of the figure edges
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")

        # Set the label colours
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")

        # Set the tick colours
        self.ax.tick_params(axis="x", colors="white", labelfontfamily="monospace", labelsize=12)
        self.ax.tick_params(axis="y", colors="white", labelfontfamily="monospace", labelsize=12)

        # Set zoom distance
        
        if belgium:
            belgium_aoi_gdf = gpd.read_file(os.path.join(self.directory, "data", "vectors", self.scale, "belgium_aoi.geojson"))
            xlim_min, ylim_min, xlim_max, ylim_max = belgium_aoi_gdf.total_bounds

        else:
            xlim_min, ylim_min, xlim_max, ylim_max = self.aoi_gdf.total_bounds
        
        
        
        self.ax.set_xlim(xlim_min, xlim_max)
        self.ax.set_ylim(ylim_min, ylim_max)
        self.ax.set_title(self.run_name, color='white')

        # Format the units so they are % 1000
        self.ax.xaxis.set_major_formatter(FuncFormatter(self._utm_formatter))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self._utm_formatter))

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

                plt.Line2D([0], [0], color='white', linewidth=0, label='------- Technologies ---------', linestyle='None'),

                Patch(color=config['colours']['mussel'], alpha=1, lw=2, label='Mussel Farms'),
                Patch(color=config['colours']['seaweed'], alpha=1, lw=2, label='Seaweed Farms'),
                Patch(color=config['colours']['monopile'], alpha=1, lw=2, label='Monopile'),
                Patch(color=config['colours']['jacket'], alpha=1, lw=2, label='Jacket'),
                Patch(color=config['colours']['fpv'], alpha=1, lw=2, label='Floating PV'),
                Patch(color=config['colours']['semisub_cat_drag'], alpha=1, lw=2, label='Semisubmersible Floating Wind'),
                Patch(color=config['colours']['spar_cat_drag'], alpha=1, lw=2, label='Spar Foundation Floating Wind'),
            ]
            
            port_legend = [
                plt.Line2D([0], [0], color='white', linewidth=0, label='------- Ports & Substations ---------', linestyle='None'),
                Line2D([0], [0], color=colors['substation'], marker='o', linestyle='None', markersize=8, label='Substation'),
                Line2D([0], [0], color=colors['ins'], marker='o', linestyle='None', markersize=8, label='Installation'),
                Line2D([0], [0], color=colors['opr'], marker='o', linestyle='None', markersize=8, label='Operational'),
                Line2D([0], [0], color=colors['both'], marker='o', linestyle='None', markersize=8, label='Both'),
            ]
            return existing_legend + port_legend

        # Now you can create the legend when needed:
        legend_handles = create_combined_legend(map_CONFIG)

        # Display the legend
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
                loc="lower right",
                bbox_to_anchor=(0.99, 0.01),  # Adjust these values as needed
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

        #self._build_text()
        outputs = TextOutputGenerator(self)
        text = outputs.generate_map_text(map_CONFIG['output_type'], install_decom_tracker, all_time_tracker, num_format)

        props = dict(boxstyle="round", facecolor="#1f1f1f", alpha=0)

        self.ax.text(
            1.02,
            1.008,
            text,
            transform=self.ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
            fontfamily="monospace",
            color="white",
        )

        self.ax.set_aspect("equal")

        if '/' in self.run_name:
            out_name = self.run_name.split('/')[1]
        else:
            out_name = self.run_name

        # if self.phasing:
        #     img_folder = os.path.join(self.directory, "test_results", self.name, 'img')


        # if not os.path.exists(img_folder):
        #     os.makedirs(img_folder)
        
        # plt.tight_layout()
        # plt.savefig(os.path.join(img_folder, out_name), dpi=300, bbox_inches='tight')



        plt.show()

        plt.close()