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
import copy


from shapely.geometry import box
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch, Rectangle


class MarinePlan:
    """
    A MarinePlan class represents the following basic and minimum requirements for specifiying a marine use plan:
    1. Area of interest to optimize for (the area available for new technologies)
    2. A set of goals and contraints to optimize for and within.
    3. A set of available technologies and their formulas to calculate metrics.
    4. A method for optimization implemented.
    5. A resulting solution including spatial layout and metric results.
    """

    def __init__(self, directory=None, sim_name=None, sim_env=None, tech_params=None):

        # Base attributes
        self.directory = directory
        self.name = sim_name
        self.run_name = sim_name

        # Sim env params
        try:
            self.aoi = "aoi.geojson"
            self.goals = sim_env["goals"]
        except:
            print("Please ensure that the simulation environment is properly formatted")

        # Tech params (these are a bit wonky implementations... but we can clean up later)
        self.technologies = tech_params
        self.all_uses = [tech for tech, info in self.technologies.items() if info['present']]
        self.final_uses = self.all_uses + ["empty"]  # hacky.. but I guess it works
        self.tech_names = {
            "OWF-M": "monopile",
            "OWF-J": "jacket",
            "FPV": "solar",
            "AQC-ML": "mussel",
            "HYD-MP": "hydrogen",
            "MUL_OWF-M_FPV": "monopile + solar",
            "MUL_OWF-M_AQC-ML": "monopile + mussel",
        }  # hack for now... there must be a better place for this info

        # Parameters set after __init__
        self.out_folder = None
        self.criteria_names = None
        self.has_targets = False

        # Marine plan metrics
        self.total_CO2_emission = 0
        self.total_CO2_mitigation = 0
        self.total_CO2 = 0
        self.total_value = 0
        self.total_opex = 0
        self.total_capex = 0
        self.total_ecosystem_benefits = 0
        self.total_energy = 0
        self.total_food = 0
        self.penalty = 1
        # Marine plan representation -- tbd

    # MARK:Prep methods

    def _set_paths(self):
        """        
        sets the output folder for the simulation

        Parameters:
            self.run_name {str}: The name of the simulation. Defaults to the name of the simulation.

        """

        # Set output folder for the named sim

        if '/' in self.run_name:
            out_name = self.run_name.split('/')[0]
        else:
            out_name = self.run_name


        self.out_folder = os.path.join(self.directory, "results", out_name)

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)


    def _load_metrics(self):
        """
        Loads in the metrics from the file path.
        These metrics are pre-processed and stored in a pickle file.
        """

        filepath = os.path.join(self.directory, "pickles", f"calculated_metrics_yearly.pkl")

        with open(filepath, "rb") as file:
            self.data = pickle.load(file)


    def _set_raster_template(self):
        """Sets the template raster, to be used for formatting and aligning the outputs."""

        self.template_r = os.path.join(self.directory, "rasters", "depth.tif")


    def _set_resolution(self):
        '''
        sets the resolution of the raster, to be used for calculating the metrics
        '''

        self.resolution = 0.81  # The current resolution of the belgium data, not what the rasters say


    def prepare_optimization(self, msg=1):
        """All work needed to prepare for a optimization to run. Includes checks and data processing."""


        self._set_paths()
        self._load_metrics()
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


    def _set_geo_constraints(self):
        """
        Sets the geospatial constraints for each technology.

        It first determines which tech are considered for each type (ie the two wind turbines, the combined tech)

        This is done in this way so the names from the original 'technologies' can be whatever the user wants

        The 'constraints' list is passed into the pulp solver and works as follows:
            - constraints.append(pulp.LpConstraint(self.x[i][tech], sense=pulp.LpConstraintEQ, rhs=0))
            - This line adds the constraint that if the criteria is met, that technology in that cell must be 0, indicating it is not present

        """

        self.wind_sheets = ["OWF-M", "OWF-J", "MUL_OWF-M_FPV", "MUL_OWF-M_AQC-ML"]
        self.solar_sheets = ["FPV"]  # Note the combined solar/monopile tech is not included here
        self.mussel_sheets = ["AQC-ML", "MUL_OWF-M_AQC-ML"]

        self.wind_names = {key: self.tech_names[key] for key in self.wind_sheets if key in self.tech_names}
        self.solar_names = {key: self.tech_names[key] for key in self.solar_sheets if key in self.tech_names}
        self.mussel_names = {key: self.tech_names[key] for key in self.mussel_sheets if key in self.tech_names}

        self.filtered_wind = {key: value for key, value in self.wind_names.items() if value in self.all_uses}
        self.filtered_solar = {key: value for key, value in self.solar_names.items() if value in self.all_uses}
        self.filtered_mussel = {key: value for key, value in self.mussel_names.items() if value in self.all_uses}

        constraints = []

        for i in self.index:

            # if the value is masked
            if self.data[self.all_uses[0]]["emi"][i] == 0:
                constraints.append(pulp.LpConstraint(self.x[i]["empty"], sense=pulp.LpConstraintEQ, rhs=1))

            # Solar cannot have a depth below 2m or a windspeed of 40m/s +, also must be within 10 km of shore
            if len(self.filtered_solar) > 0:
                for sheet, tech in self.filtered_solar.items():
                    if self.data[tech]["d"][i] <= 2 or self.data[tech]["ws"][i] >= 40:
                        constraints.append(pulp.LpConstraint(self.x[i][tech], sense=pulp.LpConstraintEQ, rhs=0))

                    if self.data[tech]["dsh"][i] >= 10:
                        constraints.append(pulp.LpConstraint(self.x[i][tech], sense=pulp.LpConstraintEQ, rhs=0))

            # Mussels cannot be present if depth is below 10 m
            if len(self.filtered_mussel) > 0:
                for sheet, tech in self.filtered_mussel.items():
                    if self.data[tech]["d"][i] <= 10:
                        constraints.append(pulp.LpConstraint(self.x[i][tech], sense=pulp.LpConstraintEQ, rhs=0))

            # Wind farms cannot be deeper than 60 m, in 'rock' substrate type, and must be 25 km away from shore
            if len(self.filtered_wind) > 0:
                for sheet, tech in self.filtered_wind.items():
                    if self.data[tech]["d"][i] >= 60:
                        constraints.append(pulp.LpConstraint(self.x[i][tech], sense=pulp.LpConstraintEQ, rhs=0))

                    if self.data[tech]["sco"][i] == 1:
                        constraints.append(pulp.LpConstraint(self.x[i][tech], sense=pulp.LpConstraintEQ, rhs=0))

                    if self.data[tech]["dsh"][i] <= 25:
                        constraints.append(pulp.LpConstraint(self.x[i][tech], sense=pulp.LpConstraintEQ, rhs=0))

        return constraints


    def _set_criteria_metrics(self):
        """
        Sets the total and tech metrics.

        Applies any modifications and resolution adjustments
        """

        # Find the value for first number in the unit density column,
        for i, value in enumerate(self.data[list(self.technologies.keys())[0]]["ud"]):
            if value != 0:
                self.first_num = i
                break

        # Calculate criteria for each tech
        for t, info in ((tech, info) for tech, info in self.technologies.items() if info['present']):
            emissions = pulp.lpSum(self.data[t]["emi"][i] * self.x[i][t] for i in self.index)
            mitigation = pulp.lpSum(self.data[t]["mit"][i] * self.x[i][t] for i in self.index)
            value = pulp.lpSum(self.data[t]["val"][i] * self.x[i][t] for i in self.index)
            opr = pulp.lpSum(self.data[t]["opr"][i] * self.x[i][t] for i in self.index)
            cap = pulp.lpSum(self.data[t]["cap"][i] * self.x[i][t] for i in self.index)

            ## THIS IS JUST FOR CHECKING SUBSTATION SENSITIVITY
            # if t == "monopile":
            #     cc = pulp.lpSum(self.data[t]["cc"][i] * self.x[i][t] for i in self.index)


            try:
                ecosystem_benefits = pulp.lpSum(self.data[t]["esb"][i] * self.x[i][t] for i in self.index)
            except:
                ecosystem_benefits = 0

            try:
                energy = pulp.lpSum(self.data[t]["aep"][i] * self.x[i][t] for i in self.index)
            except:
                energy = 0
            try:
                food = pulp.lpSum(self.data[t]["afp"][i] * self.x[i][t] for i in self.index)
            except:
                food = 0

            # Removes the space's, only works if hardcoded like this, should change eventually
            if t == "monopile + mussel":
                t = "mono_mus"
            elif t == "monopile + solar":
                t = "mono_sol"

            # Apply the criteria to respective technologies and modify to account for the resolution
            setattr(self, f"{t}_CO2_emission", emissions * self.resolution)
            setattr(self, f"{t}_value", value * self.resolution)
            setattr(self, f"{t}_opex", opr * self.resolution)
            setattr(self, f"{t}_capex", cap * self.resolution)
            setattr(self, f"{t}_food", food * self.resolution)
            setattr(self, f"{t}_ecosystem_benefits", ecosystem_benefits * self.resolution)

            # Energy is per unit of tech (ie per turbine) so this also adds the unit density per km2, thus getting the correct value
            # Mitigation shouldn't be this way, but the numbers all align when multiplying by the unit density as well
            try:
                setattr(self, f"{t}_CO2_mitigation", mitigation * (self.resolution * self.data[t]["ud"][self.first_num]))
            except:
                if t == "mono_mus":
                    setattr(
                        self,
                        f"{t}_CO2_mitigation",
                        mitigation * (self.resolution * self.data["monopile + mussel"]["ud"][self.first_num]),
                    )

                elif t == "mono_sol":
                    setattr(
                        self,
                        f"{t}_CO2_mitigation",
                        mitigation * (self.resolution * self.data["monopile + solar"]["ud"][self.first_num]),
                    )
            try:
                setattr(self, f"{t}_energy", energy * (self.resolution * self.data[t]["ud"][self.first_num]))
            except:
                if t == "mono_mus":
                    setattr(
                        self,
                        f"{t}_energy",
                        energy * (self.resolution * self.data["monopile + mussel"]["ud"][self.first_num]),
                    )

                elif t == "mono_sol":
                    setattr(
                        self, f"{t}_energy", energy * (self.resolution * self.data["monopile + solar"]["ud"][self.first_num])
                    )

            # Calculate any change in criteria
            for _criteria, _change in info['metric_mods'].items():
                name = f"{t}_{_criteria}"

                current = getattr(self, name, 0)
                modified = current * _change

                setattr(self, name, modified)

            ## THIS IS JUST FOR CHECKING SUBSTATION SENSITIVITY
            # if t == 'monopile':
            #     setattr(self, f"{t}_cables", cc * self.resolution)
            #     self.total_capex += getattr(self, f"{t}_cables", 0)

            # Add to the total values for each criteria
            self.total_CO2_emission += getattr(self, f"{t}_CO2_emission", 0)
            self.total_CO2_mitigation += getattr(self, f"{t}_CO2_mitigation", 0)
            self.total_value += getattr(self, f"{t}_value", 0)
            self.total_opex += getattr(self, f"{t}_opex", 0)
            self.total_capex += getattr(self, f"{t}_capex", 0)
            self.total_ecosystem_benefits += getattr(self, f"{t}_ecosystem_benefits", 0)

            self.total_energy += getattr(self, f"{t}_energy", 0)
            self.total_food += getattr(self, f"{t}_food", 0)
            self.total_CO2 += getattr(self, f"{t}_CO2_emission", 0) - getattr(self, f"{t}_CO2_mitigation", 0)


    def _set_goals(self):
        """
        Sets the goals or constraints. That is any minimum or maximum values for any of the metrics, total or by tech

        Review self.criteria_names for a list of all the possible metrics and their 'full' name
        """

        if len(self.goals) > 0:
            
            self._unit_conversions()

            self.z = []
            self.pen_list = []

            for criteria, info in self.goals_copy.items():

                tech = criteria.split("_")[0]
                goal_type = next(iter(info))
                target = info[goal_type]

                unit = info["unit"]

                if "_units" in criteria:

                    if goal_type == "max":
                        num_cells = target / (self.resolution * self.data[tech]["ud"][self.first_num])
                        self.prob += pulp.lpSum(self.x[i][tech] for i in self.index) <= num_cells, f"Max {tech} units"

                    elif goal_type == "min":
                        num_cells = target / (self.resolution * self.data[tech]["ud"][self.first_num])
                        self.prob += pulp.lpSum(self.x[i][tech] for i in self.index) >= num_cells, f"Min {tech} units"

                    elif goal_type == "exact":
                        num_cells = target / (self.resolution * self.data[tech]["ud"][self.first_num])
                        self.prob += pulp.lpSum(self.x[i][tech] for i in self.index) == num_cells, f"Exact {tech} units"

                    elif goal_type == "range":
                        lower = target[0] / (self.resolution * self.data[tech]["ud"][self.first_num])
                        upper = target[1] / (self.resolution * self.data[tech]["ud"][self.first_num])

                        self.prob += (
                            pulp.lpSum(self.x[i][tech] for i in self.index) >= lower,
                            f"Lower bound {target} {tech} units",
                        )
                        self.prob += (
                            pulp.lpSum(self.x[i][tech] for i in self.index) <= upper,
                            f"Upper bound {target} {tech} units",
                        )

                    elif goal_type == "target":  # BROKEN AVOID AT ALL COSTS STAY AWAY
                        print('NOT YET IMPLEMENTED -- targets for unit amounts')

                else:
                    # Get the information for the constraint

                    # Simple implementation right now
                    if goal_type == "max":
                        self.prob += getattr(self, criteria) <= target, f"Max {criteria} - {target} {unit}"
                    elif goal_type == "min":
                        self.prob += getattr(self, criteria) >= target, f"Min {criteria} - {target} {unit}"
                    elif goal_type == "exact":
                        self.prob += getattr(self, criteria) == target, f"Exact {criteria} - {target} {unit}"
                    elif goal_type == "range":
                        self.prob += getattr(self, criteria) >= target[0], f"Lower bound {criteria} - {target[0]} {unit}"
                        self.prob += getattr(self, criteria) <= target[1], f"Upper bound {criteria} - {target[1]} {unit}"

                    elif goal_type == "target":
                        try:
                            severity = info["penalty"]
                        except:
                            print(f"Penalty not set for {criteria}, defaulting to light")
                            severity = "light"

                        self.has_targets = True
                        z_var = pulp.LpVariable(f'deviation_{criteria}', lowBound=0)

                        self.prob += z_var >= getattr(self, criteria) - target
                        self.prob += z_var >= -(getattr(self, criteria) - target)

                        if severity == 'light':
                            penalty = 100
                        elif severity == 'medium':
                            penalty = 1000
                        elif severity == 'harsh':
                            penalty = 100_000_000

                        self.z.append(z_var)
                        self.pen_list.append(penalty)


    def _out_txt(self):
        """
        writes important info of the run into a txt file that stores info on all iterations

        This has not been updated yet with all the new formatting that was done to the numbers. With the change to incorporate Niko's stuff
        this may become defunct

        """

        with open(os.path.join(self.out_folder, f"{self.run_name}.txt"), "a") as file:

            file.write(f"**********************\n")
            file.write(f"Instance Name: {self.name}\n")
            file.write(f'Technology Modelled: {", ".join(self.technologies.keys())}\n')

            file.write("\nGoals:\n")

            for _criteria, goal in self.goals.items():

                if goal[0] == "min":
                    file.write(f"   {_criteria} > {goal[1]}\n")
                else:
                    file.write(f"   {_criteria} < {goal[1]}\n")

            file.write("\nValue Modifications:\n")

            for tech, change in self.technologies.items():
                for _criteria, mod in change.items():
                    if mod != 1:
                        file.write(f"   {tech} : {_criteria} : {mod}\n")

            # Cell counts
            counter = {tech: 0 for tech in self.all_uses}

            for i in self.index:
                for tech in self.all_uses:
                    if pulp.value(self.x[i][tech]) == 1:
                        counter[tech] += 1

            for tech, count in counter.items():
                file.write(f"\n{tech}: {count} cells")

            # Total values
            file.write(
                f'\n\nEmissions: {"{:,.2f}".format(round(pulp.value(self.total_CO2_emission), 2)).replace(",", " ")}\n'
            )
            file.write(
                f'Mitigation: {"{:,.2f}".format(round(pulp.value(self.total_CO2_mitigation), 2)).replace(",", " ")}\n'
            )
            file.write(f'Operational Cost: {"{:,.2f}".format(round(pulp.value(self.total_opex), 2)).replace(",", " ")}\n')
            file.write(f'Capital Cost: {"{:,.2f}".format(round(pulp.value(self.total_capex), 2)).replace(",", " ")}\n')
            file.write(
                f'Socioeconomic Value: {"{:,.2f}".format(round(pulp.value(self.total_value), 2)).replace(",", " ")}\n'
            )
            file.write(f'Energy Produced: {"{:,.2f}".format(round(pulp.value(self.total_energy), 2)).replace(",", " ")}\n')
            file.write(f'Food Produced: {"{:,.2f}".format(round(pulp.value(self.total_food), 2)).replace(",", " ")}\n')

            file.write(
                f'CO2 Total: {"{:,.2f}".format(round((pulp.value(self.total_CO2_emission) - pulp.value(self.total_CO2_mitigation)), 2)).replace(",", " ")}\n'
            )

            file.write(f"\nStatus: {self.status}\n")
            file.write(f'Optimized Value: {"{:,.2f}".format(round(self.objective, 2)).replace(",", " ")}\n')


    def _majority_filter(self, arr):
        """
        The filter itself.

        Parameters:
            arr {array}: the array to be filtered.
        """
        unique_values, counts = np.unique(arr, return_counts=True)
        majority_value = unique_values[np.argmax(counts)]
        return majority_value


    def _filter_seed(self, size=3):
        """
        Runs the filter on the seed.

        Parameters:
            size {int} the size of the filter, ie 3 = 3x3 grid for filtering.
        """

        self.filtered_seed_array = scipy.ndimage.generic_filter(self.seed_array, self._majority_filter, size=size)


    def _seed_to_gdf(self, to_file=False, filter=True):
        """
        Converts the seed from the LA to a shp for visualization or storage

        Parameters:
            to_file {bool}: Whether or not to convert the gdf to a shapefile. Defaults to False. The output is in the output folder within the wd.
            filter {bool}: Whether or not to run a majority filter over the seed array. Defaults to True.
        """

        # Converts the seed into an array
        self.seed_array = np.array(self.seed).reshape(self.n, self.n)

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

                # Whether or not to use the filtered seed
                if filter:
                    # Filters the seed
                    self._filter_seed()
                    value = self.filtered_seed_array[row, col]
                else:
                    value = self.seed_array[row, col]

                # Applies the value of each cell to the polygons
                if value != self.uDict["empty"]:  # Skip cells designated as empty
                    key = [k for k, v in self.uDict.items() if v == value][0]
                    polygon = box(minx, miny, maxx, maxy, ccw=True)
                    polygons.append((key, polygon))

        gdf = gpd.GeoDataFrame(polygons, columns=["value", "geometry"], crs=crs)

        self.seed_gdf = gdf.dissolve(by="value")
        self.seed_gdf = self.seed_gdf.reset_index()  # Resets the index so that the values are a column

        # Saves the gdf to a shp
        if to_file:

            if '/' in self.run_name:
                json_name = self.run_name.split('/')[1]
            else:
                json_name = self.run_name


            # For the original shp
            os.makedirs(os.path.join(self.out_folder, "SHP"), exist_ok=True)
            self.seed_gdf.to_file(os.path.join(self.out_folder, "SHP", f"{json_name}.geojson"), driver="GeoJSON")

            # For the shp that will be modified with tech removal
            os.makedirs(os.path.join(self.out_folder, "SHP", "mod"), exist_ok=True)
            self.seed_gdf.to_file(os.path.join(self.out_folder, "SHP", "mod", f"{json_name}.geojson"), driver="GeoJSON")


    def _generate_seed(self):
        """
        Creates the list that from the result of the optimizer.
        """

        self.seed = [0] * (self.n * self.n)
        self.seed_names = [0] * (self.n * self.n)
        self.uDict = {key: tech for tech, key in enumerate(self.final_uses)}

        for i in self.index:
            for k in self.final_uses:
                if pulp.value(self.x[i][k]) == 1:
                    self.seed[i] = self.uDict[k]
                    self.seed_names[i] = k


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

        for i in self.index:
            for tech in self.all_uses:
                if pulp.value(self.x[i][tech]) == 1:
                    counter[tech] += 1

        for tech, count in counter.items():
            inst_dict["cell_amounts"][tech] = count

        inst_dict["criteria"] = {}

        inst_dict["criteria"]["emission"] = pulp.value(self.total_CO2_emission)
        inst_dict["criteria"]["mitigation"] = pulp.value(self.total_CO2_mitigation)
        inst_dict["criteria"]["opex"] = pulp.value(self.total_opex)
        inst_dict["criteria"]["capex"] = pulp.value(self.total_capex)
        inst_dict["criteria"]["value"] = pulp.value(self.total_value)
        inst_dict["criteria"]["energy"] = pulp.value(self.total_energy)
        inst_dict["criteria"]["food"] = pulp.value(self.total_food)
        inst_dict["criteria"]["eco_ben"] = pulp.value(self.total_ecosystem_benefits)

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

    # MARK: Optimization implementation methods -- add more as we build them

    def run_linear_optimization(self, run_optimize=True, min_co2=False):
        """
        The parameters for the Linear Algorithm.

        It first prepares the data so that it is in a format for the solver to calculate. Then loads in the criteria metrics and the obejective functions, and lastly applies the constraints*

        It then runs the solver and saves the results to the instance
        """

        # Set simulation parameters
        # print(f'Setting optimization parameters for {self.name}')

        self.first = next(iter(self.data.values()))
        self.n = int(math.sqrt(len(self.first["mit"])))
        self.total_cells = sum(1 for val in self.first["mit"] if not np.isnan(val) or val == 0)
        self.shape = [self.n, self.n]
        self.num = self.n * self.n

        # Set NaN values to 0 because PuLP simply does not like NaN
        for tech, metrics in self.data.items():
            for metric, values in metrics.items():
                self.data[tech][metric] = [val if not np.isnan(val) else 0 for val in values]

        # Set whether to maximize (for the highest 'value' of metrics) or minimize (for the lowest co2) the value.
        if min_co2:
            self.prob = pulp.LpProblem("LA_Solver", pulp.LpMinimize)
        else:
            self.prob = pulp.LpProblem("LA_Solver", pulp.LpMaximize)

        # Rather than creating a set of cells with a tuple, it just creates a 1d list to match the 1d lists of each metric
        # print(f'Creating pulp LP instance for {self.name}')
        self.index = range(self.num)
        self.x = pulp.LpVariable.dicts("Tech", (self.index, self.final_uses), cat="Binary")

        # Criteria Metrics
        self._set_criteria_metrics()

        # Constraints on goals
        self._set_goals()

        # Objective functions:

        if min_co2 == True:  # Minimize co2
            self.prob += self.total_CO2_emission - self.total_CO2_mitigation

        else:  # Maximize energy, food, and socioeconomic value while minimizing ghg and cost

            self.total_added_penalty = pulp.lpSum(z_var * penalty_val for z_var, penalty_val in zip(self.z, self.pen_list))

            if self.has_targets:
                self.prob += (
                    (self.total_value
                    + self.total_ecosystem_benefits
                    - (self.total_capex / 100000)
                    - self.total_opex
                    - (self.total_CO2_emission - self.total_CO2_mitigation)
                    + self.total_energy
                    + self.total_food)
                    - self.total_added_penalty
                )
            else:
                self.prob += (
                    (self.total_value
                    #+ self.total_ecosystem_benefits
                    - (self.total_capex / 100)
                    - self.total_opex
                    #- (self.total_CO2_emission - self.total_CO2_mitigation)
                    + self.total_energy
                    + self.total_food)
                )


        # One tech per cell
        for i in self.index:
            self.prob += pulp.lpSum([self.x[i][k] for k in self.final_uses]) == 1

        # Physiographic constraints
        self.prob.extend(self._set_geo_constraints())

        if run_optimize:
            # print(f'\nBeginning optimization for {self.name}')

            self.prob.solve(pulp.PULP_CBC_CMD(timeLimit=10, msg=self.message))

            self.status = pulp.LpStatus[self.prob.status]
            self.objective = pulp.value(self.prob.objective)

            if self.status == "Optimal":
                print(f"Optimal solution found for {self.name}:", self.objective)
                
            else:
                print(f"Optimization did not find an optimal solution for {self.name} -- {self.status}.")

            self._out_seed()
            if self.status != 'infeasible':
                self._seed_to_gdf(to_file=True, filter=False)
            

    # MARK: Viz and output methods -- need to be universal to all optimization methods

    def _shorten_number(self, number):

        abs_number = abs(number)

        if abs_number < 1_000_000:
            result = f"{self._format_number(int(abs_number))}" + "  "

        elif abs_number < 1_000_000_000:
            result = f"{abs_number / 1_000_000:.2f} M"
            result = result.replace(".", ",")
        elif abs_number < 1_000_000_000_000:
            result = f"{abs_number / 1_000_000_000:.2f} B"
            result = result.replace(".", ",")
        else:
            result = f"{abs_number / 1_000_000_000_000:.2f} T"
            result = result.replace(".", ",")

        return result if number >= 0 else f"-{result}"


    def _format_number(self, number):
        if number == int(number):
            formatted_str = "{:,.0f}".format(number).replace(",", ";").replace(".", ",").replace(";", ".")
        else:
            formatted_str = "{:,.2f}".format(number).replace(",", ";").replace(".", ",").replace(";", ".")
        return formatted_str


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


    def _load_shp(self):
        """
        Loads the shapefiles for the visualization
        """

        self.substations = gpd.read_file(os.path.join(self.directory, "vectors", "substations.geojson"))
        self.aoi_gdf = gpd.read_file(os.path.join(self.directory, "vectors", self.aoi))
        self.cities_gdf = gpd.read_file(os.path.join(self.directory, "vectors", "cities.geojson"))
        self.eez_gdf = gpd.read_file(os.path.join(self.directory, "vectors", "eez.geojson"))
        self._eez_gdf = gpd.read_file(os.path.join(self.directory, "vectors", "eez.geojson"))
        self.shoreline_gdf = gpd.read_file(os.path.join(self.directory, "vectors", "Shoreline.geojson"))
        self.boundaries_gdf = gpd.read_file(os.path.join(self.directory, "vectors", "boundaries.geojson"))


    def _plot_files(self):
        """
        Plots the shapefiles for the visualization
        """
        # Geographic things
        self.shoreline_gdf.plot(ax=self.ax, color="#696969", alpha=0.8)
        self.boundaries_gdf.plot(ax=self.ax, color="black", linewidth=1.5, linestyle="--")

        self.cities_gdf.plot(ax=self.ax, marker="s", color="white", markersize=25, label="Points")
        for x, y, label in zip(self.cities_gdf.geometry.x, self.cities_gdf.geometry.y, self.cities_gdf["name"]):
            self.ax.text(x + 5000, y - 1000, label, fontsize=14, ha="left", color="white", fontfamily="monospace")

        self._eez_gdf.plot(ax=self.ax, color="#1f1f1f", linewidth=4)  # black line behind the eez dashed line to give it some more pop
        self.eez_gdf.plot(ax=self.ax, color="white", linewidth=2, linestyle="--") # dashed line for the eez

        plt.subplots_adjust(left=0, right=0.75, top=0.99, bottom=0.03)


    def _build_text(self):
        # Side text box with values for the run
        text = "----------------------------------\n"
        text += "Constraints\n"
        text += "----------------------------------\n"
        for criteria, goals in self.goals.items():

            goal_type = next(iter(goals))
            value = goals[goal_type]

            unit = goals["unit"]

            text += f"{goal_type} {criteria:<17} "

            if isinstance(value, list):
                text += f"{self._shorten_number(value[0])} to {self._shorten_number(value[1])} {unit}\n"

            else:
                text += f"{self._shorten_number(value):>10} {unit}\n"  # + self.add_unit(_criteria)

        text += "\n\n----------------------------------\n"
        text += "Modifications\n"
        text += "----------------------------------\n"
        for t, info in ((tech, info) for tech, info in self.technologies.items() if info['present']):
            for _criteria, mod in info['metric_mods'].items():
                if mod != 1:
                    combo = t + " " + _criteria
                    text += f"{combo:<17} x {mod}\n"

        text += "\n\n----------------------------------\n"
        text += "Results\n"
        text += "----------------------------------\n"
        text += f"CO2 +  {self._shorten_number(pulp.value(self.total_CO2_emission)):>12} kg CO2/y\n"
        text += f"CO2 -  {self._shorten_number(pulp.value(self.total_CO2_mitigation)):>12} kg CO2/y\n"
        text += f"OPEX   {self._shorten_number(pulp.value(self.total_opex)):>12} €/y\n"
        text += f"CAPEX  {self._shorten_number(pulp.value(self.total_capex)):>12} €\n"
        text += f"Value  {self._shorten_number(pulp.value(self.total_value)):>12} €/y\n"
        text += f"EcoBen {self._shorten_number(pulp.value(self.total_ecosystem_benefits)):>12} €/y\n"
        text += f"Energy {self._shorten_number(pulp.value(self.total_energy)):>12} GWh/y\n"
        text += f"Food   {self._shorten_number(pulp.value(self.total_food)):>12} kg/y\n\n"

        text += "\n\n----------------------------------\n"
        text += "Unit Counts\n"
        text += "----------------------------------\n"
        self.counter = {tech: 0 for tech in self.final_uses}

        for i in self.index:
            for tech in self.final_uses:
                if pulp.value(self.x[i][tech]) == 1:
                    self.counter[tech] += 1

        for tech, count in self.counter.items():

            if tech == "monopile + mussel" and count != 0:
                num_mono = self._format_number(
                    round(count * (self.resolution * self.data["monopile"]["ud"][self.first_num]))
                )
                num_mus = self._format_number(round(count * (self.resolution * self.data["mussel"]["ud"][self.first_num])))
                text += f"{num_mono:8} {'combined monopiles (with AQC)'}\n"
                text += f"{num_mus:8} {'combined aquaculture plots'}\n"

            elif tech == "monopile + solar" and count != 0:
                num_mono = self._format_number(
                    round(count * (self.resolution * self.data["monopile"]["ud"][self.first_num]))
                )
                num_mus = self._format_number(round(count * (self.resolution * self.data["mussel"]["ud"][self.first_num])))
                text += f"{num_mono:8} {'combined monopiles (with FPV)'}\n"
                text += f"{num_mus:8} {'combined FPV platforms'}\n"

            else:
                if tech == "monopile":
                    unit = "turbines (monopile)"
                elif tech == "jacket":
                    unit = "turbines (jacket)"
                elif tech == "solar":
                    unit = "FPV platforms"
                elif tech == "mussel":
                    unit = "aquaculture plots"

                if count != 0 and tech != "empty":
                    num = self._format_number(round(count * (self.resolution * self.data[tech]["ud"][self.first_num])))
                    text += f"{num:8} {unit}\n"

        props = dict(boxstyle="round", facecolor="#1f1f1f", alpha=0)

        if self.has_targets:
            text += "\n\n----------------------------------\n"
            text += "Deviation Penalties\n"
            text += "----------------------------------\n"
            text += f"Total Penalty: {self._shorten_number(pulp.value(self.total_added_penalty))} €\n"

        text += f"\nStatus: {self.status}\n"
        text += f"Optimized Value: {self._shorten_number(self.objective)}\n"

        # Add the text box to the plot
        self.ax.text(
            1.02,
            1.008,
            text,
            transform=self.ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
            fontfamily="monospace",
            color="white",
        )


    def plot_optimal_solution(self):

        # Load the appropriate files
        self._load_shp()

        # Set the formatting for the plot
        mapping = {
                'scale': 'belgium',
                'msp': {
                        'shipping': False,
                        'military': False,
                        'sand_extraction': False,
                        'nature_reserves': False,
                        'energy_zones': False,
                        'energy_zones_type': 'whole_zone',
                        'wind_farms': False,
                        'legacy_farms': False
                    },

                'colours': {
                        'mussel': 'red',
                        'monopile': 'blue',
                        'jacket': 'teal',
                        'solar': 'orange',
                        'monopile + solar': 'purple',
                        'monopile + mussel': 'pink',
                    },

                'legend': [
                        plt.Line2D([0], [0], color='white', linewidth=0, label='----------------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }

        custom_cmap = ListedColormap([mapping["colours"][value] for value in self.seed_gdf["value"].unique()]) # Set the colourmap
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.fig.patch.set_facecolor("#333333")  # Set the background color of the entire figure
        self.ax.set_facecolor("#1f1f1f")  # Set the background color of the plot area (the colour that the ocean ends up)

        # Plot the new tech locations (coloured)
        self.seed_gdf.plot(ax=self.ax, column="value", cmap=custom_cmap)

        self._plot_files()

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
        xlim_min, ylim_min, xlim_max, ylim_max = self.aoi_gdf.total_bounds
        self.ax.set_xlim(xlim_min, xlim_max)
        self.ax.set_ylim(ylim_min, ylim_max)
        self.ax.set_title(self.name)

        # Format the units so they are % 1000
        self.ax.xaxis.set_major_formatter(FuncFormatter(self._utm_formatter))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self._utm_formatter))

        # Create the legend entries for the technologies and combine them with the background entries
        legend_patches = [
            Rectangle((0, 0), 1, 1, color=color, alpha=1, label=label.capitalize())
            for label, color in mapping["colours"].items()
        ]
        all_legend_handles = mapping["legend"] + legend_patches

        # Display the legend
        self.ax.legend(
            handles=all_legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.998, 0.998),
            prop={"family": "monospace", "size": 14},
            labelcolor="white",
            facecolor="#1f1f1f",
            frameon=False,
        )

        self._build_text()

        self.ax.set_aspect("equal")

        plt.show()

        plt.close()