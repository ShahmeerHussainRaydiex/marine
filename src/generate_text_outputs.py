import pulp
import pandas as pd
import numpy as np

from src.metric_util_OLD import *
from src.config.landobject_config import north_sea_ports


def LCOE(CAPEX, OPEX, max_Capacity, lifetime, discount_rate, capacity_factor,capacity_deg=0):
    
    #initialise the exponents for LCOE calculation; lifetime of a technology
    #See the general LCOE equation in the attached email. this is exponent 't' 

    Exponent = np.arange(lifetime)
    
    #Calculate the energy term, according to the capacity of the system,
    #the capacity factor and the capacity degredation

    #Energy production/year. Maximum capacity times number of days per year times the capacity factor (=percentage of time you actually have max capacity ( around 60% for windfarm))
    Energy = np.ones(lifetime)*max_Capacity*365*24*capacity_factor

    #Reduction in energy because capacity degeneration. Every year, the wind turbine might lose capacity due to wear and tear. We'll neglect this for now and say that the capacity
    #degredation is zero
    Energy[:] = Energy[:]*((1-capacity_deg)**(Exponent[:]))

    #the final denominator of the LCOE function then becomes:
    Energy[:] = Energy[:]/((1+discount_rate)**(Exponent[:]+1))
   
    #The discount rate is the same as the WACC in our model (weighted averaged cost of capital). It basically the expected return for the investors and the banks. 
    
    #Calculate OPEX according to discount rate
    #TOTAL OPEX per year
    OPEX = OPEX/lifetime #we initialize this definition with the total OPEX remember


    #inital array of pure OPEX costs    
    OPEX = np.ones(lifetime)*(OPEX)   

    #Eventual cost every year for OPEX
    OPEX[:] = OPEX[:]/((1+discount_rate)**(Exponent[:]+1))

    #Cost is the final LCOE (€/MWh)
    Cost = (CAPEX + np.sum(OPEX))/np.sum(Energy)
    
    return Cost #this is in €/MWh


class TextOutputGenerator:
    def __init__(self, marine_plan):
        self.mp = marine_plan
        self.output = ""
        self.CO2_VALUE = 0.072  # euros per kg CO2

    def generate_map_text(self, output_type='summary', install_decom_tracker={}, all_time_tracker={}, metric_tracker={}, capacity_tracker={}, capacity_targets={}, num_format='eu'):
        if output_type == 'summary':
            return self._generate_summary_text()
        elif output_type == 'country':
            return self._generate_country_breakdown_text()
        elif output_type == 'independant_country':
            return self._generate_independant_country_breakdown_text()
        elif output_type == 'technology':
            return self._generate_technology_breakdown_text()
        elif output_type == 'cross':
            return self._generate_cross_analysis_text()
        elif output_type == 'energy_targets':
            return self._generate_energy_targets_text()
        elif output_type == 'interconnectors':
            return self._generate_interconnectors_text()
        elif output_type == 'logistics':
            return self._generate_logistics_text()
        elif output_type == 'grid_connections':
            return self._generate_grid_connections_text()
        elif output_type == 'phasing':
            return self._generate_phasing_text(install_decom_tracker, all_time_tracker, num_format)
        elif output_type == 'cursed_phasing':
            return self._generate_cursed_phasing_text(metric_tracker, capacity_tracker, capacity_targets, num_format)
        else:
            raise ValueError("Invalid output type specified")


    def _generate_summary_text(self):
        text = "----------------------------------\n"
        text += "Results Summary\n"
        text += "----------------------------------\n"
        format_string = "{:<6}{:>20}\n"
        
        # Main metrics
        text += format_string.format("CO2 +", f"{shorten_number(pulp.value(self.mp.total_CO2_emission))} €/y")
        text += format_string.format("CO2 -", f"{shorten_number(pulp.value(self.mp.total_CO2_mitigation))} €/y")
        text += format_string.format("OPEX", f"{shorten_number(pulp.value(self.mp.total_opex))} €/y")
        text += format_string.format("CAPEX", f"{shorten_number(pulp.value(self.mp.total_capex))} €/y")
        text += format_string.format("Value", f"{shorten_number(pulp.value(self.mp.total_revenue))} €/y")
        text += format_string.format("EcoBen", f"{shorten_number(pulp.value(self.mp.total_ecosystem_benefits))} €/y")
        text += format_string.format("Energy", f"{shorten_number(pulp.value(self.mp.total_energy))} GWh/y")
        text += format_string.format("Food", f"{shorten_number(pulp.value(self.mp.total_food))} kg/y")
        text += "\n"

        # Detailed technology breakdown
        text += "----------------------------------\n"
        text += "Technology Breakdown\n"
        text += "----------------------------------\n"
        
        for tech, tech_info in self.mp.technologies.items():
            if tech_info['present']:
                total_cells = sum(self.mp.counter[country][tech] for country in self.mp.countries)
                total_area = total_cells * self.mp.resolution

                if total_area > 0:
                    if tech in ['monopile', 'monopile_mussel', 'monopile_solar']:
                        num_turbines = int(total_area * self.mp.data['monopile']["unit density"][self.mp.first_num])
                        text += f"{format_number_eu(num_turbines)} monopiles covering {format_number_eu(total_area)} km²\n\n"
                    
                    if tech in ['mussel', 'monopile_mussel']:
                        num_mussels = int(total_area * self.mp.data['mussel']["unit density"][self.mp.first_num])
                        text += f"{format_number_eu(num_mussels)} mussel longlines covering {format_number_eu(total_area)} km²\n\n"
                    
                    if tech in ['solar', 'monopile_solar']:
                        num_panels = int(total_area * self.mp.data['solar']["unit density"][self.mp.first_num])
                        text += f"{format_number_eu(num_panels)} solar panels covering {format_number_eu(total_area)} km²\n\n"

        # Basic country contributions
        text += "----------------------------------\n"
        text += "Country Capacities (GW)\n"
        text += "----------------------------------\n"
        format_string = "{:<6}{:>10}{:>10}{:>10}\n"
        text += format_string.format("Country", "Current", "Added", "Total")
        for country in sorted(self.mp.countries):
            current = self.mp.wind_farms_gdf[self.mp.wind_farms_gdf['COUNTRY'] == country]['POWER_MW'].sum() / 1000
            
            # Calculate the number of turbines for this country
            num_turbines = sum(
                int(self.mp.counter[country][tech] * self.mp.resolution * self.mp.data['monopile']["unit density"][self.mp.first_num])
                for tech in ['monopile', 'monopile_mussel', 'monopile_solar']
                if tech in self.mp.technologies and self.mp.technologies[tech]['present']
            )
            
            # Calculate new energy production
            new = num_turbines * 15 / 1000
            total = current + new
            
            text += format_string.format(country, f"{current:.2f}", f"{new:.2f}", f"{total:.2f}")

        text += "----------------------------------\n"
        text += "Fishing Impacts\n"
        text += "----------------------------------\n"
        
        for type, techs in self.mp.fishing_intensities.items():

            if type == 'hours':
                text += "Average Fishing Hours:\n"
                for tech, intensity in techs.items():
                    text += f"  {tech}: {intensity:.2f}\n"
            elif type == 'surface':
                text += "Average Surface Swept Ratio:\n"
                for tech, intensity in techs.items():
                    text += f"  {tech}: {intensity:.2f}%\n"
            elif type == 'subsurface':
                text += "Average Subsurface Swept Ratio:\n"
                for tech, intensity in techs.items():
                    text += f"  {tech}: {intensity:.2f}%\n"
                    
        if self.mp.calc_interconnectors:
            text += "----------------------------------\n"
            text += "LCOE\n"
            text += "----------------------------------\n"
            
            # MEAN lcoe is the average of the LCOE column in self.mp.clusters_gdf
            avg_LCOE = self.mp.cluster_gdf['LCOE'].mean()

            text += f"Average LCOE: {avg_LCOE:.2f} €/MWh\n"

        return text
   

    def _generate_phasing_text(self, install_decom_tracker, all_time_tracker, num_format='eu'):

        text = "----------------------------------\n"
        text += "Optimization Function:\n"
        direction = self.mp.optimization_params['direction'].capitalize()
        
        positives = []
        negatives = []
        

        for category in ['positives', 'negatives']:
            for item, value in self.mp.optimization_params[category].items():
                item = item.replace('total_', '')  # Remove 'total_' prefix
                (positives if category == 'positives' else negatives).append(item)
        
        if positives and negatives:
            function_text = f"{direction}: ({' + '.join(positives)}) - ({' + '.join(negatives)})"
        elif positives:
            function_text = f"{direction}: {' + '.join(positives)}"
        elif negatives:
            function_text = f"{direction}: -({' + '.join(negatives)})"
        else:
            function_text = "No optimization terms defined"


        # Wrap the function text
        wrapped_text = []
        current_line = direction + ": "
        words = function_text[len(direction) + 2:].split()
        for word in words:
            if len(current_line) + len(word) > 34:  # 34 is the approximate width of the dashed line
                wrapped_text.append(current_line)
                current_line = "  " + word  # Add two spaces for indentation
            else:
                current_line += " " + word if current_line.strip() else word
        wrapped_text.append(current_line)

        text += "\n".join(wrapped_text)

        text += f"\nOptimizing for {self.mp.capacity_needed} GW of total energy\n"

        text += "\n----------------------------------\n"

        if self.mp.coop:
            text += "Cooperation: YES\n"
        else:
            text += "Cooperation: NO\n"

        total_achieved = 0
        total_turbines = 0

        for country in sorted(set(self.mp.countries) & set(self.mp.north_sea_offshore_wind.keys())):

            num_turbines = sum(
                int(self.mp.counter[country][tech] * self.mp.resolution * self.mp.data[tech]["unit density"][self.mp.first_num])
                for tech in ['monopile', 'jacket', 'semisub_cat_drag', 'semisub_taut_driv', 'semisub_taut_suc', 'spar_cat_drag', 'spar_taut_driv', 'spar_taut_suc']
                if tech in self.mp.technologies and self.mp.technologies[tech]['present']
            )

            model = num_turbines * 15 / 1000
            total_achieved += model
            total_turbines += num_turbines

        # Implementation for energy targets assessment
        # text += "----------------------------------\n"
        # text += "Results for the New Technologies\n"
        # text += "----------------------------------\n"
        # format_string = "{:<10}{:>12} {:<5}\n"
        

        # def format_number(value, decimals=2):
        #     return f"{value:.{decimals}f}"
        #     return f"{value:.{decimals}f}".replace('.', ',')
        
        # text += format_string.format("OPEX", shorten_number(pulp.value(self.mp.total_opex), num_format), "€/y")
        # text += format_string.format("CAPEX", shorten_number(pulp.value(self.mp.total_capex), num_format), "€/y")
        # text += format_string.format("T_CAPEX", shorten_number(pulp.value(self.mp.total_capex)*25, num_format), "€")
        # text += format_string.format("Revenue", shorten_number(pulp.value(self.mp.total_revenue), num_format), "€/y")
        # text += format_string.format("LCOE", format_number(pulp.value(self.mp.total_LCOE)/self.mp.needed_turbines), "€/MWh")
        # text += "\n"
        # text += format_string.format("Energy", shorten_number(pulp.value(self.mp.total_energy_produced), num_format), "GWh/y")
        # text += format_string.format("Food", shorten_number(pulp.value(self.mp.total_food_produced), num_format), "kg/y")
        # text += "\n"
        # text += format_string.format("CO2 +", shorten_number(pulp.value(self.mp.total_CO2_emission), num_format), "€/y")
        # text += format_string.format("CO2 -", shorten_number(pulp.value(self.mp.total_CO2_mitigation), num_format), "€/y")
        # text += format_string.format("Net CO2", shorten_number(pulp.value(self.mp.total_CO2_net), num_format), "€/y")
        # text += format_string.format("Eco Impact", shorten_number(pulp.value(self.mp.total_eco_sensitivity)*9_000_000, num_format), "")
    
        # Existing technologies

        text += "----------------------------------\n"
        text += "Summary of Capacities\n"
        text += "----------------------------------\n"

        existing_capacity_GW = self.mp.existing_tech_gdf['POWER_MW'].sum() / 1000

        modelled_capacity_GW = total_achieved

        text += f"Existing capacity: {existing_capacity_GW:.2f} GW\n"
        text += f"Added capacity: {modelled_capacity_GW:.2f} GW\n"
        text += f"Combined total capacity: {existing_capacity_GW + modelled_capacity_GW:.2f} GW\n"

        text += "----------------------------------\n"
        text += "New Installations and Decommissioning:\n"
        text += "----------------------------------\n"

        format_string = "{:<25}{:>12}\n"
        
        # Installation section
        text += "Installation:\n"
        text += format_string.format("New Turbines:", f"{shorten_number(install_decom_tracker['new_turbines'], num_format)}")
        text += format_string.format("New Foundations:", f"{shorten_number(install_decom_tracker['new_foundations'], num_format)}")
        text += format_string.format("Turbine Install Cost:", f"{shorten_number(install_decom_tracker['turbine_install_cost'], num_format)} €")
        text += format_string.format("Foundation Install Cost:", f"{shorten_number(install_decom_tracker['foundation_install_cost'], num_format)} €")
        text += format_string.format("Turbine Install Emis.:", f"{shorten_number(install_decom_tracker['turbine_install_emissions'], num_format)} €")
        text += format_string.format("Foundation Install Emis.:", f"{shorten_number(install_decom_tracker['foundation_install_emissions'], num_format)} €")
        text += "\n"

        # Decommissioning section
        text += "Decommissioning:\n"
        text += format_string.format("Decom. Turbines:", f"{shorten_number(install_decom_tracker['decomissioned_turbines'], num_format)}")
        text += format_string.format("Decom. Foundations:", f"{shorten_number(install_decom_tracker['decomissioned_foundations'], num_format)}")
        text += format_string.format("Turbine Decom Cost:", f"{shorten_number(install_decom_tracker['turbine_decom_cost'], num_format)} €")
        text += format_string.format("Foundation Decom Cost:", f"{shorten_number(install_decom_tracker['foundation_decom_cost'], num_format)} €")
        text += format_string.format("Turbine Decom Emis.:", f"{shorten_number(install_decom_tracker['turbine_decom_emissions'], num_format)} €")
        text += format_string.format("Foundation Decom Emis.:", f"{shorten_number(install_decom_tracker['foundation_decom_emissions'], num_format)} €")

        # Calculate and display totals
        total_install_cost = install_decom_tracker['turbine_install_cost'] + install_decom_tracker['foundation_install_cost']
        total_decom_cost = install_decom_tracker['turbine_decom_cost'] + install_decom_tracker['foundation_decom_cost']
        total_install_emissions = install_decom_tracker['turbine_install_emissions'] + install_decom_tracker['foundation_install_emissions']
        total_decom_emissions = install_decom_tracker['turbine_decom_emissions'] + install_decom_tracker['foundation_decom_emissions']

        text += "\nInstance Totals:\n"
        text += format_string.format("Install Cost:", f"{shorten_number(total_install_cost, num_format)} €")
        text += format_string.format("Decom Cost:", f"{shorten_number(total_decom_cost, num_format)} €")
        text += format_string.format("Install Emissions:", f"{shorten_number(total_install_emissions, num_format)} €")
        text += format_string.format("Decom Emissions:", f"{shorten_number(total_decom_emissions, num_format)} €")

        text += "----------------------------------\n"
        text += "To Date Statistics\n"
        text += "----------------------------------\n"

        total_install_cost = all_time_tracker['turbine_install_cost'] + all_time_tracker['foundation_install_cost']
        total_decom_cost = all_time_tracker['turbine_decom_cost'] + all_time_tracker['foundation_decom_cost']
        total_install_emissions = all_time_tracker['turbine_install_emissions'] + all_time_tracker['foundation_install_emissions']
        total_decom_emissions = all_time_tracker['turbine_decom_emissions'] + all_time_tracker['foundation_decom_emissions']

        text += format_string.format("Installed Turbines:", f"{shorten_number(all_time_tracker['new_turbines'], num_format)}")
        text += format_string.format("Installed Foundations:", f"{shorten_number(all_time_tracker['new_foundations'], num_format)}")
        text += format_string.format("Decom. Turbines:", f"{shorten_number(all_time_tracker['decomissioned_turbines'], num_format)}")
        text += format_string.format("Decom. Foundations:", f"{shorten_number(all_time_tracker['decomissioned_foundations'], num_format)}")

        text += "\n"
        text += format_string.format("Install Cost:", f"{shorten_number(total_install_cost, num_format)} €")
        text += format_string.format("Decom Cost:", f"{shorten_number(total_decom_cost, num_format)} €")
        text += format_string.format("Install Emissions:", f"{shorten_number(total_install_emissions, num_format)} €")
        text += format_string.format("Decom Emissions:", f"{shorten_number(total_decom_emissions, num_format)} €")

        total_costs = total_install_cost + total_decom_cost
        total_emissions = total_install_emissions + total_decom_emissions

        text += "\n"
        text += format_string.format("Total Cost:", f"{shorten_number(total_costs, num_format)} €")
        text += format_string.format("Total Emissions:", f"{shorten_number(total_emissions, num_format)} €")

        
        return text

    
    def _generate_country_breakdown_text(self):
        text = "----------------------------------\n"
        text += "Country Breakdown\n"
        text += "----------------------------------\n"
        
        format_string = "{:<15}{:>20}\n"
        sub_format = "  {:<13}{:>20}\n"
        
        for country in sorted(self.mp.countries):
            capex = pulp.value(self.mp.country_criteria[country]['capex'])
            energy = pulp.value(self.mp.country_criteria[country]['energy'])  # Convert to GWh
            
            # Skip this country if it has no CAPEX and no energy production (indicating no contribution)
            if capex == 0 and energy == 0:
                continue
            
            text += f"Country: {country}\n"
            
            # Energy
            text += format_string.format("Energy", f"{energy:.2f} GWh/y")
            
            # Food
            food = pulp.value(self.mp.country_criteria[country]['food'])
            text += format_string.format("Food", f"{shorten_number(food)} kg/y")
            
            # CO2 (now in euros per year)
            co2_emission = pulp.value(self.mp.country_criteria[country]['CO2_emission'])
            co2_mitigation = pulp.value(self.mp.country_criteria[country]['CO2_mitigation'])
            text += format_string.format("CO2 Emissions", f"{shorten_number(co2_emission)} €/y")
            text += format_string.format("CO2 Mitigation", f"{shorten_number(co2_mitigation)} €/y")
            
            # Financials
            opex = pulp.value(self.mp.country_criteria[country]['opex'])
            value = pulp.value(self.mp.country_criteria[country]['revenue'])
            eco_benefits = pulp.value(self.mp.country_criteria[country]['ecosystem_benefits'])
            
            text += format_string.format("CAPEX", f"{shorten_number(capex)} €/y")
            text += format_string.format("OPEX", f"{shorten_number(opex)} €/y")
            text += format_string.format("Value", f"{shorten_number(value)} €/y")
            text += format_string.format("Eco Benefits", f"{shorten_number(eco_benefits)} €/y")
            
            # CO2 balance (now in euros per year)
            co2_balance = co2_emission - co2_mitigation
            text += format_string.format("CO2 Balance", f"{shorten_number(co2_balance)} €/y")
            
            # Technology breakdown for this country
            text += "\nTechnology Breakdown:\n"
            for tech, count in self.mp.counter[country].items():
                if count > 0:
                    area = count * self.mp.resolution
                    if tech in ['monopile', 'monopile_mussel', 'monopile_solar']:
                        num_turbines = int(area * self.mp.data['monopile']["unit density"][self.mp.first_num])
                        text += sub_format.format(tech, f"{num_turbines} turbines")
                    elif tech in ['mussel', 'monopile_mussel']:
                        num_mussels = int(area * self.mp.data['mussel']["unit density"][self.mp.first_num])
                        text += sub_format.format(tech, f"{num_mussels} longlines")
                    elif tech in ['solar', 'monopile_solar']:
                        num_panels = int(area * self.mp.data['solar']["unit density"][self.mp.first_num])
                        text += sub_format.format(tech, f"{num_panels} panels")
                    text += sub_format.format("", f"{area:.2f} km²")
            
            text += "----------------------------------\n"
        
        return text


    def _generate_independant_country_breakdown_text(self):
        # Implementation for energy targets assessment
        text = "----------------------------------\n"
        text += "Results\n"
        text += "----------------------------------\n"
        format_string = "{:<6}{:>12} {:>5}\n"
        
        text += format_string.format("CO2 +", shorten_number(pulp.value(self.mp.total_CO2_emission)), "€/y")
        text += format_string.format("CO2 -", shorten_number(pulp.value(self.mp.total_CO2_mitigation)), "€/y")
        text += format_string.format("OPEX", shorten_number(pulp.value(self.mp.total_opex)), "€/y")
        text += format_string.format("CAPEX", shorten_number(pulp.value(self.mp.total_capex)), "€/y")
        text += format_string.format("Value", shorten_number(pulp.value(self.mp.total_revenue)), "€/y")
        text += format_string.format("EcoSens", shorten_number(pulp.value(self.mp.total_eco_sensitivity)), "")
        text += format_string.format("Energy", shorten_number(pulp.value(self.mp.total_energy_produced)), "GWh/y")
        text += format_string.format("Food", shorten_number(pulp.value(self.mp.total_food_produced)), "kg/y")
        text += "\n"

        if self.mp.energy_targets:
            text += "----------------------------------\n\n"
            text += "----------------------------------\n"
            text += "Individual Country Generation (GW) \n"
            text += "----------------------------------\n"

            total_achieved = 0
            total_turbines = 0
            target = self.mp.capacity_needed
            text += f"A total of {target} GW is required for {self.mp.optimizing_country }\n"

            for country in sorted(set(self.mp.countries) & set(self.mp.north_sea_offshore_wind.keys())):
                  
                num_turbines = sum(
                    int(self.mp.counter[country][tech] * self.mp.resolution * self.mp.data['monopile']["unit density"][self.mp.first_num])
                    for tech in ['monopile', 'jacket', 'semisub_cat_drag', 'semisub_taut_driv', 'semisub_taut_suc', 'spar_cat_drag', 'spar_taut_driv', 'spar_taut_suc']
                    if tech in self.mp.technologies and self.mp.technologies[tech]['present']
                )

                model = num_turbines * 15 / 1000
                total_achieved += model
                total_turbines += num_turbines
                text += f"{country} produces {model:.2f} GW\n"
    
            text += "----------------------------------\n"

            upfront_capex = pulp.value(self.mp.total_capex) * 25
            yearly_opex = pulp.value(self.mp.total_opex)
    

            #avg_LCOE = LCOE(upfront_capex, yearly_opex*25, total_achieved*1000, 25, WACC, capacity_factor)
            avg_LCOE = pulp.value(self.mp.total_LCOE) / total_turbines


            text += f"CAPEX: {shorten_number(upfront_capex)} €\n"
            text += f"Yearly OPEX: {shorten_number(yearly_opex)} €/y\n"
            text += f"Average LCOE: {avg_LCOE:,.2f} €/MWh\n".replace(',', 'X').replace('.', ',').replace('X', '.')
            text += f'Eco Sensitivity: {shorten_number(pulp.value(self.mp.total_eco_sensitivity))}\n'

        return text


    def _generate_technology_breakdown_text(self):
        text = "----------------------------------\n"
        text += "Technology Breakdown\n"
        text += "----------------------------------\n"
        
        format_string = "{:<15}{:>20}\n"
        sub_format = "  {:<13}{:>20}\n"
        
        for tech, tech_info in self.mp.technologies.items():
            if tech_info['present']:
                capex = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['capex']) for country in self.mp.countries)
                energy = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['energy']) for country in self.mp.countries)  
                
                # Skip this technology if it has no CAPEX and no energy production
                if capex == 0 and energy == 0:
                    continue
                
                text += f"Technology: {tech}\n"
                
                # Energy
                text += format_string.format("Energy", f"{energy:.2f} GWh/y")
                
                # Food
                food = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['food']) for country in self.mp.countries)
                text += format_string.format("Food", f"{shorten_number(food)} kg/y")
                
                # CO2 (in euros per year)
                co2_emission = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['CO2_emission']) for country in self.mp.countries)
                co2_mitigation = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['CO2_mitigation']) for country in self.mp.countries)
                text += format_string.format("CO2 Emissions", f"{shorten_number(co2_emission)} €/y")
                text += format_string.format("CO2 Mitigation", f"{shorten_number(co2_mitigation)} €/y")
                
                # Financials
                opex = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['opex']) for country in self.mp.countries)
                value = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['revenue']) for country in self.mp.countries)
                eco_benefits = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['ecosystem_benefits']) for country in self.mp.countries)
                
                text += format_string.format("CAPEX", f"{shorten_number(capex)} €/y")
                text += format_string.format("OPEX", f"{shorten_number(opex)} €/y")
                text += format_string.format("Value", f"{shorten_number(value)} €/y")
                text += format_string.format("Eco Benefits", f"{shorten_number(eco_benefits)} €/y")
                
                # Derived metrics
                if energy > 0:
                    cost_per_gwh = (capex + opex) / energy
                    text += sub_format.format("Cost per GWh", f"{shorten_number(cost_per_gwh)} €/GWh")
                    
                    co2_balance_per_gwh = (co2_emission - co2_mitigation) / energy
                    text += sub_format.format("CO2 per GWh", f"{shorten_number(co2_balance_per_gwh)} €/GWh")
                
                if food > 0:
                    cost_per_kg_food = (capex + opex) / food
                    text += sub_format.format("Cost per kg food", f"{shorten_number(cost_per_kg_food)} €/kg")
                
                # CO2 balance (in euros per year)
                co2_balance = co2_emission - co2_mitigation
                text += format_string.format("CO2 Balance", f"{shorten_number(co2_balance)} €/y")
                
                # Units and area
                total_cells = sum(self.mp.counter[country][tech] for country in self.mp.countries)
                total_area = total_cells * self.mp.resolution
                text += "\nUnits and Area:\n"
                
                if tech in ['monopile', 'monopile_mussel', 'monopile_solar']:
                    num_turbines = int(total_area * self.mp.data['monopile']["unit density"][self.mp.first_num])
                    text += sub_format.format("Turbines", f"{num_turbines}")
                
                if tech in ['mussel', 'monopile_mussel']:
                    num_mussels = int(total_area * self.mp.data['mussel']["unit density"][self.mp.first_num])
                    text += sub_format.format("Mussel lines", f"{num_mussels}")
                
                if tech in ['solar', 'monopile_solar']:
                    num_panels = int(total_area * self.mp.data['solar']["unit density"][self.mp.first_num])
                    text += sub_format.format("Solar panels", f"{num_panels}")
                
                text += sub_format.format("Total area", f"{total_area:.2f} km²")
                
                # Breakdown by country
                text += "\nBreakdown by country:\n"
                for country in sorted(self.mp.countries):
                    cells = self.mp.counter[country][tech]
                    if cells > 0:
                        area = cells * self.mp.resolution
                        country_text = f"  {country}: "
                        
                        if tech in ['monopile', 'monopile_mussel', 'monopile_solar']:
                            num_turbines = int(area * self.mp.data['monopile']["unit density"][self.mp.first_num])
                            country_text += f"{num_turbines} turbines, "
                        
                        if tech in ['mussel', 'monopile_mussel']:
                            num_mussels = int(area * self.mp.data['mussel']["unit density"][self.mp.first_num])
                            country_text += f"{num_mussels} mussel lines, "
                        
                        if tech in ['solar', 'monopile_solar']:
                            num_panels = int(area * self.mp.data['solar']["unit density"][self.mp.first_num])
                            country_text += f"{num_panels} panels, "
                        
                        country_text += f"{area:.2f} km²"
                        text += country_text + "\n"
                
                text += "----------------------------------\n"  # Add an extra newline for readability between technologies
        
        return text


    def _generate_energy_targets_text(self):

        text = "----------------------------------\n"
        text += "Optimization Function:\n"
        direction = self.mp.optimization_params['direction'].capitalize()
        
        positives = []
        negatives = []
        
        for category in ['positives', 'negatives']:
            for item, value in self.mp.optimization_params[category].items():
                item = item.replace('total_', '')  # Remove 'total_' prefix
                (positives if category == 'positives' else negatives).append(item)
        
        if positives and negatives:
            function_text = f"{direction}: ({' + '.join(positives)}) - ({' + '.join(negatives)})"
        elif positives:
            function_text = f"{direction}: {' + '.join(positives)}"
        elif negatives:
            function_text = f"{direction}: -({' + '.join(negatives)})"
        else:
            function_text = "No optimization terms defined"

        # Wrap the function text
        wrapped_text = []
        current_line = direction + ": "
        words = function_text[len(direction) + 2:].split()
        for word in words:
            if len(current_line) + len(word) > 34:  # 34 is the approximate width of the dashed line
                wrapped_text.append(current_line)
                current_line = "  " + word  # Add two spaces for indentation
            else:
                current_line += " " + word if current_line.strip() else word
        wrapped_text.append(current_line)

        text += "\n".join(wrapped_text)
        text += "\n----------------------------------\n"

        if self.mp.coop:
            text += "Cooperation: YES\n"
        else:
            text += "Cooperation: NO\n"


        # Implementation for energy targets assessment
        text += "----------------------------------\n"
        text += "Results\n"
        text += "----------------------------------\n"
        format_string = "{:<10}{:>12} {:<5}\n"
        
        def format_number(value, decimals=2):
            return f"{value:.{decimals}f}".replace('.', ',')
        

        
        text += format_string.format("OPEX", shorten_number(pulp.value(self.mp.total_opex)), "€/y")
        text += format_string.format("CAPEX", shorten_number(pulp.value(self.mp.total_capex)), "€/y")
        text += format_string.format("T_CAPEX", shorten_number(pulp.value(self.mp.total_capex)*25), "€")
        text += format_string.format("Revenue", shorten_number(pulp.value(self.mp.total_revenue)), "€/y")
        text += format_string.format("LCOE", format_number(pulp.value(self.mp.total_LCOE)/self.mp.needed_turbines), "€/MWh")
        text += "\n"
        text += format_string.format("Energy", shorten_number(pulp.value(self.mp.total_energy_produced)), "GWh/y")
        text += format_string.format("Food", shorten_number(pulp.value(self.mp.total_food_produced)), "kg/y")
        text += "\n"
        
        text += format_string.format("CO2 +", shorten_number(pulp.value(self.mp.total_CO2_emission)), "€/y")
        text += format_string.format("CO2 -", shorten_number(pulp.value(self.mp.total_CO2_mitigation)), "€/y")
        text += format_string.format("Net CO2", shorten_number(pulp.value(self.mp.total_CO2_net)), "€/y")
        text += format_string.format("Eco Impact", shorten_number(pulp.value(self.mp.total_eco_sensitivity)*9_000_000), "€/y")
        
        text += "\n"

        if self.mp.energy_targets:
            text += "----------------------------------\n\n"
            text += "----------------------------------\n"
            text += "Comparison with Energy Targets (GW)\n"
            text += "These are the constraints\n"
            text += "----------------------------------\n"
            text += f"{'Country':<6} {'Target':>8} {'Achieved':>8} {'Diff':>8}\n"

            total_target = 0
            total_achieved = 0
            total_turbines = 0

            for country in sorted(set(self.mp.countries) & set(self.mp.north_sea_offshore_wind.keys())):
                target = self.mp.north_sea_offshore_wind[country][2050]

                num_turbines = sum(
                    int(self.mp.counter[country][tech] * self.mp.resolution * self.mp.data[tech]["unit density"][self.mp.first_num])
                    for tech in ['monopile', 'jacket', 'semisub_cat_drag', 'semisub_taut_driv', 'semisub_taut_suc', 'spar_cat_drag', 'spar_taut_driv', 'spar_taut_suc']
                    if tech in self.mp.technologies and self.mp.technologies[tech]['present']
                )

                model = num_turbines * 15 / 1000

                achieved = model
                difference = achieved - target
                text += f"{country:<6} {target:8.2f} {achieved:8.2f} {difference:8.2f}\n"
                total_target += target
                total_achieved += achieved

                total_turbines += num_turbines

            text += "----------------------------------\n"
            total_difference = total_achieved - total_target
            text += f"{'Total':<6} {total_target:8.2f} {total_achieved:8.2f} {total_difference:8.2f}\n"

            text += "----------------------------------\n"
            upfront_capex = pulp.value(self.mp.total_capex) * 25
            yearly_opex = pulp.value(self.mp.total_opex)
        

            if self.mp.hubs_included:
                
                planned_capacity = self.mp.hubs_gdf['capacity'].sum()
                actual_capacity = self.mp.hubs_gdf['connected_capacity'].sum()
                ratio = actual_capacity / planned_capacity

                interconnector_CAPEX = self.mp.cable_lines_gdf['capex'].sum() * ratio # just a broad assumption that the cost of the cable will rise proportionally with the capacity
                interconnector_OPEX = self.mp.cable_lines_gdf['opex'].sum() * ratio
                interconnector_emissions = self.mp.cable_lines_gdf['co2+'].sum() * 0.072 * ratio
                # farm_to_hub_cost = self.mp.farm_to_hub['capex'].sum() * ratio
                interconnecter_total_cost = interconnector_CAPEX + interconnector_OPEX + interconnector_emissions


                hub_CAPEX = self.mp.hubs_gdf['capex'].sum()
                hub_OPEX = self.mp.hubs_gdf['opex'].sum()
                hub_emissions = self.mp.hubs_gdf['co2+'].sum() * 0.072
                hub_total_cost = hub_CAPEX + hub_OPEX + hub_emissions
                
                system_cost = interconnecter_total_cost + hub_total_cost

                text += f"Cable CAPEX: {shorten_number(interconnector_CAPEX)} €\n"
                text += f"Cable OPEX: {shorten_number(interconnector_OPEX)} €\n"
                text += f"Cable Emissions: {shorten_number(interconnector_emissions)} €\n"
                text += f"Cable Total Cost: {shorten_number(interconnecter_total_cost)} €\n"

                text += f"\nHub CAPEX: {shorten_number(hub_CAPEX)} €\n"
                text += f"Hub OPEX: {shorten_number(hub_OPEX)} €\n"
                text += f"Hub Emissions: {shorten_number(hub_emissions)} €\n"
                text += f"Hub Total Cost: {shorten_number(hub_total_cost)} €\n"

                text += f"\nSystem Cost: {shorten_number(system_cost)} €\n"

                text += "----------------------------------\n"

                upfront_capex += interconnecter_total_cost


        return text



    def _generate_cursed_phasing_text(self, metric_tracker, capacity_tracker, capacity_targets, num_format='eu'):

        text = "----------------------------------\n"
        text += "Optimization Function:\n"
        direction = self.mp.optimization_params['direction'].capitalize()
        
        positives = []
        negatives = []
        
        for category in ['positives', 'negatives']:
            for item, value in self.mp.optimization_params[category].items():
                item = item.replace('total_', '')  # Remove 'total_' prefix
                (positives if category == 'positives' else negatives).append(item)
        
        if positives and negatives:
            function_text = f"{direction}: ({' + '.join(positives)}) - ({' + '.join(negatives)})"
        elif positives:
            function_text = f"{direction}: {' + '.join(positives)}"
        elif negatives:
            function_text = f"{direction}: -({' + '.join(negatives)})"
        else:
            function_text = "No optimization terms defined"

        # Wrap the function text
        wrapped_text = []
        current_line = direction + ": "
        words = function_text[len(direction) + 2:].split()
        for word in words:
            if len(current_line) + len(word) > 34:  # 34 is the approximate width of the dashed line
                wrapped_text.append(current_line)
                current_line = "  " + word  # Add two spaces for indentation
            else:
                current_line += " " + word if current_line.strip() else word
        wrapped_text.append(current_line)

        text += "\n".join(wrapped_text)
        text += "\n----------------------------------\n"

        if self.mp.coop:
            text += "Cooperation: YES\n"
        else:
            text += "Cooperation: NO\n"


        # Implementation for energy targets assessment
        text += "----------------------------------\n"
        text += "Results\n"
        text += "----------------------------------\n"
        format_string = "{:<10}{:>12} {:<5}\n"
        
        def format_number(value, decimals=2):
            return f"{value:.{decimals}f}".replace('.', ',')
        

        total_turbines = sum(capacity_tracker.values()) * 1000 / 15

        
        text += format_string.format("OPEX", shorten_number(metric_tracker['opex']), "€/y")
        text += format_string.format("CAPEX", shorten_number(metric_tracker['capex']), "€/y")
        text += format_string.format("T_CAPEX", shorten_number(metric_tracker['capex']*25), "€")
        text += format_string.format("Revenue", shorten_number(metric_tracker['revenue']), "€/y")
        text += format_string.format("LCOE", format_number(metric_tracker['LCOE']/total_turbines), "€/MWh")
        text += "\n"
        text += format_string.format("Energy", shorten_number(metric_tracker['energy_produced']), "GWh/y")
        text += format_string.format("Food", shorten_number(metric_tracker['food_produced']), "kg/y")
        text += "\n"
        
        text += format_string.format("CO2 +", shorten_number(metric_tracker['co2+']*self.CO2_VALUE), "€/y")
        text += format_string.format("CO2 -", shorten_number(metric_tracker['co2-']*self.CO2_VALUE), "€/y")

        net_co2 = metric_tracker['co2-'] - metric_tracker['co2+']

        text += format_string.format("Net CO2", shorten_number(net_co2*self.CO2_VALUE), "€/y")
        text += format_string.format("Eco Impact", shorten_number(metric_tracker['eco_sensitivity']*9_000_000), "€/y")
        
        text += "\n"

        if self.mp.energy_targets:
            text += "----------------------------------\n\n"
            text += "----------------------------------\n"
            text += "Comparison with Energy Targets (GW)\n"
            text += "These are the constraints\n"
            text += "----------------------------------\n"
            text += f"{'Country':<6} {'Target':>8} {'Achieved':>8} {'Diff':>8}\n"

            for country, capacity in capacity_tracker.items():
                target = capacity_targets[country]
                difference = capacity - target
                text += f"{country:<6} {target:8.2f} {capacity:8.2f} {difference:8.2f}\n"


            total_achieved = sum(capacity_tracker.values())
            total_target = sum(capacity_targets.values())

            text += "----------------------------------\n"
            total_difference = total_achieved - total_target
            text += f"{'Total':<6} {total_target:8.2f} {total_achieved:8.2f} {total_difference:8.2f}\n"
    
        return text







    def _generate_interconnectors_text(self):
        text = "----------------------------------\n"
        text += "Results Summary\n"
        text += "----------------------------------\n"
        format_string = "{:<6}{:>20}\n"
        
        # Main metrics
        text += format_string.format("CO2 +", f"{shorten_number(pulp.value(self.mp.total_CO2_emission))} €/y")
        text += format_string.format("CO2 -", f"{shorten_number(pulp.value(self.mp.total_CO2_mitigation))} €/y")
        text += format_string.format("OPEX", f"{shorten_number(pulp.value(self.mp.total_opex))} €/y")
        text += format_string.format("CAPEX", f"{shorten_number(pulp.value(self.mp.total_capex))} €/y")
        text += format_string.format("Value", f"{shorten_number(pulp.value(self.mp.total_revenue))} €/y")
        text += format_string.format("EcoBen", f"{shorten_number(pulp.value(self.mp.total_ecosystem_benefits))} €/y")
        text += format_string.format("Energy", f"{shorten_number(pulp.value(self.mp.total_energy))} GWh/y")
        text += format_string.format("Food", f"{shorten_number(pulp.value(self.mp.total_food))} kg/y")
        text += "\n"

        text += "----------------------------------\n"
        text += "Interconnectors\n"
        text += "----------------------------------\n"

        total_capacity = 0
        total_foundation_cost = 0
        total_cable_cost = 0

        for interconnector, data in self.mp.summary_dict.items():
            if interconnector.startswith('Interconnector'):
                text += f"\n{interconnector}:\n"
                
                # Incoming capacity
                if data['incoming_capacity']:
                    for country, capacity in data['incoming_capacity'].items():
                        text += f"  Incoming capacity from {country}: {capacity:.2f} GW\n"
                
                # Outgoing capacity
                if data['outgoing_capacity']:
                    for country, capacity in data['outgoing_capacity'].items():
                        text += f"  Outgoing capacity to {country}: {capacity:.2f} GW\n"
                
                # Total capacity
                total_capacity += data['total_capacity']
                text += f"  Total capacity: {data['total_capacity']:.2f} GW\n"
                
                # Foundation cost
                foundation_cost = data['foundation_cost']
                total_foundation_cost += foundation_cost
                text += f"  Foundation cost: €{foundation_cost:,.2f}\n"
                
                # Cable cost
                cable_cost = data['cable_cost']
                total_cable_cost += cable_cost
                text += f"  Cable cost: €{cable_cost:,.2f}\n"

        # Summary
        text += "\nSummary:\n"
        text += f"Total interconnector capacity: {total_capacity:.2f} GW\n"
        text += f"Total foundation cost: €{total_foundation_cost:,.2f}\n"
        text += f"Total cable cost: €{total_cable_cost:,.2f}\n"
        total_cost = total_foundation_cost + total_cable_cost
        text += f"Total interconnector cost: €{total_cost:,.2f}\n"

        text += "----------------------------------\n"
        upfront_capex = pulp.value(self.mp.total_capex) * 25
        upfront_capex_interconnectors = total_cost + upfront_capex
        text += f"Upfront CAPEX: {shorten_number(upfront_capex)} €\n"
        text += f"Upfront CAPEX with interconnectors: {shorten_number(upfront_capex_interconnectors)} €\n"


        return text


    def _generate_logistics_text(self):
        text = "----------------------------------\n"
        text += "Results Summary\n"
        text += "----------------------------------\n"
        format_string = "{:<6}{:>20}\n"
        
        # Main metrics
        text += format_string.format("CO2 +", f"{shorten_number(pulp.value(self.mp.total_CO2_emission))} €/y")
        text += format_string.format("CO2 -", f"{shorten_number(pulp.value(self.mp.total_CO2_mitigation))} €/y")
        text += format_string.format("OPEX", f"{shorten_number(pulp.value(self.mp.total_opex))} €/y")
        text += format_string.format("CAPEX", f"{shorten_number(pulp.value(self.mp.total_capex))} €/y")
        text += format_string.format("Value", f"{shorten_number(pulp.value(self.mp.total_revenue))} €/y")
        text += format_string.format("EcoBen", f"{shorten_number(pulp.value(self.mp.total_ecosystem_benefits))} €/y")
        text += format_string.format("Energy", f"{shorten_number(pulp.value(self.mp.total_energy))} GWh/y")
        text += format_string.format("Food", f"{shorten_number(pulp.value(self.mp.total_food))} kg/y")
        text += "\n"

        text += "----------------------------------\n"
        text += "Logistics\n"
        text += "----------------------------------\n"

        text += self.mp._calc_logistics()

        return text
    

    def _generate_grid_connections_text(self):
        text = "----------------------------------\n"
        text += "Grid Connections Analysis\n"
        text += "----------------------------------\n"

        # Initialize substation count for each country
        substations = {country: 0 for country in self.mp.countries}

        # Count substations for each country
        for locations in north_sea_ports.values():
            for location in locations.values():
                if location['designation'] == 'substation':
                    country = location['country']
                    if country in substations:
                        substations[country] += 1

        # Define capacity scenarios
        scenarios = [1, 3, 6]  # GW per connection

        for country in sorted(self.mp.countries):
            #current = self.mp.wind_farms_gdf[self.mp.wind_farms_gdf['COUNTRY'] == country]['POWER_MW'].sum() / 1000

            num_turbines = sum(
                int(self.mp.counter[country][tech] * self.mp.resolution * self.mp.data[tech]["unit density"][self.mp.first_num])
                for tech in ['monopile', 'jacket']
                if tech in self.mp.technologies and self.mp.technologies[tech]['present']
            )
            model = num_turbines * 15 / 1000
            combined = model

            text += f"\nCountry: {country}\n"
            text += f"Energy Capacity: {combined:.2f} GW\n"
            text += f"Existing Substations: {substations[country]}\n"

            for scenario in scenarios:
                # Calculate required grid connections for this scenario
                required_connections = math.ceil(combined / scenario)

                # Calculate additional substations needed
                additional_substations = max(0, required_connections - substations[country])

                text += f"\n  Scenario: {scenario} GW per connection\n"
                text += f"  Required Grid Connections: {required_connections}\n"
                text += f"  Additional Substations Needed: {additional_substations}\n"

            text += "----------------------------------\n"

        return text


    def _add_unit(self, criteria):
        if criteria in ["co2", "mit", "emi"]:
            return " kg CO2/y\n"
        elif criteria in ["cap", "cost"]:
            return " €\n"
        elif criteria == "food":
            return " kg/y\n"
        elif criteria in ["opr", "val"]:
            return " €/y\n"
        elif "energy" in criteria:
            return " GWh/y\n"
        else:
            return "\n"


    def generate_full_report(self):
        self._add_overall_summary()
        self._add_country_breakdown()
        self._add_technology_breakdown()
        self._add_cross_analysis()
        self._add_optimization_details()
        return self.output


    def _add_section_header(self, title):
        self.output += f"\n{'=' * 50}\n{title}\n{'=' * 50}\n"


    def _add_overall_summary(self):
        self._add_section_header("Overall Summary")
        self.output += f"Total Energy Produced: {self._format_number_eu(pulp.value(self.mp.total_energy)/1000)}{self._add_unit('energy')}"
        self.output += f"Total Food Produced: {self._format_number_eu(pulp.value(self.mp.total_food))}{self._add_unit('food')}"
        self.output += f"Total CO2 Emissions: {self._format_number_eu(pulp.value(self.mp.total_CO2_emission))}{self._add_unit('emi')}"
        self.output += f"Total CO2 Mitigation: {self._format_number_eu(pulp.value(self.mp.total_CO2_mitigation))}{self._add_unit('mit')}"
        total_co2_emission = pulp.value(self.mp.total_CO2_emission)
        total_co2_mitigation = pulp.value(self.mp.total_CO2_mitigation)
        co2_balance = total_co2_emission - total_co2_mitigation
        co2_monetary_value = co2_balance * self.CO2_VALUE
        self.output += f"Total CO2 Balance: {self._format_number_eu(co2_balance)}{self._add_unit('co2')}"
        self.output += f"CO2 Monetary Value: {self._format_number_eu(co2_monetary_value)} €\n"   
        self.output += f"Total CAPEX: {self._format_number_eu(pulp.value(self.mp.total_capex))}{self._add_unit('cap')}"
        self.output += f"Total OPEX: {self._format_number_eu(pulp.value(self.mp.total_opex))}{self._add_unit('opr')}"
        self.output += f"Total Value: {self._format_number_eu(pulp.value(self.mp.total_revenue))}{self._add_unit('val')}"
        self.output += f"Total Ecosystem Benefits: {self._format_number_eu(pulp.value(self.mp.total_ecosystem_benefits))}{self._add_unit('val')}"


    def _add_country_breakdown(self):
        self._add_section_header("Country Breakdown")
        for country in self.mp.countries:
            self.output += f"\nCountry: {country}\n"
            energy = pulp.value(self.mp.country_criteria[country]['energy'])
            food = pulp.value(self.mp.country_criteria[country]['food'])
            capex = pulp.value(self.mp.country_criteria[country]['capex'])
            opex = pulp.value(self.mp.country_criteria[country]['opex'])
            co2_emission = pulp.value(self.mp.country_criteria[country]['CO2_emission'])
            co2_mitigation = pulp.value(self.mp.country_criteria[country]['CO2_mitigation'])
            value = pulp.value(self.mp.country_criteria[country]['revenue'])
            eco_benefits = pulp.value(self.mp.country_criteria[country]['ecosystem_benefits'])
            
            self.output += f"Energy Produced: {self._format_number_eu(energy)}{self._add_unit('energy')}"
            self.output += f"Food Produced: {self._format_number_eu(food)}{self._add_unit('food')}"
            self.output += f"CO2 Emissions: {self._format_number_eu(co2_emission)}{self._add_unit('emi')}"
            self.output += f"CO2 Mitigation: {self._format_number_eu(co2_mitigation)}{self._add_unit('mit')}"
            self.output += f"CAPEX: {self._format_number_eu(capex)}{self._add_unit('cap')}"
            self.output += f"OPEX: {self._format_number_eu(opex)}{self._add_unit('opr')}"
            self.output += f"Value: {self._format_number_eu(value)}{self._add_unit('val')}"
            self.output += f"Ecosystem Benefits: {self._format_number_eu(eco_benefits)}{self._add_unit('val')}"
            
            # Calculate and add derived metrics
            cost_per_gwh = (capex + opex) / energy if energy > 0 else 0
            cost_per_kg_food = (capex + opex) / food if food > 0 else 0
            co2_balance_per_gwh = (co2_emission - co2_mitigation) / energy if energy > 0 else 0
            
            self.output += f"Cost per GWh: {self._format_number_eu(cost_per_gwh)} €/GWh\n"
            self.output += f"Cost per kg of food: {self._format_number_eu(cost_per_kg_food)} €/kg\n"
            self.output += f"CO2 balance per GWh: {self._format_number_eu(co2_balance_per_gwh)} kg CO2/GWh\n"

            co2_balance = co2_emission - co2_mitigation
            co2_monetary_value = co2_balance * self.CO2_VALUE
            
            self.output += f"CO2 Balance: {self._format_number_eu(co2_balance)}{self._add_unit('co2')}"
            self.output += f"CO2 Monetary Value: {self._format_number_eu(co2_monetary_value)} €\n"


    def _add_technology_breakdown(self):
        self._add_section_header("Technology Breakdown")
        for tech, tech_info in self.mp.technologies.items():
            if tech_info['present']:
                capex = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['capex']) for country in self.mp.countries)
                if capex == 0:
                    continue  # Skip this technology if CAPEX is 0

                self.output += f"\nTechnology: {tech}\n"
                energy = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['energy']) for country in self.mp.countries)
                food = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['food']) for country in self.mp.countries)
                opex = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['opex']) for country in self.mp.countries)
                co2_emission = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['CO2_emission']) for country in self.mp.countries)
                co2_mitigation = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['CO2_mitigation']) for country in self.mp.countries)
                value = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['revenue']) for country in self.mp.countries)
                eco_benefits = sum(pulp.value(self.mp.country_tech_criteria[country][tech]['ecosystem_benefits']) for country in self.mp.countries)
                
                self.output += f"Energy Produced: {self._format_number_eu(energy)}{self._add_unit('energy')}"
                self.output += f"Food Produced: {self._format_number_eu(food)}{self._add_unit('food')}"
                self.output += f"CO2 Emissions: {self._format_number_eu(co2_emission)}{self._add_unit('emi')}"
                self.output += f"CO2 Mitigation: {self._format_number_eu(co2_mitigation)}{self._add_unit('mit')}"
                self.output += f"CAPEX: {self._format_number_eu(capex)}{self._add_unit('cap')}"
                self.output += f"OPEX: {self._format_number_eu(opex)}{self._add_unit('opr')}"
                self.output += f"Value: {self._format_number_eu(value)}{self._add_unit('val')}"
                self.output += f"Ecosystem Benefits: {self._format_number_eu(eco_benefits)}{self._add_unit('val')}"
                
                # Calculate and add derived metrics
                cost_per_gwh = (capex + opex) / energy if energy > 0 else 0
                cost_per_kg_food = (capex + opex) / food if food > 0 else 0
                co2_balance_per_gwh = (co2_emission - co2_mitigation) / energy if energy > 0 else 0
                
                self.output += f"Cost per GWh: {self._format_number_eu(cost_per_gwh)} €/GWh\n"
                self.output += f"Cost per kg of food: {self._format_number_eu(cost_per_kg_food)} €/kg\n"
                self.output += f"CO2 balance per GWh: {self._format_number_eu(co2_balance_per_gwh)} kg CO2/GWh\n"
                co2_balance = co2_emission - co2_mitigation
                co2_monetary_value = co2_balance * self.CO2_VALUE
                
                self.output += f"CO2 Balance: {self._format_number_eu(co2_balance)}{self._add_unit('co2')}"
                self.output += f"CO2 Monetary Value: {self._format_number_eu(co2_monetary_value)} €\n"
                
                # Units
                total_cells = sum(self.mp.counter[country][tech] for country in self.mp.countries)

                # Calculate total area
                self.output += f"Total number of cells: {self._format_number_eu(total_cells)}\n"

                total_area = total_cells * self.mp.resolution
                self.output += f"Total area: {self._format_number_eu(total_area)} km²\n"

                # Breakdown by country
                self.output += "Breakdown by country:\n"
                for country in self.mp.countries:
                    cells = self.mp.counter[country][tech]
                    area = cells * self.mp.resolution
                    
                    output_line = f"  {country}: "
                    
                    if tech in ['monopile', 'monopile_mussel', 'monopile_solar']:
                        num_turbines = int(area * self.mp.data['monopile']["unit density"][self.mp.first_num])
                        output_line += f"{self._format_number_eu(num_turbines)} turbines, "
                    
                    if tech in ['mussel', 'monopile_mussel']:
                        num_mussels = int(area * self.mp.data['mussel']["unit density"][self.mp.first_num])
                        output_line += f"{self._format_number_eu(num_mussels)} mussel longlines, "
                    
                    if tech in ['solar', 'monopile_solar']:
                        num_panels = int(area * self.mp.data['solar']["unit density"][self.mp.first_num])
                        output_line += f"{self._format_number_eu(num_panels)} panels, "
                    
                    output_line += f"{self._format_number_eu(area)} km²\n"
                    
                    self.output += output_line


    def _add_cross_analysis(self):
        self._add_section_header("Cross Analysis: Country and Technology")
        for country in self.mp.countries:
            for tech, tech_info in self.mp.technologies.items():
                if tech_info['present']:
                    capex = pulp.value(self.mp.country_tech_criteria[country][tech]['capex'])
                    if capex == 0:
                        continue  # Skip this technology for this country if CAPEX is 0

                    self.output += f"\nCountry: {country}, Technology: {tech}\n"
                    energy = pulp.value(self.mp.country_tech_criteria[country][tech]['energy'])
                    food = pulp.value(self.mp.country_tech_criteria[country][tech]['food'])
                    opex = pulp.value(self.mp.country_tech_criteria[country][tech]['opex'])
                    co2_emission = pulp.value(self.mp.country_tech_criteria[country][tech]['CO2_emission'])
                    co2_mitigation = pulp.value(self.mp.country_tech_criteria[country][tech]['CO2_mitigation'])
                    value = pulp.value(self.mp.country_tech_criteria[country][tech]['revenue'])
                    eco_benefits = pulp.value(self.mp.country_tech_criteria[country][tech]['ecosystem_benefits'])
                    
                    self.output += f"Energy Produced: {self._format_number_eu(energy)}{self._add_unit('energy')}"
                    self.output += f"Food Produced: {self._format_number_eu(food)}{self._add_unit('food')}"
                    self.output += f"CO2 Emissions: {self._format_number_eu(co2_emission)}{self._add_unit('emi')}"
                    self.output += f"CO2 Mitigation: {self._format_number_eu(co2_mitigation)}{self._add_unit('mit')}"
                    self.output += f"CAPEX: {self._format_number_eu(capex)}{self._add_unit('cap')}"
                    self.output += f"OPEX: {self._format_number_eu(opex)}{self._add_unit('opr')}"
                    self.output += f"Value: {self._format_number_eu(value)}{self._add_unit('val')}"
                    self.output += f"Ecosystem Benefits: {self._format_number_eu(eco_benefits)}{self._add_unit('val')}"
                    
                    # Calculate and add derived metrics
                    cost_per_gwh = (capex + opex) / energy if energy > 0 else 0
                    cost_per_kg_food = (capex + opex) / food if food > 0 else 0
                    co2_balance_per_gwh = (co2_emission - co2_mitigation) / energy if energy > 0 else 0
                    
                    self.output += f"Cost per GWh: {self._format_number_eu(cost_per_gwh)} €/GWh\n"
                    self.output += f"Cost per kg of food: {self._format_number_eu(cost_per_kg_food)} €/kg\n"
                    self.output += f"CO2 balance per GWh: {self._format_number_eu(co2_balance_per_gwh)} kg CO2/GWh\n"

                    co2_balance = co2_emission - co2_mitigation
                    co2_monetary_value = co2_balance * self.CO2_VALUE
                    
                    self.output += f"CO2 Balance: {self._format_number_eu(co2_balance)}{self._add_unit('co2')}"
                    self.output += f"CO2 Monetary Value: {self._format_number_eu(co2_monetary_value)} €\n"


    def _add_optimization_details(self):
        self._add_section_header("Optimization Details")
        self.output += f"Objective Function Value: {self._format_number_eu(self.mp.objective)}\n"
        self.output += f"Optimization Status: {self.mp.status}\n"
        # Add constraints and their status
        # for name, constraint in self.mp.prob.constraints.items():
        #     self.output += f"Constraint {name}: {constraint.sense} {self._format_number_eu(constraint.constant)}, Status: {'Satisfied' if constraint.slack == 0 else 'Not Satisfied'}\n"


    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.output)