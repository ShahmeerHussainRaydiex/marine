''' 
This is a test file to show how to structure a tech file
Adjust and add as many calcs as required
'''

from src.marine_plan.pre_compute.tech_equations.NID_specs import NID
import numpy as np
import math

class interconnector:
    def __init__(self, geo_data, NID_type=0, verbose=False):

        import numpy as np
        import math
        
        self.NID_type = NID_type
        self.verbose = verbose

        #initialising geo-spatial data dictionary
        self.distance_to_onshore_sub = geo_data['cable_distance'] #km
        self.HVDC_capacity = geo_data['HVDC_capacity']#MW
        self.interconnector_function = geo_data['interconnector_function']
        

        #Converting metrics
        self.short_ton = 907.185 #kg

        #emission factors
        self.emission_factor_MDO	= 3.53 #kgCO2eq/l
        self.emission_factor_HFO	= 3.41 #kgCO2eq/l

        #Belgian carbon intensity
        self.belgian_carbon_intensity = 172 #kgCO2eq/MWh

        #unit density and interdistance
        self.unit_density = 0.5 #units per square km
        self.turbine_rating = 15 #MW
        self.energy_density = self.unit_density*self.turbine_rating #MW/km
        self.square_km_per_turbine = 1/self.unit_density #square km per turbine
        self.interdistance_turbines = np.sqrt(self.square_km_per_turbine) #km

        #lifetime
        self.lifetime = 60 #years

        #---------------------------------- vessel library -----------------------------#
        #support vessel
        self.support_day_rate=	100000
        self.support_mobilization_days=	7
        self.support_fuel_transit	=795.2
        self.support_fuel_working	=397.6
        self.support_transit_speed=	22.7

        #-----------------------------------------------------------------------# 

        self.vars = {} # Calculated variables go in here

        self.final_metrics = {} # Final metrics go in here

    def _calculations(self):

        #---------------------------- cable library -----------------------#

        HVAC_cables = {

            "XLPE_500mm_132kV":{	
                "ac_resistance": 0.06,
                "capacitance":	200,
                "cross_sectional_area":	500,
                "current_capacity":	625,
                "inductance": 0.4,
                "linear_density": 50,
                "rated_voltage": 132,
                "cable_cost_per_km": 400_000, 
                "cable_emissions_per_km": 87_500
                },

            "XLPE_500mm_220kV":{	
                "ac_resistance": 0.03,
                "capacitance":	140,
                "cross_sectional_area":	500,
                "current_capacity":	655,
                "inductance": 0.43,
                "linear_density": 90,
                "rated_voltage": 220,
                "cable_cost_per_km": 650_000, 
                "cable_emissions_per_km": 107_500
                },
            
            "XLPE_630mm_220kV":{	
                "ac_resistance": 0.25,
                "capacitance":	160,
                "cross_sectional_area":	630,
                "current_capacity":	715,
                "inductance": 0.41,
                "linear_density": 96,
                "rated_voltage": 220,
                "cable_cost_per_km": 710_000,
                "cable_emissions_per_km": 117_500
                },

            "XLPE_800mm_220kV":{	
                "ac_resistance": 0.20,
                "capacitance":	170,
                "cross_sectional_area":	800,
                "current_capacity":	775,
                "inductance": 0.40,
                "linear_density": 105,
                "rated_voltage": 220,
                "cable_cost_per_km": 776_000,
                "cable_emissions_per_km": 132_500
                },

            "XLPE_1000mm_220kV":{	
                "ac_resistance": 0.16,
                "capacitance":	190,
                "cross_sectional_area":	1000,
                "current_capacity":	825,
                "inductance": 0.38,
                "linear_density": 115,
                "rated_voltage": 220,
                "cable_cost_per_km": 850_000,
                "cable_emissions_per_km": 150_000
                },

            "XLPE_1200mm_275kV":{	
                "ac_resistance": 0.026,
                "capacitance":	196,
                "cross_sectional_area":	1200,
                "current_capacity":	547,
                "inductance": 0.37,
                "linear_density": 148,
                "rated_voltage": 275,
                "cable_cost_per_km": 1_300_000,
                "cable_emissions_per_km": 185_000
                },

            "XLPE_1600mm_275kV":{	
                "ac_resistance": 0.022,
                "capacitance":	221,
                "cross_sectional_area":	1600,
                "current_capacity":	730,
                "inductance": 0.35,
                "linear_density": 176,
                "rated_voltage": 275,
                "cable_cost_per_km": 1_500_000,
                "cable_emissions_per_km": 215_000
                },

            "XLPE_1900mm_275kV":{	
                "ac_resistance": 0.02,
                "capacitance":	224,
                "cross_sectional_area":	1900,
                "current_capacity":	910,
                "inductance": 0.35,
                "linear_density": 185,
                "rated_voltage": 275,
                "cable_cost_per_km": 1_700_000,
                "cable_emissions_per_km": 260_000
                },

        }

        HVDC_cables = {

            "HVD_2000mm_320kV":{	
                "ac_resistance": 0,
                "capacitance":	295000,
                "cross_sectional_area":	2000,
                "current_capacity":	19000,
                "inductance": 0.127,
                "linear_density": 53,
                "rated_voltage": 320,
                "cable_cost_per_km": 500_000,
                "cable_emissions_per_km": 315_000
                },

            "HVDC_2000mm_400kV":{	
                "ac_resistance": 0,
                "capacitance":	225000,
                "cross_sectional_area":	2000,
                "current_capacity":	1900,
                "inductance": 0.141,
                "linear_density": 59,
                "rated_voltage": 400,
                "cable_cost_per_km": 620_000,
                "cable_emissions_per_km": 355_000
                },
            
            "HVDC_2000mm_525kV":{	
                "ac_resistance": 0,
                "capacitance":	227000,
                "cross_sectional_area":	2500,
                "current_capacity":	1905,
                "inductance": 0.149,
                "linear_density": 74,
                "rated_voltage": 525,
                "cable_cost_per_km": 825_000,
                "cable_emissions_per_km": 400_000
                },

            "HVDC_1200mm_300kV":{	
                "ac_resistance": 0,
                "capacitance":	0,
                "cross_sectional_area":	1200,
                "current_capacity":	1458,
                "inductance": 0,
                "linear_density": 44,
                "rated_voltage": 300,
                "cable_cost_per_km": 835_000,
                "cable_emissions_per_km": 215_000
                },
        }


        #--------------------------- ISLAND CALCULATIONS --------------------------#

        #size of the island will be determined by the connected capacity -> self.connected_capacity
        
        
        ''' 
        HVAC calculation part
        '''

       
        #-------------- electrical costs HVDC -----------------#

        #------------------ CABLES ------------------#
        #We'll assume we use the 1900mm 275kV cable for and then multiply it by how many cables we would need
        #for a certain capacity.

        #overall factors
        excess_cable_factor	= 1.05

        #cable specs
        cable_specs_HVDC = "HVDC_1200mm_300kV" # we always use this cable to estimate the costs.

        ac_resistance_HVDC  = HVDC_cables[cable_specs_HVDC ]["ac_resistance"]
        capacitance_HVDC 	= HVDC_cables[cable_specs_HVDC ]["capacitance"]
        conductor_size_HVDC  = HVDC_cables[cable_specs_HVDC ]["cross_sectional_area"]
        current_capacity_HVDC  = HVDC_cables[cable_specs_HVDC ]["current_capacity"]
        inductance_HVDC 	= HVDC_cables[cable_specs_HVDC ]["inductance"]
        linear_density_HVDC  = HVDC_cables[cable_specs_HVDC ]["linear_density"]
        voltage_HVDC 	= HVDC_cables[cable_specs_HVDC ]["rated_voltage"]
        line_frequency_HVDC  = 60 #Hz
        cable_cost_per_km_HVDC  = HVDC_cables[cable_specs_HVDC ]["cable_cost_per_km"]
        cable_emissions_per_km_HVDC = HVDC_cables[cable_specs_HVDC]["cable_emissions_per_km"]

        #cable sizing equations
        cable_power_HVDC  = current_capacity_HVDC *voltage_HVDC *2/1000
        cable_capacity_HVDC  = cable_power_HVDC 

        #economic variables
        export_cable_cost_HVDC  = cable_cost_per_km_HVDC *((self.distance_to_onshore_sub)*excess_cable_factor) 
        print('export cable cost HVDC  €' +str(f'{export_cable_cost_HVDC :,}')) if self.verbose else None

        #emission variables
        export_cable_emissions_HVDC = cable_emissions_per_km_HVDC*((self.distance_to_onshore_sub)*excess_cable_factor)
        print('export cable emissions HVDC = ' +str(f'{export_cable_emissions_HVDC:,}')) if self.verbose else None
        #----------------------------------------------------------------------------------------------#
        #------------------ electrical component cost ------------------#

        #cost calculation variables
        total_electric_HVDC_cost = 142.61*cable_power_HVDC*1000
        print('total electrical HVDC cost = €' +str(f'{total_electric_HVDC_cost:,}')) if self.verbose else None

        #emission calculation variables
        total_electric_HVDC_emissions = 8788*cable_power_HVDC
        print('total electrical HVDC emissions = ' +str(f'{total_electric_HVDC_emissions:,}')) if self.verbose else None
        #------------------------------------------------------------------#
        #----------------- onshore substation connection ------------------#

        

        interconnect_voltage_HVDC = 380 #kV
        distance_to_interconnect_HVDC = 5 #km
        onshore_substation_base_HVDC	= 6533.1*(voltage_HVDC)
        overhead_transmission_line_cost_HVDC	= 1176*interconnect_voltage_HVDC + 218257*(distance_to_interconnect_HVDC**(-0.1063))*distance_to_interconnect_HVDC
        total_electrical_cost_onshore_HVDC =	total_electric_HVDC_cost

        total_onshore_substation_cost_HVDC = onshore_substation_base_HVDC + overhead_transmission_line_cost_HVDC + total_electrical_cost_onshore_HVDC

        onshore_substation_base_emissions_HVDC	= 750*voltage_HVDC
        overhead_transmission_line_emissions_HVDC	= 750*interconnect_voltage_HVDC + 140_000*(distance_to_interconnect_HVDC**(-0.1063))*distance_to_interconnect_HVDC

        total_electrical_emissions_onshore_HVDC =	total_electric_HVDC_emissions
        
        if self.interconnector_function == 'single':
        
            total_onshore_substation_cost_HVDC = onshore_substation_base_HVDC + overhead_transmission_line_cost_HVDC + total_electrical_cost_onshore_HVDC

            total_onshore_substation_emissions_HVDC = onshore_substation_base_emissions_HVDC + overhead_transmission_line_emissions_HVDC + total_electrical_emissions_onshore_HVDC

        elif self.interconnector_function == 'double':

            total_onshore_substation_cost_HVDC = (onshore_substation_base_HVDC + overhead_transmission_line_cost_HVDC + total_electrical_cost_onshore_HVDC)*2

            total_onshore_substation_emissions_HVDC = (onshore_substation_base_emissions_HVDC + overhead_transmission_line_emissions_HVDC + total_electrical_emissions_onshore_HVDC)*2
        
        elif self.interconnector_function == 'none':

            total_onshore_substation_cost_HVDC = 0

            total_onshore_substation_emissions_HVDC = 0


        #-------------------------------------------------------------------#
        #--------------- development, contingency and insurance ------------#
        pre_capex_HVDC = ((export_cable_cost_HVDC + total_onshore_substation_cost_HVDC)*(self.HVDC_capacity/cable_power_HVDC))
        
        development_cont_insur_factor = 0.075

        total_capex_HVDC = pre_capex_HVDC + (development_cont_insur_factor*pre_capex_HVDC)*2

        yearly_OPEX_HVDC = 1000*self.HVDC_capacity

        #emissions maintenance for electrical equipment
        yearly_maintenance_trips = 5
        working_time = 0.12 #h/MW/maintenance trip
        total_working_time = working_time*self.HVDC_capacity*yearly_maintenance_trips
        fuel_HVDC = total_working_time*self.support_fuel_working
        yearly_OM_emissions_HVDC = fuel_HVDC*self.emission_factor_MDO
        print('yearly OM emissions HVDC = ' +str(f'{yearly_OM_emissions_HVDC:,}')) if self.verbose else None
        
        total_emissions_HVDC = (export_cable_emissions_HVDC + total_onshore_substation_emissions_HVDC)*(self.HVDC_capacity/cable_power_HVDC) 
        
    

        complete_CAPEX = total_capex_HVDC 
        complete_OPEX = 0

        total_emissions = total_emissions_HVDC 
        '''
        installation and decommissioning factor? not included right now (and probably impossible to find)
        '''     

        '''
        Final metrics 
        '''
        # STILL NEED TO DO TIMES THE UNIT DENSITY, THIS IS FOR ONE WHOLE TURBINE.
        self.final_metrics['capex'] = complete_CAPEX
        self.final_metrics['opex'] = complete_OPEX
        self.final_metrics['co2+'] = total_emissions
        self.final_metrics['co2-'] = 0
        self.final_metrics['unit density'] = 1

    def run(self):
        self._calculations()
        return self.final_metrics