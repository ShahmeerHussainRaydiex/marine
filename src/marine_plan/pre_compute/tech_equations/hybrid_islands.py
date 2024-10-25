''' 
This is a test file to show how to structure a tech file
Adjust and add as many calcs as required
'''

from NID_specs import NID
import numpy as np
import math

class hybrid_islands:
    def __init__(self, geo_data, NID_type=0, verbose=False):

        import numpy as np
        import math
        
        self.NID_type = NID_type
        self.verbose = verbose

        #initialising geo-spatial data dictionary
        self.depth = geo_data['depth'] #m
        self.distance_to_onshore_sub = geo_data['distance_to_onshore_sub'] #km
        self.distance_to_hydrogen_sub = geo_data['distance_to_hydrogen_sub'] #km
        self.HVAC_capacity = geo_data['HVAC_capacity'] #MW
        self.HVDC_capacity = geo_data['HVDC_capacity']#MW
        self.H2_capacity = geo_data['H2_capacity']  #MW
        

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

        if self.HVAC_capacity > 0:
            #the electrical cost is than determined with the same logic as for substations

            #-------------- electrical costs -----------------#

            #------------------ CABLES ------------------#
            #We'll assume we use the 1900mm 275kV cable for and then multiply it by how many cables we would need
            #for a certain capacity.

            #overall factors
            excess_cable_factor	= 1.05

            #cable specs

            cable_specs_HVAC = "XLPE_1900mm_275kV" #we'll always use this cable to calculate


            ac_resistance_HVAC = HVAC_cables[cable_specs_HVAC]["ac_resistance"]
            capacitance_HVAC	= HVAC_cables[cable_specs_HVAC]["capacitance"]
            conductor_size_HVAC = HVAC_cables[cable_specs_HVAC]["cross_sectional_area"]
            current_capacity_HVAC = HVAC_cables[cable_specs_HVAC]["current_capacity"]
            inductance_HVAC	= HVAC_cables[cable_specs_HVAC]["inductance"]
            linear_density_HVAC = HVAC_cables[cable_specs_HVAC]["linear_density"]
            voltage_HVAC	= HVAC_cables[cable_specs_HVAC]["rated_voltage"]
            line_frequency_HVAC = 60 #Hz
            cable_cost_per_km_HVAC = HVAC_cables[cable_specs_HVAC]["cable_cost_per_km"]
            cable_emissions_per_km_HVAC = HVAC_cables[cable_specs_HVAC]["cable_emissions_per_km"]

            #cable sizing equations
            conductance_HVAC = 1/ac_resistance_HVAC
            cable_num_HVAC = complex(ac_resistance_HVAC,2 * 3.14 * line_frequency_HVAC *inductance_HVAC,)
            cable_den_HVAC = complex(conductance_HVAC,2 * 3.14 *line_frequency_HVAC*capacitance_HVAC,)
            char_impedance_HVAC = np.sqrt(cable_num_HVAC/cable_den_HVAC)
            phase_angle_HVAC	= math.atan(np.imag(char_impedance_HVAC) / np.real(char_impedance_HVAC))
            power_factor_HVAC = math.cos(phase_angle_HVAC)
            cable_power_HVAC = (np.sqrt(3)*voltage_HVAC*current_capacity_HVAC*power_factor_HVAC/ 1000)
            cable_capacity_HVAC = cable_power_HVAC

            #economic variables
            export_cable_cost_HVAC = cable_cost_per_km_HVAC*((self.distance_to_onshore_sub)*excess_cable_factor)
            print('export cable cost €' +str(f'{export_cable_cost_HVAC:,}')) if self.verbose else None

            #emission variables
            export_cable_emissions_HVAC = cable_emissions_per_km_HVAC*((self.distance_to_onshore_sub)*excess_cable_factor)
            print('export cable emissions €' +str(f'{export_cable_emissions_HVAC:,}')) if self.verbose else None
            #----------------------------------------------------------------------------------------------#

            #------------------ electrical component cost HVAC ------------------#
            #cost calculation variables
            power_factor = 0.95

            real_power = cable_power_HVAC #MW
            rated_apparent_power = cable_power_HVAC/power_factor #MW
            mvar = np.sqrt((rated_apparent_power)**2 - (real_power)**2) #MVAr
            print(real_power)
            print(rated_apparent_power)
            print(mvar)
                
            #cost calculations electric components HVAC	
            circuit_breakers = 818.42*voltage_HVAC
            ac_switchgear = 14_018*voltage_HVAC
            transformer = 11_879*voltage_HVAC
            shunt_reactor = 35_226*mvar
            series_capacitor = 22_047*mvar
            static_var_compensator = 105_060*mvar

            total_electric_HVAC_cost = circuit_breakers + ac_switchgear + transformer + shunt_reactor + series_capacitor + static_var_compensator
            print('total electrical HVAC cost = €' +str(f'{total_electric_HVAC_cost:,}')) if self.verbose else None

            #emissions electric componentns HVAC
            circuit_breakers_emissions = 400*voltage_HVAC
            ac_switchgear_emissions = 200*voltage_HVAC
            transformer_emissions = 2500*rated_apparent_power
            shunt_reactor_emissions = 12_500*mvar
            series_capacitor_emissions = 8500*mvar
            static_var_compensator_emissions = 17_500*mvar

            total_electric_HVAC_emissions = circuit_breakers_emissions + ac_switchgear_emissions + transformer_emissions + shunt_reactor_emissions + series_capacitor_emissions + static_var_compensator_emissions
            print('total electrical HVAC emissions = ' +str(f'{total_electric_HVAC_emissions:,}')) if self.verbose else None
            #----------------------------------------------------------------------------------------------#
            #---------------------------- Fixed costs --------------------------#

            diesel_generator_backup	= 1_000_000
            workshop_accomodation_fire_protection = 2_000_000
            ancillary_cost = 3_000_000
            
            fixed_substation_costs = diesel_generator_backup + workshop_accomodation_fire_protection + ancillary_cost 
            print('fixed costs = €' +str(f'{fixed_substation_costs:,}')) if self.verbose else None

            #------------------------------------------------------------------#
            #----------------- onshore substation connection ------------------#

            interconnect_voltage_HVAC = 380 #kV
            distance_to_interconnect_HVAC = 5 #km
            onshore_substation_base_HVAC	= 6533.1*(voltage_HVAC)
            overhead_transmission_line_cost_HVAC	= 1176*interconnect_voltage_HVAC + 218257*(distance_to_interconnect_HVAC**(-0.1063))*distance_to_interconnect_HVAC
            total_electrical_cost_onshore_HVAC =	total_electric_HVAC_cost

            total_onshore_substation_cost_HVAC = onshore_substation_base_HVAC + overhead_transmission_line_cost_HVAC + total_electrical_cost_onshore_HVAC
            print('onshore sub cost = €' +str(f'{total_onshore_substation_cost_HVAC:,}')) if self.verbose else None

            onshore_substation_base_emissions_HVAC	= 750*voltage_HVAC
            overhead_transmission_line_emissions_HVAC	= 750*interconnect_voltage_HVAC + 140_000*(distance_to_interconnect_HVAC**(-0.1063))*distance_to_interconnect_HVAC
            total_electrical_emissions_onshore_HVAC =	total_electric_HVAC_emissions

            total_onshore_substation_emissions_HVAC = onshore_substation_base_emissions_HVAC + overhead_transmission_line_emissions_HVAC + total_electrical_emissions_onshore_HVAC
            print('total onshore substation emissions = ' +str(f'{total_onshore_substation_emissions_HVAC:,}')) if self.verbose else None
            #-------------------------------------------------------------------#
            #--------------- development, contingency and insurance ------------#
            pre_capex_HVAC = ((export_cable_cost_HVAC + total_electric_HVAC_cost + total_onshore_substation_cost_HVAC)*(self.HVAC_capacity/cable_power_HVAC) +
                                fixed_substation_costs)
            
            development_cont_insur_factor = 0.075

            total_capex_HVAC = pre_capex_HVAC + (development_cont_insur_factor*pre_capex_HVAC)*2
            print('total capex HVAC = €' +str(f'{total_capex_HVAC:,}')) if self.verbose else None

            yearly_OPEX_HVAC = 1000*self.HVAC_capacity

            #emissions maintenance for electrical equipment
            yearly_maintenance_trips = 5
            working_time = 0.12 #h/MW/maintenance trip
            total_working_time = working_time*self.HVAC_capacity*yearly_maintenance_trips
            fuel_HVAC = total_working_time*self.support_fuel_working
            yearly_OM_emissions_HVAC = fuel_HVAC*self.emission_factor_MDO
            print('yearly OM emissions HVAC = ' +str(f'{yearly_OM_emissions_HVAC:,}')) if self.verbose else None
            
            total_emissions_HVAC = (export_cable_emissions_HVAC + total_electric_HVAC_emissions + total_onshore_substation_emissions_HVAC)*(self.HVAC_capacity/cable_power_HVAC) + yearly_OM_emissions_HVAC*self.lifetime

        else:
            total_capex_HVAC = 0    
            yearly_OPEX_HVAC = 0
            total_emissions_HVAC = 0

        '''
        HVDC cost calculation
        '''

        if self.HVDC_capacity > 0:
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

            total_onshore_substation_emissions_HVDC = onshore_substation_base_emissions_HVDC + overhead_transmission_line_emissions_HVDC + total_electrical_emissions_onshore_HVDC
            #-------------------------------------------------------------------#
            #--------------- development, contingency and insurance ------------#
            pre_capex_HVDC = ((export_cable_cost_HVDC + total_electric_HVDC_cost + total_onshore_substation_cost_HVDC)*(self.HVDC_capacity/cable_power_HVDC))
            
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
            
            total_emissions_HVDC = (export_cable_emissions_HVDC + total_electric_HVDC_emissions + total_onshore_substation_emissions_HVDC)*(self.HVDC_capacity/cable_power_HVDC) + yearly_OM_emissions_HVDC*self.lifetime
            
        else:
            total_capex_HVDC = 0   
            yearly_OPEX_HVDC = 0
            total_emissions_HVDC = 0

            #----------------------------------------------------------------------------------#
    
        '''
        Hydrogen part
        '''

        if self.H2_capacity > 0:

            #total capacity of the hydrogen station, will be percentage of the hydrogen later to convert to cost per turbine
            capacity_hydrogen_station = self.H2_capacity #MW

            #------------------------------------#

            #all the needed components for hydrogen production (there will probably be some missing but this is the best we can do)

            #electrolyzer (PEM)
            PEM_cost_per_kW = 3000 #€/kW
            PEM_emissions_per_kW = 65 #kgCO2eq/kW
            electricty_consumption_PEM = 57.1 #kWh/kg

            total_PEM_cost = PEM_cost_per_kW*capacity_hydrogen_station*1000
            total_PEM_emissions = PEM_emissions_per_kW*capacity_hydrogen_station*1000
            print('PEM emissions = ' +str(f'{total_PEM_emissions:,}')) if self.verbose else None
            
            #desalination system (2000l/h)
            desalination_cost_per_system = 800_000 #cost per system
            desalination_emissions_per_system = 700 #kgCO2eq per system
            electricity_consumption_desalination = 0.06 #kWh/kg
            water_per_kg_hydrogen = 16 #l

            #compressor system 
            electricity_consumption_compressor = 4.45 #kWh/kg
            compressor_cost_per_kg_per_hour = 10_000 #cost per produced kg per hour
            compressor_emissions_per_kg_per_hour = 3500 #kgCO2eq per produced kg per hour

            #Total kg that can be produced per hour -> you install for the max production specs, this will often times not accur (around 60% = capacity factor of the offshore windfarm)
            max_kg_per_hour = capacity_hydrogen_station*1000/(electricity_consumption_compressor + electricity_consumption_desalination + electricty_consumption_PEM) #kg/h
            #print(max_kg_per_hour) if self.verbose else None

            #given the max kilogram produced per hour we know the dimensions of our desalination and compressor system
            total_desalination_cost = desalination_cost_per_system*(max_kg_per_hour*water_per_kg_hydrogen/2000) #units that can do 2000l per hour
            #print(total_desalination_cost)if self.verbose else None
            total_compressor_cost = compressor_cost_per_kg_per_hour*max_kg_per_hour
            #print(total_compressor_cost)if self.verbose else None
            #print(3500*1_000_000) if self.verbose else None

            total_desalination_emissions = desalination_emissions_per_system*(max_kg_per_hour*water_per_kg_hydrogen/2000) #units that can do 2000l per hour
            print('desalination emissions = ' +str(f'{total_desalination_emissions:,}')) if self.verbose else None
            total_compressor_emissions = compressor_emissions_per_kg_per_hour*max_kg_per_hour
            print('compressor emissions = ' +str(f'{total_compressor_emissions:,}')) if self.verbose else None
            
            #total cost for assembly of these components (guess)
            assembly_cost = 0.08*(total_PEM_cost + total_desalination_cost + total_compressor_cost)
            assembly_emissions = 0.08*(total_PEM_emissions + total_desalination_emissions + total_compressor_emissions)
            #-----------------------------------------------#

            #hardware cost for a 12-inch hydrogen pipeline
            pipeline_cost_per_km = 700_000 #€/km
            pipeline_emissions_per_km = 250_000 #kgCO2eq/km

            #transport capacity of pipeline
            pipeline_transport_capacity = 10_000 #kg/h

            #to make it easier on ourselves, we'll do a cost per km per kg hydrogen so we don't get stuck in minimal and maximal needed pipelines.
            pipeline_cost_per_km_per_kg = pipeline_cost_per_km/pipeline_transport_capacity
            pipeline_emissions_per_km_per_kg = pipeline_emissions_per_km/pipeline_transport_capacity

            #hardware cost pipeline
            hardware_pipeline_cost = pipeline_cost_per_km_per_kg*self.distance_to_hydrogen_sub*max_kg_per_hour
            hardware_pipeline_emissions = pipeline_emissions_per_km_per_kg*self.distance_to_onshore_sub*max_kg_per_hour

            #we'll assume the installation cost is just like the export cable, which is roughly the same cost as the hardware is
            #installation_pipeline_cost = hardware_pipeline_cost
            #installation_pipeline_emissions = 0.15*hardware_pipeline_emissions

            #decommissioning_pipeline_cost = 0.6*installation_pipeline_cost
            #decommissioning_pipeline_emissions = 0.1*hardware_pipeline_emissions

            total_pipeline_cost = hardware_pipeline_cost
            #+ installation_pipeline_cost + decommissioning_pipeline_cost

            total_pipeline_emissions = hardware_pipeline_emissions 
            print('pipeline emissions = ' +str(f'{total_pipeline_emissions:,}')) if self.verbose else None
            #+ installation_pipeline_emissions + decommissioning_pipeline_emissions

            #-----------------------------------------------#
            supporting_cost = 0.05*(total_PEM_cost + total_desalination_cost + total_compressor_cost + hardware_pipeline_cost + total_pipeline_cost + assembly_cost)
            #+ decom_cost)

            #-----------------------------------------------#

            total_capex_hydrogen = (total_PEM_cost + 
                            total_compressor_cost + 
                            total_desalination_cost + 
                            assembly_cost +
                            #decom_cost + 
                            total_pipeline_cost + 
                            supporting_cost)

            #-----------------------------------------------#

            PEM_OPEX = 0.01*total_PEM_cost #€/year
            PEM_replacement_cost = (0.6*total_PEM_cost)*(self.lifetime/14) #on time for whole lifetime
            compressor_OPEX = 0.02*total_compressor_cost #€/year
            pipeline_OPEX = 0.005*total_pipeline_cost #€/year
            desalination_OPEX = 0.05*total_desalination_cost #€/year

            total_OPEX_full_lifetime_hydrogen = (PEM_OPEX + compressor_OPEX + desalination_OPEX + pipeline_OPEX)*self.lifetime + PEM_replacement_cost
            
            PEM_OM_emissions = 0.035*total_PEM_emissions #€/year
            PEM_replacement_emissions = 0.6*total_PEM_emissions #on time for whole lifetime
            compressor_OM_emissions = 0.05*total_compressor_emissions #€/year
            pipeline_OM_emissions = 0.02*total_pipeline_emissions #€/year
            desalination_OM_emissions = 0.075*total_desalination_emissions #€/year

            total_lifetime_OM_emissions_hydrogen = (PEM_OM_emissions + compressor_OM_emissions + desalination_OM_emissions + pipeline_OM_emissions)*self.lifetime + PEM_replacement_emissions
            print('hydrogen OM emissions = ' +str(f'{total_lifetime_OM_emissions_hydrogen:,}')) if self.verbose else None
            
            total_emissions_hydrogen = total_PEM_emissions + total_compressor_emissions + total_desalination_emissions + assembly_emissions + total_pipeline_emissions + total_lifetime_OM_emissions_hydrogen

            #------------------------------------------------------------------------------#

        else:
            total_capex_hydrogen = 0   
            total_OPEX_full_lifetime_hydrogen = 0
            total_emissions_hydrogen = 0
        '''
        Island costs
        '''
        HVAC_space_per_MW = 15.125 #m^2/MW
        HVDC_space_per_MW = 21.5 #m^2/MW
        hydrogen_space_per_MW = 6 + 55 + 0.42 #m^2/MW
        
        excess_space_multiplication = 1.3 #factor to account for space for people to work

        surface_area_island = (HVAC_space_per_MW*self.HVAC_capacity + 
                               HVDC_space_per_MW*self.HVDC_capacity + 
                               hydrogen_space_per_MW*self.H2_capacity)*excess_space_multiplication
        
        print(surface_area_island)if self.verbose else None
        volume_island = surface_area_island*(self.depth+10)

        remaining_needed_sand = 0.1766*volume_island - 4503.1
        print(remaining_needed_sand)if self.verbose else None

        sand_fill_cost = 7.5*(remaining_needed_sand + volume_island)
        print(sand_fill_cost)if self.verbose else None
        sand_fill_emissions = 4*(remaining_needed_sand + volume_island)

        revetment_length = 3.3581*((surface_area_island)**0.5152)
        revetment_cost = revetment_length*200_000
        print(revetment_cost) if self.verbose else None
        revetment_emissions = revetment_length*160_000

        total_island_formation_cost = sand_fill_cost + revetment_cost
        print('island formation cost = €' +str(f'{total_island_formation_cost:,}')) if self.verbose else None     

        total_island_formation_cost = total_island_formation_cost + total_island_formation_cost*0.075*2 #contingency, insurance and development  
        total_island_formation_emissions = sand_fill_emissions + revetment_emissions
        print('island formation emissions = ' +str(f'{total_island_formation_emissions:,}')) if self.verbose else None     
        
        #-------------------------------------------------------------------#
        #------------------------ OPEX -------------------------------------#

        yearly_island_OPEX = 3_000_000 #€/year
        yearly_island_OM_emissions = 2_000_000 #kgCO2eq/year

        #------------------------------------------------------------------#

        complete_CAPEX = total_island_formation_cost + total_capex_HVDC + total_capex_HVAC + total_capex_hydrogen
        complete_OPEX = yearly_island_OPEX*self.lifetime + total_OPEX_full_lifetime_hydrogen + (yearly_OPEX_HVAC + yearly_OPEX_HVDC)*self.lifetime

        total_emissions = total_island_formation_emissions + yearly_island_OM_emissions*self.lifetime + total_emissions_HVAC + total_emissions_HVDC + total_emissions_hydrogen
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