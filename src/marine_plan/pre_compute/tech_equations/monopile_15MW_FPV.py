''' 
This is a test file to show how to structure a tech file
Adjust and add as many calcs as required
'''

from NID_specs import NID

class monopile_15MW_FPV:
    def __init__(self, geo_data, NID_type=0, verbose=False):

        import numpy as np
        import math
        
        self.NID_type = NID_type
        self.verbose = verbose

        #initialising geo-spatial data dictionary
        self.depth = geo_data['depth'] #m
        self.mean_windspeed_at_10m = geo_data['mean_wind_speed_at_10m'] #m/s at 10 m height
        self.distance_to_OPEX_port = geo_data['distance_to_OPEX_port'] #km
        self.distance_to_onshore_sub = geo_data['distance_to_onshore_sub'] #km
        self.distance_to_installation_port = geo_data['distance_to_installation_port'] #km
        self.soil_coefficient = geo_data['soil_coefficient'] #kN/m^3
        self.soil_friction_angle = geo_data['soil_friction_angle'] #angle
        self.global_tilted_irradiation = geo_data['global_tilted_irradiation']
        self.energy_revenue = geo_data['energy_revenue'] #€/MWh 
        self.WACC = geo_data['WACC'] #%
        self.cost_of_carbon = geo_data['cost_of_carbon'] #€/tonnes CO2e
        self.HVAC_distance = geo_data['HVAC_distance'] #km
        self.discount_rate_emissions = geo_data['discount_rate_emissions']

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
        self.lifetime = 25 #years

        #unit density
        self.capacity_per_square_km = 32 #MW
        self.solar_panel_capacity = 400 #W
        self.solar_per_platform = 70 #numver of solar panels
        self.capacity_per_platform = self.solar_per_platform * self.solar_panel_capacity #W
        self.amount_of_platforms_per_square_km = self.capacity_per_square_km*(1000000)/self.capacity_per_platform #number of platforms
        
        self.solar_reduction = 0.7 #reduction in space due to maintenance routs and safety etc
        self.OPEX_reduction_solar = 0.9 #OPEX reduction for solar because it can be combined with wind 


        #---------------------------------- vessel library -----------------------------# 
        # Wind turbine installation vessel

        self.WTIV_length	= 125 #m
        self.WTIV_max_hook_height = 100 #m
        self.WTIV_max_lift = 1200 #tonnes
        self.WTIV_max_windspeed = 15 #m/s
        self.WTIV_crane_rate	= 100 #m/h
        self.WTIV_leg_length = 110 #m
        self.WTIV_max_depth = 75 #m
        self.WTIV_max_extension = 85 #m
        self.WTIV_speed_above_dept =	1 #m/min
        self.WTIV_speed_below_depth=	2.5 #m/min
        self.WTIV_max_cargo = 8000 #tonnes
        self.WTIV_max_deck_load = 15 #t/m^2
        self.WTIV_max_deck_space = 4000 #m^2
        self.WTIV_max_waveheight = 3 #m
        self.WTIV_transit_speed = 20.19 #km/h
        self.WTIV_transit_speed_in_field	= 1.83 #km/h
        self.WTIV_transit_speed_loaded = 14 #km/h
        self.WTIV_day_rate = 180000 #€/day
        self.WTIV_mobilization_days = 7 #days
        self.WTIV_mobilization_mult = 1
        self.WTIV_fuel_transit_to_site = 2820 #l/h
        self.WTIV_fuel_transit_in_field = 1410 #l/h
        self.WTIV_fuel_installation = 1140 #l/h
        self.WTIV_fuel_standby = 854 #l/h

        #Scour protection vessel

        self.scour_max_cargo	= 10000 #tonnes
        self.scour_max_deck_load = 8 #t/m^2
        self.scour_max_deck_space = 600 #m^2
        self.scour_max_waveheight = 2 #m
        self.scour_max_windspeed	= 2 #m/s
        self.scour_transit_speed	= 24 #km/h
        self.scour_transit_speed_loaded = 18.5 #km/h
        self.scour_day_rate = 120000 #€/day
        self.scour_fuel_transit = 1772 #l/h
        self.scour_fuel_installation	= 2658 #l/h

        #Cable installation vessel

        self.cable_max_waveheight = 2 #m
        self.cable_max_windspeed	= 25 #m/s
        self.cable_transit_speed	= 25.5 #km/h
        self.cable_day_rate = 225000 #€/day
        self.cable_min_draft	= 8.5 #m
        self.cable_overall_length = 171 #m
        self.cable_fuel_transit = 2720 #l/h
        self.cable_fuel_installation	= 573 #l/h
        self.cable_fuel_lay_bury	= 573 #l/h
        self.cable_max_mass = 13000 #tonnes

        #Tug boat vessel

        self.tug_boat_max_waveheight	= 3 #m
        self.tug_boat_max_windspeed = 15 #m/s
        self.tug_boat_transit_speed = 21.9 #km/h
        self.tug_boat_transit_speed_with_load = 14.6 #km/h
        self.tug_boat_day_rate = 35000 #€/day
        self.tug_boat_fuel_transit = 740 #l/h

        #Barge vessel

        self.barge_max_lift = 500 #tonnes
        self.barge_class = 2 #
        self.barge_max_cargo	= 8000 #tonnes
        self.barge_max_deck_load	= 8 #t/m^2
        self.barge_max_deck_space = 1000 #m^2
        self.barge_max_waveheight = 2.5 #m
        self.barge_max_windspeed	= 20 #m/s
        self.barge_transit_speed = 6 #km/h
        self.barge_day_rate = 120000 #€/day

        #support vessel
        self.support_day_rate=	100000
        self.support_mobilization_days=	7
        self.support_fuel_transit	=795.2
        self.support_fuel_working	=397.6
        self.support_transit_speed=	22.7

        #heavy feeder vessel
        self.hfv_max_lift = 500             # t
        self.hfv_leg_length = 85            # mm
        self.hfv_max_depth= 40             # m
        self.hfv_max_extension= 60         # m
        self.hfv_speed_above_depth= 0.5    # m/min
        self.hfv_speed_below_depth= 0.5    # m/min
        self.hfv_max_cargo= 8000           # t
        self.hfv_max_deck_load= 8          # t/m^2
        self.hfv_max_deck_space=2000       # m^2
        self.hfv_max_waveheight= 2.5       # m
        self.hfv_max_windspeed= 20         # m/s
        self.hfv_transit_speed= 20         # km/h
        self.hfv_transit_speed_loaded= 14  # km/h
        self.hfv_day_rate= 120000          # USD/day
        self.hfv_fuel_transit_to_site = 2050 #l/h
        self.hfv_fuel_installation = 1140 #l/h
        self.hfv_fuel_standby = 854 #l/h

        #heavy lift vessel

        self.hlv_max_hook_height= 72       # m
        self.hlv_max_lift= 5500            # t
        self.hlv_max_windspeed= 15         # m/s
        self.hlv_leg_length= 110           # m
        self.hlv_max_depth= 75             # m
        self.hlv_max_extension= 85         # m
        self.hlv_speed_above_depth= 0.5    # m/min
        self.hlv_speed_below_depth= 0.5    # m/min
        self.hlv_max_cargo= 8000           # t
        self.hlv_max_deck_load= 15         # t/m^2
        self.hlv_max_deck_space= 4000      # m^2
        self.hlv_max_waveheight= 2.5       # m
        self.hlv_max_windspeed= 20         # m/s
        self.hlv_transit_speed= 20         # km/h
        self.hlv_transit_speed_loaded= 14  # km/h
        self.hlv_day_rate= 500000          # USD/day
        self.hlv_fuel_transit_to_site = 2050 #l/h
        self.hlv_fuel_installation = 1140 #l/h
        self.hlv_fuel_standby = 854 #l/h

        #-----------------------------------------------------------------------# 


        self.vars = {} # Calculated variables go in here

        self.final_metrics = {} # Final metrics go in here

    def _calculations(self):
        
        import numpy as np
        import math

        #------------------------------------ TURBINE HARDWARE ----------------------------#
        #dimensions of turbine 

        rotor_diameter = 236 #m

        hub_diameter = (self.turbine_rating/4) + 2 #m
        hub_height = 145 #m

        nacelle_width = hub_diameter + 1.5 #m
        nacelle_length = 2*nacelle_width #m
        rotor_nacelle_assembly_mass	= 2.082*(self.turbine_rating**2) + 44.59*self.turbine_rating + 22.48 #tonnes

        tower_diameter = (self.turbine_rating/2) + 4 #m
        tower_mass = 860 #tonnes
        tower_height = hub_height #m

        blade_weight = 50 #tonnes
        blade_width = 6 #m
        blade_length = (rotor_diameter - hub_diameter)/2 #m

        #Economic calculations
        turbine_cost_per_MW = 1_500_000 #€/MW
        turbine_cost = turbine_cost_per_MW*self.turbine_rating #€ cost for one turbine

        #Emission calculations
        turbine_emissions_per_MW = 718200 #kgCO2eq/MW
        turbine_emissions = turbine_emissions_per_MW*self.turbine_rating #kgCO2eq for one turbine
        print('turbine hardware = ' + str(turbine_cost)) if self.verbose else None
        print('turbine hardware emissions = ' + str(turbine_emissions)) if self.verbose else None
        #----------------------------------------------------------------------------------#

        #----------------------- turbine install ------------------------------------------#

        #general turbine installation times

        tower_section_fasten_time =	4 #h
        tower_section_release_time = 3 #h
        tower_section_attach_time = 6 #h
        nacelle_fasten_time	= 4 #h
        nacelle_release_time = 3 #h
        nacelle_attach_time	= 6 #h
        blade_fasten_time = 1.5 #h
        blade_release_time = 1 #h
        blade_attach_time = 3.5 #h
        turbine_sections = 2 #h

        #turbine dimensions
        turbine_weight = (blade_weight*3) + tower_mass + rotor_nacelle_assembly_mass #tonnes
        turbine_space =	(tower_diameter*hub_height) + (nacelle_width*nacelle_length) + (blade_length*blade_width) #m^2. Blades can be stacked

        #installation sequence
        amount_of_turbines_per_sequence = 1 #number
        lift_components_onto_deck = amount_of_turbines_per_sequence*3 #h
        total_tower_secure_time	= tower_section_fasten_time* turbine_sections*amount_of_turbines_per_sequence #h
        total_nacelle_secure_time = nacelle_fasten_time*amount_of_turbines_per_sequence #h
        total_blade_secure_time	= blade_fasten_time*amount_of_turbines_per_sequence*3 #h
        time_to_transit_to_site	= self.distance_to_installation_port/self.WTIV_transit_speed_loaded #h
        positioning_time_per_turbine = 2 #h
        jackup_time_per_turbine = (self.depth/self.WTIV_speed_below_depth)/60 #h
        total_jackup_and_positioning_time = (positioning_time_per_turbine + jackup_time_per_turbine)*amount_of_turbines_per_sequence #h
        total_tower_release_time = tower_section_release_time* turbine_sections*amount_of_turbines_per_sequence #h
        total_nacelle_release_time = nacelle_release_time*amount_of_turbines_per_sequence #h
        total_blade_release_time = blade_release_time*amount_of_turbines_per_sequence*3 #h
        crane_time_tower = hub_height/self.WTIV_crane_rate #h
        attach_tower_to_substructure = tower_section_attach_time #h
        attach_tower_to_tower = tower_section_attach_time #h
        change_lifting_equipment_nacelle = 1 #h
        total_tower_time = (crane_time_tower + attach_tower_to_substructure + attach_tower_to_tower + change_lifting_equipment_nacelle )*amount_of_turbines_per_sequence #h
        crane_time_nacelle = hub_height/self.WTIV_crane_rate #h
        attach_nacelle_to_tower = nacelle_attach_time #h
        total_nacelle_time = (attach_nacelle_to_tower + crane_time_nacelle)*amount_of_turbines_per_sequence #h
        crane_time_blades = (hub_height/self.WTIV_crane_rate)*3 #h
        change_lifting_equipment_blades = 1 #h
        attach_blades_to_nacelle = blade_attach_time*3 #h
        total_blades_time = (attach_blades_to_nacelle + crane_time_blades + change_lifting_equipment_blades)*amount_of_turbines_per_sequence #h
        total_attach_time = total_tower_time + total_nacelle_time + total_blades_time #h
        jackdown_time_per_turbine = ((self.depth/self.WTIV_speed_below_depth)/60)*amount_of_turbines_per_sequence #h
        transit_between_turbines = (self.interdistance_turbines/self.WTIV_transit_speed_in_field)*(amount_of_turbines_per_sequence - 1) #h
        time_to_transit_to_port	= self.distance_to_installation_port/self.WTIV_transit_speed #h

        #complete time sequence for one turbine
        complete_installation_sequence_time = lift_components_onto_deck + total_tower_secure_time + total_nacelle_secure_time + total_blade_secure_time + time_to_transit_to_site + total_jackup_and_positioning_time + total_tower_release_time + total_nacelle_release_time + total_blade_release_time + total_attach_time + jackdown_time_per_turbine + transit_between_turbines + time_to_transit_to_port #h

        #delay factor because of weather 
        weather_delay_turbine=	0.1207*self.mean_windspeed_at_10m + 0.2135 #factor

        #economic variables
        cost_of_sequence_installation	=((complete_installation_sequence_time/24) +(self.WTIV_mobilization_days/100)*2)*self.WTIV_day_rate
        cost_of_turbine_installation=	(cost_of_sequence_installation/amount_of_turbines_per_sequence)*weather_delay_turbine

        #emission variables
        fuel_consumption_transit = (time_to_transit_to_site + time_to_transit_to_port)*self.WTIV_fuel_transit_to_site + transit_between_turbines * self.WTIV_fuel_transit_in_field
        fuel_consumption_installation = (complete_installation_sequence_time - time_to_transit_to_site - transit_between_turbines - time_to_transit_to_port)*self.WTIV_fuel_installation
        fuel_consumption_standby = (complete_installation_sequence_time*weather_delay_turbine + self.WTIV_mobilization_days*2*24)*self.WTIV_fuel_standby
        emissions_turbine_installation = (fuel_consumption_transit + fuel_consumption_installation + fuel_consumption_standby)*self.emission_factor_HFO/amount_of_turbines_per_sequence
        print('turbine installation = ' + str(cost_of_turbine_installation)) if self.verbose else None
        print('turbine installation emissions = ' + str(emissions_turbine_installation )) if self.verbose else None
        #----------------------------------------------------------------------------------#

        #------------------------------ turbine decomissioning ----------------------------#

        #Installation sequence of decommissioning 
        amount_of_barges= 2

        jackup_transit_to_turbine = self.distance_to_installation_port/self.WTIV_transit_speed #h
        tug_boat_transit_to_turbine	= (self.distance_to_installation_port/self.tug_boat_transit_speed)*amount_of_barges #h
        positioning_time_per_turbine_decom = 6 #h
        jackup_time_per_turbine_decom = (self.depth/self.WTIV_speed_below_depth)/60 #h
        blade_removal = 2.5*3 #h
        nacelle_removal = 4.5 #h
        tower_segment_removal = 4.5*2 #h
        jackdown_time_per_turbine_decom	= ((self.depth/self.WTIV_speed_below_depth)/60)  #h
        transit_between_turbines_decom = (self.interdistance_turbines/self.WTIV_transit_speed_in_field) #h
        barge_transit_to_unload	= (self.distance_to_installation_port/self.tug_boat_transit_speed_with_load)*amount_of_barges #h
        barge_transit_back = (self.distance_to_installation_port/self.tug_boat_transit_speed)*amount_of_barges #h

        #total needed decomissioning time for one turbine
        total_turbine_decom_time = positioning_time_per_turbine_decom + jackup_time_per_turbine_decom + blade_removal + nacelle_removal + tower_segment_removal + jackdown_time_per_turbine_decom + transit_between_turbines_decom #h
        averaged_initial_transit = (jackup_transit_to_turbine*2)/100 #h

        #economic variables
        turbine_decommissioning_cost = ((total_turbine_decom_time + averaged_initial_transit)/24)*(self.WTIV_day_rate + (self.barge_day_rate + self.tug_boat_day_rate)*amount_of_barges) +(((self.WTIV_mobilization_days*2)))/100 

        #Emission variables
        fuel_decom_turbine_WTIV	= (averaged_initial_transit*self.WTIV_fuel_transit_to_site) + (transit_between_turbines_decom*self.WTIV_fuel_transit_in_field) + ((total_turbine_decom_time - transit_between_turbines_decom)*self.WTIV_fuel_installation) + (self.WTIV_mobilization_days*2*24*self.WTIV_fuel_standby/100) #h
        fuel_decom_turbine_barge_tug = (tug_boat_transit_to_turbine + barge_transit_to_unload)*self.tug_boat_fuel_transit #h
        emissions_turbine_decom	= (fuel_decom_turbine_WTIV*self.emission_factor_HFO) + (fuel_decom_turbine_barge_tug*self.emission_factor_MDO)
        print('turbine decom = ' + str(turbine_decommissioning_cost)) if self.verbose else None
        print('turbine decom emissions = ' + str(emissions_turbine_decom)) if self.verbose else None

        #-------------------------------------------------------------------------------------------------------------#
        #--------------------- monopile foundation hardware ----------------------------------------------------------#
        #monopile dimensioning 
        density_monopile = 7860 #kg/m^3
        monopile_length_above_sealevel = 5 #m
        diameter_at_10_meter = 0.2173*self.mean_windspeed_at_10m + 5.9056 #m
        pile_diameter = 0.0144*self.depth + (diameter_at_10_meter - 0.0144*10) #m
        pile_wall_thickness = 0.00635 + (pile_diameter/100) #m
        pile_moment_of_inertia = 0.125*((pile_diameter - pile_wall_thickness)**3)*pile_wall_thickness*3.14 #
        monopile_modulus = 200*(10**9) #Pa
        pile_embedment_length =	2*((monopile_modulus*pile_moment_of_inertia)/self.soil_coefficient)**0.2 #m
        total_pile_length = pile_embedment_length + self.depth + monopile_length_above_sealevel #m

        monopile_volume	= (3.14/4)*((pile_diameter**2) - (pile_diameter - 2*pile_wall_thickness)**2)*total_pile_length #m^3
        monopile_mass = density_monopile*monopile_volume/self.short_ton #tonnes

        #economic variables
        monopile_cost_per_tonnes = 2250 #€/tonnes
        monopile_cost = monopile_mass*monopile_cost_per_tonnes

        #emission variables
        monopile_emissions_per_tonnes = 1886
        monopile_emissions = monopile_emissions_per_tonnes*monopile_mass

        #transition piece dimensioning
        transition_piece_density = 7860 #kg/m^3
        connection_thickness = 0 #m
        transition_piece_thickness = pile_wall_thickness #m
        transition_piece_length	= 25 #m
        diameter_transition_piece = pile_diameter + (2*(transition_piece_thickness + connection_thickness)) #m

        transition_piece_mass = transition_piece_density*(pile_diameter+2*connection_thickness+transition_piece_thickness)*3.14*transition_piece_thickness*transition_piece_length/self.short_ton #tonnes

        #economic variable
        transition_piece_cost_per_tonnes = 3230
        transition_piece_cost	=transition_piece_mass*transition_piece_cost_per_tonnes


        #emission variables
        transition_piece_emissions_per_tonnes=	1886
        transition_piece_emissions=	transition_piece_emissions_per_tonnes*transition_piece_mass
        print('monopile hardware cost =' + str(monopile_cost + transition_piece_cost)) if self.verbose else None
        print('monopile hardware emissions =' + str(monopile_emissions + transition_piece_emissions)) if self.verbose else None

        #NID bird attraction measure
        bird_attraction_cost = 0
        bird_attraction_emissions = 0
        if self.NID_type == 3:
            bird_attraction_cost = NID['NID3']['bird_attraction_cost']
            bird_attraction_emissions = NID['NID3']['bird_attraction_emissions']
        print(f'bird attraction cost = {bird_attraction_cost:,.2f}') if self.verbose else None
        print(f'bird attraction emissions = {bird_attraction_emissions:,.2f}') if self.verbose else None
        
        #-------------------------------------------------------------------------------------------------------------#
        #---------------------------------- monopile installation ----------------------------------------------------#
        #installation times
        mono_drive_rate	= 20 #h
        mono_fasten_time = 12 #h
        mono_release_time =	3 #h
        tp_fasten_time = 8 #h
        tp_release_time	= 2 #h
        tp_bolt_time = 4 #h
        grout_cure_time	= 24 #h
        grout_pump_time	= 2 #h
        site_position_time = 2 #h
        rov_survey_time = 1 #h
        crane_reequip_time = 1 #h
        bird_attraction_fasten_time = 0
        bird_attraction_release_time = 0
        bird_attraction_bolt_time = 0
        if self.NID_type == 3:
            bird_attraction_fasten_time = 8
            bird_attraction_release_time = 2
            bird_attraction_bolt_time = 4

        #installation sequence
        amount_of_monopiles_per_sequence =	min(np.floor(self.WTIV_max_cargo/(monopile_mass + transition_piece_mass)),np.floor(self.WTIV_max_deck_space/((total_pile_length*pile_diameter) + (transition_piece_length*diameter_transition_piece)))) #number
        lift_monopiles_onto_deck = amount_of_monopiles_per_sequence*3 #h
        total_monopile_secure_time = mono_fasten_time*amount_of_monopiles_per_sequence #h
        total_transition_piece_secure_time = tp_fasten_time*amount_of_monopiles_per_sequence #h
        total_bird_attraction_secure_time = bird_attraction_fasten_time*amount_of_monopiles_per_sequence #h
        time_to_transit_to_site_monopile = self.distance_to_installation_port/self.WTIV_transit_speed_loaded #h
        positioning_time_per_monopile = site_position_time #h
        jackup_time_per_monopile = (self.depth/self.WTIV_speed_below_depth)/60 #h
        total_jackup_and_positioning_time_monopile = (positioning_time_per_monopile + jackup_time_per_monopile)*amount_of_monopiles_per_sequence #h
        survey_site_ROV	= rov_survey_time #h
        release_monopile = mono_release_time #h
        upend_monopile = total_pile_length/self.WTIV_crane_rate #h
        lower_monopile_to_seabed = (self.depth + 10)/(self.WTIV_crane_rate) #h
        equip_driving_equipment	= crane_reequip_time #h
        drive_monopile_into_seabed = pile_embedment_length/mono_drive_rate #h
        equip_lifting_equipment	= crane_reequip_time #h
        release_transition_piece = tp_release_time #h
        lift_transition_piece = transition_piece_length/self.WTIV_crane_rate #h
        bolt_transition_piece = tp_bolt_time #h
        release_bird_attraction = bird_attraction_release_time #h
        lift_bird_attraction = 20/self.WTIV_crane_rate #h
        bolt_bird_attraction = bird_attraction_bolt_time #h
        total_mono_time = (survey_site_ROV + release_monopile + upend_monopile + lower_monopile_to_seabed + equip_driving_equipment + drive_monopile_into_seabed + equip_lifting_equipment + release_transition_piece + lift_transition_piece + bolt_transition_piece + release_bird_attraction + lift_bird_attraction + bolt_bird_attraction)*amount_of_monopiles_per_sequence #h
        jackdown_time_per_monopile = ((self.depth/self.WTIV_speed_below_depth)/60)*amount_of_monopiles_per_sequence #h
        transit_between_monopile = (self.interdistance_turbines/self.WTIV_transit_speed_in_field)*(amount_of_monopiles_per_sequence - 1) #h
        time_to_transit_to_port_monopile = self.distance_to_installation_port/self.WTIV_transit_speed #h

        complete_installation_sequence_time_monopile = lift_monopiles_onto_deck + total_monopile_secure_time + total_transition_piece_secure_time + total_bird_attraction_secure_time + time_to_transit_to_site_monopile + total_jackup_and_positioning_time_monopile + total_mono_time + jackdown_time_per_monopile + transit_between_monopile + time_to_transit_to_port_monopile #h

        #delay factor because of weather 
        weather_delay_monopile = 0.0981*self.mean_windspeed_at_10m + 0.3612

        #economic variables
        additional_cost_equipment_and_compensation = 400_000

        cost_of_sequence_installation_monopile = (((complete_installation_sequence_time_monopile*weather_delay_monopile)/24) + (self.WTIV_mobilization_days/(1000/15))*2)*(self.WTIV_day_rate + additional_cost_equipment_and_compensation)
        cost_of_monopile_installation = (cost_of_sequence_installation_monopile/amount_of_monopiles_per_sequence)

        #emission variables
        fuel_consumption_transit_monopile=	(time_to_transit_to_site_monopile + time_to_transit_to_port_monopile)*self.WTIV_fuel_transit_to_site + transit_between_monopile * self.WTIV_fuel_transit_in_field
        fuel_consumption_installation_monopile =	(complete_installation_sequence_time_monopile - time_to_transit_to_site_monopile - transit_between_monopile - time_to_transit_to_port_monopile)*self.WTIV_fuel_installation
        fuel_consumption_standby_monopile	=(complete_installation_sequence_time_monopile*weather_delay_monopile + self.WTIV_mobilization_days*2*24)*self.WTIV_fuel_standby
        emissions_monopile_installation	=(fuel_consumption_transit_monopile + fuel_consumption_installation_monopile + fuel_consumption_standby_monopile)*self.emission_factor_HFO/amount_of_monopiles_per_sequence
        
        print('monopile install cost = ' +str(cost_of_monopile_installation)) if self.verbose else None
        print('monopile install emissions = ' +str(emissions_monopile_installation)) if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- monopile decommissioning -------------------------------------------------#
        #decommissioning times
        amount_of_barges_monopile =	2
        jackup_transit_to_monopile = self.distance_to_installation_port/self.WTIV_transit_speed #h
        tug_boat_transit_to_monopile = (self.distance_to_installation_port/self.tug_boat_transit_speed)*amount_of_barges_monopile #h
        positioning_time_per_monopile_decom = 6 #h
        jackup_time_per_monopile_decom = (self.depth/self.WTIV_speed_below_depth)/60 #h
        pumping_rate = 37
        pump_volume = (3.14/4)*(pile_diameter**2)*(2+1)
        pumping_mud	= pump_volume/pumping_rate #h
        #cutting_speed = 17 
        cut_monopile = pile_diameter*3.14/0.025/60 #h
        #cutting_speed*pile_diameter
        lift_foundation_out_sea	= 5 #h
        jackdown_time_per_monopile_decom = ((self.depth/self.WTIV_speed_below_depth)/60) #h
        transit_between_monopiles_decom = (self.interdistance_turbines/self.WTIV_transit_speed_in_field) #h
        barge_transit_to_unload_monopile = (self.distance_to_installation_port/self.tug_boat_transit_speed_with_load)*amount_of_barges_monopile #h
        barge_transit_back_monopile = (self.distance_to_installation_port/self.tug_boat_transit_speed)*amount_of_barges_monopile #h
        total_monopile_decom_time = positioning_time_per_monopile_decom + jackup_time_per_monopile_decom + pumping_mud + cut_monopile + lift_foundation_out_sea + jackdown_time_per_monopile_decom + transit_between_monopiles_decom #h
        averaged_initial_transit_monopile = (jackup_transit_to_monopile*2)/100 #h

        #economic variables
        additional_monopile_decom_equip_cost = 200_000
        monopile_decommissioning_cost = ((total_monopile_decom_time + averaged_initial_transit_monopile)/24)*(self.WTIV_day_rate + additional_monopile_decom_equip_cost  + (self.barge_day_rate + self.tug_boat_day_rate)*amount_of_barges_monopile) +(((self.WTIV_mobilization_days*2*(self.WTIV_day_rate+additional_monopile_decom_equip_cost ))))/100

        #emission varaibles
        fuel_decom_monopile_WTIV	=(averaged_initial_transit_monopile*self.WTIV_fuel_transit_to_site) + (transit_between_monopiles_decom*self.WTIV_fuel_transit_in_field) + ((total_monopile_decom_time - transit_between_monopiles_decom)*self.WTIV_fuel_installation) + (self.WTIV_mobilization_days*2*24*self.WTIV_fuel_standby/100)
        fuel_decom_monopile_barge_tug	=(tug_boat_transit_to_monopile + barge_transit_to_unload_monopile)*self.tug_boat_fuel_transit
        emissions_monopile_decom	=(fuel_decom_monopile_WTIV*self.emission_factor_HFO) + (fuel_decom_monopile_barge_tug*self.emission_factor_MDO)
        print('monopile decom cost = ' +str(monopile_decommissioning_cost)) if self.verbose else None
        print('monopile decom emissions = ' +str(emissions_monopile_decom)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- scour hardware -------------------------------------------------#
        #scour dimensioning 
        scour_depth	= 1.3*pile_diameter #m
        radius_scour_pit = (pile_diameter/2) + (scour_depth/math.tan(self.soil_friction_angle* 3.14 / 180)) #m
        scour_protection_depth = 1 #m
        scour_volume = 3.14*((radius_scour_pit**2) - (pile_diameter/2)**2)*scour_protection_depth #m^3
        scour_density = 2600 #kg/m^3
        scour_mass = (scour_volume*scour_density)/self.short_ton #tonnes
        if self.NID_type >= 1:
            scour_mass *= 1.2
        if self.NID_type == 3:
            scour_mass *= 1.075

        #economic variables
        scour_protection_material_cost = 50
        scour_protection_cost = scour_mass*scour_protection_material_cost

        corrosion_protection_cost = 230000
        
        #emission variables
        scour_protection_material_emissions = 25
        scour_protection_emissions = scour_mass*scour_protection_material_emissions

        print('scour hardware = ' + str(scour_protection_cost)) if self.verbose else None
        print('corossion hardware = ' + str(corrosion_protection_cost)) if self.verbose else None
        print('scour hardware emissions = ' + str(scour_protection_emissions)) if self.verbose else None
        
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- seabed NID hardware -------------------------------------------------#

        NID_seabed_area = 0
        NID_seabed_mass = 0
        NID_seabed_hardware_cost = 0
        NID_seabed_emissions = 0
        if self.NID_type >= 2:
            NID_seabed_area = NID['NID2']['NID_sebaed_area']
            if self.NID_type == 3:
                NID_seabed_area *= NID['NID3']['NID_seabed_area_factor']
            rock_gravel_layer_thickness = NID['NID2']['rock_gravel_layer_thickness']
            shell_layer_thickness = NID['NID2']['shell_layer_thickness']
            rock_gravel_partition = NID['NID2']['rock_gravel_partition']
            shell_partition = NID['NID2']['shell_partition']
            tree_partition = NID['NID2']['tree_partition']
            oyster_partition = NID['NID2']['oyster_partition']
            tree_cost = NID['NID2']['tree_cost']
            tree_weight = NID['NID2']['tree_weight']
            area_per_tree = NID['NID2']['area_per_tree']
            shell_density = NID['NID2']['shell_density']
            oyster_cost_per_juvenile = NID['NID2']['oyster_cost_per_juvenile']
            oyster_density = NID['NID2']['oyster_density']
            oyster_surface_density = NID['NID2']['oyster_surface_density']
            shell_emissions = NID['NID2']['shell_emissions']
            tree_emissions = NID['NID2']['tree_emissions']
            oyster_emissions = NID['NID2']['oyster_emissions']
            
            rock_gravel_cost_per_m2 = scour_protection_material_cost*scour_density*rock_gravel_layer_thickness/1000
            rock_gravel_area = NID_seabed_area*rock_gravel_partition
            total_rock_gravel_mass = rock_gravel_area*scour_density*rock_gravel_layer_thickness/1000 #tonnes
            total_rock_gravel_hardware_cost = rock_gravel_cost_per_m2*rock_gravel_area
            rock_gravel_hardware_emissions = scour_protection_material_emissions*total_rock_gravel_mass
            shell_cost_per_m2 = 0.9*scour_protection_material_cost*shell_density*shell_layer_thickness/1000
            shell_area = NID_seabed_area*shell_partition
            total_shell_mass = shell_area*shell_density*shell_layer_thickness/1000 #tonnes
            total_shell_hardware_cost = shell_cost_per_m2*shell_area
            shell_hardware_emissions = shell_emissions*total_shell_mass
            tree_cost_per_m2 = 2*tree_cost/area_per_tree
            tree_area = NID_seabed_area*tree_partition
            total_tree_mass = tree_area*2*tree_weight/(area_per_tree*1000) #tonnes
            total_tree_hardware_cost = tree_cost_per_m2*tree_area
            tree_hardware_emissions = tree_emissions*total_tree_mass
            oyster_cost_per_m2 = oyster_cost_per_juvenile*oyster_surface_density
            oyster_area = NID_seabed_area*oyster_partition
            total_oyster_mass = oyster_area*oyster_density*shell_layer_thickness/1000 #tonnes
            total_oyster_hardware_cost = oyster_cost_per_m2*oyster_area
            oyster_hardware_emissions = oyster_emissions*total_oyster_mass

            NID_seabed_mass = total_rock_gravel_mass + total_shell_mass + total_tree_mass + total_oyster_mass
            print(f'NID seabed mass = {NID_seabed_mass:,.2f}') if self.verbose else None
            NID_seabed_hardware_cost = total_rock_gravel_hardware_cost + total_shell_hardware_cost + total_tree_hardware_cost + total_oyster_hardware_cost
            NID_seabed_emissions = rock_gravel_hardware_emissions + shell_hardware_emissions + tree_hardware_emissions + oyster_hardware_emissions
        print(f'NID seabed hardware cost = {NID_seabed_hardware_cost:,.2f}') if self.verbose else None
        print(f'NID seabed emissions = {NID_seabed_emissions:,.2f}') if self.verbose else None
        
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- scour installation -------------------------------------------------#
        #installation times
        drop_rocks_time	= 30 #h
        load_rocks_time = 4 #h
        scour_position_on_site = 2 #h
        amount_of_scour_per_sequence = np.floor(self.scour_max_cargo/scour_mass)
        lift_scour_onto_deck = load_rocks_time #h
        time_to_transit_to_site_scour =	self.distance_to_installation_port/self.scour_transit_speed_loaded #h
        positioning_time_per_scour = scour_position_on_site #h
        total_scour_position_time = positioning_time_per_scour*amount_of_scour_per_sequence #h
        total_drop_material_time = drop_rocks_time*amount_of_scour_per_sequence #h
        transit_between_scour = (self.interdistance_turbines/self.scour_transit_speed_loaded)*(amount_of_scour_per_sequence - 1) #h
        time_to_transit_to_port_scour =	self.distance_to_installation_port/self.scour_transit_speed #h

        complete_installation_sequence_time_scour =	lift_scour_onto_deck + time_to_transit_to_site_scour + total_scour_position_time + total_drop_material_time + transit_between_scour + time_to_transit_to_port_scour #h

        #delay factor because of weather 
        weather_delay_scour	= 0.1866*self.mean_windspeed_at_10m - 0.0985

        #economic variable
        cost_of_sequence_installation_scour	=((complete_installation_sequence_time_scour/24))*self.scour_day_rate
        cost_of_scour_installation	=(cost_of_sequence_installation_scour/amount_of_scour_per_sequence)*weather_delay_scour

        #emission variable
        fuel_consumption_transit_scour	=(time_to_transit_to_site_scour + transit_between_scour + time_to_transit_to_port_scour)*self.scour_fuel_transit
        fuel_consumption_installation_scour	=(complete_installation_sequence_time_scour - time_to_transit_to_site_scour - transit_between_scour - time_to_transit_to_port_scour)*self.scour_fuel_installation
        emissions_scour_installation=	(fuel_consumption_transit_scour + fuel_consumption_installation_scour)*self.emission_factor_HFO/amount_of_scour_per_sequence
        print('scour installation = ' + str(cost_of_scour_installation)) if self.verbose else None
        print('scour installation emissions = ' + str(emissions_scour_installation)) if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- seabed NID installation -------------------------------------------------#
        complete_installation_sequence_time_NID_seabed = 0
        total_transit_time_NID_seabed = 0
        amount_of_NID_per_sequence = 1
        if self.NID_type >= 2:
            NID_seabed_placement_time_factor = NID['NID2']['NID_seabed_placement_time_factor']
            if self.NID_type == 3:
                NID_seabed_placement_time_factor *= NID['NID3']['NID_seabed_area_factor']
            drop_NID_time = drop_rocks_time*NID_seabed_placement_time_factor
            load_NID_time = load_rocks_time
            NID_position_on_site = scour_position_on_site
            if self.scour_max_cargo >= NID_seabed_mass:
                amount_of_NID_per_sequence = np.floor(self.scour_max_cargo/NID_seabed_mass)
            else:
                amount_of_NID_per_sequence = 1/np.ceil(NID_seabed_mass/self.scour_max_cargo)
            print(f'amount of NID per sequence = {amount_of_NID_per_sequence:,.2f}') if self.verbose else None
            lift_NID_onto_deck = load_NID_time
            time_to_transit_to_site_NID =	self.distance_to_installation_port/self.scour_transit_speed_loaded #h
            positioning_time_per_NID = NID_position_on_site #h
            total_NID_position_time = positioning_time_per_NID*amount_of_NID_per_sequence #h
            total_NID_drop_material_time = drop_NID_time*amount_of_NID_per_sequence #h
            print(f'total NID drop material time = {total_NID_drop_material_time:,.2f}') if self.verbose else None
            if amount_of_NID_per_sequence >= 1:
                transit_between_NID = (self.interdistance_turbines/self.scour_transit_speed_loaded)*(amount_of_NID_per_sequence - 1) #h
            else:
                transit_between_NID = 0
            time_to_transit_to_port_NID =	self.distance_to_installation_port/self.scour_transit_speed #h
            complete_installation_sequence_time_NID_seabed =	lift_NID_onto_deck + time_to_transit_to_site_NID + total_NID_position_time + total_NID_drop_material_time + transit_between_NID + time_to_transit_to_port_NID #h
            print(f'complete installation sequence time NID seabed = {complete_installation_sequence_time_NID_seabed:,.2f}') if self.verbose else None
            total_transit_time_NID_seabed = time_to_transit_to_site_NID + transit_between_NID + time_to_transit_to_port_NID 

        #delay factor because of weather 
        weather_delay_NID	= 0.1866*self.mean_windspeed_at_10m - 0.0985

        #economic variable
        cost_of_sequence_installation_NID_seabed	=((complete_installation_sequence_time_NID_seabed/24))*self.scour_day_rate
        cost_of_NID_seabed_installation	=(cost_of_sequence_installation_NID_seabed/amount_of_NID_per_sequence)*weather_delay_NID

        #emission variable
        fuel_consumption_transit_NID_seabed	=(total_transit_time_NID_seabed)*self.scour_fuel_transit
        fuel_consumption_installation_NID_seabed	=(complete_installation_sequence_time_NID_seabed - total_transit_time_NID_seabed)*self.scour_fuel_installation
        emissions_NID_seabed_installation=	(fuel_consumption_transit_NID_seabed + fuel_consumption_installation_NID_seabed)*self.emission_factor_HFO/amount_of_NID_per_sequence
        print(f'NID seabed installation cost = {cost_of_NID_seabed_installation:,.2f}') if self.verbose else None
        print(f'NID seabed installation emissions = {emissions_NID_seabed_installation:,.2f}') if self.verbose else None
 
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- inter-array hardware -------------------------------------------------#

        #overall factors
        burial_factor	=1
        excess_cable_factor	=1.1

        #cable specs
        ac_resistance	=0.04
        capacitance	=300
        conductor_size	=630
        current_capacity	=775
        inductance	=0.35
        linear_density	=42.5
        voltage	=66
        line_frequency = 60 #Hz

        #cable sizing equations
        conductance=	1/ac_resistance
        cable_num=	complex(ac_resistance,2 * 3.14 * line_frequency *inductance,)
        cable_den	=complex(conductance,2 * 3.14 *line_frequency*capacitance,)
        char_impedance	=np.sqrt(cable_num/cable_den)
        phase_angle	= math.atan(np.imag(char_impedance) / np.real(char_impedance))
        power_factor	=math.cos(phase_angle)
        cable_power=	(np.sqrt(3)*voltage*current_capacity*power_factor/ 1000)
        cable_capacity=	cable_power

        inter_cable_distance = self.interdistance_turbines

        #economic variables
        cable_cost_per_km	=400000
        cable_cost=	cable_cost_per_km*((inter_cable_distance + (2*self.depth/1000))*excess_cable_factor)

        #emission variables
        cable_emission_per_km=	93537
        cable_emission	=cable_emission_per_km*((inter_cable_distance + (2*self.depth/1000))*excess_cable_factor)
         
        print('inter array hardware = ' + str(cable_cost)) if self.verbose else None
        print('inter array hardware emissions = ' + str(cable_emission)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- inter-array installation -------------------------------------------------#

        #installation times
        plgr_speed = 1
        cable_load_time	= 6 #h
        cable_prep_time	= 1 #h
        cable_lower_time = 1 #h
        cable_pull_in_time = 5.5 #h
        cable_termination_time = 5.5 #h
        cable_lay_speed = 0.4 #h
        cable_lay_bury_speed = 0.062 #km/h
        cable_bury_speed = 0.4 #km/h
        cable_splice_time = 48 #h
        cable_raise_time = 0.5 #h
        load_cable_at_port = cable_load_time #h
        time_to_transit_to_site_cable =	self.distance_to_installation_port/self.cable_transit_speed #h
        cable_position_on_site = site_position_time #h
        prepare_cable =	cable_prep_time #h 
        lower_cable = cable_lower_time #h
        pull_cable_in = cable_pull_in_time #h
        test_and_terminate = cable_termination_time #h
        cable_lay_and_burial = inter_cable_distance/cable_lay_bury_speed #h
        pull_cable_in_second = cable_pull_in_time #h
        test_and_terminate_second = cable_termination_time #h
        transit_between_cable =	(self.interdistance_turbines/self.cable_transit_speed) #h
        time_to_transit_to_port_cable =	self.distance_to_installation_port/self.cable_transit_speed #h
        cable_trips_needed = 100/25 #number
        complete_installation_sequence_time_cable = (load_cable_at_port + time_to_transit_to_site_cable+ time_to_transit_to_port_cable)*cable_trips_needed + (cable_position_on_site + prepare_cable + lower_cable + pull_cable_in + test_and_terminate + cable_lay_and_burial + pull_cable_in_second + test_and_terminate_second + transit_between_cable)*100 #h

        #delay factor because of weather 
        weather_delay_inter_array =	0.1996*self.mean_windspeed_at_10m - 0.2559

        #economic variable
        cost_of_sequence_installation_cable	= ((complete_installation_sequence_time_cable/24))*self.cable_day_rate
        cost_of_cable_installation = (cost_of_sequence_installation_cable/100)*weather_delay_inter_array

        #emissions variable
        fuel_consumption_transit_cable	=(time_to_transit_to_site_cable + transit_between_cable + time_to_transit_to_port_cable)*cable_trips_needed*self.cable_fuel_transit
        fuel_consumption_lay_bury_cable=	cable_lay_and_burial*100*self.cable_fuel_lay_bury
        fuel_consumption_installation_cable	=(cable_position_on_site + prepare_cable + lower_cable + pull_cable_in + test_and_terminate + pull_cable_in_second + test_and_terminate_second + transit_between_cable)*100*self.cable_fuel_installation
        emissions_cable_installation=	(fuel_consumption_transit_cable + fuel_consumption_lay_bury_cable + fuel_consumption_installation_cable)*self.emission_factor_MDO/100
        
        print('inter array installation = ' + str(cost_of_cable_installation)) if self.verbose else None
        print('inter array installation emissions = ' + str(emissions_cable_installation)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- inter-array decommissioning -------------------------------------------------#

        #decommissioning factors
        array_decom_red_factor = 1.5

        #economic varaibles
        array_decom_days_per_km	= (complete_installation_sequence_time_cable/(24*array_decom_red_factor))/(100*inter_cable_distance)
        array_decom_cost = array_decom_days_per_km*inter_cable_distance*(self.cable_day_rate)

        #emission variables
        array_decom_emissions=	emissions_cable_installation/array_decom_red_factor
        pre_decom_section	=0
        pre_decommissioning_factor=	0.1
        print('array decom = ' + str(array_decom_cost)) if self.verbose else None
        print('array decom emissions = ' + str(array_decom_emissions)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- OPEX -------------------------------------------------#

        # general cost shares
        port_activities_share =	0.31 #percentage
        operation_and_maintenance_share = 0.53 #percentage
        other_share = 0.16 #percentage

        #rates per year
        turbine_inspection_rate_per_year = 2.5
        cable_inspection_rate_per_year = 14
        substation_inspection_rate_per_year	= 7.5
        heavy_component_replacement	= 48/20
        large_component_replacement	= 22/20
        small_component_replacement	= (29+43)/20

        #OPEX duration
        duration_of_transport = (30/self.support_transit_speed)*2 #h
        work_day = 24 #h
        percentage_of_day_used_for_transport = duration_of_transport/work_day #h

        actual_work_time_turbine_inspection	= turbine_inspection_rate_per_year*(1-percentage_of_day_used_for_transport) #h
        actual_work_time_cable_inspection = cable_inspection_rate_per_year*(1-percentage_of_day_used_for_transport) #h
        actual_work_time_substation_inspection = substation_inspection_rate_per_year*(1-percentage_of_day_used_for_transport) #h

        actual_work_time_heavy = heavy_component_replacement*(1-percentage_of_day_used_for_transport) #h
        actual_work_time_large = large_component_replacement*(1-percentage_of_day_used_for_transport) #h
        actual_work_time_small = small_component_replacement*(1-percentage_of_day_used_for_transport) #h

        turbine_inspection_transit_time = (((self.distance_to_OPEX_port*2)/(self.support_transit_speed))*turbine_inspection_rate_per_year)/24
        cable_inspection_transit_time =	(((self.distance_to_OPEX_port*2)/(self.support_transit_speed))*cable_inspection_rate_per_year)/24
        substation_inspection_transit_time = (((self.distance_to_OPEX_port*2)/(self.support_transit_speed))*substation_inspection_rate_per_year)/24

        heavy_transit_time = (((self.distance_to_OPEX_port*2)/(self.WTIV_transit_speed))*heavy_component_replacement)/24
        large_transit_time = (((self.distance_to_OPEX_port*2)/(self.support_transit_speed))*large_component_replacement)/24
        small_transit_time = (((self.distance_to_OPEX_port*2)/(self.support_transit_speed))*small_component_replacement)/24

        #amount of turbines used in the study
        OPEX_study_number_of_turbines =	100

        #additional equipment for support vessel
        additional_equip_support_vessel = 45_000

        #blade colouring for visibility for birds
        paint_cost = 0
        yearly_blade_colour_time = 0 #days
        blade_colour_transit_time = 0 #hours
        if self.NID_type >= 2:
            time_per_colouring_session = NID['NID2']['coloured_blade_days']
            colouring_frequency = NID['NID2']['blade_recolouring_frequency']
            yearly_blade_colour_time = time_per_colouring_session*colouring_frequency
            blade_colour_transit_time = self.distance_to_OPEX_port*2/self.support_transit_speed
            paint_cost = NID['NID2']['paint_cost'] * colouring_frequency

        #cost per 10 MW turbine!
        OPEX_cost_per_turbine_inspection = (actual_work_time_turbine_inspection + turbine_inspection_transit_time)*(self.support_day_rate + additional_equip_support_vessel)*OPEX_study_number_of_turbines
        OPEX_cost_per_turbine_cable_inspection = ((actual_work_time_cable_inspection + cable_inspection_transit_time)*225000)
        OPEX_cost_per_turbine_substation_inspection = ((actual_work_time_substation_inspection + substation_inspection_transit_time)*(self.support_day_rate + additional_equip_support_vessel))
        OPEX_cost_per_turbine_heavy	= ((actual_work_time_heavy + heavy_transit_time + self.WTIV_mobilization_days)*self.WTIV_day_rate )
        OPEX_cost_per_turbine_large	= ((actual_work_time_large + large_transit_time)*(self.support_day_rate + additional_equip_support_vessel) )
        OPEX_cost_per_turbine_small	= ((actual_work_time_small + small_transit_time)*(self.support_day_rate + additional_equip_support_vessel) )
        OPEX_cost_per_turbine_blade_colour = paint_cost + (yearly_blade_colour_time + blade_colour_transit_time/24)*(self.support_day_rate + additional_equip_support_vessel)

        #yearly O&M for a 10 MW turbine
        yearly_OenM_for_one_turbine	= OPEX_cost_per_turbine_inspection + OPEX_cost_per_turbine_cable_inspection + OPEX_cost_per_turbine_substation_inspection + OPEX_cost_per_turbine_heavy + OPEX_cost_per_turbine_large + OPEX_cost_per_turbine_small + OPEX_cost_per_turbine_blade_colour

        #economic variable
        #yearly OPEX for a 10 MW turbine
        yearly_OPEX_for_one_turbine = yearly_OenM_for_one_turbine*(1/operation_and_maintenance_share)/OPEX_study_number_of_turbines
        yearly_OPEX_per_MW = yearly_OPEX_for_one_turbine/10
        
        #emission variable
        fuel_consumption_transit_OM_support=	(turbine_inspection_transit_time + cable_inspection_transit_time + substation_inspection_transit_time + large_transit_time + small_transit_time)*24*self.support_fuel_transit/OPEX_study_number_of_turbines
        fuel_consumption_transit_OM_WTIV	=heavy_transit_time*24*self.WTIV_fuel_transit_to_site/OPEX_study_number_of_turbines
        fuel_consumption_working_OM_support=	(actual_work_time_turbine_inspection + actual_work_time_cable_inspection + actual_work_time_substation_inspection + actual_work_time_large + actual_work_time_small)*24*self.support_fuel_working/OPEX_study_number_of_turbines
        fuel_consumption_working_OM_WTIV	=actual_work_time_heavy*24*self.WTIV_fuel_installation/OPEX_study_number_of_turbines
        fuel_consumption_blade_colouring = (blade_colour_transit_time*self.support_fuel_transit) + (yearly_blade_colour_time*24*self.support_fuel_working)
        yearly_emissions_OM	=((fuel_consumption_transit_OM_support + fuel_consumption_working_OM_support + fuel_consumption_blade_colouring)*self.emission_factor_MDO) + ((fuel_consumption_transit_OM_WTIV + fuel_consumption_working_OM_WTIV)*self.emission_factor_HFO)
        print('OPEX = ' + str(yearly_OPEX_for_one_turbine)) if self.verbose else None
        print('OPEX emissions = ' + str(yearly_emissions_OM)) if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------- FPV ---------------------------------------------------#
        #------------------------------------------------------------------------------------------------------------#

        #----------------------- solar panels --------------------#
    
        solar_panel_cost_per_kW = 781 #€/kW
        total_solar_panel_cost = solar_panel_cost_per_kW * (self.capacity_per_square_km*1000)
        
        solar_panel_emissions = 690 #kgCO2eq/panel
        total_solar_panel_emissions = solar_panel_emissions*self.solar_per_platform
        print('total solar panel cost = ' +str(total_solar_panel_cost)) if self.verbose else None
        print('total solar panel emissions (per platform) = ' +str(total_solar_panel_emissions)) if self.verbose else None

        #----------------------- platform structure --------------#

        cost_for_one_platform = 11_366.16
        baseline_emissions_for_one_platform = 16_000
        #dependent on wind energy
        total_platform_cost = ((self.mean_windspeed_at_10m - 5.7)*2634 + cost_for_one_platform)*self.amount_of_platforms_per_square_km
        total_platform_emissions = ((self.mean_windspeed_at_10m - 5.7)*3723 + baseline_emissions_for_one_platform)
        print('total platform cost = ' +str(total_platform_cost)) if self.verbose else None
        print('total platform emissions = ' +str(total_platform_emissions)) if self.verbose else None

        #----------------------- Anchoring and mooring -----------#
        anchors_per_platform = 3
        initial_breaking_load = 100_000 #N
        cost_per_meter = 50 #€/m
        emissions_per_meter_mooring = 10 #kgCO2eq/m
        total_mooring_line_cost = (cost_per_meter*self.depth)*anchors_per_platform
        total_mooring_line_emissions = (emissions_per_meter_mooring*self.depth)*anchors_per_platform

        total_anchoring_cost = ((0.052/9.81)*initial_breaking_load)*anchors_per_platform
        total_anchoring_emissions = 1900*anchors_per_platform #kgCO2eq

        A_M_cost = total_anchoring_cost + total_mooring_line_cost
        A_M_emissions = total_mooring_line_emissions + total_anchoring_emissions

        #influence of wind

        A_M_cost_wind = ((A_M_cost/(10 - 5.7))*(self.mean_windspeed_at_10m - 5.7) + A_M_cost)*self.amount_of_platforms_per_square_km
        A_M_emissions_wind = ((A_M_emissions/(10 - 5.7))*(self.mean_windspeed_at_10m - 5.7) + A_M_emissions)
        print('total A_M cost = ' +str(A_M_cost_wind)) if self.verbose else None
        print('total A_M emisisons per platform = ' +str(A_M_emissions_wind)) if self.verbose else None
        #------------------------ Balance of system --------------#

        BOS_per_kW = 500 #€/kW
        total_BOS_cost = BOS_per_kW*(self.capacity_per_square_km*1000)
        total_BoS_emissions = 750 #kgCO2eq 1 platform
        print('total BOS cost = ' +str(total_BOS_cost)) if self.verbose else None
        print('total BOS emissions = ' +str(total_BoS_emissions)) if self.verbose else None
        #------------------------- Inter-array cable -------------------#
        #overall factors
        burial_factor	=1
        excess_cable_factor	=1.1

        #cable specs
        ac_resistance_solar	=0.04
        capacitance_solar	=300
        conductor_size_solar	=630
        current_capacity_solar	=775
        inductance_solar	=0.35
        linear_density_solar	=42.5
        voltage_solar	=66
        line_frequency_solar = 60 #Hz

        #cable sizing equations
        conductance_solar =	1/ac_resistance_solar
        cable_num_solar =	complex(ac_resistance_solar,2 * 3.14 * line_frequency_solar *inductance_solar,)
        cable_den_solar	=complex(conductance_solar,2 * 3.14 *line_frequency_solar*capacitance_solar,)
        char_impedance_solar	=np.sqrt(cable_num_solar/cable_den_solar)
        phase_angle_solar	= math.atan(np.imag(char_impedance_solar) / np.real(char_impedance_solar))
        power_factor_solar	=math.cos(phase_angle_solar)
        cable_power_solar =	(np.sqrt(3)*voltage_solar*current_capacity_solar*power_factor_solar/ 1000)
        cable_capacity_solar=	cable_power_solar
        
        ######################################################
        ############# MULTI USE ADJUSTMENT ###################
        ######################################################

        #estimated distance to shared offshore substation!
        distance_to_sub_solar = 10 #km

        #economic variables
        cable_cost_per_km_solar	=400000
        cable_cost_solar = cable_cost_per_km_solar*((distance_to_sub_solar)*excess_cable_factor)
        cable_emission_per_km	=93537
        cable_emissions_solar = cable_emission_per_km*((distance_to_sub_solar)*excess_cable_factor)
        print('total cable cost = ' +str(cable_cost)) if self.verbose else None
        print('total cable emissions = ' +str(cable_emissions_solar)) if self.verbose else None
        

        #---------------------------------------------- OPEX -----------------------------------------------------#
        ######################################################
        ############# MULTI USE ADJUSTMENT ###################
        ######################################################
        #reduction in OPEX for solar due to multiuse

        yearly_OPEX_per_kW = 15*self.OPEX_reduction_solar #€/kW/year
        yearly_OPEX_solar = yearly_OPEX_per_kW*(self.capacity_per_square_km*1000)
        total_OPEX_over_lifetime = yearly_OPEX_solar*self.lifetime
        yearly_OM_emissions = 1.1*self.OPEX_reduction_solar #kgCO2eq per platform
        total_OM_emissions_over_lifetime = yearly_OM_emissions*self.lifetime
        print('total yearly OPEX = ' +str(yearly_OPEX_solar)) if self.verbose else None
        print('OPEX over lifetime = ' +str(total_OPEX_over_lifetime)) if self.verbose else None
        print('total yearly OM emissions = ' +str(yearly_OM_emissions)) if self.verbose else None
        print('OM emissions over lifetime = ' +str(total_OM_emissions_over_lifetime)) if self.verbose else None

        #------------------------------------ installation and decommissioning -----------------------------------#
        pre_capex_solar = (total_solar_panel_cost +
                     total_platform_cost +
                     A_M_cost_wind+
                     total_BOS_cost+
                     cable_cost_solar)
    
    
        installation_percentage = 0.05
        installation_cost = pre_capex_solar*installation_percentage
        installation_emissions = 750 #kgCO2eq per platform
        print('total installation cost = ' +str(installation_cost)) if self.verbose else None
        print('total installation emissions = ' +str(installation_emissions)) if self.verbose else None

        decommissioning_percentage = 0.02
        decommissioning_cost = pre_capex_solar*decommissioning_percentage
        decommissioning_emissions = 300 #kgCO2eq per platform
        print('total decom cost = ' +str(decommissioning_cost)) if self.verbose else None
        print('total decom emissions = ' +str(decommissioning_emissions)) if self.verbose else None

        pre_capex_solar = pre_capex_solar + installation_cost + decommissioning_cost


        total_emissions_solar = (total_solar_panel_emissions + total_platform_emissions + A_M_emissions_wind + total_BoS_emissions + total_OM_emissions_over_lifetime + installation_emissions + decommissioning_emissions)*self.amount_of_platforms_per_square_km + cable_emissions_solar





        #--------------------------------- substation hardware + export cable hardware -------------------------------------------------#
        # Substation (HVAC) overall cost estimation (HVAC 275kV, 433 capacity) ---> COST PER TURBINE
        
        if(self.distance_to_onshore_sub <= self.HVAC_distance):

            ######################################################
            ############### MULTI USE ADJUSTMENT #################
            ######################################################

            substation_standard_capacity = 27*15 #previously connected 27 turbines of 15 MW to one standard substation
            #turbine normalisation was introduced to give the cost of a standard substation for one turbine.
            #one turbine was placed on 2 km^2. On 2 km^2, a certain amount of solar can be placed (32x2 MW)
            #we introduced self.solar_reduction = 0.7 meaning only 70% of our solar capacity can be placed because of
            #restrictions with maintenance etc. This value can be adjusted of course!
            #this means that the new total capacity per turbine is:

            capacity_per_turbine_with_solar = self.turbine_rating + (self.capacity_per_square_km*(1/self.unit_density))*self.solar_reduction
            print('capacity per turbine with solar = ' +str(capacity_per_turbine_with_solar)) if self.verbose else None

            turbine_normalisation = substation_standard_capacity/capacity_per_turbine_with_solar

            #export cable cost & emissions
            export_cable_cost_per_km = 1_700_000
            export_cable_cost = (export_cable_cost_per_km /turbine_normalisation)*self.distance_to_onshore_sub
            export_cable_emissions_per_km = 260_000
            export_cable_emissions = (export_cable_emissions_per_km/turbine_normalisation)*self.distance_to_onshore_sub
            #print(export_cable_cost*turbine_normalisation)

            #offshore substation cost
            electrical_cost = 30_000_000/turbine_normalisation
            topside_structure = 28_000_000/turbine_normalisation
            fixed_costs = 11_500_000/turbine_normalisation
            monopile_substation_cost = 2_300_000/turbine_normalisation
            electrical_emissions = 6_790_000/turbine_normalisation
            topside_emissions = 4_100_000/turbine_normalisation
            monopile_substation_emissions = 1_475_000/turbine_normalisation

            #onshore substation cost
            onshore_substation_cost = 33_600_000/turbine_normalisation
            onshore_substation_emissions = 7_870_000/turbine_normalisation

            total_substation_cost_per_turbine = export_cable_cost + electrical_cost + topside_structure + fixed_costs + monopile_substation_cost + onshore_substation_cost
            total_substation_emissions_per_turbine = export_cable_emissions + electrical_emissions + topside_emissions + monopile_substation_emissions + onshore_substation_emissions
            print('substation hardware = ' + str(total_substation_cost_per_turbine)) if self.verbose else None
            print('substation hardware emissions = ' + str(total_substation_emissions_per_turbine)) if self.verbose else None
            
        else:
            ######################################################
            ############### MULTI USE ADJUSTMENT #################
            ######################################################

            substation_standard_capacity = 27*15 #previously connected 27 turbines of 15 MW to one standard substation
            #turbine normalisation was introduced to give the cost of a standard substation for one turbine.
            #one turbine was placed on 2 km^2. On 2 km^2, a certain amount of solar can be placed (32x2 MW)
            #we introduced self.solar_reduction = 0.7 meaning only 70% of our solar capacity can be placed because of
            #restrictions with maintenance etc. This value can be adjusted of course!
            #this means that the new total capacity per turbine is:

            capacity_per_turbine_with_solar = self.turbine_rating + (self.capacity_per_square_km*(1/self.unit_density))*self.solar_reduction
            print('capacity per turbine with solar = ' +str(capacity_per_turbine_with_solar)) if self.verbose else None

            #turbine_normalisation = 874/15 #one cable can carry 2000MW

            turbine_normalisation = 874/capacity_per_turbine_with_solar

            

            #export cable cost HVDC
            export_cable_cost_per_km = 1_300_000
            export_cable_cost = (export_cable_cost_per_km /27)*self.distance_to_onshore_sub
            export_cable_emissions_per_km = 120_000
            export_cable_emissions = (export_cable_emissions_per_km/turbine_normalisation)*self.distance_to_onshore_sub
            #print(export_cable_cost*turbine_normalisation)

            #offshore substation cost
            electrical_cost = 124_755_228/turbine_normalisation
            topside_structure = 203_000_000/turbine_normalisation
            monopile_substation_cost = 16_800_000/turbine_normalisation
            electrical_emissions = 7_690_000/turbine_normalisation
            topside_emissions = 29_400_000/turbine_normalisation
            monopile_substation_emissions = 10_560_000/turbine_normalisation

            #onshore substation cost
            onshore_substation_cost = 128_081_719/turbine_normalisation
            onshore_substation_emissions = 8_790_000/turbine_normalisation

            total_substation_cost_per_turbine = export_cable_cost + electrical_cost + topside_structure + monopile_substation_cost + onshore_substation_cost
            total_substation_emissions_per_turbine = export_cable_emissions + electrical_emissions + topside_emissions + monopile_substation_emissions + onshore_substation_emissions
            print('substation hardware = ' + str(total_substation_cost_per_turbine)) if self.verbose else None
            print('substation hardware emissions = ' + str(total_substation_emissions_per_turbine)) if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- substation installation -------------------------------------------------#

        #installation sequence
        amount_of_monopiles_per_sequence =	1 #number
        lift_monopiles_onto_deck = amount_of_monopiles_per_sequence*3 #h
        total_monopile_secure_time = mono_fasten_time*amount_of_monopiles_per_sequence #h

        time_to_transit_to_site_monopile = self.distance_to_installation_port/self.WTIV_transit_speed_loaded #h
        positioning_time_per_monopile = site_position_time #h
        jackup_time_per_monopile = (self.depth/self.WTIV_speed_below_depth)/60 #h
        total_jackup_and_positioning_time_monopile = (positioning_time_per_monopile + jackup_time_per_monopile)*amount_of_monopiles_per_sequence #h
        survey_site_ROV	= rov_survey_time #h
        release_monopile = mono_release_time #h
        upend_monopile = self.depth/self.WTIV_crane_rate #h
        lower_monopile_to_seabed = (self.depth)/(self.WTIV_crane_rate) #h
        equip_driving_equipment	= crane_reequip_time #h
        drive_monopile_into_seabed = self.depth/mono_drive_rate #h


        total_mono_time = (survey_site_ROV + release_monopile + upend_monopile + lower_monopile_to_seabed + equip_driving_equipment + drive_monopile_into_seabed)*amount_of_monopiles_per_sequence #h
        jackdown_time_per_monopile = ((self.depth/self.WTIV_speed_below_depth)/60)*amount_of_monopiles_per_sequence #h

        time_to_transit_to_port_monopile = self.distance_to_installation_port/self.WTIV_transit_speed #h

        complete_installation_sequence_time_monopile = lift_monopiles_onto_deck + total_monopile_secure_time+ time_to_transit_to_site_monopile + total_jackup_and_positioning_time_monopile + total_mono_time + jackdown_time_per_monopile + time_to_transit_to_port_monopile #h

        #delay factor because of weather 
        weather_delay_monopile = 0.0981*self.mean_windspeed_at_10m + 0.3612

        #economic variables
        cost_of_sequence_installation_monopile = ((complete_installation_sequence_time_monopile/24) + (self.WTIV_mobilization_days/100)*2)*(self.WTIV_day_rate + 200_000)
        cost_of_monopile_installation_substation = (cost_of_sequence_installation_monopile/amount_of_monopiles_per_sequence)*weather_delay_monopile
        
        #emission variables
        fuel_consumption_transit_monopile_sub =	(time_to_transit_to_site_monopile + time_to_transit_to_port_monopile)*self.WTIV_fuel_transit_to_site
        fuel_consumption_installation_monopile_sub =	(complete_installation_sequence_time_monopile - time_to_transit_to_site_monopile - time_to_transit_to_port_monopile)*self.WTIV_fuel_installation
        fuel_consumption_standby_monopile_sub	=(complete_installation_sequence_time_monopile*weather_delay_monopile + self.WTIV_mobilization_days*2*24)*self.WTIV_fuel_standby
        emissions_monopile_installation_sub	=(fuel_consumption_transit_monopile_sub + fuel_consumption_installation_monopile_sub + fuel_consumption_standby_monopile_sub)*self.emission_factor_HFO/amount_of_monopiles_per_sequence
        

        fasten_substation = 12 #h
        transit_to_site_substation = self.distance_to_installation_port/self.hlv_transit_speed_loaded
        release_substation = 2 #
        lift_topside = 6 #h
        attach_topside = 6 #h
        transit_to_port_substation = self.distance_to_installation_port/self.hlv_transit_speed
        heavy_lift_vessel_mobilization = 7

        total_topside_sequence_time = fasten_substation + transit_to_site_substation + release_substation + lift_topside + attach_topside + transit_to_port_substation
        #economic & emissions variables
        additional_equip_cost_substation = 300_000
        cost_of_sequence_installation_substation_topside = ((total_topside_sequence_time/24) + (heavy_lift_vessel_mobilization)*2)*(self.hlv_day_rate + self.hfv_day_rate + additional_equip_cost_substation)
        fuel_consumption_transit_monopile_sub_topside =	(transit_to_site_substation + transit_to_port_substation)*(self.hlv_fuel_transit_to_site + self.hfv_fuel_transit_to_site)
        fuel_consumption_installation_monopile_sub_topside =	(total_topside_sequence_time - transit_to_site_substation - transit_to_port_substation)*(self.WTIV_fuel_installation + self.hfv_fuel_installation)
        fuel_consumption_standby_monopile_sub_topside	=(complete_installation_sequence_time_monopile*weather_delay_monopile + self.WTIV_mobilization_days*2*24)*(self.WTIV_fuel_standby + self.hfv_fuel_standby)
        emissions_monopile_installation_sub_topside	=(fuel_consumption_transit_monopile_sub_topside + fuel_consumption_installation_monopile_sub_topside + fuel_consumption_standby_monopile_sub_topside)*self.emission_factor_HFO
        
        if(self.distance_to_onshore_sub <= self.HVAC_distance):        
            total_substation_installation_cost_per_turbine = (cost_of_monopile_installation_substation + cost_of_sequence_installation_substation_topside)/29
            total_substation_installation_emissions_per_turbine = (emissions_monopile_installation_sub + emissions_monopile_installation_sub_topside)/29
        else:
            HVDC_multiplicator = 2
            total_substation_installation_cost_per_turbine = ((cost_of_monopile_installation_substation + cost_of_sequence_installation_substation_topside)/29)*HVDC_multiplicator
            total_substation_installation_emissions_per_turbine = ((emissions_monopile_installation_sub + emissions_monopile_installation_sub_topside)/29)*HVDC_multiplicator
        
        print('substation installation = ' + str(total_substation_installation_cost_per_turbine)) if self.verbose else None
        print('substation installation emissions = ' + str(total_substation_installation_emissions_per_turbine)) if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- substation decommissioning -------------------------------------------------#

        Decommissioning_to_installation_factor_substation = 2
        substation_decom_cost = total_substation_installation_cost_per_turbine*Decommissioning_to_installation_factor_substation
        substation_decom_emissions = total_substation_installation_emissions_per_turbine*Decommissioning_to_installation_factor_substation
        print('substation decom = ' + str(substation_decom_cost)) if self.verbose else None
        print('substation decom emissions = ' + str(substation_decom_emissions)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- export cable installation -------------------------------------------------#
        export_hardware_to_installation_factor = 1
        export_installation_cost = export_cable_cost*export_hardware_to_installation_factor
        export_installation_emissions = export_cable_emissions*export_hardware_to_installation_factor
        print('export install = ' + str(export_installation_cost)) if self.verbose else None
        print('export install emissions = ' + str(export_installation_emissions)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- export cable decommissioning -------------------------------------------------#

        export_install_to_decom_factor = 3/4
        export_decommissioning_cost = export_installation_cost*export_install_to_decom_factor
        export_decommissioning_emissions = export_installation_emissions*export_install_to_decom_factor
        print('export decom = ' + str(export_decommissioning_cost)) if self.verbose else None
        print('export decom emissions= ' + str(export_decommissioning_emissions)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- Total costs and emissions for major components (turbine + foundation + inter-array) -------------------------------------------------#
        #adding all costs together pre insurance etc
        pre_total_CAPEX = (turbine_cost +                   #turbine hardware
                        turbine_decommissioning_cost +   #turbine decommissioning 
                        cost_of_turbine_installation +   #turbine installation
                        monopile_cost + transition_piece_cost + bird_attraction_cost +              #foundation hardware
                        cost_of_monopile_installation +  #foundation installation
                        monopile_decommissioning_cost +  #foundation decommissioning
                        scour_protection_cost +          #scour hardware
                        cost_of_scour_installation +     #scour installation
                        NID_seabed_hardware_cost +       #NID seabed hardware
                        cost_of_NID_seabed_installation + #NID seabed installation
                        corrosion_protection_cost+       #corrosion protection hardware
                        cable_cost+                      #cable hardware
                        cost_of_cable_installation+      #cable installation
                        array_decom_cost+                #cable decom   
                        total_substation_cost_per_turbine+ #substation hardware per turbine + export cable
                        total_substation_installation_cost_per_turbine+ #substation installation
                        substation_decom_cost+ #substation decom
                        export_installation_cost+ #export cable installation
                        export_decommissioning_cost + #cable decommissioning
                        pre_capex_solar*(1/self.unit_density)*self.solar_reduction)#solar CAPEX, but make sure it is over 2km^2 because this is for one turbine!!!!!!!!!!
        
        #OPEX multiplied by lifetime of project
        total_OPEX_over_lifetime = (yearly_OPEX_for_one_turbine + yearly_OPEX_solar*(1/self.unit_density)*self.solar_reduction)*self.lifetime
        print('pre total capex = ' + str(pre_total_CAPEX)) if self.verbose else None

        total_emissions = (turbine_emissions +
                        emissions_turbine_decom + 
                        emissions_turbine_installation + 
                        monopile_emissions + transition_piece_emissions + bird_attraction_emissions + 
                        emissions_monopile_installation + 
                        emissions_monopile_decom + 
                        scour_protection_emissions + 
                        emissions_scour_installation +
                        NID_seabed_emissions + 
                        emissions_NID_seabed_installation +  
                        cable_emission + 
                        emissions_cable_installation + 
                        array_decom_emissions + 
                        total_substation_emissions_per_turbine + 
                        total_substation_installation_emissions_per_turbine + 
                        substation_decom_emissions + 
                        export_installation_emissions + 
                        export_decommissioning_emissions + 
                        yearly_emissions_OM*self.lifetime +
                        total_emissions_solar*(1/self.unit_density)*self.solar_reduction)
        print(f'total emissions = {total_emissions:,.2f}') if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- engineering costs -------------------------------------------------#
        #fixed costs
        monopile_design	= 1500000
        secondary_steel_design = 600000

        #variable costs
        staffing_and_overhead = 60 #€/kW
        engineering_cost_factor	= 0.04

        #overall engineerinh cost for one turbine
        engineering_cost_overall = (monopile_design/(1000/15) + 
                                    secondary_steel_design/(1000/15) + 
                                    staffing_and_overhead*(self.turbine_rating +(self.capacity_per_square_km*(1/self.unit_density))*self.solar_reduction)*1000 + 
                                    0.04*pre_total_CAPEX)
        print('engineering = ' + str(engineering_cost_overall)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- insurance and development -------------------------------------------------#
        
        #fixed costs
        pre_front_end_design_study = 5000000
        FEED_study = 10000000
        site_assesment = 1000000
        construction_plan = 1000000
        environmental_impact_assessment = 2000000
        physical_resource_studies =	1500000
        other_studies = 2000000

        #variable costs
        plant_commissioning = 0.01 #of total pre capex
        meteo_tower_installation = 11500 #€/installed MW
        insurance_cost = 0.075 #of total pre capex

        #development and insurance cost for one turbine
        dev_and_insurance_total = (plant_commissioning*pre_total_CAPEX + 
                                   pre_front_end_design_study/(1000/15) + 
                                   FEED_study/(1000/15) + 
                                   site_assesment/(1000/15) + 
                                   construction_plan/(1000/15) + 
                                   environmental_impact_assessment/(1000/15) +
                                   physical_resource_studies/(1000/15) + 
                                   other_studies/(1000/15) + 
                                   meteo_tower_installation*15 + 
                                   insurance_cost*pre_total_CAPEX)
        
        print('dev and insurance = ' + str(dev_and_insurance_total)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #------------------------------------------- cost of financing ----------------------------------------------#
        
        #complete CAPEX
        total_CAPEX = pre_total_CAPEX + engineering_cost_overall + dev_and_insurance_total

        #weighted average cost of capital
        WACC = 0.075

        #debt life of CAPEX repayment + interests
        debt_lifetime_CAPEX	= int(np.floor((3/4)*self.lifetime))

        #amount that is financed with equity
        equity_financing_percentage = 0.30
        import numpy_financial as np_f
        CAPEX_financing =(-(np_f.pmt(WACC,debt_lifetime_CAPEX, total_CAPEX*(1-equity_financing_percentage)))*debt_lifetime_CAPEX) - total_CAPEX*(1-equity_financing_percentage)
        

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- Energy production -------------------------------------------------#

        #calcualte the mean yearly windspeed at hub height
        mean_windspeed = self.mean_windspeed_at_10m*((hub_height/10)**(1/10)) #m/s

        #Weibull distribution variables
        k = 2 #shape factor
        c = mean_windspeed #scale factor 

        #power specs of the 15MW turbine
        cut_in_windspeed = 3 #m/s
        rated_windspeed	= 10.5 #m/s
        cut_out_windspeed = 25 #m/s

        #variables to make a powercurve
        a = self.turbine_rating/((rated_windspeed**3) - (cut_in_windspeed**3))
        b = (cut_in_windspeed**3)/((rated_windspeed**3) - (cut_in_windspeed**3))

        #calculate all energy produced
        import scipy.integrate as integrate
        middle_energy_profile = integrate.quad(lambda wind_speed: ((k/c)*((wind_speed/c)**(k-1))*np.exp(-((wind_speed/c)**k)))*((365*24))*(a*wind_speed**3 - b*self.turbine_rating), cut_in_windspeed, rated_windspeed)[0]
        last_energy_profile = integrate.quad(lambda wind_speed: ((k/c)*((wind_speed/c)**(k-1))*np.exp(-((wind_speed/c)**k)))*((365*24))*self.turbine_rating, rated_windspeed, cut_out_windspeed)[0]

        #yearly energy produced per turbine
        yearly_total_energy_produced = middle_energy_profile + last_energy_profile

        if self.NID_type >= 2:
            yearly_total_energy_produced *= 1 - NID['NID2']['energy_reduction_curtialment']

        #should be restated to LCOE
        # price_of_one_MWh	=40.5
        # yearly_revenue	=price_of_one_MWh*yearly_total_energy_produced

        #--------------------------------- Solar energy production -------------------------------------------------#

        panel_area = 1.69 * 1.046
        efficiency = 0.226
        performance_ratio = 0.75
        #global tilted in kWh/m^2

        #in MWh/year!
        yearly_total_energy_produced_solar = self.amount_of_platforms_per_square_km*self.solar_per_platform*(panel_area*efficiency*performance_ratio*self.global_tilted_irradiation)/1000
        print('yearly energy produce solar  = ' +str(yearly_total_energy_produced_solar)) if self.verbose else None
        print(self.amount_of_platforms_per_square_km) if self.verbose else None
        print(self.solar_per_platform) if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- LCOE calculation -------------------------------------------------#
        #parameters

        #CAPEX: Total cost of wind farm/unit cell/cluster

        #OPEX: Total OPEX cost over the whole lifetime. Here we add the OPEX of every year and we'll distribute it evenly over all the lifetime years to avoid working with arrays
        #for OPEX costs of specific years. Can be done later!

        def LCOE(CAPEX, OPEX, yearly_energy_produced, lifetime, discount_rate,capacity_deg=0):
            
            #initialise the exponents for LCOE calculation; lifetime of a technology
            #See the general LCOE equation in the attached email. this is exponent 't' 

            Exponent = np.arange(lifetime)
            
            #Calculate the energy term, according to the capacity of the system,
            #the capacity factor and the capacity degredation
        
            #Energy production/year. Maximum capacity times number of days per year times the capacity factor (=percentage of time you actually have max capacity ( around 60% for windfarm))
            Energy = np.ones(lifetime)*yearly_energy_produced

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

        #the max capacity a turbine has (or your complete windfarm)
        max_capacity = self.turbine_rating
        #calculated with the effective energy produced based on windspeeds and the total energy it could produce if the turbine was turning at full capacity all the time. 
        capacity_factor = yearly_total_energy_produced/(self.turbine_rating*365*24)

        LCOE_per_MWh = LCOE(total_CAPEX, 
                            total_OPEX_over_lifetime, 
                            yearly_total_energy_produced + yearly_total_energy_produced_solar,
                            self.lifetime,
                            WACC,
                            capacity_deg=0)

        #print(f'test: {self.vars["third_var"]:,.2f}') # WHenever you want to print a number just put the number like this


        '''
        Final metrics 
        '''
        # STILL NEED TO DO TIMES THE UNIT DENSITY, THIS IS FOR ONE WHOLE TURBINE.
        self.final_metrics['capex'] = total_CAPEX 
        self.final_metrics['opex'] = total_OPEX_over_lifetime
        self.final_metrics['co2+'] = total_emissions
        self.final_metrics['co2-'] = (yearly_total_energy_produced + yearly_total_energy_produced_solar)*self.belgian_carbon_intensity
        self.final_metrics['revenue'] = (yearly_total_energy_produced + yearly_total_energy_produced_solar)*40 #40 being the general price (€/MWh) of energy
        self.final_metrics['LCOE'] = LCOE_per_MWh
        self.final_metrics['unit density'] = self.unit_density


        '''                                     Calculate NPV economics for optimization                                 '''



        def NPV_calculator(capex, opex, revenue_rate, production, discount_rate, commodity_importance,lifetime):
            
            '''
            general NPV formula: 

            NPV = SUM_i [ (revenue_i - (CAPEX + OPEX)_i)/(1 + discount rate)^i]
            '''
            
            #initialising variables
            NPV_nominator = []
            NPV_denominator = []
            exponent = np.arange(lifetime)

            #calculating denominator
            NPV_denominator[:] = ((1+discount_rate)**(exponent[:]+1))

            #calculating nominator
            #revenue
            yearly_revenue = revenue_rate*production*commodity_importance #calculates the revenue of the commodity in terms, taking into account it's imporance. 
            #opex
            yearly_opex = opex/lifetime
            #capex is just one cost at the beginning

            #actual nominator will be
            NPV_nominator= np.ones(lifetime)*(yearly_revenue - yearly_opex)
            
            NPV = np.sum(NPV_nominator/NPV_denominator) - capex    

            return NPV
        
        NPV = NPV_calculator(total_CAPEX, total_OPEX_over_lifetime, self.energy_revenue, (yearly_total_energy_produced + yearly_total_energy_produced_solar), self.WACC, 1, self.lifetime)

        self.final_metrics['Economic NPV'] = NPV


        '''                                     Calculate NPV_carbon for optimization                                 '''


        def NPV_calculator_emissions(initial_emissions,
                                    total_yearly_emissions,
                                    cost_of_emissions,
                                    negative_emissions,
                                    discount_rate_emissions,
                                    carbon_intensity ,
                                    commodity_importance,
                                    lifetime):
            
            '''
            general NPV formula: 

            NPV = SUM_i [ (revenue_i - (CAPEX + OPEX)_i)/(1 + discount rate)^i]
            '''
            
            #initialising variables
            NPV_nominator = []
            NPV_denominator = []
            exponent = np.arange(lifetime)

            #calculating denominator
            NPV_denominator[:] = ((1+discount_rate_emissions)**(exponent[:]+1))

            #calculating nominator
            #revenue
            yearly_mitigation_revenue = cost_of_emissions*negative_emissions*carbon_intensity*commodity_importance #calculates the revenue of the commodity in terms, taking into account it's imporance. 
            #opex
            yearly_emission_cost = total_yearly_emissions*cost_of_emissions/lifetime
            #capex is just one cost at the beginning

            #actual nominator will be
            NPV_nominator= np.ones(lifetime)*(yearly_mitigation_revenue - yearly_emission_cost)
            
            NPV_emissions = np.sum(NPV_nominator/NPV_denominator) - initial_emissions*cost_of_emissions 

            return NPV_emissions
        
        
        NPV_emissions = NPV_calculator_emissions(total_emissions - ((yearly_emissions_OM + yearly_OM_emissions*(1/self.unit_density)*self.solar_reduction)*self.lifetime),
                                                (yearly_emissions_OM + yearly_OM_emissions*(1/self.unit_density)*self.solar_reduction)*self.lifetime,
                                                self.cost_of_carbon/1000, #cost per kg
                                                yearly_total_energy_produced,
                                                self.discount_rate_emissions,
                                                self.belgian_carbon_intensity,
                                                1,
                                                self.lifetime)

        self.final_metrics['carbon NPV'] = NPV_emissions


        

    def run(self):
        self._calculations()
        return self.final_metrics