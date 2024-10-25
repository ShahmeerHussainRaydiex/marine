''' 
This is a test file to show how to structure a tech file
Adjust and add as many calcs as required
'''

from src.marine_plan.pre_compute.tech_equations.NID_specs import NID

class jacket_15MW:
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
        self.HVAC_distance = 80 #km

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
        self.cable_fuel_installation	=573 #l/h
        self.cable_fuel_lay_bury	=573 #l/h
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
        self.hfv_max_deck_space=2000      # m^2
        self.hfv_max_waveheight= 2.5       # m
        self.hfv_max_windspeed= 20         # m/s
        self.hfv_transit_speed= 20          # km/h
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
        self.hlv_max_windspeed= 20         # km/h
        self.hlv_transit_speed_loaded= 14  # km/h
        self.hlv_transit_speed= 7          # km/h
        self.hlv_day_rate= 500000          # USD/day
        self.hlv_fuel_transit_to_site = 2050 #l/h
        self.hlv_fuel_installation = 1140 #l/h
        self.hlv_fuel_standby = 854 #l/h

        #-----------------------------------------------------------------------# 


        self.vars = {} # Calculated variables go in here

        self.final_metrics = {} # Final metrics go in here

    def _calculations(self, found_age):
        
        import numpy as np
        import math

        self.lifetime = found_age
        self.turbine_iterations = self.lifetime/25

        #------------------------------------ TURBINE HARDWARE ----------------------------#
        #dimensions of turbine 

        rotor_diameter = 263 #m

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
        turbine_cost_per_MW = 1006650 #€/MW
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
        #--------------------- jacket foundation hardware ----------------------------------------------------------#
        #hardware calculation jacket
        turbine_rating_jacket_equations = 10 #MW. the jackets are calculated with 10MW turbines because this is what ORBIT used for referencing. going higher results in blown up number.
        #we can still use this estimate since it's in the right cost range i think. 
        #we can use a corrective factor for the main lattice mass
        main_lattice_cost_per_tonnes = 4680 #€/tonnes
        main_lattice_footprint = 226 #m^2
        pile_cost_per_tonnes = 2250 #€/tonnes
        pile_length	= 293 #m
        pile_diameter = 2 #m

        main_lattice_mass_corrective_factor = 1

        main_lattice_mass =	math.exp(3.71+ 0.00176*(turbine_rating_jacket_equations)**2.5 + 0.645*np.log(self.depth))
        pile_mass = 8*(main_lattice_mass)**0.5574

        amount_of_legs = 4

        #economic variables
        main_lattice_cost = main_lattice_mass*main_lattice_cost_per_tonnes
        pile_cost = pile_mass*pile_cost_per_tonnes

        #emission variables
        jacket_emissions_per_tonnes	=1886
        jacket_emissions=	jacket_emissions_per_tonnes*(main_lattice_mass + pile_mass)

        transition_piece_mass = 1 / ((-0.0131 + 0.0381) / np.log(turbine_rating_jacket_equations) - 0.00000000227*(self.depth**3))

        #economic variable
        transition_piece_cost_per_tonnes = 4500 #€/tonnes
        transition_piece_cost = transition_piece_mass*transition_piece_cost_per_tonnes

        #emission variable
        transition_piece_emissions_per_tonnes =	1886
        transition_piece_emissions = transition_piece_emissions_per_tonnes*transition_piece_mass
        print('jacket hardware = ' + str(main_lattice_cost + transition_piece_cost + pile_cost)) if self.verbose else None
        print('jacket hardware emissions =' + str(jacket_emissions + transition_piece_emissions)) if self.verbose else None

        #NID bird attraction measure
        bird_attraction_cost = 0
        bird_attraction_emissions = 0
        if self.NID_type == 3:
            bird_attraction_cost = NID['NID3']['bird_attraction_cost']
            bird_attraction_emissions = NID['NID3']['bird_attraction_emissions']
        print(f'bird attraction cost = {bird_attraction_cost:,.2f}') if self.verbose else None
        print(f'bird attraction emissions = {bird_attraction_emissions:,.2f}') if self.verbose else None
        
        #-------------------------------------------------------------------------------------------------------------#
        #---------------------------------- jacket installation ----------------------------------------------------#
        #installation times
        jacket_fasten_time = 12 #h
        jacket_release_time	= 6 #h
        jacket_lift_time = 4 #h
        jacket_lower_time =	8 #h
        jacket_grout_time =	8 #h
        jacket_pin_template_time = 4 #h
        jacket_pile_drive_time = 6 #h
        jacket_position_pile = 6 #h
        jacket_vessel_reposition = 4 #h
        jacket_suction_bucket = 12 #h
        site_position_time = 2 #h
        rov_survey_time	= 1 #h
        crane_reequip_time = 1 #h
        bird_attraction_fasten_time = 0
        bird_attraction_release_time = 0
        bird_attraction_bolt_time = 0
        if self.NID_type == 3:
            bird_attraction_fasten_time = 8
            bird_attraction_release_time = 2
            bird_attraction_bolt_time = 4

        #installation sequence
        amount_of_jackets_per_sequence = np.floor(self.WTIV_max_cargo/(main_lattice_mass + transition_piece_mass + pile_mass)) #h
        lift_jackets_onto_deck = amount_of_jackets_per_sequence*3 #h
        total_jacket_secure_time = jacket_fasten_time*amount_of_jackets_per_sequence #h
        total_bird_attraction_secure_time = bird_attraction_fasten_time*amount_of_jackets_per_sequence #h
        time_to_transit_to_site_jacket = self.distance_to_installation_port/self.WTIV_transit_speed #h

        positioning_time_per_jacket	= site_position_time #h
        jackup_time_per_jacket = (self.depth/self.WTIV_speed_below_depth)/60 #h
        total_jackup_and_positioning_time_jacket = (positioning_time_per_jacket + jackup_time_per_jacket)*amount_of_jackets_per_sequence #h
        survey_site_ROV	= rov_survey_time #h
        release_jacket = jacket_release_time #h
        lift_jacket	= jacket_lift_time #h
        lower_jacket = jacket_lower_time #h
        grout_jacket = jacket_grout_time #h
        jacket_template = jacket_pin_template_time #h
        drive_piles	= jacket_pile_drive_time*amount_of_legs #h
        position_pile = jacket_position_pile*amount_of_legs #h
        reposition_pile = jacket_vessel_reposition*(amount_of_legs-1) #h
        release_bird_attraction = bird_attraction_release_time #h
        lift_bird_attraction = 20/self.WTIV_crane_rate #h
        bolt_bird_attraction = bird_attraction_bolt_time #h
        total_install_jacket = (survey_site_ROV + release_jacket + lift_jacket + lower_jacket + grout_jacket + jacket_template + drive_piles + position_pile + reposition_pile +release_bird_attraction + lift_bird_attraction + bolt_bird_attraction)*amount_of_jackets_per_sequence #h
        jackdown_time_per_jacket = ((self.depth/self.WTIV_speed_below_depth)/60)*amount_of_jackets_per_sequence #h
        transit_between_jackets	= (self.interdistance_turbines/self.WTIV_transit_speed_in_field)*(amount_of_jackets_per_sequence - 1) #h
        time_to_transit_to_port_jacket = self.distance_to_installation_port/self.WTIV_transit_speed #h

        #weather delay factor
        weather_delay_jacket = 0.0981*self.mean_windspeed_at_10m + 0.3612 

        complete_installation_sequence_time_jacket = lift_jackets_onto_deck + total_jacket_secure_time + total_bird_attraction_secure_time + time_to_transit_to_site_jacket + total_jackup_and_positioning_time_jacket + total_install_jacket + jackdown_time_per_jacket + transit_between_jackets + time_to_transit_to_port_jacket

        #economic calculation
        additional_cost_equipment_and_compensation = 400_000
        cost_of_sequence_installation_jacket = ((complete_installation_sequence_time_jacket/24) + (self.WTIV_mobilization_days/100)*2)*(self.WTIV_day_rate + additional_cost_equipment_and_compensation)
        cost_of_jacket_installation	= (cost_of_sequence_installation_jacket/amount_of_jackets_per_sequence)*weather_delay_jacket

        #Emission calculations
        fuel_consumption_transit_jacket=	(complete_installation_sequence_time_jacket - time_to_transit_to_site_jacket - transit_between_jackets - time_to_transit_to_port_jacket)*self.WTIV_fuel_installation
        fuel_consumption_installation_jacket=	(complete_installation_sequence_time_jacket - time_to_transit_to_site_jacket - transit_between_jackets - time_to_transit_to_port_jacket)*self.WTIV_fuel_installation
        fuel_consumption_standby_jacket	=(complete_installation_sequence_time_jacket*weather_delay_jacket + self.WTIV_mobilization_days*2*24)*self.WTIV_fuel_standby
        emissions_jacket_installation	=(fuel_consumption_transit_jacket + fuel_consumption_installation_jacket + fuel_consumption_standby_jacket)*self.emission_factor_HFO/amount_of_jackets_per_sequence

        print('jacket installation = ' + str(cost_of_jacket_installation)) if self.verbose else None
        print('jacket install emissions = ' +str(emissions_jacket_installation)) if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- jacket decommissioning -------------------------------------------------#
        #decommissioning sequence
        amount_of_barges_jacket = 2
        jackup_transit_to_jacket = self.distance_to_installation_port/self.WTIV_transit_speed #h
        tug_boat_transit_to_jacket = (self.distance_to_installation_port/self.tug_boat_transit_speed)*amount_of_barges_jacket #h
        positioning_time_per_jacket_decom = 6 #h
        jackup_time_per_jacket_decom = (self.depth/self.WTIV_speed_below_depth)/60 #h
        cut_jacket = 48 #h
        lift_foundation_out_sea	= 3 #h
        jackdown_time_per_jacket_decom = ((self.depth/self.WTIV_speed_below_depth)/60) #h
        transit_between_jackets_decom = (self.interdistance_turbines/self.WTIV_transit_speed_in_field) #h
        barge_transit_to_unload_jacket = (self.distance_to_installation_port/self.tug_boat_transit_speed_with_load)*amount_of_barges_jacket #h
        barge_transit_back_jacket = (self.distance_to_installation_port/self.tug_boat_transit_speed)*amount_of_barges_jacket #h
        total_jacket_decom_time = positioning_time_per_jacket_decom + jackup_time_per_jacket_decom + cut_jacket + lift_foundation_out_sea + jackdown_time_per_jacket_decom + transit_between_jackets_decom #h
        averaged_initial_transit_jacket	= (jackup_transit_to_jacket*2)/100#h

        #economic variable
        jacket_decommissioning_cost = ((total_jacket_decom_time + averaged_initial_transit_jacket)/24)*(self.WTIV_day_rate + (self.barge_day_rate + self.tug_boat_day_rate)*amount_of_barges_jacket) +(((self.WTIV_mobilization_days*2*self.WTIV_day_rate)))/100

        #emissions variable
        fuel_decom_jacket_WTIV	=(averaged_initial_transit_jacket*self.WTIV_fuel_transit_to_site) + (transit_between_jackets_decom*self.WTIV_fuel_transit_in_field) + ((total_jacket_decom_time - transit_between_jackets_decom)*self.WTIV_fuel_installation) + (self.WTIV_mobilization_days*2*24*self.WTIV_fuel_standby/100)
        fuel_decom_jacket_barge_tug=	(tug_boat_transit_to_jacket + barge_transit_to_unload_jacket)*self.tug_boat_fuel_transit
        emissions_jacket_decom	=(fuel_decom_jacket_WTIV*self.emission_factor_HFO) + (fuel_decom_jacket_barge_tug*self.emission_factor_MDO)
        print('jacket decom = ' + str(jacket_decommissioning_cost)) if self.verbose else None
        print('jacket decom emissions = ' +str(emissions_jacket_decom)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- scour hardware -------------------------------------------------#
        scour_protection_cost =	116_000                                                               
        scour_mass	= scour_protection_cost/50
        if self.NID_type >= 1:
            scour_mass *= 1.2
        if self.NID_type == 3:
            scour_mass *= 1.075

        print('scour protection cost hardware = €' +str(f'{scour_protection_cost:,}')) if self.verbose else None

        corrosion_protection_cost = 348000

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

            scour_protection_material_cost = 50 
            scour_density = 2600 #kg/m^3
            
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
        drop_rocks_time	= 10 #h
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
        #--------------------------------- substation hardware + export cable hardware -------------------------------------------------#
        # Substation (HVAC) overall cost estimation (HVAC 275kV, 433 capacity) ---> COST PER TURBINE
        
        if(self.distance_to_onshore_sub <= self.HVAC_distance):
            turbine_normalisation = 27

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
            jacket_substation_cost = 5_622_000/turbine_normalisation
            electrical_emissions = 6_790_000/turbine_normalisation
            topside_emissions = 4_100_000/turbine_normalisation
            jacket_substation_emissions = 2_090_000/turbine_normalisation

            #onshore substation cost
            onshore_substation_cost = 33_600_000/turbine_normalisation
            onshore_substation_emissions = 7_870_000/turbine_normalisation

            total_substation_cost_per_turbine = export_cable_cost + electrical_cost + topside_structure + fixed_costs + jacket_substation_cost + onshore_substation_cost
            total_substation_emissions_per_turbine = export_cable_emissions + electrical_emissions + topside_emissions + jacket_substation_emissions + onshore_substation_emissions
            print('substation hardware = ' + str(total_substation_cost_per_turbine)) if self.verbose else None    
            print('substation hardware emissions = ' + str(total_substation_emissions_per_turbine)) if self.verbose else None

        else:
            #HVDC

            turbine_normalisation = 874/15 #one cable can carry 2000MW

            #export cable cost HVDC
            export_cable_cost_per_km = 1_300_000
            export_cable_cost = (export_cable_cost_per_km /27)*self.distance_to_onshore_sub
            export_cable_emissions_per_km = 120_000
            export_cable_emissions = (export_cable_emissions_per_km/turbine_normalisation)*self.distance_to_onshore_sub
            #print(export_cable_cost*turbine_normalisation)

            #offshore substation cost
            electrical_cost = 124_755_228/turbine_normalisation
            topside_structure = 203_000_000/turbine_normalisation
            jacket_substation_cost = 37_210_611/turbine_normalisation
            electrical_emissions = 7_690_000/turbine_normalisation
            topside_emissions = 29_400_000/turbine_normalisation
            jacket_substation_emissions = 12_415_000/turbine_normalisation

            #onshore substation cost
            onshore_substation_cost = 128_081_719/turbine_normalisation
            onshore_substation_emissions = 8_790_000/turbine_normalisation

            total_substation_cost_per_turbine = export_cable_cost + electrical_cost + topside_structure + jacket_substation_cost + onshore_substation_cost
            total_substation_emissions_per_turbine = export_cable_emissions + electrical_emissions + topside_emissions + jacket_substation_emissions + onshore_substation_emissions
            print('substation hardware = ' + str(total_substation_cost_per_turbine)) if self.verbose else None
            print('substation hardware emissions = ' + str(total_substation_emissions_per_turbine)) if self.verbose else None
        
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- substation installation -------------------------------------------------#

        #installation times
        jacket_fasten_time = 12 #h
        jacket_release_time	= 6 #h
        jacket_lift_time = 4 #h
        jacket_lower_time =	8 #h
        jacket_grout_time =	8 #h
        jacket_pin_template_time = 4 #h
        jacket_pile_drive_time = 6 #h
        jacket_position_pile = 6 #h
        jacket_vessel_reposition = 4 #h
        jacket_suction_bucket = 12 #h
        site_position_time = 2 #h
        rov_survey_time	= 1 #h
        crane_reequip_time = 1 #h

        #installation sequence
        amount_of_jackets_per_sequence = 1
        lift_jackets_onto_deck = amount_of_jackets_per_sequence*3 #h
        total_jacket_secure_time = jacket_fasten_time*amount_of_jackets_per_sequence #h
        time_to_transit_to_site_jacket = self.distance_to_installation_port/self.WTIV_transit_speed #h

        positioning_time_per_jacket	= site_position_time #h
        jackup_time_per_jacket = (self.depth/self.WTIV_speed_below_depth)/60 #h
        total_jackup_and_positioning_time_jacket = (positioning_time_per_jacket + jackup_time_per_jacket)*amount_of_jackets_per_sequence #h
        survey_site_ROV	= rov_survey_time #h
        release_jacket = jacket_release_time #h
        lift_jacket	= jacket_lift_time #h
        lower_jacket = jacket_lower_time #h
        grout_jacket = jacket_grout_time #h
        jacket_template = jacket_pin_template_time #h
        drive_piles	= jacket_pile_drive_time*amount_of_legs #h
        position_pile = jacket_position_pile*amount_of_legs #h
        reposition_pile = jacket_vessel_reposition*(amount_of_legs-1) #h
        total_install_jacket = (survey_site_ROV + release_jacket + lift_jacket + lower_jacket + grout_jacket + jacket_template + drive_piles + position_pile + reposition_pile)*amount_of_jackets_per_sequence #h
        jackdown_time_per_jacket = ((self.depth/self.WTIV_speed_below_depth)/60)*amount_of_jackets_per_sequence #h
        transit_between_jackets	= (self.interdistance_turbines/self.WTIV_transit_speed_in_field)*(amount_of_jackets_per_sequence - 1) #h
        time_to_transit_to_port_jacket = self.distance_to_installation_port/self.WTIV_transit_speed #h

        #weather delay factor
        weather_delay_jacket = 0.0981*self.mean_windspeed_at_10m + 0.3612 

        complete_installation_sequence_time_jacket = lift_jackets_onto_deck + total_jacket_secure_time + time_to_transit_to_site_jacket + total_jackup_and_positioning_time_jacket + total_install_jacket + jackdown_time_per_jacket + transit_between_jackets + time_to_transit_to_port_jacket

        #economic calculation
        additional_cost_equipment_and_compensation = 400_000
        cost_of_sequence_installation_jacket = ((complete_installation_sequence_time_jacket/24) + (self.WTIV_mobilization_days/100)*2)*(self.WTIV_day_rate + additional_cost_equipment_and_compensation)
        cost_of_jacket_installation_substation = (cost_of_sequence_installation_jacket/amount_of_jackets_per_sequence)*weather_delay_jacket

        #Emission calculations
        fuel_consumption_transit_jacket_sub =	(time_to_transit_to_site_jacket + time_to_transit_to_port_jacket)*self.WTIV_fuel_transit_to_site
        fuel_consumption_installation_jacket_sub =	(complete_installation_sequence_time_jacket - time_to_transit_to_site_jacket - time_to_transit_to_port_jacket)*self.WTIV_fuel_installation
        fuel_consumption_standby_jacket_sub	=(complete_installation_sequence_time_jacket*weather_delay_jacket + self.WTIV_mobilization_days*2*24)*self.WTIV_fuel_standby
        emissions_jacket_installation_sub	=(fuel_consumption_transit_jacket_sub + fuel_consumption_installation_jacket_sub + fuel_consumption_standby_jacket_sub)*self.emission_factor_HFO/amount_of_jackets_per_sequence
        

        fasten_substation = 12 #h
        transit_to_site_substation = self.distance_to_installation_port/self.hlv_transit_speed
        release_substation = 2 #
        lift_topside = 6 #h
        attach_topside = 6 #h
        transit_to_port_substation = self.distance_to_installation_port/self.hlv_transit_speed
        heavy_lift_vessel_mobilization = 7

        total_topside_sequence_time = fasten_substation + transit_to_site_substation + release_substation + lift_topside + attach_topside + transit_to_port_substation
        #economic variables
        additional_equip_cost_substation = 300_000
        cost_of_sequence_installation_substation_topside = ((total_topside_sequence_time/24) + (heavy_lift_vessel_mobilization)*2)*(self.hlv_day_rate + self.hfv_day_rate + additional_equip_cost_substation)
        fuel_consumption_transit_monopile_sub_topside =	(transit_to_site_substation + transit_to_port_substation)*(self.hlv_fuel_transit_to_site + self.hfv_fuel_transit_to_site)
        fuel_consumption_installation_monopile_sub_topside =	(total_topside_sequence_time - transit_to_site_substation - transit_to_port_substation)*(self.WTIV_fuel_installation + self.hfv_fuel_installation)
        fuel_consumption_standby_monopile_sub_topside	=(complete_installation_sequence_time_jacket*weather_delay_jacket + self.WTIV_mobilization_days*2*24)*(self.WTIV_fuel_standby + self.hfv_fuel_standby)
        emissions_jacket_installation_sub_topside	=(fuel_consumption_transit_monopile_sub_topside + fuel_consumption_installation_monopile_sub_topside + fuel_consumption_standby_monopile_sub_topside)*self.emission_factor_HFO
        
        if(self.distance_to_onshore_sub <= self.HVAC_distance):
            total_substation_installation_cost_per_turbine = (cost_of_jacket_installation_substation + cost_of_sequence_installation_substation_topside)/29
            total_substation_installation_emissions_per_turbine = (emissions_jacket_installation_sub + emissions_jacket_installation_sub_topside)/29
        else:
            HVDC_mulitplier = 2
            total_substation_installation_cost_per_turbine = ((cost_of_jacket_installation_substation + cost_of_sequence_installation_substation_topside)/29)*HVDC_mulitplier
            total_substation_installation_emissions_per_turbine = ((emissions_jacket_installation_sub + emissions_jacket_installation_sub_topside)/29)*HVDC_mulitplier
        
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
        #--------------------------------- Total costs for major components (turbine + foundation + inter-array) -------------------------------------------------#
        #adding all costs together pre insurance etc

        cost_of_turbine_installation *= self.turbine_iterations
        turbine_decommissioning_cost *= self.turbine_iterations

        pre_total_CAPEX = (turbine_cost +                   #turbine hardware
                        turbine_decommissioning_cost +   #turbine decommissioning 
                        cost_of_turbine_installation +   #turbine installation
                        main_lattice_cost + pile_cost + transition_piece_cost + bird_attraction_cost +     #foundation hardware
                        cost_of_jacket_installation +  #foundation installation
                        jacket_decommissioning_cost +  #foundation decommissioning
                        scour_protection_cost +          #scour hardware
                        cost_of_scour_installation +     #scour installation
                        NID_seabed_hardware_cost +       #NID seabed hardware
                        cost_of_NID_seabed_installation + #NID seabed installation
                        corrosion_protection_cost+       #corrosion protection hardware
                        cable_cost+                      #cable hardware
                        cost_of_cable_installation+      #cable installation
                        array_decom_cost+                #cable decom   
                        total_substation_cost_per_turbine+ #substation hardware per turbine + EXPORT CABLE
                        total_substation_installation_cost_per_turbine+ #substation installation
                        substation_decom_cost+ #substation decom
                        export_installation_cost+ #export cable installation
                        export_decommissioning_cost) #cable decommissioning

        #OPEX multiplied by lifetime of project
        print('pre total capex = ' + str(pre_total_CAPEX)) if self.verbose else None
        total_OPEX_over_lifetime = (yearly_OPEX_for_one_turbine)*self.lifetime

        total_emissions = (turbine_emissions +
                        emissions_turbine_decom + 
                        emissions_turbine_installation + 
                        jacket_emissions + transition_piece_emissions + bird_attraction_emissions +
                        emissions_jacket_installation + 
                        emissions_jacket_decom + 
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
                        yearly_emissions_OM*self.lifetime)
        print('total emissions = ' + str(total_emissions)) if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- engineering costs -------------------------------------------------#
        #fixed costs
        jacket_design	= 3000000
        secondary_steel_design = 600000

        #variable costs
        staffing_and_overhead = 60 #€/kW
        engineering_cost_factor	= 0.04

        #overall engineerinh cost for one turbine
        engineering_cost_overall = jacket_design/(1000/15) + secondary_steel_design/(1000/15) + staffing_and_overhead*15*1000 + 0.04*pre_total_CAPEX
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
        dev_and_insurance_total = plant_commissioning*pre_total_CAPEX + pre_front_end_design_study/(1000/15) + FEED_study/(1000/15) + site_assesment/(1000/15) + construction_plan/(1000/15) + environmental_impact_assessment/(1000/15) +physical_resource_studies/(1000/15) + other_studies/(1000/15) + meteo_tower_installation*15 + insurance_cost*pre_total_CAPEX
        print('dev and insurance = ' + str(dev_and_insurance_total)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- cost of financing -------------------------------------------------#
        
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

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- LCOE calculation -------------------------------------------------#
        #parameters

        #CAPEX: Total cost of wind farm/unit cell/cluster

        #OPEX: Total OPEX cost over the whole lifetime. Here we add the OPEX of every year and we'll distribute it evenly over all the lifetime years to avoid working with arrays
        #for OPEX costs of specific years. Can be done later!

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

        #the max capacity a turbine has (or your complete windfarm)
        max_capacity = self.turbine_rating
        #calculated with the effective energy produced based on windspeeds and the total energy it could produce if the turbine was turning at full capacity all the time. 
        capacity_factor = yearly_total_energy_produced/(self.turbine_rating*365*24)

        LCOE_per_MWh = LCOE(total_CAPEX, total_OPEX_over_lifetime, max_capacity, self.lifetime, WACC, capacity_factor, capacity_deg=0)

        #print(f'test: {self.vars["third_var"]:,.2f}') # WHenever you want to print a number just put the number like this


        '''
        Final metrics 
        '''

        self.install_decom = {}

        # STILL NEED TO DO TIMES THE UNIT DENSITY, THIS IS FOR ONE WHOLE TURBINE.
        self.final_metrics['capex'] = total_CAPEX 
        self.final_metrics['opex'] = total_OPEX_over_lifetime / self.lifetime
        self.final_metrics['co2+'] = total_emissions
        self.final_metrics['co2-'] = yearly_total_energy_produced*self.belgian_carbon_intensity
        self.final_metrics['value'] = yearly_total_energy_produced*40 #40 being the general price (€/MWh) of energy
        self.final_metrics['LCOE'] = LCOE_per_MWh
        self.final_metrics['unit density'] = self.unit_density
        self.final_metrics['energy_produced'] = yearly_total_energy_produced
        self.final_metrics['food_produced'] = 0
        self.final_metrics['lifetime'] = 25
        self.install_decom['foundation_install_cost'] = cost_of_jacket_installation
        self.install_decom['foundation_install_emissions'] = emissions_jacket_installation
        self.install_decom['foundation_decom_cost'] = jacket_decommissioning_cost
        self.install_decom['foundation_decom_emissions'] = emissions_jacket_decom
        self.install_decom['turbine_install_cost'] = cost_of_turbine_installation
        self.install_decom['turbine_install_emissions'] = emissions_turbine_installation
        self.install_decom['turbine_decom_cost'] = turbine_decommissioning_cost
        self.install_decom['turbine_decom_emissions'] = emissions_turbine_decom


    def run(self, found_age):
        self._calculations(found_age)
        return self.final_metrics, self.install_decom