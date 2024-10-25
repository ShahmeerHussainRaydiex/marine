''' 
This is a test file to show how to structure a tech file
Adjust and add as many calcs as required
'''

from src.marine_plan.pre_compute.tech_equations.NID_specs import NID

class semisub_15MW_cat_drag:
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
        self.energy_revenue = 40 #€/MWh 
        self.WACC = 0.075 #%
        self.cost_of_carbon = 80 #€/tonnes CO2e
        self.HVAC_distance = 80
        self.discount_rate_emissions = 0.075

        #Converting metrics
        self.short_ton = 907.185 #ton

        #emission factors
        self.emission_factor_MDO	= 3.53 #
        self.emission_factor_HFO	= 3.41 #

        #Belgian carbon intensity
        self.belgian_carbon_intensity = 172 #kgCO2eq/MWh

        #unit density and interdistance
        self.unit_density = 1/3 #units per square km
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
        self.scour_fuel_transit = 1772 #
        self.scour_fuel_installation	= 2658 #

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

        #anchor handling vessel
        self.ahv_max_waveheight = 3 #m
        self.ahv_max_windspeed = 15 #m/s
        self.ahv_transit_speed = 18 #km/h
        self.ahv_towing_speed = 6 #km/s
        self.ahv_day_rate = 100_000 #€/day
        self.ahv_fuel_transit = 600 #l/h
        self.ahv_fuel_transit_tug = 1200 #l/h
        self.ahv_fuel_installation = 1000 #l/h

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
        turbine_cost_per_MW = 20_000_000*1.16/15 #€/MW
        turbine_cost = turbine_cost_per_MW*self.turbine_rating #€ cost for one turbine

        #Emission calculations
        turbine_emissions_per_MW = 718246 #kgCO2eq/MW
        turbine_emissions = turbine_emissions_per_MW*self.turbine_rating #kgCO2eq for one turbine
       
        #----------------------------------------------------------------------------------#
        #----------------------- semi sub hardware ------------------------------------------#

        #dimensioning of semi submersible
        semi_submersible_stiffened_column_mass = -0.9571*(self.turbine_rating**2) + 40.89*self.turbine_rating +802.09 #tonnes
        semi_submersible_truss_mass	= 2.7894*(self.turbine_rating**2) + 15.591*self.turbine_rating + 266.03 #tonnes 
        semi_sub_heave_plate_mass = -0.4397*(self.turbine_rating**2) + 21.545*self.turbine_rating + 177.42 #tonnes
        secondary_steel_mass = -0.153*(self.turbine_rating**2) +6.54*self.turbine_rating + 128.34 #tonnes
            
        #economic variables
        auxiliary_systems_and_corrosion_protection = 1488000 #€
        semi_submersible_stiffened_column_cost = semi_submersible_stiffened_column_mass*2932.8 
        semi_submersible_truss_cost	= 5875*semi_submersible_truss_mass
        semi_sub_heave_plate_cost = 5875*semi_sub_heave_plate_mass
        secondary_steel_cost = 6815*secondary_steel_mass	
        corossion_protection = 20_000_000*1.16/30
        total_floating_substructure_cost_for_one_turbine = auxiliary_systems_and_corrosion_protection + semi_submersible_stiffened_column_cost + semi_submersible_truss_cost + semi_sub_heave_plate_cost + secondary_steel_cost + corossion_protection

        #emission variables
        semi_submersible_stiffened_column_emissions	= 3677*semi_submersible_stiffened_column_mass
        semi_submersible_truss_emissions	= 2996*semi_submersible_truss_mass
        semi_sub_heave_plate_emissions	=  2996*semi_sub_heave_plate_mass
        secondary_steel_emissions	= 1428*secondary_steel_mass
        total_floating_substructure_emissions_for_one_turbine	= semi_submersible_stiffened_column_emissions + semi_submersible_truss_emissions + semi_sub_heave_plate_emissions + secondary_steel_emissions

        #NID bird attraction measure
        bird_attraction_cost = 0
        bird_attraction_emissions = 0
        if self.NID_type == 3:
            bird_attraction_cost = NID['NID3']['bird_attraction_cost']
            bird_attraction_emissions = NID['NID3']['bird_attraction_emissions']
        print(f'bird attraction cost = {bird_attraction_cost:,.2f}') if self.verbose else None
        print(f'bird attraction emissions = {bird_attraction_emissions:,.2f}') if self.verbose else None
        
        #----------------------------------------------------------------------------------#

        #----------------------- turbine + semisub install ------------------------------------------#

        #initial specs
        towing_day_rate	= 35000 #€/day
        ahv_day_rate = 100000 #€/day
        three_ahvs_with_combined_dayrate = 156000 #€/day	
        
        #installation sequence
        move_substructure_to_assembly = 8 #h
        prepare_for_assembly = 168 #h
        tower_sections = 2 #number
        lift_and_attach_towersection = 4 #h
        lift_and_attach_nacelle	= 12 #h
        lift_and_attach_blade =	3.5 #h
        mechanical_completion = 24 #h
        electrical_completion = 72 #h
        prepare_semi_sub_for_installation =	12 #h
        ballast_semi_sub = 6 #h
        prepare_for_tow_to_site	= 12 #h
        connect_mooring_to_float = 33 #h
        array_cable_hookup = 33 #h
        lift_and_attach_bird_attraction = 0
        if self.NID_type == 3:
            lift_and_attach_bird_attraction = 4
            
        fixed_total_hours_per_structure = move_substructure_to_assembly + (lift_and_attach_towersection)*tower_sections + lift_and_attach_nacelle + lift_and_attach_blade*3 + lift_and_attach_bird_attraction + mechanical_completion + electrical_completion + prepare_semi_sub_for_installation + ballast_semi_sub + prepare_for_tow_to_site + connect_mooring_to_float + array_cable_hookup + prepare_for_assembly/30 #h
            
        tow_to_site = self.distance_to_installation_port/self.ahv_towing_speed #h
            
        transit_back_to_shore = self.distance_to_installation_port/self.ahv_transit_speed #h

        #economic variables	
        heavy_lifting_and_moving_services =	450000 #€
        tech_services = 65000 #€	
        total_installation_floating_tur_semi = ((fixed_total_hours_per_structure + tow_to_site + transit_back_to_shore)/24)*three_ahvs_with_combined_dayrate + heavy_lifting_and_moving_services + tech_services 
        
        #emissions
        fuel_consumption_transit = (move_substructure_to_assembly + tow_to_site)*self.ahv_fuel_transit_tug + (transit_back_to_shore*self.ahv_fuel_transit)
        fuel_consumption_installation = (fixed_total_hours_per_structure - move_substructure_to_assembly)*self.ahv_fuel_installation
        emissions_semi_sub_installation	= (fuel_consumption_transit + fuel_consumption_installation)*3*self.emission_factor_MDO #times 3 since 3 vessels needed for tugigng and positioning of turbine + substructure
        
        #----------------------------------------------------------------------------------#

        #------------------------------ drag embedment hardware ----------------------------#

        #hardware dimensions
        amount_of_anchors = 3

        fit_mooring	= (-0.0004 * (self.turbine_rating**2)) + 0.0132 * self.turbine_rating + 0.0536 #number
        fIt_mooring_15MW = 0.1616 #in interval
            
        line_diameter = 0.15 #m
        line_mass_per_meter	= 0.450 #tonnes/m
        breaking_load_catenary = 419449 * (line_diameter**2) + 93415 *line_diameter  - 3577.9 #kN
        extra_line_length = 500 #m
        line_length = 0.0002 * (self.depth**2) + 1.264 * self.depth + 47.776 + extra_line_length #m
        line_mass = line_length*line_mass_per_meter #tonnes
        drag_embedment_mass	= 20 #tonnes

        #economic variables
        mooring_line_cost_per_meter = 1088 #€/m
        line_cost = line_length*mooring_line_cost_per_meter #€
        drag_embedment_cost	= breaking_load_catenary / 9.81 / 20.0 * 2000.0 #€
       
        total_mooring_anchoring_catenary_per_turbine = (drag_embedment_cost + line_cost)*amount_of_anchors #€
        
        #emission variables
        mooring_emissions = 1665
        drag_embedment_emissions = 1799
        emissions_mooring_anchoring_catenary_per_turbine = (mooring_emissions*line_mass + drag_embedment_emissions*drag_embedment_mass)*amount_of_anchors

        
        #-------------------------------------------------------------------------------------------------------------#
        #--------------------- anchoring and mooring installation ----------------------------------------------------------#
        #installation times	
        mooring_site_survey	= 4 #h
        mooring_system_load_time = 5 #h
        suction_pile_installtion_time = 11 #h
        drag_embedment_installation_time = 12 #h
        anchors_and_moorings_per_trip =	6 #number
        total_trips_needed = 100*amount_of_anchors/anchors_and_moorings_per_trip #number
        average_distance_between_anchors = 1 #km

        #installation sequence
        load_mooring_system	= mooring_system_load_time*total_trips_needed #h
        transit_to_anchor = total_trips_needed*(self.distance_to_installation_port/self.ahv_transit_speed) #h
        mooring_survey_time = mooring_site_survey*100*amount_of_anchors #h
        anchor_positioning_time	= mooring_site_survey*100*amount_of_anchors #h
        drag_anchor_instal_time = (drag_embedment_installation_time + 0.05*self.depth)*100*amount_of_anchors #h
        mooring_install_time = 2*100*amount_of_anchors #h
        transit_to_next_anchor = average_distance_between_anchors*(100-1)/self.ahv_transit_speed #h
        transit_to_harbor = total_trips_needed*(self.distance_to_installation_port/self.ahv_transit_speed) #h
            
        total_anchor_installation_time	=(load_mooring_system + transit_to_anchor + mooring_survey_time + anchor_positioning_time + drag_anchor_instal_time + mooring_install_time +  transit_to_next_anchor + transit_to_harbor)/100 #h

        #estimated weather delay
        weather_factor=	(3/2)

        additional_anchor_equipment_cost = 50_000

        #economic variable	
        anchor_installation_cost = ((total_anchor_installation_time/24)*(ahv_day_rate + additional_anchor_equipment_cost))*weather_factor
    
        #emission variable
        fuel_consumption_transit_anchor = (transit_to_anchor + transit_to_harbor + transit_to_next_anchor)*self.ahv_fuel_transit/100
        fuel_consumption_installation_anchor = (total_anchor_installation_time - (transit_to_anchor - transit_to_next_anchor - transit_to_harbor)/100)*self.ahv_fuel_installation
        emissions_anchor_installation = (fuel_consumption_transit_anchor + fuel_consumption_installation_anchor)*self.emission_factor_MDO

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- seabed NID hardware -------------------------------------------------#
        scour_protection_material_cost = 50
        scour_density = 2600 #kg/m^3
        scour_protection_material_emissions = 25
        
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
        #--------------------------------- seabed NID installation -------------------------------------------------#
        drop_rocks_time	= 30 #h
        load_rocks_time = 4 #h
        scour_position_on_site = 2 #h
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

        #preset variables
        power_factor =0.95
        burial_factor	=0
        excess_cable_factor	=1.1

        #cable specs
        #XLPE 630mm 66 kV 	
        ac_resistance = 0.04
        capacitance	= 300
        conductor_size = 630
        current_capacity = 775
        inductance = 0.35
        linear_density = 42.5
        voltage	= 66
        line_frequency = 60 #Hz

        #cable calculations
        conductance = 1/ac_resistance
        cable_num = complex(ac_resistance,2 * math.pi * line_frequency *inductance,)
        cable_den = complex(conductance,2 * math.pi *line_frequency*capacitance,)
        char_impedance = np.sqrt(cable_num/cable_den) 
        phase_angle	= math.atan(np.imag(char_impedance) / np.real(char_impedance))
        power_factor = math.cos(phase_angle)
        cable_power = (np.sqrt(3)*voltage*current_capacity*power_factor/ 1000)
            
        cable_capacity = cable_power

        inter_cable_distance = self.interdistance_turbines
            
        #calculate cable length
        system_angle = (-0.0047*self.depth) + 18.743 #angle
        dynamic_factor = 2
        catenary_length_factor = 0.5
        free_hanging_cable_length =	(((self.depth/math.cos(system_angle*3.14/180))*(catenary_length_factor+1))+190)/1000 #m

        fixed_cable_length = (inter_cable_distance*1000 - (2*math.tan(system_angle*3.14/180)*self.depth) - 70)/1000 #m

        #economic variable
        cable_cost_per_km = 400000 #€/km
        cable_accessorie_cost = 20_000_000/30 #€
        cable_cost = cable_cost_per_km*(fixed_cable_length*excess_cable_factor) + cable_cost_per_km*(free_hanging_cable_length*excess_cable_factor*dynamic_factor) + cable_accessorie_cost

        #emission variable
        cable_emission_per_km	=93537
        cable_accessorie_emission= 16000
        cable_emission=	cable_emission_per_km*(fixed_cable_length*excess_cable_factor + free_hanging_cable_length*excess_cable_factor*dynamic_factor) + cable_accessorie_emission


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
        site_position_time = 2 #h
        rov_survey_time	= 1 #h
        crane_reequip_time = 1 #h

        #installation sequence
        load_cable_at_port = cable_load_time #h
        time_to_transit_to_site_cable =	self.distance_to_installation_port/self.cable_transit_speed #h
        cable_position_on_site = site_position_time #h	
        prepare_cable = cable_prep_time #h	
        lower_cable = cable_lower_time #h
        pull_cable_in = cable_pull_in_time #h
        test_and_terminate = cable_termination_time #h
        cable_lay_and_burial = inter_cable_distance/cable_lay_bury_speed #h
        pull_cable_in_second = cable_pull_in_time #h	
        test_and_terminate_second = cable_termination_time #h
        transit_between_cable = (self.interdistance_turbines/self.cable_transit_speed) #h	
        time_to_transit_to_port_cable = self.distance_to_installation_port/self.cable_transit_speed #h	
        cable_trips_needed	=100/25 #number
                
        floating_difficulty_increase = 1.5
            
        complete_installation_sequence_time_cable	=((load_cable_at_port + time_to_transit_to_site_cable+ time_to_transit_to_port_cable)*cable_trips_needed + (cable_position_on_site + prepare_cable + lower_cable + pull_cable_in + test_and_terminate + cable_lay_and_burial + pull_cable_in_second + test_and_terminate_second + transit_between_cable)*100 )*floating_difficulty_increase

        #weather delay factos	
        weather_delay_inter_array = 0.1996*self.mean_windspeed_at_10m - 0.2559

        #economic variables
        cost_of_sequence_installation_cable=	((complete_installation_sequence_time_cable/24))*self.cable_day_rate
        cost_of_cable_installation	=(cost_of_sequence_installation_cable/100)*weather_delay_inter_array
        
        #emission variables
        fuel_consumption_transit_cable	=(time_to_transit_to_site_cable + transit_between_cable + time_to_transit_to_port_cable)*cable_trips_needed*self.cable_fuel_transit
        fuel_consumption_lay_bury_cable	=cable_lay_and_burial*100*self.cable_fuel_lay_bury
        fuel_consumption_installation_cable	=(cable_position_on_site + prepare_cable + lower_cable + pull_cable_in + test_and_terminate + pull_cable_in_second + test_and_terminate_second + transit_between_cable)*100*self.cable_fuel_installation
        emissions_cable_installation=	(fuel_consumption_transit_cable + fuel_consumption_lay_bury_cable + fuel_consumption_installation_cable)*self.emission_factor_MDO/100


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
        floating_OPEX_factor = 1.3
        yearly_OPEX_for_one_turbine = (yearly_OenM_for_one_turbine*(1/operation_and_maintenance_share)/OPEX_study_number_of_turbines)*floating_OPEX_factor

        yearly_OPEX_per_MW = yearly_OPEX_for_one_turbine/10
        
        #emission variable
        fuel_consumption_transit_OM_support=	(turbine_inspection_transit_time + cable_inspection_transit_time + substation_inspection_transit_time + large_transit_time + small_transit_time)*24*self.support_fuel_transit/OPEX_study_number_of_turbines
        fuel_consumption_transit_OM_WTIV	=heavy_transit_time*24*self.WTIV_fuel_transit_to_site/OPEX_study_number_of_turbines
        fuel_consumption_working_OM_support=	(actual_work_time_turbine_inspection + actual_work_time_cable_inspection + actual_work_time_substation_inspection + actual_work_time_large + actual_work_time_small)*24*self.support_fuel_working/OPEX_study_number_of_turbines
        fuel_consumption_working_OM_WTIV	=actual_work_time_heavy*24*self.WTIV_fuel_installation/OPEX_study_number_of_turbines
        fuel_consumption_blade_colouring = (blade_colour_transit_time*self.support_fuel_transit) + (yearly_blade_colour_time*24*self.support_fuel_working)
        yearly_emissions_OM	=((fuel_consumption_transit_OM_support + fuel_consumption_working_OM_support*floating_OPEX_factor + fuel_consumption_blade_colouring*floating_OPEX_factor)*self.emission_factor_MDO) + ((fuel_consumption_transit_OM_WTIV + fuel_consumption_working_OM_WTIV*floating_OPEX_factor)*self.emission_factor_HFO)
        print('OPEX = ' + str(yearly_OPEX_for_one_turbine)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- substation hardware + export cable hardware -------------------------------------------------#
        # Substation (HVAC) overall cost estimation (HVAC 275kV, 433 capacity) ---> COST PER TURBINE
        if(self.distance_to_onshore_sub <= self.HVAC_distance):
            turbine_normalisation = 27

            #export cable cost
            export_cable_cost_per_km = 1_700_000
            system_angle_export = -0.0047*self.depth + 18.743
            export_hang_length = ((self.depth/math.cos(system_angle_export*3.14/180))*(0.5+1))+190
            export_accessory_cost = (154666.66)+(297773)+(96666)
            dynamic_factor = 2
            export_cable_cost = (export_cable_cost_per_km*((self.distance_to_onshore_sub)*excess_cable_factor) + export_accessory_cost + (export_hang_length/1000)*dynamic_factor*export_cable_cost_per_km)/turbine_normalisation
            export_cable_emissions_per_km = 260_000
            export_accessory_emission= 16000
            export_cable_emissions = (export_cable_emissions_per_km*self.distance_to_onshore_sub*excess_cable_factor + export_accessory_emission + (export_hang_length/1000)*dynamic_factor*export_cable_emissions_per_km)/turbine_normalisation
            
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
            
        else:
            #HVDC

            turbine_normalisation = 874/15 #one cable can carry 2000MW

            #export cable cost HVDC
            export_cable_cost_per_km = 1_300_000
            system_angle_export = -0.0047*self.depth + 18.743
            export_hang_length = ((self.depth/math.cos(system_angle_export*3.14/180))*(0.5+1))+190
            export_accessory_cost = (154666.66)+(297773)+(96666)
            dynamic_factor = 2
            export_cable_cost = (export_cable_cost_per_km*((self.distance_to_onshore_sub)*excess_cable_factor) + export_accessory_cost + (export_hang_length/1000)*dynamic_factor*export_cable_cost_per_km)/27
            export_cable_emissions_per_km = 120_000
            export_accessory_emission= 16000
            export_cable_emissions = (export_cable_emissions_per_km*self.distance_to_onshore_sub*excess_cable_factor + export_accessory_emission + (export_hang_length/1000)*dynamic_factor*export_cable_emissions_per_km)/27
            

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

        amount_of_legs = 4

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
        
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- substation decommissioning -------------------------------------------------#

        Decommissioning_to_installation_factor_substation = 2
        substation_decom_cost = total_substation_installation_cost_per_turbine*Decommissioning_to_installation_factor_substation
        substation_decom_emissions = total_substation_installation_emissions_per_turbine*Decommissioning_to_installation_factor_substation
        print('substation decom = ' + str(substation_decom_cost)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- export cable installation -------------------------------------------------#
        export_hardware_to_installation_factor = 1
        export_installation_cost = export_cable_cost*export_hardware_to_installation_factor
        export_installation_emissions = export_cable_emissions*export_hardware_to_installation_factor
        print('export install = ' + str(export_installation_cost)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- export cable decommissioning -------------------------------------------------#

        export_install_to_decom_factor = 3/4
        export_decommissioning_cost = export_installation_cost*export_install_to_decom_factor
        export_decommissioning_emissions = export_installation_emissions*export_install_to_decom_factor
        print('export decom= ' + str(export_decommissioning_cost)) if self.verbose else None

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- floating decommissioning -------------------------------------------------#

        floating_wind_decom_factor = 0.8

        #economic variable	
        floating_wind_decom_cost = (total_installation_floating_tur_semi + cost_of_cable_installation + anchor_installation_cost)*floating_wind_decom_factor
        print('floating decom = ' + str(floating_wind_decom_cost)) if self.verbose else None
        #emission variable
        floating_wind_decom_emissions = (emissions_semi_sub_installation + emissions_cable_installation + emissions_anchor_installation)*floating_wind_decom_factor
        
       
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- Total costs for major components (turbine + foundation + inter-array) -------------------------------------------------#
        #adding all costs together pre insurance etc
        pre_total_CAPEX = (turbine_cost +                   #turbine hardware
                        total_floating_substructure_cost_for_one_turbine + bird_attraction_cost +  #floating cost
                        total_installation_floating_tur_semi +   #installation
                        total_mooring_anchoring_catenary_per_turbine +      #mooring and anchroing hardware
                        anchor_installation_cost +  #foundation installation
                        NID_seabed_hardware_cost +       #NID seabed hardware
                        cost_of_NID_seabed_installation + #NID seabed installation
                        cable_cost+                      #cable hardware
                        cost_of_cable_installation+      #cable installation              
                        total_substation_cost_per_turbine+ #substation hardware per turbine + export cable
                        total_substation_installation_cost_per_turbine+ #substation installation
                        substation_decom_cost+ #substation decom
                        export_installation_cost+ #export cable installation
                        export_decommissioning_cost+
                        floating_wind_decom_cost) #floating components decommissioning

        #OPEX multiplied by lifetime of project
        total_OPEX_over_lifetime = (yearly_OPEX_for_one_turbine)*self.lifetime
        print('pre total capex = ' + str(pre_total_CAPEX)) if self.verbose else None
        
        total_emissions = (turbine_emissions +
                        total_floating_substructure_emissions_for_one_turbine + bird_attraction_emissions +
                        emissions_semi_sub_installation + 
                        emissions_mooring_anchoring_catenary_per_turbine +
                        emissions_anchor_installation +  
                        NID_seabed_emissions + 
                        emissions_NID_seabed_installation +  
                        cable_emission + 
                        emissions_cable_installation + 
                        total_substation_emissions_per_turbine + 
                        total_substation_installation_emissions_per_turbine + 
                        substation_decom_emissions + 
                        export_installation_emissions + 
                        export_decommissioning_emissions + 
                        floating_wind_decom_emissions + 
                        yearly_emissions_OM*self.lifetime)
        print('total emissions = ' + str(total_emissions)) if self.verbose else None
                
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- engineering costs -------------------------------------------------#
        #fixed costs
        floating_design	= 3000000
        secondary_steel_design = 600000

        #variable costs
        staffing_and_overhead = 60 #€/kW
        engineering_cost_factor	= 0.04

        #overall engineerinh cost for one turbine
        engineering_cost_overall = floating_design/(1000/15) + secondary_steel_design/(1000/15) + staffing_and_overhead*15*1000 + 0.04*pre_total_CAPEX
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
        # STILL NEED TO DO TIMES THE UNIT DENSITY, THIS IS FOR ONE WHOLE TURBINE.
        self.final_metrics['capex'] = total_CAPEX 
        self.final_metrics['opex'] = total_OPEX_over_lifetime
        self.final_metrics['co2+'] = total_emissions
        self.final_metrics['co2-'] = yearly_total_energy_produced*self.belgian_carbon_intensity
        self.final_metrics['revenue'] = yearly_total_energy_produced*40 #40 being the general price (€/MWh) of energy
        self.final_metrics['LCOE'] = LCOE_per_MWh
        self.final_metrics['unit density'] = self.unit_density
        self.final_metrics['energy_produced'] = yearly_total_energy_produced
        self.final_metrics['food_produced'] = 0
        self.final_metrics['lifetime'] = self.lifetime



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
        
        NPV = NPV_calculator(total_CAPEX, total_OPEX_over_lifetime, self.energy_revenue, yearly_total_energy_produced, self.WACC, 1, self.lifetime)

        # self.final_metrics['Economic NPV'] = NPV


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
        
        
        NPV_emissions = NPV_calculator_emissions(total_emissions - (yearly_emissions_OM*self.lifetime),
                                                yearly_emissions_OM*self.lifetime,
                                                self.cost_of_carbon/1000, #cost per kg
                                                yearly_total_energy_produced,
                                                self.discount_rate_emissions,
                                                self.belgian_carbon_intensity,
                                                1,
                                                self.lifetime)

        # self.final_metrics['carbon NPV'] = NPV_emissions

    def run(self, found_age):
        self._calculations()
        return self.final_metrics, {}