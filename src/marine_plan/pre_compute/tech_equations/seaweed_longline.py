''' 
This is a test file to show how to structure a tech file
Adjust and add as many calcs as required
'''

class seaweed_longline:
    def __init__(self, geo_data, verbose = False):

        import numpy as np
        import math

        self.verbose = verbose

        #initialising geo-spatial data dictionary
        self.depth = geo_data['depth'] #m
        self.distance_to_OPEX_port = geo_data['distance_to_OPEX_port'] #km
        self.food_revenue = 3 
        self.WACC = 0.075
        self.cost_of_carbon = 80
        self.discount_rate_emissions = 0.075

        #Converting metrics
        self.short_ton = 907.185 #kg

        #emission factors
        self.emission_factor_MDO	= 3.53 #kgCO2eq/l
        self.emission_factor_HFO	= 3.41 #kgCO2eq/l

        #unit density and interdistance
        self.unit_density = 36 #units per square km

        #Belgian carbon intensity
        self.belgian_carbon_intensity = 172 #kgCO2eq/MWh

        #lifetime
        self.lifetime_primary = 12 #years
        self.lifetime_secondary = 3 #years

        #workday
        self.workday = 12 #h
        self.sailing_time_workday = self.workday-2 #h

        #---------------------------------- vessel library -----------------------------# 
        # Large workboat

        self.large_workboat_transit_speed = 25.9 #km/h
        self.large_workboat_day_rate = 3000 #€/day
        self.large_workboat_fuel_transit = 200 #l/h
        self.large_workboat_fuel_installation = 80 #l/h

        # Small workboat

        self.small_workboat_transit_speed = 22.2 #km/h
        self.small_workboat_day_rate = 3000 #€/day
        self.small_workboat_fuel_transit = 120 #l/h
        self.small_workboat_fuel_installation = 30 #l/h

        # Small fast boat

        self.small_fast_boat_transit_speed = 33.3 #km/h
        self.small_fast_boat_day_rate = 3000 #€/day
        self.small_fast_boat_fuel_transit = 100 #l/h
        self.small_fast_boat_fuel_installation = 25 #l/h


        #-----------------------------------------------------------------------# 


        self.vars = {} # Calculated variables go in here

        self.final_metrics = {} # Final metrics go in here

    def _calculations(self):
        
        import numpy as np
        import math
        #------------------------------------ plot design ----------------------------#
        longline_length = 100 #m
        longlines_per_plot = 5
        net_length = 96 #m
        net_depth = 3 #m
        net_surface = net_length * net_depth
        total_net_surface_per_plot = net_surface*longlines_per_plot
        anchors_per_plot = longlines_per_plot*2
        small_buoys_per_plot = 10*longlines_per_plot

        #------------------------------------ primary structure hardware ----------------------------#
        #longline dimensions -> material used = propylene (PP)
        longline_density = 910 #kg/m^3
        longline_diameter = 0.02 + 0.048*(self.depth-10)/20 #m
        longline_mass_per_m = math.pi*((longline_diameter/2)**2)*longline_density #kg/m
        total_longline_mass = longline_mass_per_m*longline_length #kg

        #Mooring line dimensions -> material used = grill anchor chain/steel
        mooring_length = 1.5*self.depth #m
        mooring_mass_per_m = 4.5 + 7.5*(self.depth-10)/20 #kg/m
        total_mooring_mass = mooring_mass_per_m*mooring_length #kg

        #Anchor dimensions -> mateiral used = steel
        anchor_mass = 500 + 1500*(self.depth-10)/20 #kg

        #Buoy dimensions -> used material = high-density polyethelene (HDPE)
        corner_buoy_mass = 50 + 400*(self.depth-10)/20 #kg
        small_buoy_mass = 10 + 30*(self.depth-10)/20 #kg

        #Economic calculations
        #longlines
        longline_cost_per_kg = 5 #€
        longline_cost_per_plot = longline_cost_per_kg*total_longline_mass*longlines_per_plot

        longline_replacement_lifetime_cost = longline_cost_per_plot*(self.lifetime_primary/self.lifetime_secondary) #4 replacements over 12 years

        #buoys
        spar_marker_buoy_cost = 127 #per buoy
        marker_buoy_cost_per_plot = spar_marker_buoy_cost*small_buoys_per_plot #total cost for buoys per plot
        bouy_replacement = marker_buoy_cost_per_plot*(0.10)*self.lifetime_primary #every year 10% of bouys are replaced

        #support line cost
        support_line_cost_per_meter_longline = 8.16 #€/m longline
        support_line_cost_per_plot = support_line_cost_per_meter_longline*longline_length*longlines_per_plot

        #floating support
        floating_support_cost_per_meter_longline = 16 #€/m longline
        floating_support_cost_per_plot = floating_support_cost_per_meter_longline*longline_length*longlines_per_plot

        #anchoring and mooring costs
        steel_cost = 2.75 #€/kg
        mooring_cost = total_mooring_mass*steel_cost*anchors_per_plot
        anchor_cost = anchor_mass*steel_cost*anchors_per_plot

        #other costs
        pre_capex = (longline_cost_per_plot +
                     longline_replacement_lifetime_cost +
                     marker_buoy_cost_per_plot +
                     bouy_replacement +
                     support_line_cost_per_plot +
                     floating_support_cost_per_plot +
                     mooring_cost +
                     anchor_cost)
        
        precontruction_cost = 0.027 * pre_capex
        insurance_cost = 0.013 * pre_capex

        #Emission calculations
        emission_factor_PP = 2.75 #kgCO2eq/kg
        emission_factor_steel = 2.2 #kgCO2eq/kg
        emission_factor_HDPE = 2.35 #kgCO2eq/kg
        primary_structure_emissions_per_plot = total_longline_mass*longlines_per_plot*(self.lifetime_primary/self.lifetime_secondary)*emission_factor_PP + (total_mooring_mass+anchor_mass)*anchors_per_plot*emission_factor_steel + (corner_buoy_mass*anchors_per_plot + small_buoy_mass*small_buoys_per_plot + small_buoy_mass*small_buoys_per_plot*0.1*self.lifetime_primary)*emission_factor_HDPE
        print(f'primary structure hardware emissions = {primary_structure_emissions_per_plot:,.2f}') if self.verbose else None
        
        #------------------------------------ secondary structure hardware ----------------------------#
        #longline dimensions -> material used = propylene (PP)
        net_density = 0.4 #kg/m^2
        net_mass = net_density*net_surface #kg

        #Economic calculations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        net_cost_per_kg = 5 #€
        total_net_cost = net_mass*net_cost_per_kg*longlines_per_plot*(self.lifetime_primary/self.lifetime_secondary)
        

        #Emission calculations
        secondary_structure_emissions_per_plot = net_mass*longlines_per_plot*(self.lifetime_primary/self.lifetime_secondary)*emission_factor_PP
        print(f'secondary structure hardware emissions = {secondary_structure_emissions_per_plot:,.2f}') if self.verbose else None

        #----------------------------------------------------------------------------------#

        #----------------------- installation - O&M - decommissioing ------------------------------------------#

        #activity times
        installation_time_corner =	2 #h
        installation_time_net = 4 #h
        monitor_time_per_plot = 30/3600 #h
        maintenance_time_per_net = installation_time_net/2 #h
        cleaning_time_per_net = 0.64 #h
        harvest_time_per_per_net = 1.6 #h
        
        yearly_monitoring = 5 #times per year
        yearly_maintenance = 5 #times per year
        yearly_cleaning = 2 #times per year

        total_installation_time_primary = installation_time_corner*anchors_per_plot #h
        total_installation_time_secondary = installation_time_net*longlines_per_plot*math.ceil(self.lifetime_primary/self.lifetime_secondary) #h
        
        total_yearly_monitoring = monitor_time_per_plot*yearly_monitoring #h/year
        total_yearly_maintenance = maintenance_time_per_net*longlines_per_plot*yearly_maintenance #h/year
        total_yearly_cleaning = cleaning_time_per_net*longlines_per_plot*yearly_cleaning #h/year
        total_yearly_harvest = harvest_time_per_per_net*longlines_per_plot #h/year
        print(f'total yearly monitoring = {total_yearly_monitoring:,.2f}') if self.verbose else None
        print(f'total yearly maintenance = {total_yearly_maintenance:,.2f}') if self.verbose else None
        print(f'total yearly cleaning = {total_yearly_cleaning:,.2f}') if self.verbose else None
        print(f'total yearly harvest = {total_yearly_harvest:,.2f}') if self.verbose else None

        total_decom_time_primary = installation_time_corner*anchors_per_plot #h
        total_decom_time_secondary = installation_time_net*longlines_per_plot*(math.ceil(self.lifetime_primary/self.lifetime_secondary)-1) #h

        #large workboat
        large_workboat_transit_time = 2*self.distance_to_OPEX_port/self.large_workboat_transit_speed #h
        large_workboat_available_activity_time = self.sailing_time_workday - large_workboat_transit_time #h
        large_workboat_total_activity_time = total_installation_time_primary + total_decom_time_primary #h
        large_workboat_trips = large_workboat_total_activity_time/large_workboat_available_activity_time

        #large workboat
        small_workboat_transit_time = 2*self.distance_to_OPEX_port/self.small_workboat_transit_speed #h
        small_workboat_available_activity_time = self.sailing_time_workday - small_workboat_transit_time #h
        small_workboat_activity_time_instal_decom = total_installation_time_secondary + total_decom_time_secondary #h
        small_workboat_yearly_activity_time_operation = total_yearly_maintenance + total_yearly_cleaning + total_yearly_harvest #h/year
        small_workboat_total_activity_time = small_workboat_activity_time_instal_decom + self.lifetime_primary*small_workboat_yearly_activity_time_operation #h
        small_workboat_trips_install_decom = small_workboat_activity_time_instal_decom/small_workboat_available_activity_time
        small_workboat_yearly_trips_operation = small_workboat_yearly_activity_time_operation/small_workboat_available_activity_time #per year
        small_workboat_total_trips = small_workboat_total_activity_time/small_workboat_available_activity_time

        #small fast boat
        small_fast_boat_transit_time = 2*self.distance_to_OPEX_port/self.small_fast_boat_transit_speed #h
        small_fast_boat_available_activity_time = self.sailing_time_workday - small_fast_boat_transit_time #h
        small_fast_boat_yearly_activity_time = total_yearly_monitoring #h
        small_fast_boat_total_activity_time = self.lifetime_primary*small_fast_boat_yearly_activity_time
        small_fast_boat_yearly_trips = small_fast_boat_yearly_activity_time/small_fast_boat_available_activity_time
        small_fast_boat_total_trips = small_fast_boat_total_activity_time/small_fast_boat_available_activity_time

        #Economic calculations
        large_workboat_cost = large_workboat_trips*self.large_workboat_day_rate #$
        small_workboat_cost_instal_decom = small_workboat_trips_install_decom*self.small_workboat_day_rate #$

        small_workboat_cost_operation = small_workboat_yearly_trips_operation*self.small_workboat_day_rate #$/year
        small_fast_boat_cost_operation = small_fast_boat_yearly_trips*self.small_workboat_day_rate #$/year

        #Emission calculation
        large_workboat_fuel_transit = large_workboat_transit_time*large_workboat_trips*self.large_workboat_fuel_transit #l
        large_workboat_fuel_installation = large_workboat_total_activity_time*self.large_workboat_fuel_installation #l
        large_workboat_emissions = (large_workboat_fuel_transit + large_workboat_fuel_installation)*self.emission_factor_MDO #kgCO2eq
        print(f'large workboat emissions = {large_workboat_emissions:,.2f}') if self.verbose else None

        small_workboat_fuel_transit = small_workboat_transit_time*small_workboat_total_trips*self.small_workboat_fuel_transit #l
        small_workboat_fuel_installation = small_workboat_total_activity_time*self.small_workboat_fuel_installation #l
        small_workboat_emissions = (small_workboat_fuel_transit + small_workboat_fuel_installation)*self.emission_factor_MDO #kgCO2eq
        print(f'small workboat emissions = {small_workboat_emissions:,.2f}') if self.verbose else None

        small_fast_boat_fuel_transit = small_fast_boat_transit_time*small_fast_boat_total_trips*self.small_fast_boat_fuel_transit #l
        small_fast_boat_fuel_installation = small_fast_boat_total_activity_time*self.small_fast_boat_fuel_installation #l
        small_fast_boat_emissions = (small_fast_boat_fuel_transit + small_fast_boat_fuel_installation)*self.emission_factor_MDO #kgCO2eq
        print(f'small fast boat emissions = {small_fast_boat_emissions:,.2f}') if self.verbose else None

        total_vessel_emissions = large_workboat_emissions + small_workboat_emissions + small_fast_boat_emissions
        print(f'total vessel emissions = {total_vessel_emissions:,.2f}') if self.verbose else None
      
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- cost of financing -------------------------------------------------#
        
        #complete CAPEX
        

        #weighted average cost of capital
        

        #debt life of CAPEX repayment + interests
        

        #amount that is financed with equity
        
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- total cost and emissions -------------------------------------------------#
        #total costs
        total_CAPEX = pre_capex + precontruction_cost + insurance_cost +large_workboat_cost + small_workboat_cost_instal_decom + total_net_cost
        total_OPEX_over_lifetime = (small_workboat_cost_operation + small_fast_boat_cost_operation)*self.lifetime_primary
        
        #total emissions
        total_emissions = primary_structure_emissions_per_plot + secondary_structure_emissions_per_plot + total_vessel_emissions

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- food production -------------------------------------------------#

        #Yield
        seaweed_yield = 8 #kg/m/year (per running meter dropper line)

        #Loss during mussel processing
        processing_loss = 0.2 #20% loss

        #food production
        yearly_food_production = seaweed_yield*(1-processing_loss)*total_net_surface_per_plot #kg/year 


        '''
        Final metrics 
        '''
        self.final_metrics['capex'] = total_CAPEX 
        self.final_metrics['opex'] = total_OPEX_over_lifetime
        self.final_metrics['co2+'] = total_emissions
        self.final_metrics['co2-'] = 0 
        self.final_metrics['revenue'] = yearly_food_production*3*self.lifetime_primary #3 being the general price (€/kg) of mussels
        self.final_metrics['unit density'] = 36 # 36 units per square km
        self.final_metrics['food_produced'] = yearly_food_production
        self.final_metrics['energy_produced'] = 0
        self.final_metrics['LCOE'] = 0
        self.final_metrics['lifetime'] = self.lifetime_primary

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
        
        NPV = NPV_calculator(total_CAPEX, total_OPEX_over_lifetime, self.food_revenue, yearly_food_production, self.WACC, 1, self.lifetime_primary)

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
        
        
        NPV_emissions = NPV_calculator_emissions(total_emissions,
                                                0, #change this
                                                self.cost_of_carbon/1000, #cost per kg
                                                0,
                                                self.discount_rate_emissions,
                                                self.belgian_carbon_intensity,
                                                1,
                                                self.lifetime_primary)

        # self.final_metrics['carbon NPV'] = NPV_emissions

    def run(self, found_age):
        self._calculations()
        return self.final_metrics, {}