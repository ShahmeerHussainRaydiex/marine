''' 
This is a test file to show how to structure a tech file
Adjust and add as many calcs as required
'''

class mussel_longline:
    def __init__(self, geo_data, verbose=False):

        import numpy as np
        import math

        self.verbose = verbose

        #initialising geo-spatial data dictionary
        self.depth = geo_data['depth'] #m
        self.distance_to_OPEX_port = geo_data['distance_to_OPEX_port'] #km
        # self.temperature = geo_data['temp'] #°C -> surface temperature
        # self.chla = geo_data['chla'] #µg/L -> concentration of chlrophyll a
        # self.pom = geo_data['pom'] #mg/L -> concentration of paticulate organic matter

        self.temperature = 15
        self.chla = 0.002
        self.pom = 3




        #self.soil_coefficient = geo_data['soil_coefficient'] #kN/m^3
        #self.soil_friction_angle = geo_data['soil_friction_angle'] #angle

        #Converting metrics
        self.short_ton = 907.185 #kg

        #emission factors
        self.emission_factor_MDO	= 3.53 #kgCO2eq/l
        self.emission_factor_HFO	= 3.41 #kgCO2eq/l

        #unit density and interdistance
        self.unit_density = 35 #units per square km

        #lifetime
        self.lifetime_primary = 12 #years
        self.lifetime_secondary = 3 #years -> lifetime of longline and cultivation system

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
        dropper_line_interdistance = 1.5 #m
        droppers_per_plot = longlines_per_plot*(longline_length//dropper_line_interdistance)
        dropper_length = 5 #m
        total_dropper_length_per_plot = dropper_length*droppers_per_plot
        print(total_dropper_length_per_plot) if self.verbose else None
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
        #longline dimensions -> material used = polyester
        dropper_density = 1380 #kg/m^3
        dropper_diameter = 0.008 + 0.008*(self.depth-10)/20 #m
        dropper_mass_per_m = math.pi*((dropper_diameter/2)**2)*dropper_density #kg/m
        total_dropper_mass = dropper_mass_per_m*dropper_length #kg

        #sock dimensions -> mterial used = polypropylene (PP)
        sock_mass_per_m = 0.2 #kg/m
        total_sock_mass = sock_mass_per_m*dropper_length

        #Economic calculations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #still need droper cost -> @Bernd is this neglegible? 
        dropper_cost_per_kg = 5 #€
        total_dropper_cost = total_dropper_mass*dropper_cost_per_kg*droppers_per_plot*(self.lifetime_primary/self.lifetime_secondary)
        

        #Emission calculations
        emission_factor_ployester = 3.8 #kgCO2eq/kg
        secondary_structure_emissions_per_plot = total_dropper_mass*droppers_per_plot*(self.lifetime_primary/self.lifetime_secondary)*emission_factor_ployester + total_sock_mass*droppers_per_plot*self.lifetime_primary*emission_factor_PP
        print(f'secondary structure hardware emissions = {secondary_structure_emissions_per_plot:,.2f}') if self.verbose else None

        #----------------------------------------------------------------------------------#

        #----------------------- installation - O&M - decommissioing ------------------------------------------#

        #activity times
        installation_time_corner =	2 #h
        installation_time_dropper = 0.064 #h
        monitor_time_per_plot = 30/3600 #h
        maintenance_time_per_dropper	= installation_time_dropper/2 #h
        cleaning_time_per_dropper = 7.2*dropper_length/3600 #h
        resocking_time_per_dropper	= 18*dropper_length/3600 #h
        harvest_time_per_dropper	= 18*dropper_length/3600 #h
        
        yearly_monitoring = 5 #times per year
        yearly_maintenance = 5 #times per year
        yearly_cleaning = 4 #times per year

        total_installation_time_primary = installation_time_corner*anchors_per_plot #h
        total_installation_time_secondary = installation_time_dropper*droppers_per_plot*math.ceil(self.lifetime_primary/self.lifetime_secondary) #h
        
        total_yearly_monitoring = monitor_time_per_plot*yearly_monitoring #h/year
        total_yearly_maintenance = maintenance_time_per_dropper*droppers_per_plot*yearly_maintenance #h/year
        total_yearly_cleaning = cleaning_time_per_dropper*droppers_per_plot*yearly_cleaning #h/year
        total_yearly_resocking = resocking_time_per_dropper*droppers_per_plot #h/year
        total_yearly_harvest = harvest_time_per_dropper*droppers_per_plot #h/year
        print(f'total yearly monitoring = {total_yearly_monitoring:,.2f}') if self.verbose else None
        print(f'total yearly maintenance = {total_yearly_maintenance:,.2f}') if self.verbose else None
        print(f'total yearly cleaning = {total_yearly_cleaning:,.2f}') if self.verbose else None
        print(f'total yearly harvest = {total_yearly_harvest:,.2f}') if self.verbose else None

        total_decom_time_primary = installation_time_corner*anchors_per_plot #h
        total_decom_time_secondary = installation_time_dropper*droppers_per_plot*(math.ceil(self.lifetime_primary/self.lifetime_secondary)-1) #h

        #large workboat
        large_workboat_transit_time = 2*self.distance_to_OPEX_port/self.large_workboat_transit_speed #h
        large_workboat_available_activity_time = self.sailing_time_workday - large_workboat_transit_time #h
        large_workboat_total_activity_time = total_installation_time_primary + total_decom_time_primary #h
        large_workboat_trips = large_workboat_total_activity_time/large_workboat_available_activity_time

        #large workboat
        small_workboat_transit_time = 2*self.distance_to_OPEX_port/self.small_workboat_transit_speed #h
        small_workboat_available_activity_time = self.sailing_time_workday - small_workboat_transit_time #h
        small_workboat_activity_time_instal_decom = total_installation_time_secondary + total_decom_time_secondary #h
        small_workboat_yearly_activity_time_operation = total_yearly_maintenance + total_yearly_cleaning + total_yearly_resocking + total_yearly_harvest #h/year
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
        total_CAPEX = pre_capex + precontruction_cost + insurance_cost +large_workboat_cost + small_workboat_cost_instal_decom + total_dropper_cost
        total_OPEX_over_lifetime = (small_workboat_cost_operation + small_fast_boat_cost_operation)*self.lifetime_primary
        
        #total emissions
        total_emissions = primary_structure_emissions_per_plot + secondary_structure_emissions_per_plot + total_vessel_emissions

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- food production -------------------------------------------------#

        #mussel growth calculation
        clearance_rate = 4.825 - 0.013*(self.temperature - 18.954)**2
        #print(f'clearance rate = {clearance_rate:,.2f}')
        clearance_rate_15C = 4.825 - 0.013*(15 - 18.954)**2
        #print(f'clearance rate 15°C = {clearance_rate_15C:,.2f}')
        tef = clearance_rate/clearance_rate_15C #temperature_feeding_effect
        #print(f'TEF = {tef:,.2f}')
        tem = math.exp(0.074*self.temperature)/math.exp(0.074*15) #temperature maintenance effect
        #print(f'TEM = {tem:,.2f}')

        SOM_as_food = self.chla*12/0.38 #mg/l -> suspended organic matter available as food
        #SOM_as_food = self.chla*50/0.38 #mg/l -> suspended organic matter available as food
        #print(f'SELORG = {SOM_as_food:,.2f}')
        remaining_OM = self.pom - SOM_as_food #mg/l -> remiaing organic matter
        #print(f'REMORG = {remaining_OM:,.2f}')

        energy_content_of_food = 8.25 + (21.24*(1 - math.exp(-2.793*SOM_as_food))) + (-0.1743*remaining_OM) #J/mg
        #print(f'EREM = {energy_content_of_food:,.2f}')

        net_ingestion_SOM = (-0.16 + (3.57*SOM_as_food))*tef #mg/h
        #print(f'NIRSELORG = {net_ingestion_SOM:,.2f}')
        net_ingestion_remaining_OM = 7.1 * (1 - math.exp(-0.31*remaining_OM)) * tef #mg/h
        #print(f'NIRREMORG = {net_ingestion_remaining_OM:,.2f}')

        nea = ((net_ingestion_SOM*23.5) + (net_ingestion_remaining_OM*0.15*energy_content_of_food))*0.8*24 #J/day -> net energy absorption
        #nea = (net_ingestion_SOM*23.5)*0.8*24 #J/day -> net energy absorption
        #print(f'NEA = {nea:,.2f}')

        mnea = 1250 #J/day/g dry wieght -> maximum rate net energy absorption
        oxygen_nitrogen_ratio = 10 + ((200-10)/mnea)*nea
        #print(f'O:N = {oxygen_nitrogen_ratio:,.2f}')

        mta = 0.68 # -> mean tissue allocation
        
        """
        #values at juvenile stage
        dstw = 0.003 #g -> dry soft tissue weight at seeding
        dsw = 0.006 #g -> dry shell weight at seeding

        sl = 2.654*(dsw**0.327)
        #print(f'sl = {sl:,.5f}')
        ste = 23.5*1000*dstw
        se = 1.035*1000*dsw

        mhl = 4.005*tef*tem*24#*dstw
        thl = mhl + (0.23*nea)
        el = 14*1000*thl/(14.06*16*oxygen_nitrogen_ratio)
        neb = nea - thl - (el*0.02428)
        
        tg = mta*neb
        sg = (1-mta)*neb       

        t = 75

        factor_energy_to_weight = 0.02
        while sl < 2:
            t+=1
            dstw += factor_energy_to_weight*ste/(23.5*1000)
            #print(f'dstw = {dstw:,.5f}')
            dsw += factor_energy_to_weight*se/(1.035*1000)
            sl = 2.654*(dsw**0.327)
            #print(f'sl = {sl:,.5f}')
            #mhl = 4.005*dstw*tef*tem*24
            #hl = mhl + (0.23*nea)
            #el = 14*1000*thl/(14.06*16*oxygen_nitrogen_ratio)
            #neb = nea - thl - (el*0.02428)
            if ste/(ste+se) >= mta:
                tg = mta*neb
                sg = (1-mta)*neb
            else:
                tg = neb
                sg = 0
            ste += tg
            se += sg
        print(f'time to reach 2cm shell length = {t:,.2f}')

        spawn_count = 0

        
        while t<365:
            t+=1
            spawn = 0
            if spawn_count < 2:
                if self.temperature >= 15:
                    if ste/(ste+se) >= 0.95*mta:
                        spawn = dstw*0.18*23.5
            dstw += factor_energy_to_weight*ste/(23.5*1000)
            dsw += factor_energy_to_weight*se/(1.035*1000)
            sl = 2.654*(dsw**0.327)
            #mhl = 4.005*dstw*tef*tem*24
            #hl = mhl + (0.23*nea)
            #el = 14*1000*thl/(14.06*16*oxygen_nitrogen_ratio)
            #neb = nea - thl - (el*0.02428)
            if ste/(ste+se) >= mta:
                tg = mta*neb
                sg = (1-mta)*neb
            else:
                tg = neb
                sg = 0
            ste += tg
            se += sg

        print(f'dstw = {dstw:,.5f}')

        ww_1_year = ((dsw*(1+0.048)) + (dstw*(1+0.804)))*1.485
        print(f'total mussel wet weight = {ww_1_year:,.2f}')
        

        """
        mhl = 4.005*tef*tem*24 #J/day maintenance heat losses
        #print(f'MHL = {mhl:,.2f}')
        thl = mhl + (0.23*nea) #J/day -> total heat losses
        #print(f'THL = {thl:,.2f}')

        el = 14*1000*thl/(14.06*16*oxygen_nitrogen_ratio) #µm NH4/day -> excretion losses
        #print(f'EL = {el:,.2f}')

        neb = nea - thl - (el*0.02428) #J/day -> net energy balance
        #print(f'NEB = {neb:,.2f}')

        ea_soft_tissue = mta*neb #J/day -> energy allocation to total soft tissue
        ea_shell = (1-mta)*neb
        print(f'TG = {ea_soft_tissue:,.2f}') if self.verbose else None
        print(f'SG = {ea_shell:,.2f}') if self.verbose else None

        t_juv = 75

        dstw_juv = 0.003 #g -> dry soft tissue weight at seeding
        dsw_juv = 0.006 #g -> dry shell weight at seeding
 
        ste_juv = 23.5*1000*dstw_juv #J -> soft tissue energy at seeding
        se_juv = 1.035*1000*dsw_juv #J -> shell energy at seeding
        print(f'ste juv = {ste_juv:,.2f}') if self.verbose else None
        print(f'se juv = {se_juv:,.2f}') if self.verbose else None

        time_2cm = t_juv + (((2/2.654) - dsw_juv)*1.035*1000 - se_juv)/ea_shell #days for shell to reach 2cm
        print(f'time to reach 2cm shell length = {time_2cm:,.2f}') if self.verbose else None

        dstw_2cm = dstw_juv + (ste_juv + ea_soft_tissue*(time_2cm-t_juv))/(23.5*1000)
        dsw_2cm = dsw_juv + (se_juv + ea_shell*(time_2cm-t_juv))/(1.035*1000)
        print(f'DSTW 2 cm = {dstw_2cm:,.2f}') if self.verbose else None
        print(f'DSW 2 cm = {dsw_2cm:,.2f}') if self.verbose else None

        ste_2cm = ste_juv + ea_soft_tissue*(time_2cm-t_juv)
        se_2cm = se_juv + ea_shell*(time_2cm-t_juv)

        if self.temperature < 13:
            el_spawn = 0
        else:
            el_spawn = dstw_2cm*0.18*23.5
        
        dstw_1_year = dstw_2cm + (ste_2cm + ea_soft_tissue*(365-time_2cm) - el_spawn)/(23.5*1000)
        dsw_1_year = dsw_2cm + (se_2cm + ea_shell*(365-time_2cm))/(1.035*1000)
        print(f'DSTW 1 year = {dstw_1_year:,.2f}') if self.verbose else None
        print(f'DSW 1 year = {dsw_1_year:,.2f}') if self.verbose else None

        ww_1_year = ((dsw_1_year*(1+0.048)) + (dstw_1_year*(1+0.804)))*1.485
        print(f'total mussel wet weight = {ww_1_year:,.2f}') if self.verbose else None
        
        mussel_density = 750 #mussel per meter dropper line

        #Yield
        mussel_yield = 10 #kg/m/year (per running meter dropper line)
        #mussel_yield = ww_1_year/1000*mussel_density #kg/m/year (per running meter dropper line)
        print(f'mussel yield = {mussel_yield:,.2f}') if self.verbose else None

        #Loss during mussel processing
        processing_loss = 0.2 #20% loss

        #food production
        yearly_food_production = mussel_yield*(1-processing_loss)*total_dropper_length_per_plot #kg/year 

        
        '''
        Final metrics 
        '''
        # STILL NEED TO DO TIMES THE UNIT DENSITY, THIS IS FOR ONE PLOT.
        self.final_metrics['capex'] = total_CAPEX 
        self.final_metrics['opex'] = total_OPEX_over_lifetime
        self.final_metrics['co2+'] = total_emissions
        self.final_metrics['co2-'] = 0 
        self.final_metrics['value'] = yearly_food_production*3*self.lifetime_primary #3 being the general price (€/kg) of mussels
        self.final_metrics['unit density'] = 36 # 36 units per square km
        self.final_metrics['food_produced'] = yearly_food_production
        self.final_metrics['energy_produced'] = 0
        self.final_metrics['LCOE'] = 0
        self.final_metrics['lifetime'] = self.lifetime_primary

    def run(self, found_age):
        self._calculations()
        return self.final_metrics, {}