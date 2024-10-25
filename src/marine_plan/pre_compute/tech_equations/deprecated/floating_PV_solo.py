''' 
This is a test file to show how to structure a tech file
Adjust and add as many calcs as required
'''

class floating_PV_solo:
    def __init__(self, geo_data, verbose=False):

        import numpy as np
        import math

        self.verbose = verbose

        #initialising geo-spatial data dictionary
        self.depth = geo_data['depth'] #m
        self.mean_windspeed_at_10m = geo_data['mean_wind_speed_at_10m'] #m/s at 10 m height
        self.distance_to_OPEX_port = geo_data['distance_to_OPEX_port'] #km
        self.distance_to_onshore_sub = geo_data['distance_to_onshore_sub'] #km
        self.distance_to_installation_port = geo_data['distance_to_installation_port'] #km
        self.soil_coefficient = geo_data['soil_coefficient'] #kN/m^3
        self.soil_friction_angle = geo_data['soil_friction_angle'] #angle
        self.global_tilted_irradiation = 1100

        #Converting metrics
        self.short_ton = 907.185 #ton

        #emission factors
        self.emission_factor_MDO	= 3.53 #
        self.emission_factor_HFO	= 3.41 #

        #Belgian carbon intensity
        self.belgian_carbon_intensity = 172 #kgCO2eq/MWh

        #unit density
        self.capacity_per_square_km = 32 #MWh
        self.solar_panel_capacity = 400 #W
        self.solar_per_platform = 70 #numver of solar panels
        self.capacity_per_platform = self.solar_per_platform * self.solar_panel_capacity #W
        self.amount_of_platforms_per_square_km = self.capacity_per_square_km*(1000000)/self.capacity_per_platform #number of platforms
        #lifetime
        self.lifetime = 25 #years

        self.vars = {} # Calculated variables go in here

        self.final_metrics = {} # Final metrics go in here

    def _calculations(self):
        import numpy as np
        import math
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
        #------------------------- Export Cable -------------------#
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

        #economic variables
        cable_cost_per_km	=400000
        cable_cost =	cable_cost_per_km*((self.distance_to_onshore_sub)*excess_cable_factor)
        cable_emission_per_km	=93537
        cable_emissions = cable_emission_per_km*((self.distance_to_onshore_sub)*excess_cable_factor)
        print('total cable cost = ' +str(cable_cost)) if self.verbose else None
        print('total cable emissions = ' +str(cable_emissions)) if self.verbose else None
        

        #---------------------------------------------- OPEX -----------------------------------------------------#
        yearly_OPEX_per_kW = 15 #€/kW/year
        yearly_OPEX = yearly_OPEX_per_kW*(self.capacity_per_square_km*1000)
        total_OPEX_over_lifetime = yearly_OPEX*self.lifetime
        yearly_OM_emissions = 1.12 #kgCO2eq per platform
        total_OM_emissions_over_lifetime = yearly_OM_emissions*self.lifetime
        print('total yearly OPEX = ' +str(yearly_OPEX)) if self.verbose else None
        print('OPEX over lifetime = ' +str(total_OPEX_over_lifetime)) if self.verbose else None
        print('total yearly OM emissions = ' +str(yearly_OM_emissions)) if self.verbose else None
        print('OM emissions over lifetime = ' +str(total_OM_emissions_over_lifetime)) if self.verbose else None

        #------------------------------------ installation and decommissioning -----------------------------------#
        pre_capex = (total_solar_panel_cost +
                     total_platform_cost +
                     A_M_cost_wind+
                     total_BOS_cost+
                     cable_cost)
    
    
        installation_percentage = 0.05
        installation_cost = pre_capex*installation_percentage
        installation_emissions = 750 #kgCO2eq per platform
        print('total installation cost = ' +str(installation_cost)) if self.verbose else None
        print('total installation emissions = ' +str(installation_emissions)) if self.verbose else None

        decommissioning_percentage = 0.02
        decommissioning_cost = pre_capex*decommissioning_percentage
        decommissioning_emissions = 300 #kgCO2eq per platform
        print('total decom cost = ' +str(decommissioning_cost)) if self.verbose else None
        print('total decom emissions = ' +str(decommissioning_emissions)) if self.verbose else None

        pre_capex = pre_capex + installation_cost + decommissioning_cost


        total_emissions = (total_solar_panel_emissions + total_platform_emissions + A_M_emissions_wind + total_BoS_emissions + total_OM_emissions_over_lifetime + installation_emissions + decommissioning_emissions)*self.amount_of_platforms_per_square_km + cable_emissions

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- engineering costs -------------------------------------------------#
    
        engineering_cost_factor	= 0.04

        #overall engineerinh cost for one turbine
        engineering_cost_overall = 0.04*pre_capex
        print('total engineering cost = ' +str(engineering_cost_overall)) if self.verbose else None
        

        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- insurance and development -------------------------------------------------#
        
        plant_commissioning = 0.01 #of total pre capex
        insurance_cost = 0.075 #of total pre capex

        #development and insurance cost for one turbine
        dev_and_insurance_total = plant_commissioning*pre_capex + insurance_cost*pre_capex
        print('dev and insurance cost = ' +str(dev_and_insurance_total)) if self.verbose else None
        #------------------------------------------------------------------------------------------------------------#
        #--------------------------------- cost of financing -------------------------------------------------#
        
        #complete CAPEX
        total_CAPEX = pre_capex + engineering_cost_overall + dev_and_insurance_total

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

        panel_area = 1.69 * 1.046
        efficiency = 0.226
        performance_ratio = 0.75
        #global tilted in kWh/m^2

        #in MWh/year!
        yearly_total_energy_produced = self.amount_of_platforms_per_square_km*self.solar_per_platform*(panel_area*efficiency*performance_ratio*self.global_tilted_irradiation)/1000
        print('yearly energy produced = ' +str(yearly_total_energy_produced)) if self.verbose else None
        print(self.amount_of_platforms_per_square_km) if self.verbose else None
        print(self.solar_per_platform) if self.verbose else None

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
        max_capacity = self.capacity_per_square_km
        print('max capacity = ' +str(max_capacity)) if self.verbose else None
        #calculated with the effective energy produced based on windspeeds and the total energy it could produce if the turbine was turning at full capacity all the time. 
        capacity_factor = yearly_total_energy_produced/(self.capacity_per_square_km*365*24)
        print('capacity factor = ' +str(capacity_factor)) if self.verbose else None

        LCOE_per_MWh = LCOE(total_CAPEX, total_OPEX_over_lifetime, max_capacity, self.lifetime, WACC, capacity_factor, capacity_deg=0)

        #print(f'test: {self.vars["third_var"]:,.2f}') # WHenever you want to print a number just put the number like this


        '''
        Final metrics 
        '''
        # STILL NEED TO DO TIMES THE UNIT DENSITY, THIS IS FOR ONE WHOLE TURBINE. -> FOR FPV THIS IS OK
        self.final_metrics['capex'] = total_CAPEX 
        self.final_metrics['opex'] = total_OPEX_over_lifetime
        self.final_metrics['co2+'] = total_emissions
        self.final_metrics['co2-'] = yearly_total_energy_produced*self.belgian_carbon_intensity
        self.final_metrics['value'] = yearly_total_energy_produced*40 #40 being the general price (€/MWh) of energy
        self.final_metrics['LCOE'] = LCOE_per_MWh
        self.final_metrics['unit density'] = 1
        self.final_metrics['energy_produced'] = yearly_total_energy_produced
        self.final_metrics['food_produced'] = 0
        self.final_metrics['lifetime'] = self.lifetime

    def run(self, found_age=25):
        self._calculations()
        return self.final_metrics, {}