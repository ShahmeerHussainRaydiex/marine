import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

def set_map_config(config, scale):

    print('Setting map config for:', config, 'at scale:', scale)

    if scale == 'belgium':
        
        if config == 'whole area':
            map_CONGIF = {
                'scale': 'belgium',
                'msp': {
                        'shipping': False,
                        'military': False,
                        'sand_extraction': False,
                        'nature_reserves': False,
                        'energy_zones': False,
                        'energy_zones_type': 'whole_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
                        'wind_farms': False,
                        'legacy_farms': False
                    },

                'colours': {
                        'mussel': 'red',
                        'monopile': 'blue',
                        'jacket': 'teal',
                        'solar': 'orange',
                        'monopile_solar': 'purple',
                        'monopile_mussel': 'pink',
                        'wind': 'white'
                        #'hydrogen': 'black'
                    },

                'legend': [
                        plt.Line2D([0], [0], color='white', linewidth=0, label='----------------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }

        elif config == 'full PE zone':
            map_CONGIF = {
                'scale': 'belgium',
                'msp': {
                        'shipping': False,
                        'military': False,
                        'sand_extraction': False,
                        'nature_reserves': False,
                        'energy_zones': True,
                        'energy_zones_type': 'whole_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
                        'wind_farms': False,
                        'legacy_farms': False
                    },

                'colours': {
                        'mussel': 'red',
                        'monopile': 'blue',
                        'jacket': 'teal',
                        'solar': 'orange',
                        'monopile_solar': 'purple',
                        'monopile_mussel': 'pink',
                        'wind': 'white'
                        #'hydrogen': 'black'
                    },

                'legend': [
                        Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
                        Line2D([0], [0], color='red', alpha=1, lw=2, linestyle='-', label='Energy Zones'),
                        plt.Line2D([0], [0], color='white', linewidth=0, label='----------------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }
        
        elif config in ['PE zone 1', 'PE zone 2', 'PE zone 3']:
            map_CONGIF = {
                'scale': 'belgium',
                'msp': {
                        'shipping': False,
                        'military': False,
                        'sand_extraction': False,
                        'nature_reserves': False,
                        'energy_zones': True,
                        'energy_zones_type': 'split_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
                        'wind_farms': False,
                        'legacy_farms': False
                    },

                'colours': {
                        'mussel': 'red',
                        'monopile': 'blue',
                        'jacket': 'teal',
                        'solar': 'orange',
                        'monopile_solar': 'purple',
                        'monopile_mussel': 'pink',
                        'wind': 'white'
                        #'hydrogen': 'black'
                    },

                'legend': [
                        Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
                        Line2D([0], [0], color='red', alpha=1, lw=2, linestyle='-', label='Energy Zones'),
                        Patch(color='#F5F5DC', alpha=0.5, lw=2, hatch='//', label='Sand Extraction'),
                        plt.Line2D([0], [0], color='white', linewidth=0, label='----------------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }
        
        elif config == 'renewable energy zones':
            map_CONGIF = {
                'scale': 'belgium',
                'msp': {
                        'shipping': False,
                        'military': False,
                        'sand_extraction': False,
                        'nature_reserves': False,
                        'energy_zones': True,
                        'energy_zones_type': 'whole_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
                        'wind_farms': False,
                        'legacy_farms': False
                    },

                'colours': {
                        'mussel': 'red',
                        'monopile': 'blue',
                        'jacket': 'teal',
                        'solar': 'orange',
                        'monopile_solar': 'purple',
                        'monopile_mussel': 'pink',
                        'wind': 'white'
                        #'hydrogen': 'black'
                    },

                'legend': [
                        Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
                        Line2D([0], [0], color='red', alpha=1, lw=2, linestyle='-', label='Energy Zones'),
                        plt.Line2D([0], [0], color='white', linewidth=0, label='----------------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }

        elif config == 'msp':
            map_CONGIF = {
                'scale': 'belgium',
                'msp': {
                        'shipping': True,
                        'military': True,
                        'sand_extraction': True,
                        'nature_reserves': True,
                        'energy_zones': True,
                        'energy_zones_type': 'whole_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
                        'wind_farms': True,
                        'legacy_farms': False
                    },

                'colours': {
                        'mussel': '#00BFFF',
                        'monopile': '#4169E1',
                        'jacket': '#4682B4',
                        'solar': '#FFA500',
                        # 'monopile_solar': '#FFBF00',
                        # 'monopile_mussel': '#9370DB',
                        #'hydrogen': 'black'
                    },

                'legend': [

                        plt.Line2D([0], [0], color='white', linewidth=0, label='------- MSP ---------', linestyle='None'), # Keep this line, it seperates the background files from the tech
                        Patch(color='#333333', alpha=1, lw=2, label='Shipping Routes'),
                        Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
                        Patch(color='#8FBC8F', alpha=0.5, lw=2, hatch='/', label='Nature Reserves'),
                        Patch(color='#F4A460', alpha=0.5, lw=2, hatch='//', label='Sand Extraction'),
                        Patch(color='#800000', alpha=0.5, hatch='x', lw=2, label='Military Shooting'),


                        plt.Line2D([0], [0], color='white', linewidth=0, label='------- Wind Farms ---------', linestyle='None'),
                        Patch(color='#008080', hatch='|', lw=1, alpha=0.5, label='Operational'),
                        # Patch(color='#FF7F50', alpha=0.5, lw=1, label='Under Construction'),
                        Patch(color='#FFD700', alpha=0.5, lw=1, hatch='|', label='Approved'),
                        # Patch(color='#EEE8AA', alpha=0.5, lw=1, hatch='|', label='Planned'),

                        
                        plt.Line2D([0], [0], color='white', linewidth=0, label='------- Technologies ---------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }
        
        elif config == 'exclude shipping':
            map_CONGIF = {
                'scale': 'belgium',
                'msp': {
                        'shipping': True,
                        'military': False,
                        'sand_extraction': False,
                        'nature_reserves': False,
                        'energy_zones': False,
                        'energy_zones_type': 'whole_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
                        'wind_farms': False,
                        'legacy_farms': False
                    },

                'colours': {
                        'mussel': 'red',
                        'monopile': 'blue',
                        'jacket': 'teal',
                        'solar': 'orange',
                        'monopile_solar': 'purple',
                        'monopile_mussel': 'pink',
                        'wind': 'white'
                        #'hydrogen': 'black'
                    },

                'legend': [
                        Patch(color='#333333', alpha=1, lw=2, label='Shipping Routes'),
                        Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
                        
                        plt.Line2D([0], [0], color='white', linewidth=0, label='----------------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }
    
    elif scale == 'international':
       
        # if config == 'whole area':
        #     map_CONGIF = {
        #         'scale': 'international',
        #         'msp': {
        #                 'shipping': False,
        #                 'military': False,
        #                 'sand_extraction': False,
        #                 'nature_reserves': False,
        #                 'energy_zones': False,
        #                 'energy_zones_type': 'whole_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
        #                 'wind_farms': False,
        #                 'legacy_farms': False
        #             },

        #         'colours': {
        #                 'mussel': 'red',
        #                 'monopile': 'blue',
        #                 'jacket': 'teal',
        #                 'solar': 'orange',
        #                 'monopile_solar': 'purple',
        #                 'monopile_mussel': 'pink',
        #                 'wind': 'white'
        #                 #'hydrogen': 'black'
        #             },

        #         'legend': [
        #                 plt.Line2D([0], [0], color='white', linewidth=0, label='----------------', linestyle='None') # Keep this line, it seperates the background files from the tech
        #             ],
        #     }

        if config == 'whole area':
            map_CONGIF = {
                'scale': 'international',
                'msp': {
                        'shipping': False,
                        'military': False,
                        'sand_extraction': False,
                        'nature_reserves': False,
                        'energy_zones': False,
                        'energy_zones_type': 'whole_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
                        'wind_farms': False,
                        'legacy_farms': False,
                        'interconnectors': False,
                        'cables': False
                    },

                'colours': {
                        'mussel': '#00BFFF',
                        'monopile': '#4169E1',
                        'jacket': '#4682B4',
                        'fpv': '#FFA500',
                        'semisub_cat_drag': '#FFBF00',
                        'semisub_taut_driv': '#9370DB',
                        'semisub_taut_suc': '#FFD700',
                        'spar_cat_drag': '#EEE8AA',
                        'spar_taut_driv': '#008080',
                        'spar_taut_suc': '#FF7F50',
                        # 'monopile_solar': '#800000',
                        # 'monopile_mussel': '#00FF00',
                        # 'wind': '#00FFFF'
                        #'hydrogen': 'black'
                    },

                'legend': [

                        plt.Line2D([0], [0], color='white', linewidth=0, label='------- MSP ---------', linestyle='None'), # Keep this line, it seperates the background files from the tech
                        # Patch(color='#333333', alpha=1, lw=2, label='Shipping Routes'),
                        Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
                        # Patch(color='#8FBC8F', alpha=0.5, lw=2, hatch='/', label='Nature Reserves'),
                        # Patch(color='#F4A460', alpha=0.5, lw=2, hatch='//', label='Sand Extraction'),
                        # Patch(color='#800000', alpha=0.5, hatch='x', lw=2, label='Military Shooting'),


                        # plt.Line2D([0], [0], color='white', linewidth=0, label='------- Wind Farms ---------', linestyle='None'),
                        # Patch(color='#008080', hatch='|', lw=1, alpha=0.5, label='Operational'),
                        # Patch(color='#FF7F50', alpha=0.5, lw=1, label='Under Construction'),
                        # Patch(color='#FFD700', alpha=0.5, lw=1, hatch='|', label='Approved'),
                        # Patch(color='#EEE8AA', alpha=0.5, lw=1, hatch='|', label='Planned'),

                        # plt.Line2D([0], [0], color='white', linewidth=0, label='------- Interconnects ---------', linestyle='None'), # Keep this line, it seperates the background files from the tech
                        # Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Interconnector to Land'),
                        # Line2D([0], [0], color='#FFD700', linestyle='--', linewidth=3, label='Substation to Interconnector'),
                        # Line2D([0], [0], color='yellow', linewidth=2, label='Substation to Land (Current and Modelled)'),
                        # Line2D([0], [0], marker='o', color='red', markersize=10, markeredgecolor='white', 
                        #     markeredgewidth=1, linestyle='None', label='Interconnectors'),
                        plt.Line2D([0], [0], color='white', linewidth=0, label='------- Technologies ---------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }

        elif config == 'msp':
            map_CONGIF = {
                'scale': 'international',
                'msp': {
                        'shipping': True,
                        'military': True,
                        'sand_extraction': False,
                        'nature_reserves': True,
                        'energy_zones': False,
                        'energy_zones_type': 'whole_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
                        'wind_farms': False,
                        'legacy_farms': False,
                        'interconnectors': False,
                        'cables': False
                    },

                'colours': {
                        'mussel': '#00BFFF',
                        'monopile': '#4169E1',
                        'jacket': '#4682B4',
                        'fpv': '#FFA500',
                        'semisub_cat_drag': '#FFBF00',
                        'semisub_taut_driv': '#9370DB',
                        'semisub_taut_suc': '#FFD700',
                        'spar_cat_drag': '#EEE8AA',
                        'spar_taut_driv': '#008080',
                        'spar_taut_suc': '#FF7F50',
                        # 'monopile_solar': '#800000',
                        # 'monopile_mussel': '#00FF00',
                        # 'wind': '#00FFFF'
                        #'hydrogen': 'black'
                    },

                'legend': [

                        plt.Line2D([0], [0], color='white', linewidth=0, label='------- MSP ---------', linestyle='None'), # Keep this line, it seperates the background files from the tech
                        # Patch(color='#333333', alpha=1, lw=2, label='Shipping Routes'),
                        Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
                        # Patch(color='#8FBC8F', alpha=0.5, lw=2, hatch='/', label='Nature Reserves'),
                        # Patch(color='#F4A460', alpha=0.5, lw=2, hatch='//', label='Sand Extraction'),
                        # Patch(color='#800000', alpha=0.5, hatch='x', lw=2, label='Military Shooting'),


                        # plt.Line2D([0], [0], color='white', linewidth=0, label='------- Wind Farms ---------', linestyle='None'),
                        # Patch(color='#008080', hatch='|', lw=1, alpha=0.5, label='Operational'),
                        # Patch(color='#FF7F50', alpha=0.5, lw=1, label='Under Construction'),
                        # Patch(color='#FFD700', alpha=0.5, lw=1, hatch='|', label='Approved'),
                        # Patch(color='#EEE8AA', alpha=0.5, lw=1, hatch='|', label='Planned'),

                        # plt.Line2D([0], [0], color='white', linewidth=0, label='------- Interconnects ---------', linestyle='None'), # Keep this line, it seperates the background files from the tech
                        # Line2D([0], [0], color='red', linestyle='--', linewidth=4, label='Interconnector to Land'),
                        # Line2D([0], [0], color='#FFD700', linestyle='--', linewidth=3, label='Substation to Interconnector'),
                        # Line2D([0], [0], color='yellow', linewidth=2, label='Substation to Land (Current and Modelled)'),
                        # Line2D([0], [0], marker='o', color='red', markersize=10, markeredgecolor='white', 
                        #     markeredgewidth=1, linestyle='None', label='Interconnectors'),
                        plt.Line2D([0], [0], color='white', linewidth=0, label='------- Technologies ---------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }


  

        elif config == 'exclude shipping':
             map_CONGIF = {
                'scale': 'international',
                'msp': {
                        'shipping': True,
                        'military': False,
                        'sand_extraction': False,
                        'nature_reserves': False,
                        'energy_zones': False,
                        'energy_zones_type': 'whole_zone', # options: whole_zone OR split_zone (this splits it into the 3 individual zones)
                        'wind_farms': False,
                        'legacy_farms': False
                    },

                'colours': {
                        'mussel': 'red',
                        'monopile': 'blue',
                        'jacket': 'teal',
                        'solar': 'orange',
                        'monopile_solar': 'purple',
                        'monopile_mussel': 'pink',
                        'wind': 'white'
                        #'hydrogen': 'black'
                    },

                'legend': [
                        Patch(color='#333333', alpha=1, lw=2, label='Shipping Routes'),
                        Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='EEZ'),
                        
                        plt.Line2D([0], [0], color='white', linewidth=0, label='----------------', linestyle='None') # Keep this line, it seperates the background files from the tech
                    ],
            }
             
    return map_CONGIF